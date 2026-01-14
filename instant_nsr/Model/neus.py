import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from nerfacc import (
    ContractionType, OccupancyGrid, ray_marching, 
    render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
)
from nerfacc.intersection import ray_aabb_intersect


def update_module_step(m, epoch, global_step):
    """Update module step for modules that have update_step method."""
    if hasattr(m, 'update_step'):
        m.update_step(epoch, global_step)


class VarianceNetwork(nn.Module):
    """Variance network for SDF to density conversion."""
    def __init__(self, init_val: float = 0.3, modulate: bool = False, 
                 mod_start_steps: int = 0, reach_max_steps: int = 0, max_inv_s: float = 0.0):
        super(VarianceNetwork, self).__init__()
        self.init_val = init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
        self.modulate = modulate
        if self.modulate:
            self.mod_start_steps = mod_start_steps
            self.reach_max_steps = reach_max_steps
            self.max_inv_s = max_inv_s
            self.do_mod = False
            self.prev_inv_s = 0.0
            self.mod_val = 0.0

    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s

    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min(
                    (global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, 
                    self.max_inv_s
                )

def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    """Process batch in chunks to avoid OOM."""
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    if B is None:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                B = v.shape[0]
                break
    assert B is not None
    
    out = {}
    out_type = None
    for i in range(0, B, chunk_size):
        args_chunk = [arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs_chunk = {k: v[i:i+chunk_size] if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        out_chunk = func(*args_chunk, **kwargs_chunk)
        if out_type is None:
            out_type = type(out_chunk)
        if isinstance(out_chunk, dict):
            for k, v in out_chunk.items():
                if k not in out:
                    out[k] = []
                v_ = v.cpu() if move_to_cpu and isinstance(v, torch.Tensor) else v
                out[k].append(v_)
        elif isinstance(out_chunk, torch.Tensor):
            if 'tensor' not in out:
                out['tensor'] = []
            v_ = out_chunk.cpu() if move_to_cpu else out_chunk
            out['tensor'].append(v_)

    if out_type == dict:
        return {k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v for k, v in out.items()}
    elif out_type == torch.Tensor:
        return torch.cat(out['tensor'], dim=0)


class NeusModel(nn.Module):
    """
    NeuS Model without PyTorch Lightning dependency.
    This is the core rendering model that combines geometry and texture networks.
    """
    def __init__(
        self,
        geometry: nn.Module,
        texture: nn.Module,
        variance_config: Dict[str, Any],
        radius: float = 1.5,
        num_samples_per_ray: int = 1024,
        grid_prune: bool = True,
        grid_prune_occ_thre: float = 0.001,
        randomized: bool = True,
        ray_chunk: int = 4096,
        cos_anneal_end: int = 20000,
        learned_background: bool = False,
        geometry_bg: Optional[nn.Module] = None,
        texture_bg: Optional[nn.Module] = None,
        num_samples_per_ray_bg: int = 256,
        grid_prune_occ_thre_bg: float = 0.01,
    ):
        super().__init__()
        
        self.geometry = geometry
        self.texture = texture
        self.geometry.contraction_type = ContractionType.AABB
        
        self.learned_background = learned_background
        if self.learned_background:
            self.geometry_bg = geometry_bg
            self.texture_bg = texture_bg
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg = 0.1
            self.far_plane_bg = 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01
        
        # Build variance network
        self.variance = VarianceNetwork(
            init_val=variance_config.get('init_val', 0.3),
            modulate=variance_config.get('modulate', False),
            mod_start_steps=variance_config.get('mod_start_steps', 0),
            reach_max_steps=variance_config.get('reach_max_steps', 0),
            max_inv_s=variance_config.get('max_inv_s', 0.0),
        )
        
        self.radius = radius
        self.num_samples_per_ray = num_samples_per_ray
        self.grid_prune = grid_prune
        self.grid_prune_occ_thre = grid_prune_occ_thre
        self.grid_prune_occ_thre_bg = grid_prune_occ_thre_bg
        self.randomized = randomized
        self.ray_chunk = ray_chunk
        self.cos_anneal_end = cos_anneal_end
        self.cos_anneal_ratio = 1.0
        
        # Scene bounding box
        self.register_buffer(
            'scene_aabb', 
            torch.tensor([-radius, -radius, -radius, radius, radius, radius], dtype=torch.float32)
        )
        
        # Occupancy grid for acceleration
        if self.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
            if self.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        
        self.render_step_size = 1.732 * 2 * radius / num_samples_per_ray
        self.background_color = None
        
        # Config for geometry grad type
        self._geometry_grad_type = getattr(self.geometry, 'grad_type', 'analytic')
    
    def update_step(self, epoch: int, global_step: int):
        """Update model state based on training progress."""
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)
        
        self.cos_anneal_ratio = 1.0 if self.cos_anneal_end == 0 else min(1.0, global_step / self.cos_anneal_end)
        
        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[..., None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[..., None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            return density[..., None] * self.render_step_size_bg
        
        if self.training and self.grid_prune:
            self.occupancy_grid.every_n_step(
                step=global_step, 
                occ_eval_fn=occ_eval_fn, 
                occ_thre=self.grid_prune_occ_thre
            )
            if self.learned_background:
                self.occupancy_grid_bg.every_n_step(
                    step=global_step, 
                    occ_eval_fn=occ_eval_fn_bg, 
                    occ_thre=self.grid_prune_occ_thre_bg
                )
    
    def get_alpha(self, sdf, normal, dirs, dists):
        """Convert SDF to alpha using the NeuS formulation."""
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(sdf.shape[0], 1)
        
        true_cos = (dirs * normal).sum(-1, keepdim=True)
        
        # Cos anneal strategy for better convergence at early training
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)
        
        estimated_next_sdf = sdf[..., None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[..., None] - iter_cos * dists.reshape(-1, 1) * 0.5
        
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        
        p = prev_cdf - next_cdf
        c = prev_cdf
        
        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha
    
    def forward_bg_(self, rays):
        """Forward pass for background rendering."""
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        
        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry_bg(positions)
            return density[..., None]
        
        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None,
                grid=self.occupancy_grid_bg if self.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=near_plane, 
                far_plane=self.far_plane_bg,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        intervals = t_ends - t_starts
        
        density, feature = self.geometry_bg(positions)
        rgb = self.texture_bg(feature, t_dirs)
        
        weights = render_weight_from_density(t_starts, t_ends, density[..., None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)
        
        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }
        
        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })
        
        return out
    
    def forward_(self, rays):
        """Single forward pass for ray rendering."""
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
        
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.grid_prune else None,
                alpha_fn=None,
                near_plane=None, 
                far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts
        
        if self._geometry_grad_type == 'finite_difference':
            sdf, sdf_grad, feature, sdf_laplace = self.geometry(
                positions, with_grad=True, with_feature=True, with_laplace=True
            )
        else:
            sdf, sdf_grad, feature = self.geometry(positions, with_grad=True, with_feature=True)
            sdf_laplace = None
        
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[..., None]
        rgb = self.texture(feature, t_dirs, normal)
        
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        
        out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }
        
        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1)
            })
            if sdf_laplace is not None:
                out.update({'sdf_laplace_samples': sdf_laplace})
        
        # Background
        if self.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None, :].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }
        
        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }
        
        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }
    
    def forward(self, rays):
        """Forward pass with optional chunking for inference."""
        if self.training:
            out = self.forward_(rays)
        else:
            out = chunk_batch(self.forward_, self.ray_chunk, True, rays)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }
    
    def train(self, mode=True):
        self.randomized = mode and True  # Enable randomization only during training
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        """Compute regularization losses."""
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses
    
    def isosurface(self):
        """Extract isosurface mesh."""
        mesh = self.geometry.isosurface()
        return mesh
    
    @torch.no_grad()
    def export(self, export_config: Dict[str, Any]):
        """Export mesh with optional vertex colors."""
        mesh = self.isosurface()
        if export_config.get('export_vertex_color', True):
            device = next(self.parameters()).device
            _, sdf_grad, feature = chunk_batch(
                self.geometry, 
                export_config.get('chunk_size', 2097152), 
                False, 
                mesh['v_pos'].to(device), 
                with_grad=True, 
                with_feature=True
            )
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            rgb = self.texture(feature, -normal, normal)
            mesh['v_rgb'] = rgb.cpu()
        return mesh
