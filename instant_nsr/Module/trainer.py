import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dataclasses import dataclass
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, List, Callable
from torch_efficient_distloss import flatten_eff_distloss

from instant_nsr.Model.neus import NeusModel
from instant_nsr.Metric.psnr import PSNR
from instant_nsr.Loss.bce import binary_cross_entropy


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    """Generate ray directions for all pixels."""
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)
    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)
    return directions


def get_rays(directions, c2w, keepdim=False):
    """Generate ray origins and directions from camera parameters."""
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        assert c2w.ndim == 3  # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)
        rays_o = c2w[:, :, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(-1)
            rays_o = c2w[None, None, :, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)
            rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""
    # Training settings
    max_steps: int = 20000
    log_every_n_steps: int = 100
    val_check_interval: int = 10000

    # Ray sampling
    train_num_rays: int = 256
    max_train_num_rays: int = 8192
    num_samples_per_ray: int = 1024
    num_samples_per_ray_bg: int = 256
    dynamic_ray_sampling: bool = True
    batch_image_sampling: bool = True

    # Loss weights
    lambda_rgb_mse: float = 10.0
    lambda_rgb_l1: float = 0.0
    lambda_mask: float = 0.1
    lambda_eikonal: float = 0.1
    lambda_curvature: float = 0.0
    lambda_sparsity: float = 0.0
    lambda_distortion: float = 0.0
    lambda_distortion_bg: float = 0.0
    lambda_opaque: float = 0.0
    sparsity_scale: float = 1.0

    # Model settings
    background_color: str = 'random'  # 'white' or 'random'
    learned_background: bool = False

    # Optimizer settings
    lr: float = 0.01
    lr_geometry: float = 0.01
    lr_texture: float = 0.01
    lr_variance: float = 0.001
    betas: tuple = (0.9, 0.99)
    eps: float = 1e-15
    warmup_steps: int = 500

    # Mixed precision
    use_amp: bool = True

    # Save settings
    save_dir: str = './exp'
    ckpt_dir: str = './ckpt'

    # Export settings
    export_chunk_size: int = 2097152
    export_vertex_color: bool = True
    isosurface_method: str = 'mc'
    isosurface_resolution: int = 512


@dataclass
class DatasetInfo:
    """Dataset information container."""
    all_images: torch.Tensor
    all_c2w: torch.Tensor
    all_fg_masks: torch.Tensor
    directions: torch.Tensor
    w: int
    h: int
    img_wh: tuple
    has_mask: bool = True
    apply_mask: bool = True


class Trainer:
    """
    NeuS Trainer without PyTorch Lightning dependency.
    Provides complete training loop, validation, and mesh export functionality.
    """
    def __init__(
        self,
        model: NeusModel,
        config: TrainerConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.train_num_rays = config.train_num_rays
        self.train_num_samples = config.train_num_rays * (
            config.num_samples_per_ray + (config.num_samples_per_ray_bg if config.learned_background else 0)
        )
        
        # Metrics
        self.psnr = PSNR()
        
        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config.use_amp else None
        
        # Dataset
        self.dataset: Optional[DatasetInfo] = None
        
        # Logging
        self.log_dict = {}
        
        # Create directories
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.ckpt_dir, exist_ok=True)

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        param_groups = [
            {'params': self.model.geometry.parameters(), 'lr': self.config.lr_geometry, 'name': 'geometry'},
            {'params': self.model.texture.parameters(), 'lr': self.config.lr_texture, 'name': 'texture'},
            {'params': self.model.variance.parameters(), 'lr': self.config.lr_variance, 'name': 'variance'},
        ]
        
        if self.config.learned_background:
            param_groups.extend([
                {'params': self.model.geometry_bg.parameters(), 'lr': self.config.lr_geometry, 'name': 'geometry_bg'},
                {'params': self.model.texture_bg.parameters(), 'lr': self.config.lr_texture, 'name': 'texture_bg'},
            ])
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.lr,
            betas=self.config.betas,
            eps=self.config.eps
        )
        
        # Linear warmup + exponential decay
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return 0.01 + 0.99 * step / self.config.warmup_steps
            else:
                decay_rate = 0.1 ** (1.0 / (self.config.max_steps - self.config.warmup_steps))
                return decay_rate ** (step - self.config.warmup_steps)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def set_dataset(self, dataset_info: DatasetInfo):
        """Set the dataset for training."""
        self.dataset = dataset_info

    def preprocess_data(self, stage: str = 'train') -> Dict[str, torch.Tensor]:
        """Preprocess data for training or validation."""
        assert self.dataset is not None, "Dataset not set. Call set_dataset() first."
        
        if stage == 'train':
            # Random sampling for training
            if self.config.batch_image_sampling:
                index = torch.randint(
                    0, len(self.dataset.all_images), 
                    size=(self.train_num_rays,), 
                    device=self.dataset.all_images.device
                )
            else:
                index = torch.randint(
                    0, len(self.dataset.all_images), 
                    size=(1,), 
                    device=self.dataset.all_images.device
                )
            
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device)
            y = torch.randint(0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device)
            
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.device)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.device)
            
            # Set background color
            if self.config.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.device)
            elif self.config.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.device)
            else:
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.device)
        else:
            # Full image for validation/test
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.device)
            return None  # Return None for validation, handle separately
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[..., None] + self.model.background_color * (1 - fg_mask[..., None])
        
        return {
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask
        }

    def preprocess_validation_data(self, index: int) -> Dict[str, torch.Tensor]:
        """Preprocess data for a single validation image."""
        assert self.dataset is not None, "Dataset not set."
        
        self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.device)
        
        c2w = self.dataset.all_c2w[index]
        if self.dataset.directions.ndim == 3:
            directions = self.dataset.directions
        elif self.dataset.directions.ndim == 4:
            directions = self.dataset.directions[index]
        
        rays_o, rays_d = get_rays(directions, c2w)
        rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.device)
        fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.device)
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[..., None] + self.model.background_color * (1 - fg_mask[..., None])
        
        return {
            'rays': rays.to(self.device),
            'rgb': rgb,
            'fg_mask': fg_mask,
            'index': index
        }

    def get_loss_weight(self, value):
        """Get loss weight (potentially scheduled)."""
        if isinstance(value, (int, float)):
            return value
        elif isinstance(value, list) and len(value) >= 3:
            if len(value) == 3:
                value = [0] + value
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
            else:
                current_step = self.current_epoch
            progress = max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            return start_value + (end_value - start_value) * progress
        return value

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute a single training step."""
        out = self.model(batch['rays'])
        
        loss = torch.tensor(0.0, device=self.device)
        
        # Dynamic ray sampling
        if self.config.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))
            self.train_num_rays = min(
                int(self.train_num_rays * 0.9 + train_num_rays * 0.1), 
                self.config.max_train_num_rays
            )
        
        # RGB MSE loss
        valid_mask = out['rays_valid_full'][..., 0]
        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][valid_mask], batch['rgb'][valid_mask])
        self.log_dict['train/loss_rgb_mse'] = loss_rgb_mse.item()
        loss = loss + loss_rgb_mse * self.get_loss_weight(self.config.lambda_rgb_mse)
        
        # RGB L1 loss
        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][valid_mask], batch['rgb'][valid_mask])
        self.log_dict['train/loss_rgb_l1'] = loss_rgb_l1.item()
        loss = loss + loss_rgb_l1 * self.get_loss_weight(self.config.lambda_rgb_l1)
        
        # Eikonal loss
        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.) ** 2).mean()
        self.log_dict['train/loss_eikonal'] = loss_eikonal.item()
        loss = loss + loss_eikonal * self.get_loss_weight(self.config.lambda_eikonal)
        
        # Mask loss
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1e-3, 1. - 1e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log_dict['train/loss_mask'] = loss_mask.item()
        if self.dataset.has_mask:
            loss = loss + loss_mask * self.get_loss_weight(self.config.lambda_mask)
        
        # Opaque loss
        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log_dict['train/loss_opaque'] = loss_opaque.item()
        loss = loss + loss_opaque * self.get_loss_weight(self.config.lambda_opaque)
        
        # Sparsity loss
        loss_sparsity = torch.exp(-self.config.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log_dict['train/loss_sparsity'] = loss_sparsity.item()
        loss = loss + loss_sparsity * self.get_loss_weight(self.config.lambda_sparsity)
        
        # Curvature loss
        lambda_curvature = self.get_loss_weight(self.config.lambda_curvature)
        if lambda_curvature > 0 and 'sdf_laplace_samples' in out:
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log_dict['train/loss_curvature'] = loss_curvature.item()
            loss = loss + loss_curvature * lambda_curvature
        
        # Distortion loss
        lambda_distortion = self.get_loss_weight(self.config.lambda_distortion)
        if lambda_distortion > 0:
            loss_distortion = flatten_eff_distloss(
                out['weights'], out['points'], out['intervals'], out['ray_indices']
            )
            self.log_dict['train/loss_distortion'] = loss_distortion.item()
            loss = loss + loss_distortion * lambda_distortion
        
        # Background distortion loss
        if self.config.learned_background:
            lambda_distortion_bg = self.get_loss_weight(self.config.lambda_distortion_bg)
            if lambda_distortion_bg > 0:
                loss_distortion_bg = flatten_eff_distloss(
                    out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg']
                )
                self.log_dict['train/loss_distortion_bg'] = loss_distortion_bg.item()
                loss = loss + loss_distortion_bg * lambda_distortion_bg
        
        # Model regularizations
        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log_dict[f'train/loss_{name}'] = value.item()
            loss = loss + value * self.get_loss_weight(getattr(self.config, f'lambda_{name}', 0.0))
        
        self.log_dict['train/inv_s'] = out['inv_s'].item()
        self.log_dict['train/num_rays'] = float(self.train_num_rays)
        
        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute a single validation step."""
        self.model.eval()
        out = self.model(batch['rays'])
        
        psnr = self.psnr(out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        
        W, H = self.dataset.img_wh
        
        result = {
            'psnr': psnr.item(),
            'index': batch['index'],
            'rgb_gt': batch['rgb'].view(H, W, 3),
            'rgb_pred': out['comp_rgb_full'].view(H, W, 3),
            'depth': out['depth'].view(H, W),
            'normal': out['comp_normal'].view(H, W, 3),
        }
        
        if self.config.learned_background:
            result['rgb_bg'] = out['comp_rgb_bg'].view(H, W, 3)
            result['rgb_fg'] = out['comp_rgb'].view(H, W, 3)
        
        self.model.train()
        return result

    def save_image(self, filename: str, img: torch.Tensor, data_format='HWC', data_range=(0, 1)):
        """Save image to file."""
        img = img.cpu().numpy()
        if data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = np.clip(img, data_range[0], data_range[1])
        img = ((img - data_range[0]) / (data_range[1] - data_range[0]) * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        save_path = os.path.join(self.config.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    def save_depth(self, filename: str, depth: torch.Tensor):
        """Save depth map to file."""
        depth = depth.cpu().numpy()
        depth = np.nan_to_num(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = (depth * 255.).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        
        save_path = os.path.join(self.config.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, depth)

    def save_normal(self, filename: str, normal: torch.Tensor):
        """Save normal map to file."""
        normal = normal.cpu().numpy()
        normal = (normal + 1) / 2  # Map from [-1, 1] to [0, 1]
        normal = (normal * 255.).astype(np.uint8)
        normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        
        save_path = os.path.join(self.config.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, normal)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_num_rays': self.train_num_rays,
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = os.path.join(self.config.ckpt_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.train_num_rays = checkpoint.get('train_num_rays', self.config.train_num_rays)
        
        print(f"Checkpoint loaded from {filepath}, resuming from step {self.global_step}")

    def export_mesh(self, filename: str = None):
        """Export mesh from trained model."""
        print("Starting mesh export...")
        
        self.model.eval()
        export_config = {
            'chunk_size': self.config.export_chunk_size,
            'export_vertex_color': self.config.export_vertex_color,
        }
        
        mesh = self.model.export(export_config)
        
        if filename is None:
            filename = f"it{self.global_step}-{self.config.isosurface_method}{self.config.isosurface_resolution}.obj"
        
        save_path = os.path.join(self.config.save_dir, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        import trimesh
        trimesh_mesh = trimesh.Trimesh(
            vertices=mesh['v_pos'].cpu().numpy(),
            faces=mesh['t_pos_idx'].cpu().numpy(),
            vertex_colors=mesh.get('v_rgb', None).cpu().numpy() if mesh.get('v_rgb') is not None else None
        )
        trimesh_mesh.export(save_path)
        
        print(f"Mesh exported to {save_path}")
        self.model.train()
        return save_path

    def train(
        self,
        num_steps: int = None,
        val_indices: List[int] = None,
        progress_callback: Callable = None,
    ):
        """
        Main training loop.
        
        Args:
            num_steps: Number of training steps. Defaults to config.max_steps.
            val_indices: Indices of validation images.
            progress_callback: Optional callback for progress updates.
        """
        if num_steps is None:
            num_steps = self.config.max_steps
        
        if self.optimizer is None:
            self.setup_optimizer()
        
        self.model.train()
        
        pbar = tqdm(range(self.global_step, num_steps), desc="Training")
        
        for step in pbar:
            self.global_step = step
            
            # Update model step
            self.model.update_step(self.current_epoch, self.global_step)
            
            # Prepare batch
            batch = self.preprocess_data('train')
            
            # Forward and backward
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler is not None:
                with autocast():
                    result = self.training_step(batch)
                    loss = result['loss']
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                result = self.training_step(batch)
                loss = result['loss']
                loss.backward()
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            if step % self.config.log_every_n_steps == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{lr:.2e}",
                    'rays': self.train_num_rays,
                    'inv_s': f"{self.log_dict.get('train/inv_s', 0):.2f}"
                })
                
                if progress_callback:
                    progress_callback(step, self.log_dict)
            
            # Validation
            if val_indices and step > 0 and step % self.config.val_check_interval == 0:
                self.validate(val_indices)
            
            # Save checkpoint
            if step > 0 and step % self.config.val_check_interval == 0:
                self.save_checkpoint(f"step_{step}.ckpt")
        
        # Final validation and export
        if val_indices:
            self.validate(val_indices)
        
        self.save_checkpoint("final.ckpt")
        self.export_mesh()
        
        print("Training completed!")

    def validate(self, indices: List[int]):
        """Run validation on specified image indices."""
        print(f"Running validation on {len(indices)} images...")
        
        psnr_values = []
        
        for idx in indices:
            batch = self.preprocess_validation_data(idx)
            result = self.validation_step(batch)
            psnr_values.append(result['psnr'])
            
            # Save validation images
            self.save_image(
                f"val_it{self.global_step}_{idx}_gt.png",
                result['rgb_gt']
            )
            self.save_image(
                f"val_it{self.global_step}_{idx}_pred.png",
                result['rgb_pred']
            )
            self.save_depth(
                f"val_it{self.global_step}_{idx}_depth.png",
                result['depth']
            )
            self.save_normal(
                f"val_it{self.global_step}_{idx}_normal.png",
                result['normal']
            )
        
        mean_psnr = np.mean(psnr_values)
        print(f"Validation PSNR: {mean_psnr:.2f} dB")
        self.log_dict['val/psnr'] = mean_psnr
        
        return mean_psnr
