import os
import torch
import trimesh
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from datasets.colmap_utils import read_cameras_text, read_cameras_binary, read_images_text, read_images_binary


def get_center(pts):
    """Calculate center point with outlier filtering."""
    center = pts.mean(0)
    dis = (pts - center[None, :]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & \
            (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center


def normalize_poses(poses, pts, up_est_method='camera', center_est_method='lookat'):
    """
    Normalize camera poses and point cloud.
    
    Args:
        poses: Camera poses (N, 3, 4)
        pts: Point cloud (M, 3)
        up_est_method: 'ground' or 'camera'
        center_est_method: 'camera', 'lookat', or 'point'
    
    Returns:
        Normalized poses and point cloud
    """
    if center_est_method == 'camera':
        center = poses[..., 3].mean(0)
    elif center_est_method == 'lookat':
        cams_ori = poses[..., 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0., 0., -1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) * t[:, None, :] + 
                  torch.stack([cams_ori, cams_ori.roll(1, 0)], dim=-1)).mean((0, 2))
    elif center_est_method == 'point':
        center = poses[..., 3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01)
        plane_eq = torch.as_tensor(plane_eq)
        z = F.normalize(plane_eq[:3], dim=-1)
        signed_distance = (torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z
    elif up_est_method == 'camera':
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    # New axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # Rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0., 0., 0., 1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0., 0., 0.]]).T], dim=1), 
                               torch.as_tensor([[0., 0., 0., 1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None])[:, :3, 0]

        # Translation and scaling
        poses_min, poses_max = poses_norm[..., 3].min(0)[0], poses_norm[..., 3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:, 0]) & (pts[:, 0] < poses_max[0]) & 
                     (poses_min[1] < pts[:, 1]) & (pts[:, 1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0., 0., 0., 1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0., 0., 0., 1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None])[:, :3, 0]
        pts = pts / scale
    else:
        # Rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([poses, torch.as_tensor([[[0., 0., 0., 1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0., 0., 0., 1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:, :3]

        # Scaling
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale

        # Apply transformation to point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None])[:, :3, 0]
        pts = pts / scale

    return poses_norm, pts


def get_ray_directions(W, H, fx, fy, cx, cy):
    """Generate ray directions for all pixels."""
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    directions = torch.stack([
        (i - cx) / fx,
        -(j - cy) / fy,
        -torch.ones_like(i)
    ], dim=-1)  # (H, W, 3)
    return directions


def load_colmap_dataset(
    root_dir: str,
    img_wh: tuple = None,
    img_downscale: float = None,
    up_est_method: str = 'camera',
    center_est_method: str = 'lookat',
    apply_mask: bool = True,
    device: str = 'cuda'
):
    """
    Load COLMAP dataset.
    
    Args:
        root_dir: Path to dataset root (should contain 'sparse/0/' and 'images/')
        img_wh: Image width and height tuple, e.g., (800, 800)
        img_downscale: Downscale factor for images (alternative to img_wh)
        up_est_method: Up direction estimation method ('camera' or 'ground')
        center_est_method: Center estimation method ('camera', 'lookat', or 'point')
        apply_mask: Whether to apply mask if available
        device: Device to load data to
    
    Returns:
        Dictionary containing dataset info
    """
    sparse_dir = os.path.join(root_dir, 'sparse/0')
    
    # Read camera data (try text format first, then binary)
    cameras_txt = os.path.join(sparse_dir, 'cameras.txt')
    cameras_bin = os.path.join(sparse_dir, 'cameras.bin')
    
    if os.path.exists(cameras_txt):
        camdata = read_cameras_text(cameras_txt)
    elif os.path.exists(cameras_bin):
        camdata = read_cameras_binary(cameras_bin)
    else:
        raise FileNotFoundError(f"No camera file found in {sparse_dir}")
    
    # Get first camera (assuming single camera)
    cam = camdata[list(camdata.keys())[0]]
    H = int(cam.height)
    W = int(cam.width)
    
    # Determine image size
    if img_wh is not None:
        w, h = img_wh
        assert round(W / w * h) == H, "img_wh aspect ratio doesn't match original images"
    elif img_downscale is not None:
        w, h = int(W / img_downscale + 0.5), int(H / img_downscale + 0.5)
    else:
        w, h = W, H
    
    factor = w / W
    
    # Parse camera intrinsics
    if cam.model == 'SIMPLE_RADIAL':
        fx = fy = cam.params[0] * factor
        cx = cam.params[1] * factor
        cy = cam.params[2] * factor
    elif cam.model in ['PINHOLE', 'OPENCV']:
        fx = cam.params[0] * factor
        fy = cam.params[1] * factor
        cx = cam.params[2] * factor
        cy = cam.params[3] * factor
    elif cam.model == 'SIMPLE_PINHOLE':
        fx = fy = cam.params[0] * factor
        cx = cam.params[1] * factor
        cy = cam.params[2] * factor
    else:
        raise ValueError(f"Please parse the intrinsics for camera model {cam.model}!")
    
    # Generate ray directions
    directions = get_ray_directions(w, h, fx, fy, cx, cy).to(device)
    
    # Read image data
    images_txt = os.path.join(sparse_dir, 'images.txt')
    images_bin = os.path.join(sparse_dir, 'images.bin')
    
    if os.path.exists(images_txt):
        imdata = read_images_text(images_txt)
    elif os.path.exists(images_bin):
        imdata = read_images_binary(images_bin)
    else:
        raise FileNotFoundError(f"No images file found in {sparse_dir}")
    
    # Check for masks
    mask_dir = os.path.join(root_dir, 'masks')
    has_mask = os.path.exists(mask_dir)
    
    all_c2w, all_images, all_fg_masks = [], [], []
    
    for i, d in enumerate(imdata.values()):
        # Convert quaternion to rotation matrix and get camera pose
        R = d.qvec2rotmat()
        t = d.tvec.reshape(3, 1)
        c2w = torch.from_numpy(np.concatenate([R.T, -R.T @ t], axis=1)).float()
        c2w[:, 1:3] *= -1.  # COLMAP => OpenGL convention
        all_c2w.append(c2w)
        
        # Load image
        img_path = os.path.join(root_dir, 'images', d.name)
        if not os.path.exists(img_path):
            # Try alternative path
            img_path = os.path.join(root_dir, d.name)
        
        img = Image.open(img_path)
        img = img.resize((w, h), Image.BICUBIC)
        img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]  # (H, W, 3)
        all_images.append(img)
        
        # Load mask if available
        if has_mask:
            mask_paths = [
                os.path.join(mask_dir, d.name),
                os.path.join(mask_dir, os.path.splitext(d.name)[0] + '.png'),
                os.path.join(mask_dir, d.name.replace('.jpg', '.png')),
            ]
            if d.name.startswith('img'):
                mask_paths.append(os.path.join(mask_dir, d.name[3:]))
            
            mask_path = None
            for mp in mask_paths:
                if os.path.exists(mp):
                    mask_path = mp
                    break
            
            if mask_path:
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((w, h), Image.BICUBIC)
                mask = TF.to_tensor(mask)[0]  # (H, W)
            else:
                mask = torch.ones_like(img[..., 0])
        else:
            mask = torch.ones_like(img[..., 0])
        
        all_fg_masks.append(mask)
    
    all_c2w = torch.stack(all_c2w, dim=0)
    
    # Load point cloud for pose normalization
    pts3d_ply = os.path.join(sparse_dir, 'points3D.ply')
    pts3d_bin = os.path.join(sparse_dir, 'points3D.bin')
    pts3d_txt = os.path.join(sparse_dir, 'points3D.txt')
    
    if os.path.exists(pts3d_ply):
        pcd = trimesh.load(pts3d_ply)
        pts3d = torch.from_numpy(pcd.vertices).float()
    elif os.path.exists(pts3d_bin):
        from datasets.colmap_utils import read_points3d_binary
        pts3d_data = read_points3d_binary(pts3d_bin)
        pts3d = torch.from_numpy(np.array([p.xyz for p in pts3d_data.values()])).float()
    elif os.path.exists(pts3d_txt):
        from datasets.colmap_utils import read_points3D_text
        pts3d_data = read_points3D_text(pts3d_txt)
        pts3d = torch.from_numpy(np.array([p.xyz for p in pts3d_data.values()])).float()
    else:
        # If no point cloud, use camera positions
        pts3d = all_c2w[..., 3].clone()
    
    # Normalize poses
    all_c2w, pts3d = normalize_poses(all_c2w, pts3d, up_est_method=up_est_method, center_est_method=center_est_method)
    
    # Stack tensors and move to device
    all_c2w = all_c2w.float().to(device)
    all_images = torch.stack(all_images, dim=0).float().to(device)
    all_fg_masks = torch.stack(all_fg_masks, dim=0).float().to(device)
    
    return {
        'all_images': all_images,
        'all_c2w': all_c2w,
        'all_fg_masks': all_fg_masks,
        'directions': directions,
        'w': w,
        'h': h,
        'img_wh': (w, h),
        'has_mask': has_mask,
        'apply_mask': apply_mask and has_mask,
        'pts3d': pts3d,
    }
