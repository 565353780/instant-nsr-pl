"""
Demo script showing how to use NeuS trainer without PyTorch Lightning.
This demonstrates training NeuS on COLMAP dataset.
"""
import os
import torch

from instant_nsr.Model.neus import NeusModel
from instant_nsr.Module.trainer import Trainer, TrainerConfig, DatasetInfo
from instant_nsr.Dataset.colmap import load_colmap_dataset


def load_colmap_data(
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
        DatasetInfo object
    """
    data = load_colmap_dataset(
        root_dir=root_dir,
        img_wh=img_wh,
        img_downscale=img_downscale,
        up_est_method=up_est_method,
        center_est_method=center_est_method,
        apply_mask=apply_mask,
        device=device
    )
    
    return DatasetInfo(
        all_images=data['all_images'],
        all_c2w=data['all_c2w'],
        all_fg_masks=data['all_fg_masks'],
        directions=data['directions'],
        w=data['w'],
        h=data['h'],
        img_wh=data['img_wh'],
        has_mask=data['has_mask'],
        apply_mask=data['apply_mask']
    )


def build_model(config: dict, device='cuda'):
    """
    Build NeuS model from config.
    
    This function shows how to create the geometry and texture networks
    and combine them into a NeusModel.
    
    Note: This requires the original models module to be available.
    """
    # Import original models module for geometry and texture networks
    import models

    # Create geometry network
    geometry = models.make(config['geometry']['name'], config['geometry'])
    
    # Create texture network
    texture = models.make(config['texture']['name'], config['texture'])
    
    # Optional: background networks
    geometry_bg = None
    texture_bg = None
    if config.get('learned_background', False):
        geometry_bg = models.make(config['geometry_bg']['name'], config['geometry_bg'])
        texture_bg = models.make(config['texture_bg']['name'], config['texture_bg'])
    
    # Create NeusModel
    model = NeusModel(
        geometry=geometry,
        texture=texture,
        variance_config=config['variance'],
        radius=config.get('radius', 1.5),
        num_samples_per_ray=config.get('num_samples_per_ray', 1024),
        grid_prune=config.get('grid_prune', True),
        grid_prune_occ_thre=config.get('grid_prune_occ_thre', 0.001),
        randomized=config.get('randomized', True),
        ray_chunk=config.get('ray_chunk', 4096),
        cos_anneal_end=config.get('cos_anneal_end', 20000),
        learned_background=config.get('learned_background', False),
        geometry_bg=geometry_bg,
        texture_bg=texture_bg,
        num_samples_per_ray_bg=config.get('num_samples_per_ray_bg', 256),
        grid_prune_occ_thre_bg=config.get('grid_prune_occ_thre_bg', 0.01),
    )

    return model.to(device)


def demo():
    """Main training function demonstrating usage."""
    # Configuration
    device = torch.device('cuda:2')

    home = os.environ['HOME']

    # 小妖怪头
    shape_id="003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
    # 女人上半身
    shape_id="017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
    # 长发男人头
    shape_id="0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"

    dataset_root = home + "/chLi/Dataset/pixel_align/" + shape_id + "/colmap/"

    # Check if dataset exists
    if not os.path.exists(dataset_root):
        print(f"Dataset not found at {dataset_root}")
        print("Please ensure the COLMAP dataset directory exists with 'sparse/0/' and 'images/' subdirectories.")
        return
    
    print("Loading dataset...")
    train_dataset = load_colmap_data(
        root_dir=dataset_root,
        img_downscale=4,  # Downscale images by 4x
        up_est_method='camera',
        center_est_method='lookat',
        apply_mask=True,
        device=device
    )
    
    # Model configuration (simplified version of neus-blender.yaml)
    model_config = {
        'radius': 1.5,
        'num_samples_per_ray': 1024,
        'grid_prune': True,
        'grid_prune_occ_thre': 0.001,
        'randomized': True,
        'ray_chunk': 4096,
        'cos_anneal_end': 20000,
        'learned_background': False,
        'variance': {
            'init_val': 0.3,
            'modulate': False,
        },
        'geometry': {
            'name': 'volume-sdf',
            'radius': 1.5,
            'feature_dim': 13,
            'grad_type': 'analytic',
            'isosurface': {
                'method': 'mc',
                'resolution': 512,
                'chunk': 2097152,
                'threshold': 0.,
            },
            'xyz_encoding_config': {
                'otype': 'HashGrid',
                'n_levels': 16,
                'n_features_per_level': 2,
                'log2_hashmap_size': 19,
                'base_resolution': 32,
                'per_level_scale': 1.3195079107728942,
                'include_xyz': True,
            },
            'mlp_network_config': {
                'otype': 'VanillaMLP',
                'activation': 'ReLU',
                'output_activation': 'none',
                'n_neurons': 64,
                'n_hidden_layers': 1,
                'sphere_init': True,
                'sphere_init_radius': 0.5,
                'weight_norm': True,
            },
        },
        'texture': {
            'name': 'volume-radiance',
            'input_feature_dim': 16,  # feature_dim + 3 (normal)
            'dir_encoding_config': {
                'otype': 'SphericalHarmonics',
                'degree': 4,
            },
            'mlp_network_config': {
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'none',
                'n_neurons': 64,
                'n_hidden_layers': 2,
            },
            'color_activation': 'sigmoid',
        },
    }

    print("Building model...")
    model = build_model(model_config, device=device)

    # Trainer configuration
    trainer_config = TrainerConfig(
        max_steps=20000,
        log_every_n_steps=100,
        val_check_interval=5000,
        train_num_rays=256,
        max_train_num_rays=8192,
        dynamic_ray_sampling=True,
        batch_image_sampling=True,
        lambda_rgb_mse=10.0,
        lambda_rgb_l1=0.0,
        lambda_mask=0.1,
        lambda_eikonal=0.1,
        background_color='random',
        lr=0.01,
        lr_geometry=0.01,
        lr_texture=0.01,
        lr_variance=0.001,
        warmup_steps=500,
        use_amp=True,
        save_dir='./exp/neus_demo/save',
        ckpt_dir='./exp/neus_demo/ckpt',
    )
    
    # Create trainer
    trainer = Trainer(model=model, config=trainer_config, device=device)
    trainer.set_dataset(train_dataset)
    
    print("Starting training...")
    
    # Train with validation on first 2 images
    trainer.train(
        num_steps=trainer_config.max_steps,
        val_indices=[0, 1],
    )
    
    print("Training completed!")
    print(f"Results saved to {trainer_config.save_dir}")
    return True
