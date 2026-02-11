"""
Setup Helpers - Reusable utilities for notebooks and scripts

Consolidates common setup patterns to eliminate code duplication.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

logger = logging.getLogger(__name__)


def setup_config(config_path: str, auto_resolve_paths: bool = True) -> Dict[str, Any]:
    """
    Load and prepare configuration with automatic path resolution.
    
    Args:
        config_path: Path to YAML config file
        auto_resolve_paths: Automatically resolve relative paths to absolute
        
    Returns:
        Configuration dictionary with resolved paths
    """
    from .config import load_config, validate_config, get_project_root
    
    # Load config
    config = load_config(config_path)
    
    if auto_resolve_paths:
        root = get_project_root()
        
        # Resolve data root directory
        if 'data' in config and 'root_dir' in config['data']:
            data_root = Path(config['data']['root_dir'])
            if not data_root.is_absolute():
                config['data']['root_dir'] = str(root / data_root)
        
        # Resolve pretrained weights path
        if 'model' in config and 'encoder' in config['model']:
            encoder_cfg = config['model']['encoder']
            if 'pretrained_path' in encoder_cfg and encoder_cfg['pretrained_path']:
                weight_path = Path(encoder_cfg['pretrained_path'])
                if not weight_path.is_absolute():
                    # Handle ./ prefix
                    path_str = str(weight_path)
                    if path_str.startswith('./'):
                        path_str = path_str[2:]
                    config['model']['encoder']['pretrained_path'] = str(root / path_str)
    
    # Validate
    validate_config(config)
    
    # Cast scheduler params to correct types (YAML may return strings)
    if 'training' in config and 'scheduler' in config['training']:
        sched = config['training']['scheduler']
        if 'T_0' in sched:
            sched['T_0'] = int(sched['T_0'])
        if 'T_mult' in sched:
            sched['T_mult'] = int(sched['T_mult'])
        if 'eta_min' in sched:
            sched['eta_min'] = float(sched['eta_min'])
    
    return config


def setup_device(verbose: bool = True) -> str:
    """
    Setup and display device information.
    
    Args:
        verbose: Print device information
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"Using device: {device}")
        
        if device == 'cuda':
            print(f"\nGPU Information:")
            print(f"  Name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            used_mem = torch.cuda.memory_allocated(0) / 1e9
            print(f"  Total VRAM: {total_mem:.2f} GB")
            print(f"  Available VRAM: {total_mem - used_mem:.2f} GB")
            
            # Clear cache
            torch.cuda.empty_cache()
            print("\n✓ GPU cache cleared")
        else:
            print("⚠️ GPU not available. Training will be slow on CPU.")
    
    return device


def create_model_from_config(config: Dict[str, Any], device: str = 'cuda', verbose: bool = True):
    """
    Create and setup model from configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to move model to
        verbose: Print model information
        
    Returns:
        Model instance on specified device
    """
    from ..models import SatMAERestoration
    
    if verbose:
        print("Creating model...")
    
    model = SatMAERestoration(config)
    model = model.to(device)
    
    if verbose:
        # Print parameter counts
        params = model.count_parameters()
        print(f"\nModel Parameters:")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable: {params['trainable']:,} ({params['trainable_percent']:.1f}%)")
        print(f"  Frozen: {params['frozen']:,}")
        
        # Profile memory usage
        if device == 'cuda':
            print("\nProfiling VRAM usage...")
            memory_stats = model.profile_memory(batch_size=config['training']['micro_batch_size'])
            print(f"  Model weights: {memory_stats['model_weights_gb']:.2f} GB")
            print(f"  Optimizer states: {memory_stats['optimizer_gb']:.2f} GB")
            print(f"  Activations: {memory_stats['activations_gb']:.2f} GB")
            print(f"  Gradients: {memory_stats['gradients_gb']:.2f} GB")
            print(f"  Peak VRAM: {memory_stats['peak_vram_gb']:.2f} GB")
            
            if memory_stats['peak_vram_gb'] > 5.5:
                print("\n⚠️ WARNING: Estimated VRAM exceeds 5.5GB!")
                print("   Consider reducing micro_batch_size or enabling more optimizations.")
            else:
                print(f"\n✓ VRAM usage OK ({memory_stats['peak_vram_gb']:.2f}GB / 6GB available)")
    
    return model


def create_training_components(model, config: Dict[str, Any], verbose: bool = True):
    """
    Create optimizer, scheduler, and loss function from config.
    
    Args:
        model: Model instance
        config: Configuration dictionary
        verbose: Print component information
        
    Returns:
        Tuple of (optimizer, scheduler, criterion)
    """
    from ..training import CombinedLoss
    
    if verbose:
        print("Setting up training components...")
    
    # Create optimizer with parameter groups
    param_groups = model.get_optimizer_param_groups(
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    optimizer = AdamW(
        param_groups,
        betas=config['training']['optimizer']['betas']
    )
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config['training']['scheduler']['T_0']),
        T_mult=int(config['training']['scheduler']['T_mult']),
        eta_min=float(config['training']['scheduler']['eta_min'])
    )
    
    # Create loss function
    criterion = CombinedLoss(
        mse_weight=config['training']['loss']['mse_weight'],
        ssim_weight=config['training']['loss']['ssim_weight']
    )
    
    if verbose:
        print(f"  Optimizer: AdamW")
        print(f"  Param groups: {len(param_groups)}")
        print(f"  Scheduler: CosineAnnealingWarmRestarts")
        print(f"  Loss: MSE ({config['training']['loss']['mse_weight']}) + SSIM ({config['training']['loss']['ssim_weight']})")
        print("\n✓ Training components ready")
    
    return optimizer, scheduler, criterion


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: str = 'cuda',
    load_optimizer: bool = False,
    optimizer: Optional[Any] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint with optional optimizer state.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer instance (required if load_optimizer=True)
        verbose: Print loading information
        
    Returns:
        Checkpoint dictionary with metadata
    """
    if verbose:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Note: weights_only=False is used because checkpoints often contain numpy scalars
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if verbose:
        print(f"✓ Model loaded from: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        if 'val_psnr' in checkpoint:
            print(f"  Val PSNR: {checkpoint['val_psnr']:.2f} dB")
        if 'val_ssim' in checkpoint:
            print(f"  Val SSIM: {checkpoint['val_ssim']:.4f}")
    
    return checkpoint


def setup_training_session(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Any, Any, Any, Any, str, Dict[str, Any]]:
    """
    Complete training session setup in one call.
    
    Combines all setup steps:
    1. Load and resolve config
    2. Setup device
    3. Create model
    4. Create training components
    5. Optionally load checkpoint
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Optional checkpoint to resume from
        device: Device to use (auto-detected if None)
        verbose: Print setup information
        
    Returns:
        Tuple of (model, optimizer, scheduler, criterion, device, config)
    """
    # Load config
    config = setup_config(config_path, auto_resolve_paths=True)
    
    # Setup device
    if device is None:
        device = setup_device(verbose=verbose)
    
    # Create model
    model = create_model_from_config(config, device=device, verbose=verbose)
    
    # Create training components
    optimizer, scheduler, criterion = create_training_components(model, config, verbose=verbose)
    
    # Load checkpoint if provided
    if checkpoint_path:
        load_checkpoint(
            model, checkpoint_path, device=device,
            load_optimizer=True, optimizer=optimizer, verbose=verbose
        )
    
    return model, optimizer, scheduler, criterion, device, config


def print_config_summary(config: Dict[str, Any]):
    """
    Print a formatted summary of key configuration settings.
    
    Args:
        config: Configuration dictionary
    """
    print("\nConfiguration Summary:")
    print("=" * 60)
    
    # Training settings
    if 'training' in config:
        print("Training:")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Micro Batch Size: {config['training']['micro_batch_size']}")
        if 'gradient_accumulation_steps' in config['training']:
            eff_batch = config['training']['micro_batch_size'] * config['training']['gradient_accumulation_steps']
            print(f"  Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")
            print(f"  Effective Batch Size: {eff_batch}")
        print(f"  Mixed Precision: {config['training'].get('mixed_precision', False)}")
        if 'optimizer' in config['training']:
            print(f"  Learning Rate: {config['training']['optimizer']['lr']}")
    
    # Model settings
    if 'model' in config:
        print("\nModel:")
        print(f"  Image Size: {config['data']['image_size']}")
        print(f"  Input Bands: {config['data']['num_bands']}")
        if 'encoder' in config['model']:
            print(f"  Encoder Dim: {config['model']['encoder'].get('embed_dim', 768)}")
            freeze = config['model']['encoder'].get('freeze_layers', [])
            print(f"  Frozen Layers: {len(freeze)} layers")
            print(f"  Gradient Checkpointing: {config['model']['encoder'].get('gradient_checkpointing', False)}")
    
    # Noise settings
    if 'noise' in config:
        print("\nNoise Simulation:")
        print(f"  Gaussian σ: {config['noise']['gaussian_sigma']}")
        print(f"  Speckle σ: {config['noise']['speckle_sigma']}")
        print(f"  Dead Band Prob: {config['noise']['dead_band_prob']}")
    
    print("=" * 60)
