"""
Main Training Script for SatMAE Multi-Spectral Denoiser

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/experiments/low_noise.yaml --resume
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.data import get_dataloaders
from src.models import SatMAERestoration
from src.training import Trainer, CombinedLoss
from src.utils import load_config, CheckpointManager, validate_config
from src.utils.download import setup_project_data, verify_downloads

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('outputs/logs/training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SatMAE Restoration Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to specific checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to train on'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("SatMAE Multi-Spectral Denoiser Training")
    logger.info("=" * 80)
    
    # Auto-download data if not present
    logger.info("Verifying project data...")
    status = verify_downloads()
    
    if not status['dataset'] or not status['weights']:
        logger.info("Missing required data. Starting automatic download...")
        try:
            paths = setup_project_data()
            logger.info("✓ Data setup complete!")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            logger.error("Please download manually:")
            logger.error("  Dataset: https://madm.dfki.de/files/sentinel/EuroSATallBands.zip")
            logger.error("  Weights: https://zenodo.org/record/7338613/files/pretrain-vit-base-e199.pth")
            logger.error("  Extract dataset to: ./data/EuroSAT_MS/")
            logger.error("  Place weights at: ./weights/satmae_pretrain.pth")
            raise
    
    logger.info("=" * 80)
    logger.info("Using TRUE 13-band Multi-Spectral Dataset!")
    logger.info("Dataset: EuroSAT Sentinel-2 (13 bands, 64x64)")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    model = SatMAERestoration(config)
    
    # Print parameter counts
    params = model.count_parameters()
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {params['total']:,}")
    logger.info(f"  Trainable: {params['trainable']:,} ({params['trainable_percent']:.1f}%)")
    
    # Profile memory usage
    if device == 'cuda':
        logger.info("Profiling VRAM usage...")
        memory_stats = model.profile_memory(batch_size=config['training']['micro_batch_size'])
        logger.info(f"  Model weights: {memory_stats['model_weights_gb']:.2f} GB")
        logger.info(f"  Peak VRAM: {memory_stats['peak_vram_gb']:.2f} GB")
    
    # Create optimizer
    logger.info("Creating optimizer...")
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
        T_0=config['training']['scheduler']['T_0'],
        T_mult=config['training']['scheduler']['T_mult'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # Create loss function
    criterion = CombinedLoss(
        mse_weight=config['training']['loss']['mse_weight'],
        ssim_weight=config['training']['loss']['ssim_weight']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        scheduler=scheduler,
        device=device
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir='outputs/checkpoints',
        keep_top_k=config['checkpoint']['keep_top_k'],
        metric=config['checkpoint']['metric'],
        mode='max' if 'psnr' in config['checkpoint']['metric'] else 'min'
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume or args.checkpoint:
        if args.checkpoint:
            logger.info(f"Resuming from checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        else:
            logger.info("Resuming from latest checkpoint...")
            checkpoint_manager.load_latest(model, optimizer, scheduler, device=device)
        
        start_epoch = trainer.current_epoch + 1
    
    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        save_dir='outputs/checkpoints'
    )
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Best validation {config['checkpoint']['metric']}: {trainer.best_val_metric:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
