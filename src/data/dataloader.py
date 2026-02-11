"""
DataLoader Factory

Creates train/val dataloaders with appropriate configurations
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import logging

from .dataset import EuroSATMultiSpectral
from .transforms import AddSensorNoise

logger = logging.getLogger(__name__)


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        config: Configuration dictionary from YAML
    
    Returns:
        train_loader, val_loader
    """
    # Create noise transform
    noise_transform = AddSensorNoise(
        gaussian_sigma=config['noise']['gaussian_sigma'],
        speckle_sigma=config['noise']['speckle_sigma'],
        dead_band_prob=config['noise']['dead_band_prob'],
        thermal_scale=config['noise']['thermal_noise_scale']
    )
    
    logger.info(f"Noise transform: {noise_transform}")
    
    # Create datasets
    train_dataset = EuroSATMultiSpectral(
        root_dir=config['data']['root_dir'],
        split='train',
        target_size=config['data']['image_size'],
        noise_transform=noise_transform,
        normalize=True,
        train_split=config['data']['train_split']
    )
    
    val_dataset = EuroSATMultiSpectral(
        root_dir=config['data']['root_dir'],
        split='val',
        target_size=config['data']['image_size'],
        noise_transform=noise_transform,
        normalize=True,
        train_split=config['data']['train_split']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['micro_batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True,  # For consistent gradient accumulation
        persistent_workers=True if config['data']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['micro_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    logger.info(
        f"Created dataloaders: "
        f"train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples, "
        f"batch_size={config['training']['micro_batch_size']}"
    )
    
    return train_loader, val_loader


def get_test_dataloader(
    test_dir: str, 
    config: Dict,
    batch_size: int = 1
) -> DataLoader:
    """
    Create test dataloader (no noise injection)
    
    Args:
        test_dir: Path to test dataset
        config: Configuration dictionary
        batch_size: Batch size for inference
    
    Returns:
        test_loader
    """
    test_dataset = EuroSATMultiSpectral(
        root_dir=test_dir,
        split='train',  # Use all data
        target_size=config['data']['image_size'],
        noise_transform=None,  # No noise for testing
        normalize=True,
        train_split=1.0  # Use 100% of data
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    logger.info(f"Created test dataloader: {len(test_dataset)} samples")
    
    return test_loader
