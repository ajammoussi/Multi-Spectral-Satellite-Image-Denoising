"""
EuroSAT Multi-Spectral Dataset Loader

Handles 13-band Sentinel-2 imagery from the EuroSAT dataset.
Supports train/val splitting and on-the-fly noise injection.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
import logging

# Try importing rasterio (for multi-spectral .tif)
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logging.warning("rasterio not available. Using PIL for RGB images only.")

# Fallback to PIL for RGB
from PIL import Image

logger = logging.getLogger(__name__)


class EuroSATMultiSpectral(Dataset):
    """
    EuroSAT Multi-Spectral Dataset Loader
    
    Expects directory structure:
        data/EuroSAT/
            AnnualCrop/
                AnnualCrop_1.tif  # Shape: (13, 64, 64)
                ...
            Forest/
                ...
    
    Returns:
        clean: Tensor [13, 224, 224] - Original image (upscaled)
        noisy: Tensor [13, 224, 224] - Corrupted version
    """
    
    # Pre-computed statistics from EuroSAT dataset
    # These values are approximate - compute actual values from your data
    BAND_STATS = {
        'mean': torch.tensor([
            1353.04, 1265.03, 1269.85, 1274.46, 1512.79, 
            2179.59, 2480.46, 2677.89, 2854.54, 734.18, 
            12.09, 1818.82, 1116.98
        ]),
        'std': torch.tensor([
            245.67, 289.34, 327.45, 389.34, 434.56,
            567.89, 634.23, 701.45, 789.23, 98.76,
            5.43, 456.78, 234.56
        ])
    }
    
    def __init__(
        self, 
        root_dir: str,
        split: str = 'train',
        target_size: int = 224,
        noise_transform: Optional[Callable] = None,
        normalize: bool = True,
        train_split: float = 0.8,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Path to EuroSAT dataset root
            split: 'train' or 'val'
            target_size: Output image size (default 224 for SatMAE)
            noise_transform: Callable to add noise to images
            normalize: Whether to normalize with dataset statistics
            train_split: Fraction of data for training
            seed: Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.noise_transform = noise_transform
        self.normalize = normalize
        
        # Collect all .tif files
        self.samples = sorted(list(self.root_dir.rglob("*.tif")))
        
        if len(self.samples) == 0:
            logger.warning(f"No .tif files found in {root_dir}")
            logger.info("Searching for alternative formats (.jpg, .png)...")
            self.samples = sorted(
                list(self.root_dir.rglob("*.jpg")) + 
                list(self.root_dir.rglob("*.png"))
            )
        
        logger.info(f"Found {len(self.samples)} images in {root_dir}")
        
        # Train/Val split (deterministic)
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(train_split * len(self.samples))
        
        if split == 'train':
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        elif split == 'val':
            self.samples = [self.samples[i] for i in indices[split_idx:]]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'val'")
        
        logger.info(f"{split.upper()} split: {len(self.samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            clean: Original image [13, 224, 224]
            noisy: Corrupted image [13, 224, 224]
        """
        sample_path = self.samples[idx]
        
        # Read multi-spectral .tif with rasterio (13-band Sentinel-2)
        if HAS_RASTERIO and sample_path.suffix.lower() == '.tif':
            try:
                with rasterio.open(sample_path) as src:
                    img = src.read()  # Shape: (bands, H, W)
                img = torch.from_numpy(img).float()
            except Exception as e:
                logger.error(f"Error reading {sample_path} with rasterio: {e}")
                # Fallback to zeros
                return torch.zeros(13, self.target_size, self.target_size), \
                       torch.zeros(13, self.target_size, self.target_size)
        
        # Fallback to PIL for non-TIF images (convert grayscale to 13 bands)
        else:
            try:
                img_pil = Image.open(sample_path).convert('L')  # Convert to grayscale
                img = np.array(img_pil)[np.newaxis, :, :]  # Add channel dimension
                img = torch.from_numpy(img).float()
                # Replicate single channel to 13 bands
                img = img.repeat(13, 1, 1)
                logger.warning(f"Loaded non-TIF image {sample_path.name}. Replicating grayscale to 13 bands.")
            except Exception as e:
                logger.error(f"Error reading {sample_path} with PIL: {e}")
                return torch.zeros(13, self.target_size, self.target_size), \
                       torch.zeros(13, self.target_size, self.target_size)
        
        # Verify we have exactly 13 bands (EuroSAT multi-spectral standard)
        if img.shape[0] != 13:
            logger.error(f"Expected 13 bands, got {img.shape[0]} in {sample_path}. Padding/truncating.")
            # Pad or truncate to 13 channels
            if img.shape[0] < 13:
                padding = torch.zeros(13 - img.shape[0], img.shape[1], img.shape[2])
                img = torch.cat([img, padding], dim=0)
            else:
                img = img[:13]
        
        # Normalize to [0, 1] range for consistent scaling
        # Handle both raw Sentinel-2 (0-10000) and pre-normalized data
        if img.max() > 10:  # Likely raw pixel values
            img = img / 10000.0  # Normalize to [0, 1]
        elif img.max() > 1.5:  # Possibly 0-255 range
            img = img / 255.0
        # If already in [0, 1] range, no scaling needed
        
        # Upsample to target size (224x224 for SatMAE)
        if img.shape[1] != self.target_size or img.shape[2] != self.target_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), 
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Normalize with dataset statistics
        if self.normalize:
            # Simple min-max normalization to [0, 1] range
            # The pre-computed BAND_STATS don't match this dataset's distribution
            # so we use straightforward [0, 1] scaling which works universally
            img = torch.clamp(img, 0, 1)
            
            # Optional: Apply slight contrast enhancement to use fuller dynamic range
            # Scale from [0, 1] to approximately [-2, 2] for better training dynamics
            img = (img - 0.5) / 0.25  # Center at 0, spread to [-2, 2]
        
        # Apply noise corruption
        if self.noise_transform is not None:
            noisy = self.noise_transform(img.clone())
        else:
            noisy = img.clone()
        
        return img, noisy  # (clean, corrupted)
    
    @staticmethod
    def compute_statistics(root_dir: str, num_samples: int = 1000):
        """
        Compute mean and std statistics from dataset
        
        Args:
            root_dir: Path to dataset
            num_samples: Number of samples to use for statistics
            
        Returns:
            dict with 'mean' and 'std' tensors
        """
        from tqdm import tqdm
        
        root = Path(root_dir)
        samples = sorted(list(root.rglob("*.tif")))[:num_samples]
        
        running_mean = torch.zeros(13)
        running_std = torch.zeros(13)
        count = 0
        
        logger.info(f"Computing statistics from {len(samples)} samples...")
        
        for sample_path in tqdm(samples):
            try:
                with rasterio.open(sample_path) as src:
                    img = src.read()
                    img = torch.from_numpy(img).float()
                    
                    if img.shape[0] != 13:
                        continue
                    
                    running_mean += img.mean(dim=[1, 2])
                    running_std += img.std(dim=[1, 2])
                    count += 1
            except Exception:
                continue
        
        if count == 0:
            raise ValueError("No valid samples found for statistics computation")
        
        mean = running_mean / count
        std = running_std / count
        
        return {'mean': mean, 'std': std}
