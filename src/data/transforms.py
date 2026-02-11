"""
Sensor Noise Transforms for Multi-Spectral Imagery

Simulates realistic satellite sensor degradation including:
- Gaussian noise (thermal/electronic)
- Speckle noise (multiplicative)
- Dead bands (channel dropout)
- Thermal noise (wavelength-dependent)
"""

import torch
import torch.nn as nn
from typing import Optional, Sequence
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AddSensorNoise(nn.Module):
    """
    Simulates multi-spectral satellite sensor degradation
    
    Noise Types:
    1. Gaussian noise - Additive thermal/electronic noise
    2. Speckle noise - Multiplicative coherent noise
    3. Dead bands - Complete channel dropout (sensor failure)
    4. Thermal noise - Wavelength-dependent noise (affects SWIR more)
    
    Usage:
        >>> transform = AddSensorNoise(
        ...     gaussian_sigma=0.05,
        ...     dead_band_prob=0.25
        ... )
        >>> noisy_img = transform(clean_img)
    """
    
    def __init__(
        self,
        gaussian_sigma: float = 0.05,
        speckle_sigma: float = 0.03,
        dead_band_prob: float = 0.25,
        thermal_scale: float = 0.01,
        enable_striping: bool = False,
        stripe_prob: float = 0.1
    ):
        """
        Args:
            gaussian_sigma: Std dev of Gaussian noise
            speckle_sigma: Std dev of multiplicative speckle
            dead_band_prob: Probability of each band being dead (0-1)
            thermal_scale: Scale of wavelength-dependent thermal noise
            enable_striping: Whether to add striping artifacts
            stripe_prob: Probability of striping per image
        """
        super().__init__()
        self.gaussian_sigma = gaussian_sigma
        self.speckle_sigma = speckle_sigma
        self.dead_band_prob = dead_band_prob
        self.thermal_scale = thermal_scale
        self.enable_striping = enable_striping
        self.stripe_prob = stripe_prob
        
        logger.debug(
            f"Initialized AddSensorNoise: "
            f"gaussian={gaussian_sigma:.3f}, "
            f"dead_bands={dead_band_prob:.2f}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sensor noise to clean image
        
        Args:
            x: Clean image [C, H, W] or [B, C, H, W]
        
        Returns:
            Corrupted image with same shape
        """
        # Handle both batched and unbatched inputs
        is_batched = x.ndim == 4
        if not is_batched:
            x = x.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        B, C, H, W = x.shape
        corrupted = x.clone()
        
        # 1. Gaussian noise (additive white noise)
        if self.gaussian_sigma > 0:
            gaussian_noise = torch.randn_like(corrupted) * self.gaussian_sigma
            corrupted = corrupted + gaussian_noise
        
        # 2. Speckle noise (multiplicative)
        if self.speckle_sigma > 0:
            speckle = 1.0 + torch.randn_like(corrupted) * self.speckle_sigma
            corrupted = corrupted * speckle
        
        # 3. Dead bands (random channel dropout)
        if self.dead_band_prob > 0:
            # Each sample in batch can have different dead bands
            for b in range(B):
                dead_mask = torch.rand(C, device=x.device) < self.dead_band_prob
                num_dead = dead_mask.sum().item()
                if num_dead > 0:
                    corrupted[b, dead_mask, :, :] = 0.0
                    logger.debug(f"Sample {b}: {num_dead}/{C} dead bands")
        
        # 4. Thermal noise (wavelength-dependent)
        # Longer wavelengths (bands 10-13: SWIR) have more thermal noise
        if self.thermal_scale > 0:
            # Linear increase from band 0 to band 12
            thermal_weights = torch.linspace(
                1.0, 2.0, C, device=x.device
            ).view(1, C, 1, 1)
            
            thermal_noise = (
                torch.randn_like(corrupted) * 
                self.thermal_scale * 
                thermal_weights
            )
            corrupted = corrupted + thermal_noise
        
        # 5. Striping artifacts (optional, common in push-broom sensors)
        if self.enable_striping and torch.rand(1).item() < self.stripe_prob:
            corrupted = self._add_striping(corrupted)
        
        # Clip to reasonable range (assumes normalized data)
        corrupted = torch.clamp(corrupted, -3.0, 3.0)
        
        # Return to original shape
        if not is_batched:
            corrupted = corrupted.squeeze(0)
        
        return corrupted
    
    def _add_striping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add columnar striping artifacts (push-broom sensor artifact)
        
        Args:
            x: Image tensor [B, C, H, W]
        
        Returns:
            Image with striping [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Random stripe intensity per column
        stripe_intensity = torch.randn(B, C, 1, W, device=x.device) * 0.02
        
        # Apply stripes (broadcast across height dimension)
        striped = x + stripe_intensity
        
        return striped
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gaussian_sigma={self.gaussian_sigma:.3f}, "
            f"speckle_sigma={self.speckle_sigma:.3f}, "
            f"dead_band_prob={self.dead_band_prob:.2f}, "
            f"thermal_scale={self.thermal_scale:.3f})"
        )


class RandomBandPermutation(nn.Module):
    """
    Randomly permute spectral bands for data augmentation
    
    Note: Use with caution - may break spectral relationships
    """
    
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.prob:
            return x
        
        C = x.shape[0] if x.ndim == 3 else x.shape[1]
        perm = torch.randperm(C)
        
        if x.ndim == 3:
            return x[perm]
        else:
            return x[:, perm]


class NormalizeSpectral(nn.Module):
    """
    Normalize multi-spectral image with per-band statistics
    """
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [C, H, W] or [B, C, H, W]
        """
        if x.ndim == 3:
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        else:
            mean = self.mean[None, :, None, None]
            std = self.std[None, :, None, None]
        
        return (x - mean) / std
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization"""
        if x.ndim == 3:
            mean = self.mean[:, None, None]
            std = self.std[:, None, None]
        else:
            mean = self.mean[None, :, None, None]
            std = self.std[None, :, None, None]
        
        return x * std + mean


def extract_rgb_bands(img: torch.Tensor or np.ndarray, bands: Sequence[int] = (3, 2, 1)) -> np.ndarray:
    """Extract RGB visualization image from multi-spectral input.

    Args:
        img: Tensor or ndarray, shape [C, H, W] or [H, W, C]
        bands: sequence of three band indices to use as RGB
               Default (3,2,1) = Sentinel-2 Red/Green/Blue bands

    Returns:
        HxWx3 float32 numpy image (not yet contrast-normalized)
    """
    # Accept torch tensors or numpy arrays
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.array(img)

    # Convert from [C,H,W] to [H,W,C]
    if arr.ndim == 3 and arr.shape[0] in (3, 13):
        arr = np.transpose(arr, (1, 2, 0))

    H, W, C = arr.shape
    # Safe band selection
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i, b in enumerate(bands):
        if b < 0 or b >= C:
            raise IndexError(f"Requested band {b} out of range (0..{C-1})")
        rgb[..., i] = arr[..., b].astype(np.float32)

    return rgb


def normalize_for_display(img: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Normalize image for display (per-channel percentile stretch to [0,1]).

    Args:
        img: HxWx3 numpy image (float)
        low_pct: lower percentile for contrast stretching
        high_pct: upper percentile

    Returns:
        HxWx3 float image in [0,1]
    """
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        channel = img[..., c]
        lo = np.percentile(channel, low_pct)
        hi = np.percentile(channel, high_pct)
        if hi - lo < 1e-6:
            out[..., c] = np.clip(channel, 0.0, 1.0)
        else:
            out[..., c] = (channel - lo) / (hi - lo)
        out[..., c] = np.clip(out[..., c], 0.0, 1.0)

    return out
