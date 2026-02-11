"""
Evaluation Metrics for Multi-Spectral Image Restoration

Includes:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- SAM: Spectral Angle Mapper (multi-spectral specific)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 6.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    PSNR = 10 * log10(MAX^2 / MSE)
    
    Higher is better (typically 20-40 dB for images)
    
    Args:
        pred: Predicted image [B, C, H, W] or [C, H, W]
        target: Ground truth image [B, C, H, W] or [C, H, W]
        data_range: Maximum possible value (for normalized data)
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target).item()
    
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10((data_range ** 2) / mse)
    
    return psnr


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 6.0,
    window_size: int = 11
) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    SSIM measures perceptual quality (0-1, higher is better)
    
    Args:
        pred: Predicted image [B, C, H, W] or [C, H, W]
        target: Ground truth image [B, C, H, W] or [C, H, W]
        data_range: Maximum possible value
        window_size: Size of Gaussian window
    
    Returns:
        SSIM value (0-1)
    """
    # Ensure 4D tensors
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    B, C, H, W = pred.shape
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        x = torch.arange(size).float() - size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()
    
    # Choose a common dtype for the convolution kernel.
    # If pred and target have different dtypes (e.g., one is FP16 from AMP),
    # promote to float32 for stability and to avoid mismatched-type conv errors.
    if pred.dtype == target.dtype:
        kernel_dtype = pred.dtype
    else:
        kernel_dtype = torch.float32
    
    # Ensure inputs match kernel dtype
    if pred.dtype != kernel_dtype:
        pred = pred.type(kernel_dtype)
    if target.dtype != kernel_dtype:
        target = target.type(kernel_dtype)

    # Ensure the Gaussian window matches the input's device and chosen dtype
    _1D = gaussian_window(window_size).to(pred.device).type(kernel_dtype)
    _2D = _1D.unsqueeze(1).mm(_1D.unsqueeze(0))
    window = _2D.unsqueeze(0).unsqueeze(0).expand(C, 1, window_size, window_size).to(pred.device).type(kernel_dtype)
    
    # Constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Compute statistics
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8
) -> float:
    """
    Calculate Spectral Angle Mapper (SAM)
    
    SAM measures spectral similarity by computing the angle between
    spectral signatures. Lower is better (0 = perfect match).
    
    Useful for multi-spectral imagery where spectral relationships matter.
    
    SAM = arccos( (x · y) / (||x|| ||y||) )
    
    Args:
        pred: Predicted image [B, C, H, W] or [C, H, W]
        target: Ground truth image [B, C, H, W] or [C, H, W]
        epsilon: Small value for numerical stability
    
    Returns:
        Mean SAM in degrees
    """
    # Ensure 4D tensors
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Reshape to [B*H*W, C] for spectral vectors
    B, C, H, W = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
    
    # Compute dot product
    dot_product = (pred_flat * target_flat).sum(dim=1)
    
    # Compute norms
    pred_norm = torch.norm(pred_flat, dim=1) + epsilon
    target_norm = torch.norm(target_flat, dim=1) + epsilon
    
    # Compute angle (in radians)
    cos_angle = dot_product / (pred_norm * target_norm)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # Numerical stability
    angle = torch.acos(cos_angle)
    
    # Convert to degrees and return mean
    sam_degrees = torch.rad2deg(angle).mean().item()
    
    return sam_degrees


def calculate_rmse(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE)
    
    Args:
        pred: Predicted image
        target: Ground truth image
    
    Returns:
        RMSE value
    """
    mse = F.mse_loss(pred, target).item()
    rmse = np.sqrt(mse)
    return rmse


def calculate_mae(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        pred: Predicted image
        target: Ground truth image
    
    Returns:
        MAE value
    """
    mae = F.l1_loss(pred, target).item()
    return mae


class MetricsTracker:
    """
    Track multiple metrics during training/validation
    
    Usage:
        >>> tracker = MetricsTracker()
        >>> for pred, target in dataloader:
        ...     tracker.update(pred, target)
        >>> metrics = tracker.compute()
    """
    
    def __init__(self, data_range: float = 6.0):
        self.data_range = data_range
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.sam_sum = 0.0
        self.rmse_sum = 0.0
        self.count = 0
    
    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a new batch
        
        Args:
            pred: Predicted images [B, C, H, W]
            target: Ground truth images [B, C, H, W]
        """
        batch_size = pred.shape[0]
        
        # Calculate metrics for each sample in batch
        for i in range(batch_size):
            pred_i = pred[i:i+1]
            target_i = target[i:i+1]
            
            self.psnr_sum += calculate_psnr(pred_i, target_i, self.data_range)
            self.ssim_sum += calculate_ssim(pred_i, target_i, self.data_range)
            self.sam_sum += calculate_sam(pred_i, target_i)
            self.rmse_sum += calculate_rmse(pred_i, target_i)
            
            self.count += 1
    
    def compute(self) -> dict:
        """
        Compute average metrics
        
        Returns:
            Dictionary with averaged metrics
        """
        if self.count == 0:
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'sam': 0.0,
                'rmse': 0.0
            }
        
        return {
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count,
            'sam': self.sam_sum / self.count,
            'rmse': self.rmse_sum / self.count
        }
    
    def __repr__(self) -> str:
        metrics = self.compute()
        return (
            f"PSNR: {metrics['psnr']:.2f} dB, "
            f"SSIM: {metrics['ssim']:.4f}, "
            f"SAM: {metrics['sam']:.2f}°, "
            f"RMSE: {metrics['rmse']:.4f}"
        )
