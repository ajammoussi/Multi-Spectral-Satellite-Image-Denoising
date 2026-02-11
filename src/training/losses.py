"""
Loss Functions for Multi-Spectral Image Restoration

Combines MSE for pixel accuracy with SSIM for perceptual quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined MSE + SSIM loss for multi-spectral restoration
    
    Loss = α * MSE + β * (1 - SSIM)
    
    MSE: Pixel-wise accuracy
    SSIM: Structural similarity (perceptual quality)
    
    Usage:
        >>> criterion = CombinedLoss(ssim_weight=0.1)
        >>> loss = criterion(pred, target)
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.1,
        data_range: float = 6.0  # Range of normalized data (~-3 to 3)
    ):
        """
        Args:
            mse_weight: Weight for MSE loss
            ssim_weight: Weight for SSIM loss
            data_range: Dynamic range of data (for SSIM calculation)
        """
        super().__init__()
        
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(data_range=data_range, channel=13)
        
        logger.info(
            f"Initialized CombinedLoss: "
            f"MSE weight={mse_weight}, SSIM weight={ssim_weight}"
        )
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, 13, H, W]
            target: Ground truth image [B, 13, H, W]
        
        Returns:
            Combined loss (scalar)
        """
        # MSE loss
        mse_loss = self.mse(pred, target)
        
        # SSIM loss (averaged across all bands)
        ssim_loss = self.ssim(pred, target)
        
        # Combined
        total_loss = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
        
        return total_loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss
    
    SSIM measures perceptual quality by comparing:
    - Luminance
    - Contrast
    - Structure
    
    Returns: 1 - SSIM (so lower is better)
    """
    
    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        data_range: float = 6.0,
        channel: int = 13
    ):
        super().__init__()
        
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.data_range = data_range
        
        # Create Gaussian window
        self.window = self._create_window(window_size, channel)
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create Gaussian window for SSIM calculation"""
        def gaussian(window_size, sigma):
            x = torch.arange(window_size).float() - window_size // 2
            gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def _ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window: torch.Tensor,
        window_size: int,
        channel: int,
        size_average: bool = True
    ) -> torch.Tensor:
        """Calculate SSIM between two images"""
        # Constants for stability
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=window_size // 2, groups=channel
        ) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=window_size // 2, groups=channel
        ) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=window_size // 2, groups=channel
        ) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img1, img2: Images [B, C, H, W]
        
        Returns:
            1 - SSIM (loss, lower is better)
        """
        # Move window to same device
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        # Calculate SSIM
        ssim_value = self._ssim(
            img1, img2, self.window, 
            self.window_size, self.channel, 
            self.size_average
        )
        
        # Return as loss (1 - SSIM)
        return 1 - ssim_value


class L1Loss(nn.Module):
    """
    L1 (MAE) Loss - more robust to outliers than MSE
    """
    
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features
    
    Note: Disabled by default to save VRAM
    Only use if you have >8GB VRAM
    """
    
    def __init__(self, layer_weights: Optional[dict] = None):
        super().__init__()
        
        try:
            import torchvision.models as models
            
            # Load pre-trained VGG16
            vgg = models.vgg16(pretrained=True).features.eval()
            
            # Freeze parameters
            for param in vgg.parameters():
                param.requires_grad = False
            
            self.vgg = vgg
            
            # Default layer weights
            self.layer_weights = layer_weights or {
                '3': 1.0,   # relu1_2
                '8': 1.0,   # relu2_2
                '15': 1.0,  # relu3_3
                '22': 1.0   # relu4_3
            }
            
            logger.info("Initialized PerceptualLoss with VGG16")
            
        except Exception as e:
            logger.error(f"Failed to initialize PerceptualLoss: {e}")
            self.vgg = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: Images [B, C, H, W]
        
        Note: VGG expects 3-channel RGB, so we average spectral bands
        """
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Convert multi-spectral to 3-channel
        pred_rgb = pred[:, :3, :, :]  # Use first 3 bands
        target_rgb = target[:, :3, :, :]
        
        # Extract features
        pred_features = self._extract_features(pred_rgb)
        target_features = self._extract_features(target_rgb)
        
        # Compute loss
        loss = 0.0
        for layer, weight in self.layer_weights.items():
            loss += weight * F.mse_loss(
                pred_features[layer], 
                target_features[layer]
            )
        
        return loss
    
    def _extract_features(self, x: torch.Tensor) -> dict:
        """Extract VGG features from specific layers"""
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layer_weights:
                features[name] = x
        return features
