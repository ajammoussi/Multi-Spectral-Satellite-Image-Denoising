"""
Reusable Convolutional Building Blocks

Provides efficient conv blocks for the decoder network
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvBlock(nn.Module):
    """
    Basic convolutional block with residual connection
    
    Architecture:
        Conv3x3 -> BN -> GELU -> Conv3x3 -> BN -> (+skip) -> GELU
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 
                kernel_size, stride, padding, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels, 
                kernel_size, 1, padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection (1x1 conv if channel mismatch)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=1, stride=stride, 
                bias=False
            )
        else:
            self.skip = nn.Identity()
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv(x)
        out = out + residual
        out = self.activation(out)
        return out


class ResidualBlock(nn.Module):
    """
    Lightweight residual block for feature refinement
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class UpsampleBlock(nn.Module):
    """
    Upsampling block with 2x resolution increase
    
    Uses transposed convolution for learned upsampling
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        use_pixel_shuffle: bool = False
    ):
        super().__init__()
        
        if use_pixel_shuffle:
            # Pixel shuffle (sub-pixel convolution)
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, 3, padding=1),
                nn.PixelShuffle(2),  # 2x upscale
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            # Transposed convolution
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=2, stride=2
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for parameter efficiency
    
    Params: in_ch * k^2 + in_ch * out_ch
    vs Regular Conv: in_ch * out_ch * k^2
    
    Savings: ~8x for k=3
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        
        # Depthwise: each input channel convolved separately
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size, stride, padding,
            groups=in_channels,  # Key: groups = channels
            bias=False
        )
        
        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    
    Useful for emphasizing important spectral bands
    """
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        reduced_channels = max(channels // reduction, 8)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Conv2d(channels, reduced_channels, 1),
            nn.GELU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Recalibrated features [B, C, H, W]
        """
        attention = self.se(x)  # [B, C, 1, 1]
        return x * attention  # Channel-wise scaling
