"""
Lightweight CNN Decoder for Multi-Spectral Reconstruction

Progressive upsampling from ViT features to full resolution:
    14×14 → 28×28 → 56×56 → 112×112 → 224×224

Optimized for 6GB VRAM: ~4-5M parameters
"""

import torch
import torch.nn as nn
from typing import List
import logging

from .blocks import ConvBlock, UpsampleBlock, ResidualBlock

logger = logging.getLogger(__name__)


class LightweightDecoder(nn.Module):
    """
    Efficient decoder for multi-spectral image reconstruction
    
    Architecture:
        Input:  [B, 768, 14, 14]  (ViT features)
        ↓ 4 Upsample blocks (2x each)
        Output: [B, 13, 224, 224]  (Reconstructed image)
    
    Memory: ~4.2M parameters (lightweight!)
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        channels: List[int] = [384, 192, 96, 48],
        out_channels: int = 13,
        num_residual_blocks: int = 2,
        use_pixel_shuffle: bool = False
    ):
        """
        Args:
            in_channels: Input feature dimension from encoder (768 for ViT-Base)
            channels: Channel counts at each upsampling stage
            out_channels: Number of output spectral bands (13)
            num_residual_blocks: Residual blocks per stage
            use_pixel_shuffle: Use pixel shuffle instead of transposed conv
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = len(channels)
        
        # Build progressive upsampling path
        layers = []
        prev_ch = in_channels
        
        for stage_idx, ch in enumerate(channels):
            # Upsample block (2x resolution)
            layers.append(
                UpsampleBlock(
                    prev_ch, ch, 
                    use_pixel_shuffle=use_pixel_shuffle
                )
            )
            
            # Refinement with residual blocks
            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(ch))
            
            prev_ch = ch
        
        self.upsample_blocks = nn.Sequential(*layers)
        
        # Final projection to spectral bands
        self.head = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.GELU(),
            nn.Conv2d(channels[-1], out_channels, kernel_size=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(
            f"Initialized LightweightDecoder: "
            f"{self.count_parameters():,} parameters"
        )
    
    def _init_weights(self, m):
        """Kaiming initialization for conv layers"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder features [B, 768, 14, 14]
        
        Returns:
            Reconstructed image [B, 13, 224, 224]
        
        Resolution progression:
            14→28→56→112→224 (4 stages, 2x each)
        """
        # Progressive upsampling
        x = self.upsample_blocks(x)  # [B, 48, 224, 224]
        
        # Project to spectral bands
        x = self.head(x)  # [B, 13, 224, 224]
        
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetDecoder(nn.Module):
    """
    U-Net style decoder with skip connections
    
    Note: Requires encoder to return multi-scale features
    Currently not used (no skip connections from ViT)
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        channels: List[int] = [384, 192, 96, 48],
        out_channels: int = 13
    ):
        super().__init__()
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        prev_ch = in_channels
        
        for ch in channels:
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_ch, ch, 2, stride=2),
                    ConvBlock(ch, ch)
                )
            )
            prev_ch = ch
        
        # Final projection
        self.head = nn.Conv2d(channels[-1], out_channels, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        skip_features: List[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Bottleneck features [B, 768, 14, 14]
            skip_features: Multi-scale encoder features (optional)
        
        Returns:
            Reconstructed image [B, 13, 224, 224]
        """
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            
            # Add skip connection if available
            if skip_features is not None and i < len(skip_features):
                x = x + skip_features[-(i+1)]
        
        return self.head(x)


class AttentionDecoder(nn.Module):
    """
    Decoder with spatial attention for emphasizing important regions
    
    More parameters but better quality - use if VRAM allows
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        channels: List[int] = [384, 192, 96, 48],
        out_channels: int = 13
    ):
        super().__init__()
        
        # Upsampling with attention
        self.stages = nn.ModuleList()
        prev_ch = in_channels
        
        for ch in channels:
            stage = nn.Sequential(
                UpsampleBlock(prev_ch, ch),
                SpatialAttention(ch),
                ConvBlock(ch, ch)
            )
            self.stages.append(stage)
            prev_ch = ch
        
        self.head = nn.Conv2d(channels[-1], out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Attended features [B, C, H, W]
        """
        attention_map = self.attention(x)  # [B, 1, H, W]
        return x * attention_map
