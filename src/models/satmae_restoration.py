"""
SatMAE Multi-Spectral Restoration Model

Complete architecture combining:
- Pre-trained SatMAE ViT encoder (with transfer learning)
- Lightweight CNN decoder for reconstruction

Optimized for RTX 4050 (6GB VRAM)
"""

import torch
import torch.nn as nn
from typing import Dict
import logging

from .encoder import SatMAEEncoder
from .decoder import LightweightDecoder

logger = logging.getLogger(__name__)


class SatMAERestoration(nn.Module):
    """
    Transfer Learning Architecture for Multi-Spectral Denoising
    
    Architecture Flow:
        Input [B, 13, 224, 224]
            ↓ SatMAE Encoder (ViT-Base)
        Patch Embeddings [B, 196, 768]
            ↓ Reshape to Spatial
        Feature Map [B, 768, 14, 14]
            ↓ Lightweight Decoder
        Output [B, 13, 224, 224]
    
    Memory Optimizations:
    - Gradient checkpointing on encoder (~30% savings)
    - Frozen early layers (~50% trainable params)
    - Lightweight decoder (<5M params)
    - Mixed precision training (FP16)
    
    Total: ~3GB VRAM for batch_size=8
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary from YAML
        """
        super().__init__()
        
        self.config = config
        self.patch_size = config['model']['encoder']['patch_size']
        self.image_size = config['data']['image_size']
        
        # Calculate grid size after patchification
        self.grid_size = self.image_size // self.patch_size  # 224/16 = 14
        
        # Load pre-trained SatMAE encoder
        self.encoder = SatMAEEncoder(
            in_channels=config['model']['encoder']['input_channels'],
            image_size=self.image_size,
            patch_size=config['model']['encoder']['patch_size'],
            embed_dim=config['model']['encoder']['embed_dim'],
            depth=config['model']['encoder']['depth'],
            num_heads=config['model']['encoder']['num_heads'],
            pretrained_path=config['model']['encoder'].get('pretrained_path'),
            freeze_layers=config['model']['encoder'].get('freeze_layers'),
            use_checkpointing=config['model']['encoder']['gradient_checkpointing']
        )
        
        # Lightweight decoder
        self.decoder = LightweightDecoder(
            in_channels=config['model']['encoder']['embed_dim'],
            channels=config['model']['decoder']['channels'],
            out_channels=config['model']['decoder']['output_channels']
        )
        
        # Log architecture summary
        self._log_architecture()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and decoder
        
        Args:
            x: Noisy input [B, 13, 224, 224]
        
        Returns:
            Reconstructed image [B, 13, 224, 224]
        """
        B, C, H, W = x.shape
        
        # Validate input shape
        assert H == self.image_size and W == self.image_size, \
            f"Expected {self.image_size}x{self.image_size}, got {H}x{W}"
        assert C == self.config['data']['num_bands'], \
            f"Expected {self.config['data']['num_bands']} bands, got {C}"
        
        # Encoder: Image → Patch Embeddings
        patch_embeddings = self.encoder(x)  # [B, N_patches, D]
        # N_patches = (224/16)^2 = 196, D = 768
        
        # Reshape to spatial feature map
        # [B, 196, 768] → [B, 768, 14, 14]
        features = self._reshape_to_spatial(patch_embeddings, B)
        
        # Decoder: Features → Full Resolution
        output = self.decoder(features)  # [B, 13, 224, 224]
        
        return output
    
    def _reshape_to_spatial(
        self, 
        patch_embeddings: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:
        """
        Reshape patch embeddings to spatial feature map
        
        Args:
            patch_embeddings: [B, N, D] where N = grid_size^2
            batch_size: Batch size
        
        Returns:
            Spatial features [B, D, grid_size, grid_size]
        """
        # [B, N, D] → [B, D, N]
        features = patch_embeddings.transpose(1, 2)
        
        # [B, D, N] → [B, D, H, W]
        features = features.reshape(
            batch_size, -1, self.grid_size, self.grid_size
        )
        
        return features
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Get detailed parameter counts for profiling
        
        Returns:
            Dictionary with parameter statistics
        """
        encoder_params = self.encoder.get_num_params()
        
        decoder_total = sum(p.numel() for p in self.decoder.parameters())
        decoder_trainable = sum(
            p.numel() for p in self.decoder.parameters() 
            if p.requires_grad
        )
        
        total = encoder_params['total'] + decoder_total
        trainable = encoder_params['trainable'] + decoder_trainable
        
        return {
            'encoder_total': encoder_params['total'],
            'encoder_trainable': encoder_params['trainable'],
            'encoder_frozen': encoder_params['frozen'],
            'decoder_total': decoder_total,
            'decoder_trainable': decoder_trainable,
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_percent': 100 * trainable / total if total > 0 else 0
        }
    
    def _log_architecture(self):
        """Log architecture details"""
        params = self.count_parameters()
        
        logger.info("=" * 60)
        logger.info("SatMAE Restoration Model Architecture")
        logger.info("=" * 60)
        logger.info(f"Input:  [{self.config['data']['num_bands']}, "
                   f"{self.image_size}, {self.image_size}]")
        logger.info(f"Output: [{self.config['data']['num_bands']}, "
                   f"{self.image_size}, {self.image_size}]")
        logger.info("-" * 60)
        logger.info("Encoder (SatMAE ViT-Base):")
        logger.info(f"  Total params:     {params['encoder_total']:>12,}")
        logger.info(f"  Trainable params: {params['encoder_trainable']:>12,}")
        logger.info(f"  Frozen params:    {params['encoder_frozen']:>12,}")
        logger.info("-" * 60)
        logger.info("Decoder (Lightweight CNN):")
        logger.info(f"  Total params:     {params['decoder_total']:>12,}")
        logger.info(f"  Trainable params: {params['decoder_trainable']:>12,}")
        logger.info("-" * 60)
        logger.info("Total Model:")
        logger.info(f"  Total params:     {params['total']:>12,}")
        logger.info(f"  Trainable params: {params['trainable']:>12,} "
                   f"({params['trainable_percent']:.1f}%)")
        logger.info(f"  Frozen params:    {params['frozen']:>12,}")
        logger.info("=" * 60)
    
    def get_optimizer_param_groups(self, lr, weight_decay: float = 0.05):
        """
        Create parameter groups with different learning rates
        
        Strategy:
        - Encoder: Lower LR (fine-tuning pre-trained weights)
        - Decoder: Higher LR (training from scratch)
        """
        
        if isinstance(lr, str):
            lr = float(lr)
        elif isinstance(lr, (list, tuple)):
            lr = float(lr[0])
            
        encoder_params = []
        decoder_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Simple string matching to separate encoder/decoder
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                # Default to decoder group for anything else (decoder, head, etc.)
                decoder_params.append(param)
        
        param_groups = [
            {
                'params': encoder_params,
                'lr': lr * 0.1,  # 10x lower LR for encoder
                'weight_decay': weight_decay
            },
            {
                'params': decoder_params,
                'lr': lr,
                'weight_decay': weight_decay
            }
        ]
        
        logger.info(
            f"Created optimizer groups: "
            f"encoder (lr={lr*0.1:.2e}), "
            f"decoder (lr={lr:.2e})"
        )
        
        return param_groups

    @torch.no_grad()
    def profile_memory(self, batch_size: int = 8) -> Dict[str, float]:
        """
        Profile VRAM usage for a given batch size
        
        Args:
            batch_size: Batch size to test
        
        Returns:
            Dictionary with memory statistics (in GB)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, cannot profile memory")
            return {}
        
        device = next(self.parameters()).device
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # Measure model weight memory (bytes)
        model_memory = sum(
            p.nelement() * p.element_size()
            for p in self.parameters()
        )

        # Measure trainable parameter memory (used for optimizer states & gradients)
        trainable_memory = sum(
            p.nelement() * p.element_size()
            for p in self.parameters() if p.requires_grad
        )
        
        # Forward pass
        dummy_input = torch.randn(
            batch_size,
            self.config['data']['num_bands'],
            self.image_size,
            self.image_size,
            device=device
        )
        
        _ = self.forward(dummy_input)
        
        # Get peak memory (bytes)
        peak_memory = torch.cuda.max_memory_allocated(device)

        # Estimate optimizer states memory.
        # Common optimizers (Adam/AdamW) keep two state tensors per parameter
        # (exp_avg, exp_avg_sq). We approximate optimizer_state = 2 * trainable
        optimizer_state_bytes = trainable_memory * 2

        # Estimate gradients memory (one tensor per trainable parameter)
        gradients_bytes = trainable_memory

        # Estimate activations as the remainder of peak memory after accounting
        # for model weights, optimizer states and gradients. This is an estimate
        # because other CUDA allocations (buffers, cached memory) may contribute.
        activations_bytes = peak_memory - (model_memory + optimizer_state_bytes + gradients_bytes)
        if activations_bytes < 0:
            activations_bytes = 0

        return {
            'model_weights_gb': model_memory / 1e9,
            'optimizer_gb': optimizer_state_bytes / 1e9,
            'activations_gb': activations_bytes / 1e9,
            'gradients_gb': gradients_bytes / 1e9,
            'peak_vram_gb': peak_memory / 1e9,
            'batch_size': batch_size
        }
