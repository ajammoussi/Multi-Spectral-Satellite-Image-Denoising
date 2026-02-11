"""
Visualization Tools for Multi-Spectral Imagery

Provides functions to visualize restoration results and spectral signatures
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range for display using smart scaling.
    """
    # 1. Sanity Check for "Blob" artifacts (Low contrast noise)
    img_min, img_max = img.min(), img.max()
    range_span = img_max - img_min
    
    if range_span < 0.1 and img_max <= 1.5 and img_min >= -0.5:
        # If image is flat/noise, clip it to avoid blobs
        return np.clip(img, 0, 1)

    # 2. Relaxed Robust Normalization (0.5% - 99.5%)
    # We relaxed this from 1%-99% to bring back some vibrancy ("pop")
    p05, p995 = np.percentile(img, (0.5, 99.5))
    
    if p995 - p05 < 1e-8:
        normalized = img
    else:
        normalized = (img - p05) / (p995 - p05)
        
    return np.clip(normalized, 0, 1)


def extract_rgb_bands(tensor: torch.Tensor, bands: List[int] = [3, 2, 1]) -> np.ndarray:
    """
    Extract RGB-like bands from multi-spectral tensor.
    Default [3, 2, 1] corresponds to Red, Green, Blue in Sentinel-2 (0-indexed).
    """
    tensor = tensor.cpu().detach()
    
    # Handle batched input [B, C, H, W] -> [B, H, W, 3]
    if tensor.ndim == 4:
        rgb = tensor[:, bands, :, :].permute(0, 2, 3, 1).numpy()
    # Handle single input [C, H, W] -> [H, W, 3]
    else:
        rgb = tensor[bands, :, :].permute(1, 2, 0).numpy()
    
    return rgb


def visualize_samples_grid(
    samples: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    bands: List[int] = [3, 2, 1],
    normalize: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
):
    """
    Display multiple images in a grid
    """
    n_samples = len(samples)
    if figsize is None:
        figsize = (5 * n_samples, 5)
    
    fig, axes = plt.subplots(1, n_samples, figsize=figsize)
    if n_samples == 1:
        axes = [axes]
    
    for idx, sample in enumerate(samples):
        rgb = extract_rgb_bands(sample, bands)
        if normalize:
            rgb = normalize_for_display(rgb)
        
        # Use bicubic interpolation to hide checkerboard/grid artifacts
        axes[idx].imshow(rgb, interpolation='bicubic')
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()
    plt.close()


def visualize_restoration(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    restored: torch.Tensor,
    save_path: Optional[str] = None,
    bands_to_show: List[int] = [3, 2, 1],
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize noisy, clean, and restored images side-by-side.
    Uses bicubic interpolation to reduce checkerboard/grid artifacts.
    """
    # Ensure tensors are on CPU
    noisy = noisy.cpu().detach()
    clean = clean.cpu().detach()
    restored = restored.cpu().detach()
    
    # Extract RGB
    noisy_rgb = extract_rgb_bands(noisy, bands_to_show)
    clean_rgb = extract_rgb_bands(clean, bands_to_show)
    restored_rgb = extract_rgb_bands(restored, bands_to_show)
    
    # Normalize
    noisy_rgb = normalize_for_display(noisy_rgb)
    clean_rgb = normalize_for_display(clean_rgb)
    restored_rgb = normalize_for_display(restored_rgb)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Noisy: Nearest neighbor to show grain/pixels clearly
    axes[0].imshow(noisy_rgb, interpolation='nearest')
    axes[0].set_title("Noisy Input")
    axes[0].axis('off')
    
    # Clean: Bicubic for smooth look
    axes[1].imshow(clean_rgb, interpolation='bicubic')
    axes[1].set_title("Clean Target")
    axes[1].axis('off')
    
    # Restored: Bicubic to smooth out checkerboard/grid artifacts
    axes[2].imshow(restored_rgb, interpolation='bicubic')
    axes[2].set_title("Restored Output")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spectral_signatures(
    clean: torch.Tensor,
    restored: torch.Tensor,
    pixel_coords: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None,
    band_names: Optional[List[str]] = None
):
    """
    Plot spectral signatures at specific pixel locations
    """
    clean = clean.cpu().detach()
    restored = restored.cpu().detach()
    
    C, H, W = clean.shape
    
    if pixel_coords is None:
        pixel_coords = [
            (H//2, W//2),  # Center
            (H//4, W//4),  # Top-left
            (H//4, 3*W//4),  # Top-right
            (3*H//4, W//4),  # Bottom-left
            (3*H//4, 3*W//4)  # Bottom-right
        ]
    
    if band_names is None:
        band_names = [f'B{i+1}' for i in range(C)]
    
    n_pixels = len(pixel_coords)
    fig, axes = plt.subplots(1, n_pixels, figsize=(5*n_pixels, 4))
    
    if n_pixels == 1:
        axes = [axes]
    
    for idx, (y, x) in enumerate(pixel_coords):
        clean_spectrum = clean[:, y, x].numpy()
        restored_spectrum = restored[:, y, x].numpy()
        
        axes[idx].plot(range(C), clean_spectrum, 'o-', label='Clean', linewidth=2, markersize=6)
        axes[idx].plot(range(C), restored_spectrum, 's--', label='Restored', linewidth=2, markersize=6)
        
        axes[idx].set_xlabel('Band Index')
        axes[idx].set_ylabel('Reflectance')
        axes[idx].set_title(f'Pixel ({y}, {x})')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        if len(band_names) == C:
            axes[idx].set_xticks(range(C))
            axes[idx].set_xticklabels(band_names, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved spectral signatures to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves
    """
    epochs = range(1, len(train_metrics) + 1)
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    train_loss = [m['loss'] for m in train_metrics]
    val_loss = [m['loss'] for m in val_metrics]
    ax1.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PSNR
    ax2 = fig.add_subplot(gs[0, 1])
    train_psnr = [m['psnr'] for m in train_metrics]
    val_psnr = [m['psnr'] for m in val_metrics]
    ax2.plot(epochs, train_psnr, 'o-', label='Train PSNR', linewidth=2)
    ax2.plot(epochs, val_psnr, 's-', label='Val PSNR', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Peak Signal-to-Noise Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # SSIM
    ax3 = fig.add_subplot(gs[1, 0])
    train_ssim = [m['ssim'] for m in train_metrics]
    val_ssim = [m['ssim'] for m in val_metrics]
    ax3.plot(epochs, train_ssim, 'o-', label='Train SSIM', linewidth=2)
    ax3.plot(epochs, val_ssim, 's-', label='Val SSIM', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.set_title('Structural Similarity Index')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # SAM
    ax4 = fig.add_subplot(gs[1, 1])
    val_sam = [m.get('sam', 0) for m in val_metrics]
    ax4.plot(epochs, val_sam, 's-', label='Val SAM', linewidth=2, color='orange')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('SAM (degrees)')
    ax4.set_title('Spectral Angle Mapper')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_band_comparison(
    clean: torch.Tensor,
    restored: torch.Tensor,
    num_bands: int = 13,
    save_path: Optional[str] = None
):
    """
    Plot all spectral bands side-by-side
    """
    clean = clean.cpu().detach()
    restored = restored.cpu().detach()
    
    fig, axes = plt.subplots(2, num_bands, figsize=(num_bands * 2, 4))
    
    for i in range(num_bands):
        clean_band = clean[i].numpy()
        axes[0, i].imshow(clean_band, cmap='gray')
        axes[0, i].set_title(f'Band {i+1}')
        axes[0, i].axis('off')
        
        restored_band = restored[i].numpy()
        axes[1, i].imshow(restored_band, cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].text(-0.1, 0.5, 'Clean', transform=axes[0, 0].transAxes, rotation=90, va='center', fontsize=12, fontweight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Restored', transform=axes[1, 0].transAxes, rotation=90, va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved band comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_grid(
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    save_path: Optional[str] = None,
    bands_to_show: List[int] = [3, 2, 1]
):
    """
    Create a grid of comparison images
    """
    n_samples = len(samples)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (noisy, clean, restored) in enumerate(samples):
        noisy_rgb = extract_rgb_bands(noisy, bands_to_show)
        clean_rgb = extract_rgb_bands(clean, bands_to_show)
        restored_rgb = extract_rgb_bands(restored, bands_to_show)
        
        noisy_rgb = normalize_for_display(noisy_rgb)
        clean_rgb = normalize_for_display(clean_rgb)
        restored_rgb = normalize_for_display(restored_rgb)
        
        # Noisy: Nearest to see grain
        axes[idx, 0].imshow(noisy_rgb, interpolation='nearest')
        axes[idx, 0].set_title(f"Sample {idx+1}: Noisy")
        axes[idx, 0].axis('off')
        
        # Clean: Bicubic for reference
        axes[idx, 1].imshow(clean_rgb, interpolation='bicubic')
        axes[idx, 1].set_title(f"Sample {idx+1}: Clean")
        axes[idx, 1].axis('off')
        
        # Restored: Bicubic to smooth checkerboard
        axes[idx, 2].imshow(restored_rgb, interpolation='bicubic')
        axes[idx, 2].set_title(f"Sample {idx+1}: Restored")
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison grid to {save_path}")
    else:
        plt.show()
    
    plt.close()
