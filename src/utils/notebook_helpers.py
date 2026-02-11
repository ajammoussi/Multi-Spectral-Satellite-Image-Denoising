"""
Notebook Visualization Helpers

Reusable visualization utilities specifically for Jupyter notebooks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from IPython.display import clear_output

from .visualization import extract_rgb_bands, normalize_for_display
from pathlib import Path
from PIL import Image
import os



def visualize_sample_batch(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    num_samples: int = 3,
    bands: List[int] = [3, 2, 1],  # Sentinel-2 Red/Green/Blue
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Visualize clean vs noisy images from a batch.
    
    Args:
        clean: Clean images [B, C, H, W]
        noisy: Noisy images [B, C, H, W]
        num_samples: Number of samples to display
        bands: Band indices to use as RGB
        figsize: Figure size (auto if None)
    """
    if figsize is None:
        figsize = (8, num_samples * 3)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Use centralized utilities
        clean_rgb = extract_rgb_bands(clean[i], bands=bands)
        noisy_rgb = extract_rgb_bands(noisy[i], bands=bands)
        
        clean_rgb = normalize_for_display(clean_rgb)
        noisy_rgb = normalize_for_display(noisy_rgb)
        
        axes[i, 0].imshow(clean_rgb)
        axes[i, 0].set_title(f"Clean Image {i+1}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy_rgb)
        axes[i, 1].set_title(f"Noisy Image {i+1}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_restoration_comparison(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    restored: torch.Tensor,
    num_samples: int = 4,
    bands: List[int] = [0, 1, 2],
    figsize: Tuple[int, int] = (12, None)
):
    """
    Visualize clean, noisy, and restored images with metrics.
    
    Args:
        clean: Clean images [B, C, H, W]
        noisy: Noisy images [B, C, H, W]
        restored: Restored images [B, C, H, W]
        num_samples: Number of samples to display
        bands: Band indices to use as RGB
        figsize: Figure size (height auto-calculated if None)
    """
    from ..training.metrics import calculate_psnr, calculate_ssim
    
    if figsize[1] is None:
        figsize = (figsize[0], num_samples * 3)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Extract and normalize RGB
        clean_rgb = normalize_for_display(extract_rgb_bands(clean[i], bands=bands))
        noisy_rgb = normalize_for_display(extract_rgb_bands(noisy[i], bands=bands))
        restored_rgb = normalize_for_display(extract_rgb_bands(restored[i], bands=bands))
        
        # Calculate per-sample metrics
        psnr = calculate_psnr(restored[i:i+1], clean[i:i+1])
        ssim = calculate_ssim(restored[i:i+1], clean[i:i+1])
        
        axes[i, 0].imshow(clean_rgb)
        axes[i, 0].set_title("Clean (Ground Truth)")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy_rgb)
        axes[i, 1].set_title("Noisy (Input)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(restored_rgb)
        axes[i, 2].set_title(f"Restored\nPSNR: {psnr:.1f}dB, SSIM: {ssim:.3f}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_progress(
    history: dict,
    epoch: int,
    total_epochs: int,
    best_loss: float,
    best_psnr: float,
    clear_prev: bool = True
):
    """
    Plot training curves with live updates (for use in training loop).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_psnr', 'val_ssim', 'learning_rate'
        epoch: Current epoch number
        total_epochs: Total number of epochs
        best_loss: Best validation loss so far
        best_psnr: Best PSNR so far
        clear_prev: Clear previous output (for live updates)
    """
    if clear_prev:
        clear_output(wait=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation', marker='s')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR
    axes[0, 1].plot(epochs, history['val_psnr'], label='PSNR', color='green', marker='o')
    axes[0, 1].axhline(y=30, color='r', linestyle='--', linewidth=1, label='Target (30dB)', alpha=0.5)
    axes[0, 1].set_title('Validation PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM
    axes[1, 0].plot(epochs, history['val_ssim'], label='SSIM', color='orange', marker='o')
    axes[1, 0].axhline(y=0.90, color='r', linestyle='--', linewidth=1, label='Target (0.90)', alpha=0.5)
    axes[1, 0].set_title('Validation SSIM')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], label='LR', color='red', marker='o')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nEpoch {epoch}/{total_epochs}")
    print(f"Best Loss: {best_loss:.4f} | Best PSNR: {best_psnr:.2f} dB")


def plot_spectral_comparison(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    restored: torch.Tensor,
    sample_idx: int = 0,
    center_only: bool = True
):
    """
    Plot spectral signatures comparing clean, noisy, and restored.
    
    Args:
        clean: Clean image [B, C, H, W] or [C, H, W]
        noisy: Noisy image [B, C, H, W] or [C, H, W]
        restored: Restored image [B, C, H, W] or [C, H, W]
        sample_idx: Batch index to use
        center_only: Only plot center pixel (faster)
    """
    from ..training.metrics import calculate_sam
    
    # Handle batched input
    if clean.ndim == 4:
        clean = clean[sample_idx]
        noisy = noisy[sample_idx]
        restored = restored[sample_idx]
    
    # Extract center pixel
    h, w = clean.shape[1], clean.shape[2]
    center_y, center_x = h // 2, w // 2
    
    clean_spectrum = clean[:, center_y, center_x].cpu().numpy()
    noisy_spectrum = noisy[:, center_y, center_x].cpu().numpy()
    restored_spectrum = restored[:, center_y, center_x].cpu().numpy()
    
    # Sentinel-2 band info
    wavelengths = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
    band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
    
    plt.figure(figsize=(14, 6))
    
    # Spectral signatures
    plt.subplot(1, 2, 1)
    plt.plot(wavelengths, clean_spectrum, 'o-', label='Clean', linewidth=2, markersize=8)
    plt.plot(wavelengths, noisy_spectrum, 's-', label='Noisy', alpha=0.7, markersize=6)
    plt.plot(wavelengths, restored_spectrum, '^-', label='Restored', linewidth=2, markersize=6)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title('Spectral Signature Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Band-wise error
    plt.subplot(1, 2, 2)
    error_noisy = np.abs(clean_spectrum - noisy_spectrum)
    error_restored = np.abs(clean_spectrum - restored_spectrum)
    
    x = np.arange(len(band_names))
    width = 0.35
    
    plt.bar(x - width/2, error_noisy, width, label='Noisy Error', alpha=0.7, color='orange')
    plt.bar(x + width/2, error_restored, width, label='Restored Error', alpha=0.7, color='green')
    plt.xlabel('Band', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Band-wise Absolute Error', fontsize=14)
    plt.xticks(x, band_names, rotation=45)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print SAM
    sam_noisy = calculate_sam(noisy.unsqueeze(0), clean.unsqueeze(0))
    sam_restored = calculate_sam(restored.unsqueeze(0), clean.unsqueeze(0))
    
    print(f"\nSpectral Angle Mapper (SAM):")
    print(f"  Noisy vs Clean: {sam_noisy:.2f}°")
    print(f"  Restored vs Clean: {sam_restored:.2f}°")
    print(f"  Improvement: {sam_noisy - sam_restored:.2f}°")


def print_dataset_info(train_loader, val_loader):
    """
    Print formatted dataset statistics.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    print("\nDataloader Statistics:")
    print("=" * 60)
    print(f"Training:")
    print(f"  Batches: {len(train_loader)}")
    print(f"  Samples: {len(train_loader.dataset)}")
    print(f"\nValidation:")
    print(f"  Batches: {len(val_loader)}")
    print(f"  Samples: {len(val_loader.dataset)}")
    
    # Inspect a sample batch
    sample_clean, sample_noisy = next(iter(train_loader))
    print(f"\nSample Batch Shape:")
    print(f"  Clean: {sample_clean.shape}")
    print(f"  Noisy: {sample_noisy.shape}")
    print(f"  Value Range: [{sample_clean.min():.3f}, {sample_clean.max():.3f}]")
    print("=" * 60)
    
    return sample_clean, sample_noisy


def print_evaluation_summary(avg_psnr: float, avg_ssim: float, avg_sam: float):
    """
    Print formatted evaluation results.
    
    Args:
        avg_psnr: Average PSNR
        avg_ssim: Average SSIM
        avg_sam: Average SAM
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average SAM:  {avg_sam:.2f}°")
    print("="*60)
    
    # Performance assessment
    if avg_psnr > 30 and avg_ssim > 0.90:
        print("\n✅ EXCELLENT: Model meets performance targets!")
    elif avg_psnr > 25 and avg_ssim > 0.85:
        print("\n✓ GOOD: Model shows good performance")
    else:
        print("\n⚠️ NEEDS IMPROVEMENT: Consider training longer or tuning hyperparameters")


def visualize_noise_impact(
    clean_sample: torch.Tensor,
    noise_configs: list = None
):
    """
    Visualize the impact of different noise levels.
    
    Args:
        clean_sample: Clean input tensor [C, H, W]
        noise_configs: List of noise configuration dictionaries
    """
    if noise_configs is None:
        noise_configs = [
            {'name': 'Low', 'gaussian': 0.005, 'speckle': 0.002, 'dead_band': 0.02},
            {'name': 'Medium', 'gaussian': 0.015, 'speckle': 0.005, 'dead_band': 0.08},
            {'name': 'High', 'gaussian': 0.035, 'speckle': 0.01, 'dead_band': 0.15}
        ]
        
    from src.data.transforms import AddSensorNoise, extract_rgb_bands, normalize_for_display
    
    clean_rgb = extract_rgb_bands(clean_sample, bands=[0, 1, 2])
    clean_rgb = normalize_for_display(clean_rgb)
    
    rows = len(noise_configs)
    
    # Create subplots
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    
    # Handle single row case (axes would be 1D array)
    if rows == 1:
        axes = [axes]
        
    for i, noise_cfg in enumerate(noise_configs):
        # Create noise transform
        # Scale thermal noise based on level
        thermal_scale = 0.002
        if noise_cfg['name'] == 'Medium':
            thermal_scale = 0.005
        elif noise_cfg['name'] == 'High':
            thermal_scale = 0.01
            
        noise_transform = AddSensorNoise(
            gaussian_sigma=noise_cfg['gaussian'],
            speckle_sigma=noise_cfg['speckle'],
            dead_band_prob=noise_cfg['dead_band'],
            thermal_scale=thermal_scale
        )
        
        # Apply noise
        noisy_sample = noise_transform(clean_sample.clone())
        noisy_rgb = extract_rgb_bands(noisy_sample, bands=(3, 2, 1))
        noisy_rgb = normalize_for_display(noisy_rgb)
        
        # Plot Clean
        ax_clean = axes[i][0] if rows > 1 else axes[0]
        ax_clean.imshow(clean_rgb)
        ax_clean.set_title(f'{noise_cfg["name"]} Noise - Clean')
        ax_clean.axis('off')
        
        # Plot Noisy
        ax_noisy = axes[i][1] if rows > 1 else axes[1]
        ax_noisy.imshow(noisy_rgb)
        ax_noisy.set_title(
            f'{noise_cfg["name"]} Noise - Corrupted\n'
            f'(σ_g={noise_cfg["gaussian"]}, σ_s={noise_cfg["speckle"]}, p_dead={noise_cfg["dead_band"]})'
        )
        ax_noisy.axis('off')
        
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot final training history curves.
    
    Args:
        history: Dictionary with history data
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # PSNR
    axes[0, 1].plot(epochs, history['val_psnr'], label='PSNR', color='green')
    axes[0, 1].axhline(y=30, color='r', linestyle='--', label='Target (30dB)')
    axes[0, 1].set_title('Validation PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # SSIM
    axes[1, 0].plot(epochs, history['val_ssim'], label='SSIM', color='orange')
    axes[1, 0].axhline(y=0.90, color='r', linestyle='--', label='Target (0.90)')
    axes[1, 0].set_title('Validation SSIM')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], label='LR', color='red')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Training curves saved to {save_path}")
        
    plt.show()


def visualize_noise_impact(
    clean_sample: torch.Tensor,
    noise_configs: list = None
):
    """
    Visualize the impact of different noise levels.
    
    Args:
        clean_sample: Clean input tensor [C, H, W]
        noise_configs: List of noise configuration dictionaries
    """
    if noise_configs is None:
        noise_configs = [
            {'name': 'Low', 'gaussian': 0.005, 'speckle': 0.002, 'dead_band': 0.02},
            {'name': 'Medium', 'gaussian': 0.015, 'speckle': 0.005, 'dead_band': 0.08},
            {'name': 'High', 'gaussian': 0.035, 'speckle': 0.01, 'dead_band': 0.15}
        ]
        
    from src.data.transforms import AddSensorNoise, extract_rgb_bands, normalize_for_display
    
    # Use correct Sentinel-2 RGB bands (Red, Green, Blue)
    clean_rgb = extract_rgb_bands(clean_sample, bands=(3, 2, 1))
    clean_rgb = normalize_for_display(clean_rgb)
    
    rows = len(noise_configs)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows))
    
    # Handle single row case
    if rows == 1:
        axes = [axes]
        
    for i, noise_cfg in enumerate(noise_configs):
        # Create noise transform
        thermal_scale = 0.002 if noise_cfg['name'] == 'Low' else (0.005 if noise_cfg['name'] == 'Medium' else 0.01)
        
        noise_transform = AddSensorNoise(
            gaussian_sigma=noise_cfg['gaussian'],
            speckle_sigma=noise_cfg['speckle'],
            dead_band_prob=noise_cfg['dead_band'],
            thermal_scale=thermal_scale
        )
        
        # Apply noise
        noisy_sample = noise_transform(clean_sample.clone())
        noisy_rgb = extract_rgb_bands(noisy_sample, bands=(3, 2, 1))
        noisy_rgb = normalize_for_display(noisy_rgb)
        
        # Plot
        axes[i][0].imshow(clean_rgb)
        axes[i][0].set_title(f'{noise_cfg["name"]} Noise - Clean')
        axes[i][0].axis('off')
        
        axes[i][1].imshow(noisy_rgb)
        axes[i][1].set_title(
            f'{noise_cfg["name"]} Noise - Corrupted\n'
            f'(σ_g={noise_cfg["gaussian"]}, σ_s={noise_cfg["speckle"]}, p_dead={noise_cfg["dead_band"]})'
        )
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot final training history curves.
    
    Args:
        history: Dictionary with history data
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # PSNR
    axes[0, 1].plot(epochs, history['val_psnr'], label='PSNR', color='green')
    axes[0, 1].axhline(y=30, color='r', linestyle='--', label='Target (30dB)')
    axes[0, 1].set_title('Validation PSNR')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # SSIM
    axes[1, 0].plot(epochs, history['val_ssim'], label='SSIM', color='orange')
    axes[1, 0].axhline(y=0.90, color='r', linestyle='--', label='Target (0.90)')
    axes[1, 0].set_title('Validation SSIM')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], label='LR', color='red')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Training curves saved to {save_path}")
        
    plt.show()



def explore_dataset(dataset_path: str, show_samples: bool = True):
    """
    Print dataset structure and visualize samples.
    
    Args:
        dataset_path: Path to dataset root directory
        show_samples: Whether to display one sample per class
    """
    path = Path(dataset_path)
    if not path.exists():
        print(f"Dataset not found at {path}")
        return

    # Count classes and images
    classes = sorted([d.name for d in path.iterdir() if d.is_dir()])
    print(f"\nEuroSAT Classes ({len(classes)}):")
    
    total_images = 0
    images_per_class = {}
    
    for cls in classes:
        images = list((path / cls).glob('*.jpg')) + list((path / cls).glob('*.tif'))
        print(f"  {cls:20s}: {len(images):5d} images")
        total_images += len(images)
        images_per_class[cls] = images[0] if images else None
    
    print(f"\nTotal Images: {total_images}")
    
    if show_samples and classes:
        print("\nSample Images per Class:")
        
        # Import rasterio for 13-band TIFFs
        try:
            import rasterio
            has_rasterio = True
        except ImportError:
            has_rasterio = False
            print("Warning: rasterio not available, using PIL (may not work for multi-band TIFFs)")
        
        n_classes = len(classes)
        rows = (n_classes + 4) // 5
        fig, axes = plt.subplots(rows, 5, figsize=(15, 3*rows))
        axes = axes.flatten()
        
        for i, cls in enumerate(classes):
            img_path = images_per_class[cls]
            if img_path:
                try:
                    # Handle multi-band TIF files with rasterio
                    if img_path.suffix.lower() == '.tif' and has_rasterio:
                        with rasterio.open(img_path) as src:
                            # Read all bands
                            img_data = src.read()  # Shape: (bands, H, W)
                        
                        # Extract RGB bands (Sentinel-2: Red=4, Green=3, Blue=2, 1-indexed)
                        # In 0-indexed: Red=3, Green=2, Blue=1
                        if img_data.shape[0] >= 4:  # Has at least 4 bands
                            rgb = np.stack([img_data[3], img_data[2], img_data[1]], axis=-1)
                        elif img_data.shape[0] == 3:  # Already RGB
                            rgb = np.transpose(img_data, (1, 2, 0))
                        else:  # Single band, replicate to RGB
                            rgb = np.repeat(img_data[0:1], 3, axis=0).transpose(1, 2, 0)
                        
                        # Normalize for display (simple min-max)
                        rgb = rgb.astype(np.float32)
                        rgb_min, rgb_max = rgb.min(), rgb.max()
                        if rgb_max > rgb_min:
                            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
                        else:
                            rgb = np.clip(rgb, 0, 1)
                        
                        axes[i].imshow(rgb)
                    else:
                        # Fallback to PIL for JPG or if rasterio not available
                        img = Image.open(img_path)
                        axes[i].imshow(img)
                    
                    axes[i].set_title(cls, fontsize=9)
                    axes[i].axis('off')
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    axes[i].text(0.5, 0.5, 'Load Error', ha='center', va='center')
                    axes[i].axis('off')
            else:
                axes[i].axis('off')
                
        # Turn off remaining axes
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
