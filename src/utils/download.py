"""
Automatic Download Utilities

Handles downloading of datasets and model weights
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def download_file(url: str, destination: str, description: str = "Downloading") -> str:
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        description: Description for progress bar
    
    Returns:
        Path to downloaded file
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if destination.exists():
        logger.info(f"File already exists: {destination}")
        return str(destination)
    
    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {destination}")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    
    logger.info(f"✓ Download complete: {destination}")
    return str(destination)


def download_and_extract_zip(url: str, extract_to: str, description: str = "Downloading") -> str:
    """
    Download and extract a zip file
    
    Args:
        url: URL to zip file
        extract_to: Directory to extract to
        description: Description for progress bar
    
    Returns:
        Path to extracted directory
    """
    extract_to = Path(extract_to)
    
    # Check if already extracted
    if extract_to.exists() and any(extract_to.iterdir()):
        logger.info(f"Dataset already exists: {extract_to}")
        return str(extract_to)
    
    # Download zip to temp location
    zip_path = extract_to.parent / f"{extract_to.name}.zip"
    download_file(url, str(zip_path), description)
    
    # Extract
    logger.info(f"Extracting to {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Remove zip file
    zip_path.unlink()
    logger.info(f"✓ Extraction complete: {extract_to}")
    
    return str(extract_to)


def download_eurosat_dataset(data_dir: str | None = None) -> str:
    """
    Download EuroSAT Multi-Spectral dataset (13-band)
    
    Downloads the full 13-band multi-spectral Sentinel-2 imagery.
    
    Args:
        data_dir: Directory to save dataset
    
    Returns:
        Path to dataset directory
    """
    url = "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip"
    
    logger.info("=" * 60)
    logger.info("Downloading EuroSAT Multi-Spectral Dataset")
    logger.info("="*60)
    logger.info("This is the FULL 13-band multi-spectral version!")
    logger.info("Dataset: Sentinel-2 imagery (13 bands, 64x64)")
    logger.info("Size: ~2.5GB (this may take a while)")
    logger.info("="*60)
    
    from .config import get_project_root

    if data_dir is None:
        data_dir = str(get_project_root() / 'data')

    extract_path = Path(data_dir) / "EuroSAT_MS"
    dataset_path = download_and_extract_zip(
        url, 
        str(extract_path),
        "Downloading EuroSAT Multi-Spectral dataset (~2.5GB)"
    )
    
    # The extracted structure is nested: /ds/images/remote_sensing/otherDatasets/sentinel_2/tif/
    # Need to find the tif folder
    extracted_dir = Path(dataset_path)
    tif_folder = extracted_dir / "ds" / "images" / "remote_sensing" / "otherDatasets" / "sentinel_2" / "tif"
    
    if tif_folder.exists():
        # Move tif contents to extract_path root for easier access
        import shutil
        for class_folder in tif_folder.iterdir():
            if class_folder.is_dir():
                dest = extracted_dir / class_folder.name
                if not dest.exists():
                    shutil.move(str(class_folder), str(dest))
        
        # Clean up nested structure
        shutil.rmtree(extracted_dir / "ds", ignore_errors=True)
    
    subdirs = list(extracted_dir.glob("*/"))
    
    logger.info(f"✓ Dataset downloaded: {len(subdirs)} classes found")
    logger.info(f"  Classes: {[d.name for d in subdirs[:5]]}...")
    
    return dataset_path


def download_satmae_weights(weights_dir: str | None = None) -> str:
    """
    Download SatMAE pre-trained weights (multi-spectral)
    
    Args:
        weights_dir: Directory to save weights
    
    Returns:
        Path to weights file
    """
    # Multi-spectral SatMAE weights
    url = "https://zenodo.org/record/7338613/files/pretrain-vit-base-e199.pth"
    
    logger.info("=" * 60)
    logger.info("Downloading SatMAE Pre-trained Weights")
    logger.info("=" * 60)
    logger.info("Model: ViT-Base pre-trained on multi-spectral Sentinel-2")
    logger.info("Source: Zenodo (SatMAE official)")
    logger.info("=" * 60)
    
    from .config import get_project_root

    if weights_dir is None:
        weights_dir = str(get_project_root() / 'weights')

    weights_path = Path(weights_dir) / "satmae_pretrain.pth"
    
    download_file(
        url,
        str(weights_path),
        "Downloading SatMAE weights (~330MB)"
    )
    
    logger.info(f"✓ Weights saved to: {weights_path}")
    
    return str(weights_path)


def setup_project_data(data_dir: str | None = None, weights_dir: str | None = None) -> dict:
    """
    Download all required data for the project
    
    Args:
        data_dir: Directory for datasets
        weights_dir: Directory for model weights
    
    Returns:
        Dictionary with paths to downloaded resources
    """
    logger.info("=" * 60)
    logger.info("Setting up project data")
    logger.info("=" * 60)
    
    paths = {}
    
    # Download dataset
    try:
        paths['dataset'] = download_eurosat_dataset(data_dir)
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        paths['dataset'] = None
    
    # Download weights
    try:
        paths['weights'] = download_satmae_weights(weights_dir)
    except Exception as e:
        logger.error(f"Failed to download weights: {e}")
        paths['weights'] = None
    
    logger.info("=" * 60)
    logger.info("Setup complete!")
    logger.info("=" * 60)
    logger.info(f"Dataset: {paths['dataset']}")
    logger.info(f"Weights: {paths['weights']}")
    
    return paths


def verify_downloads() -> dict:
    """
    Verify that all required files are present
    
    Returns:
        Dictionary with verification status
    """
    status = {
        'dataset': False,
        'weights': False,
        'dataset_path': None,
        'weights_path': None
    }
    
    from .config import get_project_root

    # Check dataset
    dataset_path = get_project_root() / "data" / "EuroSAT_MS"
    if dataset_path.exists():
        # Check for class directories
        subdirs = list(dataset_path.glob("*/"))
        if len(subdirs) >= 9:  # EuroSAT has 10 classes (sometimes 9)
            status['dataset'] = True
            status['dataset_path'] = str(dataset_path)
            logger.info(f"✓ Dataset found: {len(subdirs)} classes")
    
    # Check weights
    weights_path = get_project_root() / "weights" / "satmae_pretrain.pth"
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / 1e6
        if size_mb > 100:  # SatMAE weights should be ~330MB
            status['weights'] = True
            status['weights_path'] = str(weights_path)
            logger.info(f"✓ Weights found: {size_mb:.1f}MB")
    
    if not status['dataset']:
        logger.warning("✗ Dataset not found")
    if not status['weights']:
        logger.warning("✗ Model weights not found")
    
    return status


def inspect_checkpoint(weights_path: str):
    """
    Inspect the contents of a checkpoint file.
    
    Args:
        weights_path: Path to the checkpoint file
    """
    import torch
    import argparse
    from pathlib import Path
    
    path = Path(weights_path)
    
    if not path.exists():
        print(f"Weights not found at {path}")
        return

    # Try the safest, recommended approach first: load only tensor weights (weights_only=True)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        print("Loaded checkpoint with weights_only=True")
    except Exception as e:
        print("weights_only load failed:", e)
        print("Attempting safe full load by allowlisting argparse.Namespace (only do this for trusted checkpoints)")
        # Allowlist common safe globals required by some legacy checkpoints
        try:
            torch.serialization.add_safe_globals([argparse.Namespace])
        except Exception:
            # older/newer PyTorch variants may expose a different helper
            try:
                torch.serialization.safe_globals([argparse.Namespace])
            except Exception:
                print("Warning: could not add safe globals via helper; will retry full load (ensure checkpoint is trusted)")

        # Full load (may execute code contained in the checkpoint) — use only for trusted sources
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    print("Checkpoint Info:")
    print(f"  File Size: {path.stat().st_size / 1e6:.1f} MB")

    if isinstance(checkpoint, dict):
        # Heuristics to find state dict inside common container formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # If weights_only=True succeeded the object may already be a state dict
            state_dict = checkpoint

        try:
            print(f"  Parameters: {len(state_dict)} tensors")
            print(f"\n  Sample keys:")
            # Print first 5 keys
            for i, key in enumerate(list(state_dict.keys())[:5]):
                try:
                    tensor = state_dict[key]
                    if hasattr(tensor, 'shape'):
                         print(f"    {key}: {tensor.shape}")
                    else:
                         print(f"    {key}: <non-tensor>")
                except Exception:
                    print(f"    {key}: <unknown>")
        except Exception:
            print("  Could not introspect state dict structure")

    print("\n✓ Weights inspected!")

