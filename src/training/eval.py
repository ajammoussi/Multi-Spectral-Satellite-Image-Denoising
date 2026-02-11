
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Union
import logging

from .metrics import MetricsTracker

logger = logging.getLogger(__name__)

def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    metrics_tracker: Optional[MetricsTracker] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        metrics_tracker: Optional existing MetricsTracker (will create new if None)
        verbose: Whether to print progress and results
        
    Returns:
        Dictionary of computed metrics
    """
    model.eval()
    model.to(device)
    
    if metrics_tracker is None:
        metrics_tracker = MetricsTracker()
    else:
        metrics_tracker.reset()
        
    if verbose:
        logger.info(f"Evaluating on {len(dataloader)} batches...")
        
    iterator = tqdm(dataloader, desc="Evaluating", leave=False) if verbose else dataloader
    
    with torch.no_grad():
        for clean, noisy in iterator:
            clean = clean.to(device)
            noisy = noisy.to(device)
            
            # Forward pass
            restored = model(noisy)
            
            # Update metrics
            metrics_tracker.update(restored, clean)
            
    metrics = metrics_tracker.compute()
    
    if verbose:
        logger.info("Evaluation Results:")
        for k, v in metrics.items():
            if 'psnr' in k:
                logger.info(f"  {k.upper()}: {v:.2f} dB")
            elif 'ssim' in k:
                logger.info(f"  {k.upper()}: {v:.4f}")
            elif 'sam' in k:
                logger.info(f"  {k.upper()}: {v:.2f}°")
            else:
                logger.info(f"  {k}: {v:.4f}")
                
    return metrics
