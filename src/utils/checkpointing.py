"""
Checkpoint Management

Handles saving, loading, and managing model checkpoints
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
import logging
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints with automatic cleanup
    
    Features:
    - Save best model based on metric
    - Keep only top-K checkpoints
    - Automatic cleanup of old checkpoints
    - Resume training from checkpoint
    
    Usage:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='outputs/checkpoints',
        ...     keep_top_k=3,
        ...     metric='val_psnr',
        ...     mode='max'
        ... )
        >>> manager.save(model, optimizer, epoch, metrics)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_top_k: int = 3,
        metric: str = 'val_psnr',
        mode: str = 'max'
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_top_k: Keep only top K checkpoints
            metric: Metric to use for ranking (e.g., 'val_psnr', 'val_loss')
            mode: 'max' for higher is better, 'min' for lower is better
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        
        # Track saved checkpoints
        self.checkpoints = []  # List of (metric_value, filepath) tuples
        
        logger.info(
            f"Initialized CheckpointManager: "
            f"dir={checkpoint_dir}, metric={metric}, mode={mode}"
        )
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        additional_info: Optional[Dict] = None
    ) -> str:
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Validation metrics
            scheduler: LR scheduler (optional)
            scaler: AMP scaler (optional)
            additional_info: Additional info to save
        
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Get metric value for this checkpoint
        metric_value = metrics.get(self.metric, 0.0)
        
        # Create checkpoint filename
        filename = f"checkpoint_epoch_{epoch:03d}_{self.metric}_{metric_value:.4f}.pth"
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filename}")
        
        # Add to tracked checkpoints
        self.checkpoints.append((metric_value, filepath))
        
        # Check if this is the best checkpoint
        is_best = self._is_best(metric_value)
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy2(filepath, best_path)
            logger.info(f"New best model! {self.metric}={metric_value:.4f}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return str(filepath)
    
    def _is_best(self, metric_value: float) -> bool:
        """Check if current metric is the best so far"""
        if not self.checkpoints:
            return True
        
        best_value = max(m for m, _ in self.checkpoints) if self.mode == 'max' \
                      else min(m for m, _ in self.checkpoints)
        
        if self.mode == 'max':
            return metric_value >= best_value
        else:
            return metric_value <= best_value
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only top-K"""
        if len(self.checkpoints) <= self.keep_top_k:
            return
        
        # Sort checkpoints by metric
        reverse = (self.mode == 'max')
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)
        
        # Remove worst checkpoints
        to_remove = self.checkpoints[self.keep_top_k:]
        
        for _, filepath in to_remove:
            if filepath.exists() and filepath.name != "best_model.pth":
                filepath.unlink()
                logger.info(f"Removed old checkpoint: {filepath.name}")
        
        # Keep only top-K
        self.checkpoints = self.checkpoints[:self.keep_top_k]
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load the latest checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state (optional)
            scheduler: Scheduler to load state (optional)
            scaler: AMP scaler to load state (optional)
            device: Device to load to
        
        Returns:
            Checkpoint dictionary
        """
        # Find latest checkpoint
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda x: x.stat().st_mtime
        )
        
        if not checkpoints:
            logger.warning("No checkpoints found")
            return {}
        
        latest_checkpoint = checkpoints[-1]
        return self.load_checkpoint(
            str(latest_checkpoint), model, optimizer, scheduler, scaler, device
        )
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load the best checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state (optional)
            scheduler: Scheduler to load state (optional)
            scaler: AMP scaler to load state (optional)
            device: Device to load to
        
        Returns:
            Checkpoint dictionary
        """
        best_path = self.checkpoint_dir / "best_model.pth"
        
        if not best_path.exists():
            logger.warning("Best model checkpoint not found")
            return {}
        
        return self.load_checkpoint(
            str(best_path), model, optimizer, scheduler, scaler, device
        )
    
    @staticmethod
    def load_checkpoint(
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = 'cuda'
    ) -> Dict:
        """
        Load checkpoint from file
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state (optional)
            scheduler: Scheduler to load state (optional)
            scaler: AMP scaler to load state (optional)
            device: Device to load to
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model weights from {filepath}")
        
        # Load optimizer state
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state")
        
        # Load scaler state
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info("Loaded AMP scaler state")
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Resumed from epoch {epoch}")
        logger.info(f"Checkpoint metrics: {metrics}")
        
        return checkpoint
