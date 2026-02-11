#!/usr/bin/env python
"""
Evaluation Script

Evaluates trained model on validation/test set
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.data import get_dataloaders
from src.models import SatMAERestoration
from src.training.metrics import MetricsTracker
from src.utils import load_config, visualize_restoration
from src.utils.download import verify_downloads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_visualizations', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Verify data is present
    logger.info("Verifying project data...")
    status = verify_downloads()
    if not status['dataset'] or not status['weights']:
        logger.warning("Missing data! Please run: python scripts/train.py (will auto-download)")
        logger.warning("Or manually download from:")
        logger.warning("  Dataset: https://madm.dfki.de/files/sentinel/EuroSAT.zip")
        logger.warning("  Weights: https://zenodo.org/record/7338613/files/pretrain-vit-base-e199.pth")
    
    # Load config and model
    config = load_config(args.config)
    # Skip encoder pretrain when loading a full checkpoint
    config['model']['encoder']['pretrained_path'] = None
    model = SatMAERestoration(config)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Get dataloader
    _, val_loader = get_dataloaders(config)
    
    # Evaluate
    from src.training.eval import evaluate_model
    evaluate_model(model, val_loader, args.device, verbose=True)

    # Optional: Save visualizations logic remains here as it's specific to this script's args
    if args.save_visualizations:
        logger.info("Saving visualizations...")
        with torch.no_grad():
            for idx, (clean, noisy) in enumerate(tqdm(val_loader)):
                if idx >= 5: break
                clean = clean.to(args.device)
                noisy = noisy.to(args.device)
                output = model(noisy)
                
                visualize_restoration(
                    noisy[0], clean[0], output[0],
                    save_path=f'outputs/visualizations/sample_{idx}.png'
                )


if __name__ == '__main__':
    main()

