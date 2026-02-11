#!/usr/bin/env python
"""
ONNX Export Script

Exports trained PyTorch model to ONNX format
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models import SatMAERestoration
from src.deployment import export_to_onnx, verify_onnx_model, compare_pytorch_onnx_outputs
from src.utils import load_config
from src.utils.download import verify_downloads

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/onnx/model.onnx')
    parser.add_argument('--opset', type=int, default=14)
    parser.add_argument('--verify', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Verify weights are present (optional check)
    status = verify_downloads()
    if not status['weights']:
        logger.warning("SatMAE weights not found. Model may not be properly initialized.")
    
    # Load model
    config = load_config(args.config)
    # Skip encoder pretrain when loading a full checkpoint
    config['model']['encoder']['pretrained_path'] = None
    model = SatMAERestoration(config)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Export to ONNX
    input_shape = (1, config['data']['num_bands'], 
                   config['data']['image_size'], config['data']['image_size'])
    
    export_to_onnx(
        model=model,
        output_path=args.output,
        input_shape=input_shape,
        opset_version=args.opset,
        verify=args.verify
    )
    
    # Compare outputs
    if args.verify:
        test_input = torch.randn(*input_shape)
        match = compare_pytorch_onnx_outputs(model, args.output, test_input)
        
        if match:
            logger.info("✓ Export successful - outputs match!")
        else:
            logger.warning("⚠ Outputs do not match exactly")


if __name__ == '__main__':
    main()
