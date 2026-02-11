"""
Model Optimization for Deployment

Includes quantization and ONNX graph optimization
"""

import onnx
from pathlib import Path
import logging
import subprocess
import shutil

logger = logging.getLogger(__name__)


def optimize_onnx(
    input_path: str,
    output_path: str,
    optimization_level: int = 2
) -> str:
    """
    Optimize ONNX model graph using onnxslim
    
    Args:
        input_path: Input ONNX model
        output_path: Output optimized model
        optimization_level: 0-2 (higher = more aggressive, currently unused)
    
    Returns:
        Path to optimized model
    """
    logger.info(f"Optimizing ONNX model with onnxslim...")
    
    # Compare sizes before optimization
    original_size = Path(input_path).stat().st_size / 1e6
    
    # Try onnxslim CLI first
    onnxslim_cmd = shutil.which('onnxslim')
    
    if onnxslim_cmd:
        try:
            subprocess.run([onnxslim_cmd, input_path, output_path], check=True)
            optimized_size = Path(output_path).stat().st_size / 1e6
            
            logger.info(f"✓ Optimization complete")
            logger.info(f"  Original: {original_size:.2f} MB")
            logger.info(f"  Optimized: {optimized_size:.2f} MB")
            logger.info(f"  Reduction: {(1 - optimized_size / original_size) * 100:.1f}%")
            
            return output_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"onnxslim CLI failed: {e}")
    
    # Try onnxslim Python API
    try:
        import onnxslim as _onnxslim
        
        if hasattr(_onnxslim, 'simplify'):
            _onnxslim.simplify(input_path, output_path)
            optimized_size = Path(output_path).stat().st_size / 1e6
            
            logger.info(f"✓ Optimization complete")
            logger.info(f"  Original: {original_size:.2f} MB")
            logger.info(f"  Optimized: {optimized_size:.2f} MB")
            logger.info(f"  Reduction: {(1 - optimized_size / original_size) * 100:.1f}%")
            
            return output_path
    except Exception as e:
        logger.warning(f"onnxslim not available: {e}")
    
    # No optimizer available - copy original file
    logger.warning("No ONNX optimizer available. Copying original file.")
    shutil.copy(input_path, output_path)
    
    return output_path


def quantize_model(
    input_path: str,
    output_path: str,
    use_dynamic: bool = True
) -> str:
    """
    Quantize ONNX model to INT8 for faster inference
    
    Note: May reduce accuracy slightly
    
    Args:
        input_path: Input ONNX model (FP32)
        output_path: Output quantized model (INT8)
        use_dynamic: Use dynamic quantization
    
    Returns:
        Path to quantized model
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static
        from onnxruntime.quantization import QuantType
        
        logger.info("Quantizing model to INT8...")
        
        if use_dynamic:
            quantize_dynamic(
                input_path,
                output_path,
                weight_type=QuantType.QInt8
            )
        else:
            # Static quantization requires calibration data
            logger.warning("Static quantization not implemented yet")
            return input_path
        
        # Compare sizes
        original_size = Path(input_path).stat().st_size / 1e6
        quantized_size = Path(output_path).stat().st_size / 1e6
        
        logger.info(f"✓ Quantization complete")
        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  Quantized: {quantized_size:.2f} MB")
        logger.info(f"  Reduction: {(1 - quantized_size / original_size) * 100:.1f}%")
        
        return output_path
        
    except ImportError:
        logger.error("onnxruntime quantization tools not available")
        return input_path
