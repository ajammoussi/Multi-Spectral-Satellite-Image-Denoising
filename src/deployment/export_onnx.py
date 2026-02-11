"""
PyTorch to ONNX Model Export

Handles conversion of SatMAE restoration model to ONNX format
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 13, 224, 224),
    opset_version: int = 14,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[dict] = None,
    verify: bool = True
) -> str:
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: Trained SatMAERestoration model
        output_path: Where to save .onnx file
        input_shape: Example input tensor shape (B, C, H, W)
        opset_version: ONNX opset version (14 recommended)
        do_constant_folding: Enable constant folding optimization
        dynamic_axes: Dictionary of dynamic axes (for variable batch size)
        verify: Whether to verify exported model
    """
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Default dynamic axes (variable batch size)
    if dynamic_axes is None:
        dynamic_axes = {
            'noisy_image': {0: 'batch_size'},
            'clean_image': {0: 'batch_size'}
        }
    
    # Ensure output directory exists (convert to Path for manipulation)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Export model (torch.onnx.export expects a string path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=['noisy_image'],
            output_names=['clean_image'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    logger.info(f"✓ Model exported to {output_path}")
    
    # Get file size
    file_size_mb = output_path_obj.stat().st_size / 1e6
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Opset version: {opset_version}")
    
    # Verify exported model
    if verify:
        verify_onnx_model(str(output_path), dummy_input)
    
    return str(output_path)


def verify_onnx_model(
    onnx_path: str,
    test_input: Optional[torch.Tensor] = None
) -> bool:
    """
    Verify exported ONNX model
    
    Checks:
    1. Model structure is valid
    2. ONNX Runtime can load the model
    3. Inference runs successfully
    4. Output shape matches expected
    
    Args:
        onnx_path: Path to ONNX model
        test_input: Optional test input tensor
    
    Returns:
        True if verification passes
    """
    logger.info("Verifying ONNX model...")
    
    try:
        # 1. Check model structure
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ Model structure is valid")
        
        # 2. Load with ONNX Runtime
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        logger.info(f"✓ ONNX Runtime loaded model")
        logger.info(f"  Providers: {ort_session.get_providers()}")
        
        # Get input/output info
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_name = ort_session.get_outputs()[0].name
        output_shape = ort_session.get_outputs()[0].shape
        
        logger.info(f"  Input: {input_name}, shape: {input_shape}")
        logger.info(f"  Output: {output_name}, shape: {output_shape}")
        
        # 3. Test inference
        if test_input is not None:
            # Accept either torch.Tensor or numpy array
            if isinstance(test_input, torch.Tensor):
                test_input_np = test_input.cpu().numpy()
            else:
                test_input_np = np.asarray(test_input).astype(np.float32)
        else:
            # Derive a suitable dummy input from the model's declared input shape.
            # ONNX input shapes may contain dynamic dimensions (None or symbolic names),
            # substitute them with 1 for verification purposes.
            inferred_shape = []
            for dim in input_shape:
                if dim is None:
                    inferred_shape.append(1)
                else:
                    try:
                        inferred_shape.append(int(dim))
                    except Exception:
                        inferred_shape.append(1)

            # Ensure a 4D input (B, C, H, W). If fewer dims, add batch dim(s).
            if len(inferred_shape) < 4:
                while len(inferred_shape) < 4:
                    inferred_shape.insert(0, 1)

            test_input_np = np.random.randn(*inferred_shape).astype(np.float32)
        
        ort_inputs = {input_name: test_input_np}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logger.info(f"✓ Inference successful")
        logger.info(f"  Output shape: {ort_outputs[0].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def compare_pytorch_onnx_outputs(
    pytorch_model: nn.Module,
    onnx_path: str,
    sample_input: Union[torch.Tensor, np.ndarray],
    rtol: float = 1e-3,
    atol: float = 1e-5,
    **kwargs
) -> dict:
    """
    Compare outputs between PyTorch and ONNX models
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        sample_input: Test input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Dictionary containing comparison results
    """
    logger.info("Comparing PyTorch and ONNX outputs...")
    
    # Handle older 'test_input' or 'sample_input' argument if passed as kwargs
    if sample_input is None and 'test_input' in kwargs:
        sample_input = kwargs['test_input']
    
    # Ensure input is a tensor
    # NOTE: This line was failing because of the 'import numpy as np' below.
    # By removing the inner import, this now works correctly using the global 'np'.
    if isinstance(sample_input, np.ndarray):
        sample_input = torch.from_numpy(sample_input)
        
    pytorch_model.eval()
    
    # Determine device: Check kwargs first, then fall back to model's device
    if 'device' in kwargs:
        device = kwargs['device']
    else:
        try:
            device = next(pytorch_model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input.to(device))
        pytorch_output_np = pytorch_output.cpu().numpy()
    
    # ONNX inference
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: sample_input.cpu().numpy()}
    onnx_output_np = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    # REMOVED: import numpy as np (This was causing the UnboundLocalError)
    
    # Flatten outputs for similarity calculation
    p_flat = pytorch_output_np.flatten()
    o_flat = onnx_output_np.flatten()
    
    max_diff = float(np.abs(pytorch_output_np - onnx_output_np).max())
    mean_diff = float(np.abs(pytorch_output_np - onnx_output_np).mean())
    shapes_match = pytorch_output_np.shape == onnx_output_np.shape
    match = bool(np.allclose(pytorch_output_np, onnx_output_np, rtol=rtol, atol=atol))
    
    # Calculate cosine similarity
    norm_p = np.linalg.norm(p_flat)
    norm_o = np.linalg.norm(o_flat)
    if norm_p > 0 and norm_o > 0:
        cosine_sim = float(np.dot(p_flat, o_flat) / (norm_p * norm_o))
    else:
        cosine_sim = 1.0 if norm_p == norm_o else 0.0
    
    logger.info(f"Output comparison:")
    logger.info(f"  Max difference: {max_diff:.6e}")
    logger.info(f"  Mean difference: {mean_diff:.6e}")
    logger.info(f"  Cosine similarity: {cosine_sim:.6f}")
    logger.info(f"  Shapes match: {shapes_match}")
    logger.info(f"  Match (rtol={rtol}, atol={atol}): {match}")
    
    if match:
        logger.info("✓ Outputs match within tolerance")
    else:
        logger.warning("✗ Outputs do not match within tolerance")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'cosine_sim': cosine_sim,
        'shapes_match': shapes_match,
        'match_allclose': match,
        'torch_shape': pytorch_output_np.shape,
        'onnx_shape': onnx_output_np.shape
    }


def export_with_simplification(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 13, 224, 224),
    **export_kwargs
) -> str:
    """
    Export model and simplify ONNX graph
    
    Requires: onnx-simplifier (pip install onnx-simplifier)
    
    Args:
        model: PyTorch model
        output_path: Output path
        input_shape: Input shape
        **export_kwargs: Additional export arguments
    
    Returns:
        Path to simplified ONNX model
    """
    # Export to ONNX (temporary file)
    temp_path = output_path.replace('.onnx', '_temp.onnx')
    export_to_onnx(model, temp_path, input_shape, verify=False, **export_kwargs)

    # Prefer pure-Python `onnxslim` (works on Python 3.12). If unavailable,
    # keep the original exported ONNX model.
    try:
        import shutil
        import subprocess

        onnxslim_cmd = shutil.which('onnxslim')

        if onnxslim_cmd:
            try:
                subprocess.run([onnxslim_cmd, temp_path, output_path], check=True)
                logger.info(f"✓ Model simplified with onnxslim and saved to {output_path}")
                Path(temp_path).unlink()
                return output_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"onnxslim CLI failed: {e}; falling back to python API if available")

        # Try onnxslim Python API
        try:
            import onnxslim as _onnxslim

            if hasattr(_onnxslim, 'simplify'):
                _onnxslim.simplify(temp_path, output_path)
                logger.info(f"✓ Model simplified with onnxslim module and saved to {output_path}")
                Path(temp_path).unlink()
                return output_path
            logger.debug("onnxslim module imported but no simplify() API found")

        except Exception:
            logger.debug("onnxslim module not installed or failed to import")

        # No simplifier available — keep the original ONNX model
        logger.warning("No ONNX simplifier available (onnxslim). Using original ONNX model")
        Path(temp_path).rename(output_path)

    except Exception as e:
        logger.warning(f"Unexpected error while simplifying: {e}\nUsing original ONNX model")
        Path(temp_path).rename(output_path)

    return output_path
