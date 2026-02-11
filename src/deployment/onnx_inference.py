"""
ONNX Runtime Inference Session

Provides optimized inference for deployed ONNX models
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class ONNXInferenceSession:
    """
    Wrapper for ONNX Runtime inference
    
    Features:
    - GPU acceleration (if available)
    - Batch inference
    - Performance profiling
    
    Usage:
        >>> session = ONNXInferenceSession('model.onnx')
        >>> output = session.predict(input_array)
    """
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        use_gpu: bool = True
    ):
        """
        Args:
            model_path: Path to ONNX model
            providers: List of execution providers
            use_gpu: Whether to use GPU if available
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Setup providers
        if providers is None:
            if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        # Create inference session
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        logger.info(f"Loaded ONNX model from {model_path}")
        logger.info(f"  Provider: {self.session.get_providers()[0]}")
        logger.info(f"  Input: {self.input_name}, shape: {self.input_shape}")
        logger.info(f"  Output: {self.output_name}, shape: {self.output_shape}")
    
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on input
        
        Args:
            input_array: Input numpy array [B, C, H, W]
        
        Returns:
            Output numpy array [B, C, H, W]
        """
        # Validate input shape
        if input_array.ndim != 4:
            raise ValueError(f"Expected 4D input, got {input_array.ndim}D")
        
        # Ensure float32
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)
        
        # Run inference
        ort_inputs = {self.input_name: input_array}
        ort_outputs = self.session.run(None, ort_inputs)
        
        return ort_outputs[0]
    
    def benchmark(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 13, 224, 224),
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> dict:
        """
        Benchmark inference performance
        
        Args:
            input_shape: Input shape for testing
            num_iterations: Number of iterations
            warmup_iterations: Warmup iterations
        
        Returns:
            Performance statistics
        """
        logger.info(f"Benchmarking with shape {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = self.predict(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.predict(dummy_input)
            end = time.time()
            times.append(end - start)
        
        times = np.array(times)
        
        stats = {
            'mean_ms': times.mean() * 1000,
            'std_ms': times.std() * 1000,
            'min_ms': times.min() * 1000,
            'max_ms': times.max() * 1000,
            'fps': 1.0 / times.mean(),
            'throughput_imgs_per_sec': input_shape[0] / times.mean()
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Mean: {stats['mean_ms']:.2f} ms")
        logger.info(f"  Std:  {stats['std_ms']:.2f} ms")
        logger.info(f"  FPS:  {stats['fps']:.2f}")
        
        return stats
