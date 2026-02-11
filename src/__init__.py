"""
SatM AE Multi-Spectral Denoiser

Production-ready implementation optimized for RTX 4050 (6GB VRAM)
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Avoid importing heavy submodules at package import time so that
# lightweight utilities (e.g., config helpers) can be imported
# without triggering large dependencies like PyTorch or rasterio.

__all__ = []
