# Multi-Spectral Satellite Image Denoising with SatMAE

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Production-ready implementation of multi-spectral satellite image restoration using transfer learning with SatMAE (Masked Autoencoder for Satellite Imagery) on TRUE 13-band Sentinel-2 data.

**Optimized for NVIDIA RTX 4050 (6GB VRAM)** through mixed-precision training, gradient checkpointing, and gradient accumulation.

---

## 🎯 Project Overview

This project restores corrupted multi-spectral satellite imagery affected by:
- **Sensor Dropout**: Dead spectral bands from sensor malfunction (8% probability)
- **Gaussian/Speckle Noise**: Thermal and electronic noise
- **Thermal Artifacts**: Wavelength-dependent noise in SWIR bands

### Achieved Results

After 48 epochs of training on NVIDIA RTX 4050:

| Metric | Value | Status |
|--------|-------|--------|
| **PSNR** | **41.5 dB** | ✅ Excellent (>30 dB target) |
| **SSIM** | **0.988** | ✅ Excellent (>0.90 target) |
| **SAM** | **1.86°** | ✅ Excellent (<5° target) |
| **Inference (PyTorch)** | 11.7 ms/image | ✅ Real-time capable |
| **Inference (ONNX)** | 10.1 ms/image | ✅ 1.16x speedup |
| **VRAM Usage** | ~3 GB | ✅ Fits 6GB cards |
| **Training Time** | ~6 hours (100 epochs) | ✅ Efficient |

### Key Features

**Transfer Learning**: Pre-trained SatMAE ViT-Base encoder with frozen early layers  
**Memory Efficient**: Optimized for 6GB VRAM through:
- Mixed Precision (FP16) training (~40% VRAM savings)
- Gradient Checkpointing (~30% savings)
- Gradient Accumulation (effective batch size 64 from micro-batch 8)
- Selective layer freezing (50% trainable parameters)

**Production Ready**: ONNX export with 1.16x inference speedup  
**Modular Architecture**: Easy experimentation and extension  
**Comprehensive Metrics**: PSNR, SSIM, SAM (Spectral Angle Mapper)

### Visual Results

![Restoration Results](output.png)

The model successfully restores spectral information with high fidelity. However, some visual artifacts are present in edge regions due to the upscaling process (64px→192px) and MSE-based loss function. See [Known Limitations](#known-limitations) for details.

---

## 🔬 Known Limitations

While achieving excellent numerical metrics (PSNR 41.5 dB, SSIM 0.988), the model exhibits some visual artifacts:

### 1. Color Fringing at Edges
**Observation**: Slight pink/green halos around sharp edges in RGB visualization.

**Cause**: The model treats the 13 spectral bands somewhat independently during reconstruction. Combined with wavelength-dependent noise simulation, different bands (Red, Green, Blue) are restored with slightly different edge profiles. When composited into RGB, these sub-pixel misalignments manifest as color fringes.

**Why metrics don't catch it**: PSNR is calculated per-band or averaged. A 1-pixel shift in the Red band relative to Green only slightly reduces PSNR but creates visually obvious colored outlines that the human eye immediately notices.

**Mitigation**: Use perceptual loss or reduce MSE weight in favor of SSIM. Apply post-processing edge alignment.

### 2. Ringing Artifacts
**Observation**: Subtle "ghosting" or double-edge effects, particularly visible in the first few samples.

**Cause**: The EuroSAT dataset consists of native 64×64 images upscaled to 192×192 to match the Vision Transformer architecture. Standard bicubic interpolation creates overshoot/undershoot (ringing) at sharp edges. Since the ground truth itself contains these upscaling artifacts, the model learns to reproduce them. The model is effectively "rewarded" (lower MSE) for generating these artifacts.

**Why metrics look good**: The model accurately reconstructs the ground truth, including its artifacts. If it produced "cleaner" edges, PSNR would actually drop.

**Mitigation**: Use native resolution data or train on downsampled-then-up-scaled pairs. Apply perceptual supervision that penalizes unnatural edges.

### 3. MSE vs. Perception Trade-off
**Observation**: Some edges appear slightly blurred or duplicated.

**Cause**: The loss function is dominated by MSE with small SSIM weight (0.1). MSE minimizes prediction error by averaging possibilities, leading to blur. SSIM encourages local contrast. These competing objectives can cause the model to create sharp edges at slightly wrong positions, leaving ghost outlines.

**Mitigation**: Increase SSIM weight, add perceptual loss using pre-trained features, or use adversarial training for sharper edges.

### 4. Catastrophic Failures on Dead Bands
**Observation**: Occasional samples with PSNR ~10 dB (vs. typical 40+ dB).

**Cause**: When critical spectral bands (e.g., Green) are completely zeroed out by the dead band simulation (8% probability), the model sometimes fails to hallucinate the missing information, outputting incorrect intensity ranges or patterns.

**Mitigation**: Use spectral correlation priors, increase dead band probability during training for robustness, or add auxiliary reconstruction loss on individual bands.

### Recommendations for Production Use

1. **For visual quality**: Reduce MSE weight, increase SSIM/perceptual loss weight
2. **For edge sharpness**: Use native resolution data or perceptual supervision  
3. **For robustness**: Increase dead band training probability, add band-wise reconstruction loss
4. **Post-processing**: Apply spectral band alignment and edge-aware filtering

Despite these artifacts, the model demonstrates strong spectral reconstruction capability and is suitable for many remote sensing applications where spectral fidelity (SAM: 1.86°) is more critical than perfect edge rendering.

---

## 🏗️ Architecture Details

### Model Overview

The restoration model combines:
1. **Pre-trained SatMAE ViT-Base Encoder** (86M parameters)
2. **Lightweight CNN Decoder** (4.2M parameters)
3. **Two-Stage Training Strategy** for optimal transfer learning

**Total Parameters**: ~90M (only 45M trainable after freezing)

### Encoder: SatMAE Vision Transformer

**Architecture**: ViT-Base adapted for 13-band multi-spectral input

```
Input [B, 13, 192, 192]
    ↓ Patch Embedding (16×16 patches)
Patch Tokens [B, 144, 768]  (12×12 grid)
    ↓ 12 Transformer Blocks
    │   Each block:
    │   - Multi-Head Self-Attention (12 heads)
    │   - Feed-Forward Network (MLP)
    │   - Layer Normalization
    │   - Residual Connections
    ↓
Feature Embeddings [B, 144, 768]
```

**Key Features**:
- **Pre-trained Weights**: Trained on Sentinel-2 imagery (unsupervised masked autoencoding)
- **Gradient Checkpointing**: Reduces VRAM by ~30% at cost of 20% slower training
- **Selective Freezing**: First 6 blocks frozen (preserve low-level features)
- **Input Adaptation**: Modified first layer to accept 13 channels (vs. original 3)

**Parameter Breakdown**:
- Total: 86M parameters
- Frozen (blocks 0-5): 43M parameters
- Trainable (blocks 6-11): 43M parameters

### Decoder: Lightweight CNN

**Architecture**: Progressive upsampling with residual refinement

```
Input [B, 768, 12, 12]  (Reshaped encoder output)
    ↓
Stage 1: Upsample → 24×24 (384 channels)
    ↓ 2× Residual Blocks
Stage 2: Upsample → 48×48 (192 channels)
    ↓ 2× Residual Blocks
Stage 3: Upsample → 96×96 (96 channels)
    ↓ 2× Residual Blocks
Stage 4: Upsample → 192×192 (48 channels)
    ↓ 2× Residual Blocks
Final Conv: 48 → 13 channels
    ↓
Output [B, 13, 192, 192]
```

**Key Features**:
- **Lightweight**: Only 4.2M parameters (~5% of total model)
- **Residual Connections**: 2 residual blocks per stage for feature refinement
- **Skip Connections**: Optional U-Net style skip connections (disabled by default for memory)
- **Efficient Upsampling**: Uses transposed convolutions (2× per stage)

### Two-Stage Training Strategy

The model uses a progressive training approach for optimal transfer learning:

#### **Stage A: Decoder Training** (25 epochs, ~2.5 hours)

**Purpose**: Rapidly train the decoder while preserving pre-trained encoder knowledge

**Configuration** (`configs/experiments/stage_a_decoder.yaml`):
```yaml
model:
  encoder:
    freeze_layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # ALL frozen
training:
  epochs: 25
  optimizer:
    lr: 1e-4  # Higher learning rate for decoder
```

**Rationale**:
- Encoder already learned powerful features from Sentinel-2 pre-training
- Decoder needs to learn reconstruction from scratch
- Freezing encoder prevents catastrophic forgetting
- Higher LR enables fast decoder convergence

**Expected Results**:
- PSNR: ~38-39 dB
- SSIM: ~0.96-0.97
- Training time: ~2.5 hours (RTX 4050)

#### **Stage B: Full Fine-Tuning** (30-40 epochs, ~3.5 hours)

**Purpose**: Fine-tune the encoder's deeper layers to adapt features for denoising

**Configuration** (`configs/experiments/stage_b_finetune.yaml`):
```yaml
model:
  encoder:
    freeze_layers: [0, 1, 2, 3, 4, 5]  # Unfreeze last 6 blocks
    pretrained_path: null  # Resume from Stage A checkpoint
training:
  epochs: 30
  optimizer:
    lr: 1e-5  # Lower LR for stable fine-tuning
```

**Rationale**:
- Shallow layers (0-5) capture low-level features (edges, textures) - keep frozen
- Deep layers (6-11) learn task-specific representations - fine-tune these
- Lower LR prevents large weight updates that could break pre-trained features
- Decoder already converged, so only gentle refinement needed

**Expected Results**:
- PSNR: ~41-42 dB (+3 dB improvement)
- SSIM: ~0.985-0.990
- Training time: ~3.5 hours (RTX 4050)

#### **Why Two Stages?**

**Alternative (Single-Stage Training)**:
- Train everything from scratch: PSNR ~35 dB, takes 12+ hours
- Train with all frozen: PSNR ~38 dB, decoder limited by frozen features

**Our Approach**:
- Stage A: Fast decoder convergence (2.5 hrs) → 38 dB
- Stage B: Encoder adaptation (3.5 hrs) → 41.5 dB
- **Total**: 6 hours, 41.5 dB ✅

**Benefits**:
1. **Faster convergence**: 6 hours vs 12+ hours
2. **Better final performance**: 41.5 dB vs 35 dB
3. **Stable training**: No catastrophic forgetting
4. **Lower risk**: Stage A produces usable model; Stage B only improves

### Training the Two-Stage Model

**Full Pipeline**:
```bash
# Stage A: Train decoder only (25 epochs)
python scripts/train.py --config configs/experiments/stage_a_decoder.yaml

# Stage B: Fine-tune encoder + decoder (30 epochs)
python scripts/train.py --config configs/experiments/stage_b_finetune.yaml \
    --resume outputs/checkpoints/stage_a/best_model_psnr.pth

# Or use the base config which already implements this strategy
python scripts/train.py --config configs/base.yaml
```

**Monitoring Progress**:
```bash
# Watch metrics during training
tensorboard --logdir outputs/logs

# Expected progression:
# Stage A: Loss 1.8 → 0.02, PSNR 15 → 38 dB (rapid improvement)
# Stage B: Loss 0.02 → 0.01, PSNR 38 → 41.5 dB (gradual refinement)
```

### Memory Optimizations

The architecture achieves ~3GB VRAM usage through:

1. **Gradient Checkpointing** (~30% savings)
   - Recompute activations during backward pass
   - Trade compute for memory

2. **Layer Freezing** (~25% savings)
   - Frozen layers don't store gradients
   - 50% of encoder frozen = 43M fewer gradient tensors

3. **Mixed Precision FP16** (~40% savings)
   - Activations stored in FP16 (2 bytes) vs FP32 (4 bytes)
   - Safe for most operations with automatic loss scaling

4. **Gradient Accumulation** (enables larger batch size)
   - Micro-batch 8 → accumulate 8 steps = effective batch 64
   - Improves convergence without OOM

5. **Lightweight Decoder** (~5% of total parameters)
   - Only 4.2M parameters vs 86M encoder
   - Most VRAM used by encoder

**VRAM Breakdown** (batch_size=8, FP16):
- Model weights: ~0.7 GB
- Activations: ~1.2 GB
- Gradients: ~1.0 GB
- Optimizer states: ~0.3 GB
- **Total**: ~3.2 GB ✅

### Forward Pass Flow

Complete data flow through the model:

```python
# Input: Noisy 13-band image
x = torch.randn(8, 13, 192, 192)  # [batch, channels, height, width]

# 1. Patch Embedding
patches = patch_embed(x)          # [8, 144, 768]  (12×12 patches of 16×16)

# 2. Positional Encoding
patches = patches + pos_embed      # Add learned position information

# 3. Transformer Encoder (12 blocks)
for block in transformer_blocks:
    patches = block(patches)       # Self-attention + FFN

# 4. Reshape to Spatial
features = patches.reshape(8, 768, 12, 12)

# 5. Progressive Upsampling
x = upsample_stage1(features)     # [8, 384, 24, 24]
x = residual_blocks(x)

x = upsample_stage2(x)            # [8, 192, 48, 48]
x = residual_blocks(x)

x = upsample_stage3(x)            # [8, 96, 96, 96]
x = residual_blocks(x)

x = upsample_stage4(x)            # [8, 48, 192, 192]
x = residual_blocks(x)

# 6. Final Projection
output = final_conv(x)            # [8, 13, 192, 192]

# Output: Restored 13-band image
- Progressive upsampling: 14→28→56→112→224
- Channels: 384→192→96→48
- ~4M parameters
    ↓
Output [B, 13, 224, 224]
```

### Memory Optimizations

| Technique | VRAM Savings | Implementation |
|-----------|-------------|----------------|
| Mixed Precision (FP16) | ~40% | `torch.cuda.amp` |
| Gradient Checkpointing | ~30% | `torch.utils.checkpoint` |
| Frozen Layers | ~50% params | `param.requires_grad = False` |
| Gradient Accumulation | Effective 8x batch | `accumulation_steps=8` |

**Total VRAM**: ~3GB (leaves 3GB safety margin on 6GB GPU)

---

## 📁 Project Structure

```
Multi-Spectral-Satellite-Image-Denoising/
├── configs/                         # YAML configurations
│   ├── base.yaml                   # Base configuration (192x192, RTX 4050 optimized)
│   ├── deployment.yaml             # ONNX export settings
│   └── experiments/                # Experiment-specific configs
│       ├── high_noise.yaml        # Challenging noise levels
│       ├── low_noise.yaml         # Light noise for quick testing
│       ├── medium_noise.yaml      # Balanced noise simulation
│       ├── quick_test.yaml        # 10-epoch sanity check
│       ├── stage_a_decoder.yaml   # Decoder-only training
│       └── stage_b_finetune.yaml  # Full model fine-tuning
│
├── data/                           # Dataset storage
│   └── EuroSAT_MS/                # 13-band Sentinel-2 imagery
│       ├── AnnualCrop/            # 2,700 images per class
│       ├── Forest/
│       ├── HerbaceousVegetation/
│       ├── Highway/
│       ├── Industrial/
│       ├── Pasture/
│       ├── PermanentCrop/
│       ├── Residential/
│       ├── River/
│       └── SeaLake/
│
├── notebooks/                      # Jupyter notebooks
│   ├── 00_quick_setup_test.ipynb  # Setup, data download, quick test
│   ├── 02_training.ipynb          # Interactive training with visualization
│   └── 03_evaluation.ipynb        # Evaluation, ONNX export, benchmarking
│
├── outputs/                        # Training outputs
│   ├── checkpoints/               # Model checkpoints
│   │   ├── best_model.pth
│   │   ├── best_model_loss.pth
│   │   ├── best_model_psnr.pth
│   │   ├── stage_a/              # Decoder-only training checkpoints
│   │   └── stage_b/              # Fine-tuned model checkpoints
│   ├── logs/                      # Training logs
│   │   ├── evaluation_results.json
│   │   ├── training_history.json
│   │   └── training_history.npz
│   └── onnx/                      # Exported ONNX models
│       └── satmae_restoration.onnx
│
├── scripts/                        # Command-line scripts
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Evaluation script
│   └── export.py                  # ONNX export script
│
├── src/                            # Source code
│   ├── data/                      # Dataset & data loading
│   │   ├── dataset.py            # EuroSATMultiSpectral dataset
│   │   ├── dataloader.py         # DataLoader wrappers
│   │   └── transforms.py         # Noise simulation & augmentation
│   ├── models/                    # Model architecture
│   │   ├── satmae_restoration.py # Main restoration model
│   │   ├── encoder.py            # SatMAE ViT encoder
│   │   ├── decoder.py            # UNet-style decoder
│   │   └── blocks.py             # Reusable building blocks
│   ├── training/                  # Training utilities
│   │   ├── trainer.py            # Training loop
│   │   ├── eval.py               # Evaluation functions
│   │   ├── losses.py             # Loss functions (MSE + SSIM)
│   │   └── metrics.py            # PSNR, SSIM, SAM, RMSE, MAE
│   ├── deployment/                # Deployment tools
│   │   ├── export_onnx.py        # PyTorch → ONNX conversion
│   │   ├── onnx_inference.py     # ONNX Runtime wrapper
│   │   └── optimize_model.py     # Model optimization & quantization
│   └── utils/                     # Utilities
│       ├── config.py             # Configuration management
│       ├── checkpointing.py      # Checkpoint saving/loading
│       ├── visualization.py      # Plotting utilities
│       ├── notebook_helpers.py   # Jupyter notebook helpers
│       ├── setup_helpers.py      # Model/optimizer setup
│       └── download.py           # Data download automation
│
├── weights/                        # Pre-trained weights
│   └── satmae_pretrain.pth        # SatMAE ViT-Base (1.3GB)
│
├── tests/                          # Unit tests
│
├── GETTING_STARTED.md              # Quick start guide
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
└── .gitignore                      # Git ignore rules
```
---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ajammoussi/Multi-Spectral-Satellite-Image-Denoising.git
cd Multi-Spectral-Satellite-Image-Denoising

# Create conda environment
conda create -n satmae python=3.10
conda activate satmae

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Optional: Install rasterio for .tif support
conda install -c conda-forge rasterio
```

### 2. Download Data & Weights

**Option A: Automated (Recommended)**
```python
# Run setup notebook
jupyter notebook notebooks/00_quick_setup_test.ipynb
```

**Option B: Manual**
```bash
# Download EuroSAT Multi-Spectral dataset
# Place in: data/EuroSAT_MS/

# Download SatMAE weights from:
# https://github.com/sustainlab-group/SatMAE
# Place in: weights/satmae_pretrain.pth
```

### 3. Train the Model

**Option A: Single-Stage Training (Recommended for Beginners)**

Quick test (10 epochs, ~1 hour):
```bash
python scripts/train.py --config configs/experiments/quick_test.yaml
```

Full training with base config (100 epochs, ~6 hours on RTX 4050):
```bash
python scripts/train.py --config configs/base.yaml
```

**Option B: Two-Stage Training (Recommended for Best Results)**

Stage A - Train decoder only (25 epochs, ~2.5 hours):
```bash
python scripts/train.py --config configs/experiments/stage_a_decoder.yaml
```

Stage B - Fine-tune encoder + decoder (30 epochs, ~3.5 hours):
```bash
python scripts/train.py --config configs/experiments/stage_b_finetune.yaml \
    --resume outputs/checkpoints/stage_a/best_model_psnr.pth
```

**Why two stages?**
- Faster convergence (6 hours total vs 12+ hours single-stage)
- Better final performance (41.5 dB vs 35 dB)
- Lower risk of catastrophic forgetting
- See [Two-Stage Training Strategy](#two-stage-training-strategy) for details

**Option C: Interactive Training with Visualization**
```bash
jupyter notebook notebooks/02_training.ipynb
```

### 4. Evaluate & Export

```bash
# Evaluate on validation set
python scripts/evaluate.py --checkpoint outputs/checkpoints/stage_b/best_model_psnr.pth

# Export to ONNX
python scripts/export.py --checkpoint outputs/checkpoints/stage_b/best_model_psnr.pth

# Or use the comprehensive evaluation notebook
jupyter notebook notebooks/03_evaluation.ipynb
```

### 5. Use Trained Model

```python
from src.deployment import ONNXInferenceSession
from src.data import EuroSATMultiSpectral

# Load ONNX model
session = ONNXInferenceSession('outputs/onnx/satmae_restoration.onnx')

# Run inference
restored_image = session.predict(noisy_image)  # Input: [1, 13, 192, 192]
```

---

## 📊 Training Configuration

### Base Configuration (`configs/base.yaml`)

- **Image Size**: 192×192 pixels (matches checkpoint)
- **Batch Size**: 8 (micro) × 8 (accumulation) = 64 effective
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.05)
- **Scheduler**: Cosine Annealing with Warm Restarts
- **Loss**: MSE (1.0) + SSIM (0.1)
- **Mixed Precision**: FP16 enabled
- **Gradient Checkpointing**: Enabled
- **Frozen Layers**: First 6 encoder blocks (50% parameters)

### Training Strategies

**Single-Stage (base.yaml)**:
- Train for 100 epochs with partially frozen encoder
- Good balance between speed and performance
- PSNR: ~40-41 dB

**Two-Stage (stage_a + stage_b)**:
- **Stage A**: Train decoder only, all encoder frozen (25 epochs, lr=1e-4)
- **Stage B**: Fine-tune last 6 encoder blocks + decoder (30 epochs, lr=1e-5)
- Best performance: PSNR ~41.5 dB
- Total time: ~6 hours (faster than single-stage to same quality)

**Quick Test (quick_test.yaml)**:
- 10 epochs, lower resolution
- Good for pipeline validation
- Time: ~1 hour

### Noise Simulation

Realistic sensor degradation modeling:
- **Gaussian Noise**: σ=0.015 (thermal/electronic noise)
- **Speckle Noise**: σ=0.008 (coherent noise)
- **Dead Bands**: 8% probability (sensor malfunction)
- **Thermal Scaling**: 0.005 (wavelength-dependent)

---

```bash
# Create data directory
mkdir -p data/EuroSAT

# Download and extract
# https://github.com/phelber/EuroSAT
# Expected structure:
# data/EuroSAT/
#   AnnualCrop/*.tif
#   Forest/*.tif
#   ...
```

### 4. Train Model

```bash
# Train with base configuration
python scripts/train.py --config configs/base.yaml

# Train with low noise experiment
python scripts/train.py --config configs/experiments/low_noise.yaml

# Resume training
python scripts/train.py --config configs/base.yaml --resume

# Monitor training with TensorBoard
tensorboard --logdir outputs/logs
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
    --config configs/base.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --save_visualizations
```

### 6. Export to ONNX

```bash
python scripts/export.py \
    --config configs/base.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --output outputs/onnx/model.onnx \
    --verify
```

---

## ⚙️ Configuration

Edit `configs/base.yaml` to customize:

```yaml
# Key parameters for RTX 4050 optimization
training:
  micro_batch_size: 8           # Actual GPU batch size
  gradient_accumulation_steps: 8 # Effective batch = 64
  mixed_precision: true          # FP16 training
  
model:
  encoder:
    freeze_layers: [0,1,2,3,4,5] # Freeze 50% of encoder
    gradient_checkpointing: true  # Save VRAM

noise:
  gaussian_sigma: 0.05           # Noise intensity
  dead_band_prob: 0.25           # 25% channel dropout
```

---

## 📊 Expected Results

### Performance (RTX 4050)

| Metric | Value |
|--------|-------|
| Training Time | ~6 hours (100 epochs, 27K samples) |
| VRAM Usage | ~3.0 GB (batch_size=8, FP16) |
| Inference Speed | ~120 ms/image (224×224×13) |

### Restoration Quality

| Noise Level | PSNR (dB) | SSIM | SAM (°) |
|-------------|-----------|------|---------|
| Low (σ=0.01) | 38.5 | 0.96 | 2.3 |
| Medium (σ=0.05) | 32.1 | 0.91 | 4.8 |
| High (σ=0.1) | 28.4 | 0.85 | 7.2 |

---



## 🔬 Advanced Usage

### Custom Noise Profiles

```python
from src.data.transforms import AddSensorNoise

# Create custom noise
custom_noise = AddSensorNoise(
    gaussian_sigma=0.08,
    dead_band_prob=0.3,
    thermal_scale=0.015,
    enable_striping=True  # Push-broom sensor artifacts
)
```

### Multi-GPU Training

```python
# In train.py
model = nn.DataParallel(model)  # Simple multi-GPU
# Or use DistributedDataParallel for better performance
```

### Hyperparameter Tuning

```bash
# Create new experiment config
cp configs/base.yaml configs/experiments/custom.yaml

# Edit learning rate, architecture, etc.
# Train with new config
python scripts/train.py --config configs/experiments/custom.yaml
```

---

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir outputs/logs --port 6006
```

Tracks:
- Training/Validation Loss
- PSNR, SSIM, SAM metrics
- Learning rate schedule
- VRAM usage

### Weights & Biases (Optional)

```python
# Add to train.py
import wandb

wandb.init(project="satmae-denoiser", config=config)
wandb.watch(model)
```

---

## 🚢 Deployment

### ONNX Inference

```python
from src.deployment import ONNXInferenceSession

# Load ONNX model
session = ONNXInferenceSession('outputs/onnx/model.onnx')

# Run inference
import numpy as np
noisy_image = np.random.randn(1, 13, 224, 224).astype(np.float32)
clean_image = session.predict(noisy_image)

# Benchmark performance
stats = session.benchmark(num_iterations=100)
print(f"FPS: {stats['fps']:.2f}")
```

### Model Optimization

```python
from src.deployment import quantize_model, optimize_onnx

# Optimize ONNX graph
optimize_onnx('outputs/onnx/model.onnx', 'outputs/onnx/model_optimized.onnx')

# Quantize to INT8 (faster inference, ~4x smaller)
quantize_model('outputs/onnx/model.onnx', 'outputs/onnx/model_int8.onnx')
```

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{cong2022satmae,
  title={SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery},
  author={Cong, Yezhen and Khanna, Samar and Meng, Chenlin and Liu, Patrick and Rozi, Erik and He, Yutong and Burke, Marshall and Lobell, David B and Ermon, Stefano},
  journal={NeurIPS},
  year={2022}
}
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce micro_batch_size in config
training:
  micro_batch_size: 4  # Instead of 8
  gradient_accumulation_steps: 16  # Maintain effective batch_size
```

### Slow Training

- Check `num_workers` in dataloader (set to 4-8)
- Enable `pin_memory: true` in config
- Verify CUDA is being used: `torch.cuda.is_available()`

### Poor Restoration Quality

- Increase training epochs
- Try different freeze_layers configuration
- Adjust loss weights (increase SSIM weight)

---

## 🙏 Acknowledgments

- **SatMAE**: Pre-trained weights from [sustainlab-group/SatMAE](https://github.com/sustainlab-group/SatMAE)
- **EuroSAT**: Dataset from [EuroSAT](https://github.com/phelber/EuroSAT)
- **timm**: Vision Transformer implementation

---

**Built with ❤️ for Earth Observation**
