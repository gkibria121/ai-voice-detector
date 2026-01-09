# AI Voice Detector — Comprehensive Documentation

> **Version:** 1.0 | **Last Updated:** January 2026

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Dataset: Fake-or-Real](#dataset-fake-or-real)
5. [Feature Extraction Methodologies](#feature-extraction-methodologies)
6. [Data Augmentation Pipeline](#data-augmentation-pipeline)
7. [Model Architectures](#model-architectures)
8. [Training Methodology](#training-methodology)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Visualization Tools](#visualization-tools)
11. [Streamlit Web Application](#streamlit-web-application)
12. [Configuration System](#configuration-system)
13. [Performance Optimizations](#performance-optimizations)
14. [CLI Reference](#cli-reference)
15. [Developer Guide](#developer-guide)
16. [Examples & Quick Commands](#examples--quick-commands)

---

## Project Overview

AI Voice Detector is a research and production-ready codebase for **detecting synthetic/deepfake audio**. This repository implements state-of-the-art deep learning architectures for audio spoofing detection, focusing on the **Fake-or-Real** dataset benchmark (2-second audio clips).

### Key Capabilities

- **Multiple Model Architectures**: EfficientNetB2, SEResNet, LCNN, RawNet3, SimpleCNN, FusionNet, and ensemble models
- **Rich Feature Extraction**: Raw waveform, Mel-spectrogram, LFCC, MFCC, and CQT representations
- **Comprehensive Augmentation**: RIR simulation, MUSAN-style noise, SpecAugment, pitch shift, time stretch, and more
- **Production Optimizations**: PyTorch 2.x optimizations, mixed precision training (AMP), TF32, BF16, `torch.compile()`
- **End-to-End Pipeline**: Training, evaluation, visualization, and deployment via Streamlit

---

## Repository Layout

```
ai-voice-detector/
├── main.py                    # Main training/evaluation entrypoint
├── app.py                     # Streamlit web application for inference
├── cli.py                     # Command-line argument parser
├── realtime.py                # Real-time inference utilities
├── data_utils.py              # Feature extraction & augmentation functions
├── dataset_factory.py         # Dataset loaders for Fake-or-Real
├── evaluation.py              # EER, t-DCF computation and evaluation
├── metrics.py                 # MetricsTracker, plotting, and visualization
├── utils.py                   # Optimizers, schedulers, seeding utilities
├── feature_analysis.py        # Audio feature analysis and visualization
├── visualize_results.py       # Training results visualization and comparison
├── download_dataset.py        # Dataset download helper
├── notebook.ipynb             # Interactive Jupyter notebook with experiments
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container definition
├── docker-compose.yml         # Docker Compose for CPU
├── docker-compose-gpu.yml     # Docker Compose for GPU
│
├── models/                    # Neural network architectures
│   ├── EfficientNetB2.py      # EfficientNet-B2 with optional attention
│   ├── SEResNet.py            # SE-ResNet with attentive statistics pooling
│   ├── LCNN.py                # Light CNN with Max-Feature-Map activation
│   ├── RawNet3.py             # Raw waveform model with SincConv
│   ├── SimpleCNN.py           # Lightweight baseline CNN
│   └── FusionNet.py           # Multi-modal fusion architecture
│
├── config/                    # Model configuration files (JSON)
│   ├── EfficientNetB2.conf
│   ├── EfficientNetB2_Attention.conf
│   ├── SEResNet.conf
│   ├── LCNN.conf
│   ├── LCNN_Large.conf
│   ├── RawNet3.conf
│   ├── SimpleCNN.conf
│   └── ensemble.conf
│
├── exp_result/                # Experiment outputs (checkpoints, metrics)
├── comparison_plots/          # Generated model comparison visualizations
├── fake_or_real/              # Dataset directory
│   ├── for-original/
│   ├── for-norm/
│   ├── for-2sec/              # Default 2-second clips
│   └── for-rerec/
└── bin/
    └── start.sh               # Startup script
```

---

## Installation

### Requirements

- **Python**: 3.8+ (3.10+ recommended)
- **CUDA**: 11.7+ for GPU acceleration (optional)
- **Hardware**: NVIDIA GPU with compute capability 7.0+ (T4, V100, A100, RTX 30/40 series)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-voice-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `torchvision` | Pre-trained models (EfficientNet) |
| `torchcontrib` | Stochastic Weight Averaging (SWA) |
| `librosa` | Audio processing and feature extraction |
| `soundfile` | Audio file I/O |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualizations |
| `tensorboard` | Training monitoring |
| `tqdm` | Progress bars |
| `pandas` | Data manipulation |
| `kagglehub` | Dataset downloading |

### Docker Deployment

```bash
# CPU-only deployment
docker-compose up

# GPU deployment (requires nvidia-docker)
docker-compose -f docker-compose-gpu.yml up
```

---

## Dataset: Fake-or-Real

### Overview

The **Fake-or-Real** dataset is a binary classification benchmark for detecting AI-generated speech. It contains real human speech recordings and synthetic audio generated by various TTS/voice cloning systems.

### Dataset Versions

| Version | Flag | Description | Path |
|---------|------|-------------|------|
| 1 | `--dataset_version 1` | Original full-length clips | `for-original/` |
| 2 | `--dataset_version 2` | Normalized audio | `for-norm/` |
| **3** | `--dataset_version 3` | **2-second clips (default)** | `for-2sec/` |
| 4 | `--dataset_version 4` | Re-recorded audio | `for-rerec/` |

### Expected Directory Structure

```
fake_or_real/for-2sec/for-2seconds/
├── training/
│   ├── real/
│   │   ├── audio_001.wav
│   │   └── ...
│   └── fake/
│       ├── audio_001.wav
│       └── ...
├── validation/
│   ├── real/
│   └── fake/
└── testing/
    ├── real/
    └── fake/
```

### Dataset Classes

- **`Dataset_FakeOrReal_train`**: Training dataset with augmentation support
- **`Dataset_FakeOrReal_devNeval`**: Validation/test dataset (no augmentation)

Both classes handle automatic feature extraction and padding/cropping to fixed lengths.

---

## Feature Extraction Methodologies

The system supports **5 feature types** for representing audio:

### Feature Type 0: Raw Waveform

- **Description**: Direct audio signal as 1D time series
- **Shape**: `(samples,)` typically 64,600 samples (~4 seconds at 16kHz)
- **Use Case**: End-to-end models like RawNet3
- **Advantages**: Preserves all audio information; no feature engineering
- **Disadvantages**: Requires more model parameters; longer training

### Feature Type 1: Mel-Spectrogram (Recommended)

- **Description**: Time-frequency representation using mel-scaled filterbank
- **Parameters**: 128 mel bins, 512-sample FFT, 160-sample hop length
- **Shape**: `(128, T)` where T depends on audio duration
- **Use Case**: CNN-based models (EfficientNetB2, LCNN, SEResNet)
- **Advantages**: Perceptually motivated; mimics human hearing; compact representation
- **Processing**:
  ```python
  mel_spec = librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=128, n_fft=512, hop_length=160)
  mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
  ```

### Feature Type 2: LFCC (Linear Frequency Cepstral Coefficients)

- **Description**: Cepstral coefficients using linear frequency scale filterbank
- **Parameters**: 20 linear filters, 13 cepstral coefficients
- **Shape**: `(13, T)`
- **Use Case**: Spoofing detection (captures synthesis artifacts)
- **Processing**:
  1. Compute STFT power spectrum
  2. Apply linear-spaced triangular filterbank
  3. Log compression
  4. DCT to obtain cepstral coefficients

### Feature Type 3: MFCC (Mel-Frequency Cepstral Coefficients)

- **Description**: Classic speech features using mel filterbank + DCT
- **Parameters**: 13 coefficients, 512-sample FFT, 160-sample hop
- **Shape**: `(13, T)`
- **Use Case**: Traditional speech processing; compact representation
- **Processing**:
  ```python
  mfcc = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13, n_fft=512, hop_length=160)
  ```

### Feature Type 4: CQT (Constant-Q Transform)

- **Description**: Variable-resolution time-frequency representation
- **Parameters**: 84 bins, 12 bins per octave, 512-sample hop
- **Shape**: `(84, T)`
- **Use Case**: Harmonic content analysis; detecting pitch artifacts
- **Advantages**: Better frequency resolution at low frequencies
- **Processing**:
  ```python
  cqt = librosa.cqt(y=waveform, sr=16000, hop_length=512, n_bins=84, bins_per_octave=12)
  cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
  ```

### Feature Type 8: Prosodic Features
- **Description**: High-level prosodic features capturing intonation and rhythm
- **Components**: F0 (pitch), Energy (RMS), Zero-Crossing Rate, Voiced Probability, Speaking Rate
- **Shape**: `(7, T)`
- **Use Case**: Detecting TTS artifacts (unnatural pitch/rhythm)
- **Processing**:
  ```python
  # Uses librosa.pyin for pitch tracking
  features = extract_feature(waveform, feature_type=8)
  ```

### Multimodal Feature Fusion


The system supports combining multiple feature types:

```bash
python main.py --feature_type 1,2,4  # Mel-spec + LFCC + CQT fusion
```

Features are stacked along the channel dimension and processed by a `MultimodalFusionWrapper`.

---

## Data Augmentation Pipeline

Augmentation is enabled via `--random_noise` flag. The system applies **composed augmentations** for robust training.

### Augmentation Types

| ID | Name | Description | Parameters |
|----|------|-------------|------------|
| 0 | None | No augmentation | - |
| 1 | Gaussian Noise | Additive white noise | SNR: 10-25 dB |
| 2 | Background Noise | White/pink noise | Factor: 0.01-0.05 |
| 3 | Reverberation | Echo/room simulation | Factor: 0.3-0.8, 50ms delay |
| 4 | Pitch Shift | Pitch alteration | ±4 semitones |
| 5 | Time Stretch | Speed perturbation | Rate: 0.85-1.15x |
| 6 | Gain | Volume adjustment | ±6 dB |
| 7 | Low-pass Filter | High frequency attenuation | Cutoff: 2000-6000 Hz |
| 8 | High-pass Filter | Low frequency attenuation | Cutoff: 50-300 Hz |
| 9 | RIR Simulation | Room Impulse Response | RT60: 0.1-0.5s |
| 10 | MUSAN-style Noise | Babble/music/ambient | SNR: 5-25 dB |

### Composed Augmentation

Applied via `apply_composed_augmentation()`:

```python
def apply_composed_augmentation(waveform, sr=16000, num_augmentations=2, augment_prob=0.8):
    # 80% chance to apply augmentation
    # Randomly select 1-2 augmentation types
    # Apply sequentially
```

### SpecAugment for Spectrograms

Applied via `apply_spectrogram_augmentation()`:

- **Frequency Masking**: Masks random frequency bands (up to 20 bins)
- **Time Masking**: Masks random time segments (up to 50 frames)
- **Probability**: 50% each for frequency and time masking

---

## Model Architectures

### EfficientNetB2 (`models/EfficientNetB2.py`)

**Architecture**: Pre-trained EfficientNet-B2 backbone adapted for single-channel spectrogram input.

**Key Features**:
- Transfer learning from ImageNet weights
- Modified first conv layer for single-channel input
- Custom classifier head with dropout and batch normalization
- Optional backbone freezing for fine-tuning

**Variants**:
- `Model`: Standard EfficientNetB2 with global average pooling
- `ModelWithAttention`: Adds attention-based spatial pooling for better temporal modeling

**Parameters**: ~9.2M

**Config**:
```json
{
  "architecture": "EfficientNetB2",
  "model_variant": "attention",
  "dropout": 0.4,
  "pretrained": true,
  "freeze_backbone": false,
  "att_bottleneck": 128
}
```

### SEResNet (`models/SEResNet.py`)

**Architecture**: Squeeze-and-Excitation ResNet with Attentive Statistics Pooling.

**Key Features**:
- SE blocks for channel-wise attention
- Residual connections for deep feature learning
- Attentive statistics pooling (mean + std with attention weights)

**Components**:
- `SEBlock`: Channel attention module
- `BasicBlockSE`: SE-ResNet residual block
- `AttentiveStatPool`: Attention-weighted pooling

**Config**:
```json
{
  "architecture": "SEResNet",
  "block_channels": [64, 128, 256, 512],
  "layers": [3, 4, 6, 3],
  "dropout": 0.3,
  "att_bottleneck": 128
}
```

### LCNN (`models/LCNN.py`)

**Architecture**: Light CNN with Max-Feature-Map (MFM) activation.

**Key Features**:
- MFM activation: Competitive feature suppression for robust learning
- Lightweight design suitable for real-time deployment
- Residual blocks with MFM

**Components**:
- `MaxFeatureMap2D`: Splits channels, takes element-wise max
- `LCNNBlock`: Conv → MFM → BatchNorm
- `LCNNResBlock`: Residual connection with MFM
- `AttentiveStatisticsPooling`: Attention-weighted statistics

**Variants**:
- `Model`: Standard LCNN
- `ModelLarge`: Deeper variant with more channels

**Reference**: Based on designs from top ASVspoof challenge systems.

### RawNet3 (`models/RawNet3.py`)

**Architecture**: End-to-end raw waveform model with learnable filterbank.

**Key Features**:
- **SincConv**: Learnable sinc-function filterbank (no hand-crafted features)
- **Res2NetBlock**: Multi-scale residual processing
- Self-attention pooling

**Components**:
- `SincConv`: Parametric sinc filters with mel-spaced initialization
- `Res2NetBlock`: Hierarchical residual connections with multiple scales
- `RawNet3Backbone`: End-to-end feature extraction

**Config**:
```json
{
  "architecture": "RawNet3",
  "nb_samp": 64600,
  "channels": [64, 128, 256, 512]
}
```

### SimpleCNN (`models/SimpleCNN.py`)

**Architecture**: Minimal 3-layer 1D CNN for baseline experiments.

**Use Case**: Fast prototyping, sanity checks, embedded deployment.

### FusionNet (`models/FusionNet.py`)

**Architecture**: Multi-modal fusion network with attention.

**Key Features**:
- Separate feature extractors per modality
- Self-attention fusion across modalities
- Supports variable input shapes

**Components**:
- `FeatureCNN`: Per-modality CNN feature extractor
- `AttentionFusion`: Cross-modal attention mechanism

### Wav2Vec 2.0 (`models/Wav2Vec2.py`)
**Architecture**: Wrapper for HuggingFace `Wav2Vec2Model`.
**Key Features**:
- Self-supervised pre-training on large speech corpora
- Contextualized representations from raw audio
- Fine-tuning or frozen encoder modes
**Config**: `config/Wav2Vec2.conf`

### HuBERT (`models/HuBERT.py`)
**Architecture**: Wrapper for HuggingFace `HubertModel`.
**Key Features**:
- Masked prediction of hidden units
- Weighted layer sum for feature combination
**Config**: `config/HuBERT.conf`

### MobileViT (`models/MobileViT.py`)
**Architecture**: Lightweight Transformer-CNN hybrid.
**Key Features**:
- ~2M parameters (vs >10M for others)
- Mobile-friendly, efficient inference
- Global processing via transformer blocks
**Config**: `config/MobileViT.conf`

### Ensemble Models


The system supports **ensemble learning** via configuration:

```json
{
  "model_config": [
    {"architecture": "LCNN", ...},
    {"architecture": "SEResNet", ...},
    {"architecture": "EfficientNetB2", ...}
  ]
}
```

**Ensemble Wrapper** (`EnsembleModel`):
- Runs each sub-model independently
- Averages logits for final prediction
- Projects embeddings to common dimension for unified representation

---

## Training Methodology

### Training Pipeline

1. **Data Loading**: `create_dataset_loaders()` creates train/dev/eval loaders
2. **Forward Pass**: Mixed precision (AMP) for efficiency
3. **Loss Computation**: Weighted Cross-Entropy (weight: [0.1, 0.9] for class imbalance)
4. **Backward Pass**: Gradient scaling for FP16 stability
5. **Optimization**: Adam/SGD with learning rate scheduling
6. **Validation**: EER computation on dev set each epoch
7. **Checkpointing**: Save best model based on dev EER
8. **SWA**: Stochastic Weight Averaging for improved generalization

### Loss Function

**Weighted Cross-Entropy**:
```python
weight = torch.FloatTensor([0.1, 0.9]).to(device)  # [fake, real]
criterion = nn.CrossEntropyLoss(weight=weight)
```

The weighting addresses potential class imbalance in spoofing detection.

### Optimizers

Configured in `config/*.conf`:

**Adam** (default):
```json
{
  "optimizer": "adam",
  "base_lr": 0.0001,
  "betas": [0.9, 0.999],
  "weight_decay": 0.0001,
  "amsgrad": false
}
```

**SGD**:
```json
{
  "optimizer": "sgd",
  "base_lr": 0.01,
  "momentum": 0.9,
  "weight_decay": 0.0001,
  "nesterov": false
}
```

### Learning Rate Schedulers

| Scheduler | Description | Config |
|-----------|-------------|--------|
| `cosine` | Cosine annealing decay | `"scheduler": "cosine"` |
| `sgdr` | SGD with warm restarts | `"scheduler": "sgdr", "T0": 10, "Tmult": 2` |
| `multistep` | Step decay at milestones | `"scheduler": "multistep", "milestones": [10, 15]` |
| `keras_decay` | Keras-style inverse decay | `"scheduler": "keras_decay"` |

### Stochastic Weight Averaging (SWA)

Enabled via `--weight_avg`:

- Collects model snapshots on best epochs and late epochs
- Averages weights for better generalization
- Updates BatchNorm statistics after averaging
- Saves to `weights/swa.pth`

### Gradient Clipping

```json
{"grad_clip": 5.0}
```

Prevents gradient explosion during training.

---

## Evaluation Metrics

### Equal Error Rate (EER)

Primary metric for spoofing detection:

- **Definition**: Operating point where False Acceptance Rate = False Rejection Rate
- **Computation**: `evaluation.compute_eer(bona_scores, spoof_scores)`
- **Lower is better** (0% = perfect separation)

### Accuracy

Computed at EER threshold:
```python
correct_bona = sum(bona_scores >= threshold)
correct_spoof = sum(spoof_scores < threshold)
accuracy = (correct_bona + correct_spoof) / total_samples * 100
```

### Detection Error Trade-off (DET) Curve

Visualizes trade-off between FAR and FRR across thresholds.

### ROC-AUC

Area Under the ROC Curve for overall discrimination ability.

### Score File Format

```
filename label score
training/real/audio_001.wav bonafide 2.3456
training/fake/audio_002.wav spoof -1.2345
```

---

## Visualization Tools

### MetricsTracker (`metrics.py`)

Tracks and saves training metrics:
- Training loss per epoch
- Dev EER, t-DCF, accuracy
- Eval EER, t-DCF, accuracy
- Best metrics

**Outputs**:
- `metrics.json`: Complete metrics history
- `metrics.csv`: Tabular format
- `training_metrics.png`: Training curves
- `final_metrics.png`: Final performance summary

### visualize_results.py

Generate comparative visualizations:

```bash
# Single experiment
python visualize_results.py --path exp_result/model_name/metrics

# Compare multiple experiments
python visualize_results.py --path "exp_result/*/metrics" --compare --output ./comparison_plots
```

**Generated Plots**:
- Training loss curves
- Dev/Eval EER comparison
- Model comparison bar charts
- Accuracy trends

### feature_analysis.py

Analyze audio features:

```bash
python -c "from feature_analysis import analyze_and_visualize_features; \
           analyze_and_visualize_features('audio.wav', feature_type=1, save_dir='analysis')"
```

**Visualizations**:
- Waveform plot
- Feature spectrogram
- Feature statistics
- Value distribution histogram
- Temporal dynamics

---

## Streamlit Web Application

### Overview

`app.py` provides a web interface for audio classification using trained models.

### Usage

```bash
streamlit run app.py -- --config config/EfficientNetB2_Attention.conf --eval_model_weights path/to/weights.pth
```

### Features

- Upload multiple audio files (WAV, FLAC, MP3)
- Audio playback in browser
- Real-time classification (Fake/Real)
- Confidence scores
- Debug mode for tensor/logit inspection

### Implementation Details

- Uses `torch.inference_mode()` for fast inference
- Automatic preprocessing (resampling, padding)
- Supports all model architectures
- Cached model loading with `@st.cache_resource`

---

---

## Real-time Classification (`realtime.py`)

The system includes a production-ready streaming pipeline for live audio detection.

### Features
- **Sliding Window**: Processes audio in overlapping chunks (e.g., 2s window, 0.5s step)
- **EMA Smoothing**: Stabilizes predictions using Exponential Moving Average
- **Low Latency**: Optimized for real-time feedback

### Usage
```bash
# Microphone input
python realtime.py --model_path weights.pth --config config.conf

# File processing
python realtime.py --file audio.wav --model_path weights.pth
```

---

## Domain Adaptation (`domain_adaptation.py`)

Tools for improving generalization across different datasets.

**Methods**:
- **CORAL**: Correlation Alignment (matches feature covariances)
- **MMD**: Maximum Mean Discrepancy (kernel-based matching)
- **DANN**: Domain Adversarial Neural Network (adversarial training)

### Usage
```python
from domain_adaptation import DomainAdaptationTrainer
trainer = DomainAdaptationTrainer(model, adaptation_method='coral')
```

---

## Knowledge Distillation (`models/DistillationTrainer.py`)

Compress large models (Teacher) into lightweight models (Student).

### Usage
```python
from models.DistillationTrainer import distill_model
distill_model(teacher, student, train_loader, ...)
```

---

## Configuration System


### Configuration File Structure

All hyperparameters are stored in JSON config files (`config/*.conf`):

```json
{
  "database_path": "./LA/",
  "batch_size": 24,
  "num_epochs": 25,
  "loss": "CCE",
  "eval_output": "eval_scores.txt",
  
  "use_amp": true,
  "use_bf16": true,
  "use_compile": false,
  "use_channels_last": true,
  "grad_clip": 5.0,
  
  "model_config": {
    "architecture": "EfficientNetB2",
    "model_variant": "attention",
    "dropout": 0.4,
    "pretrained": true
  },
  
  "optim_config": {
    "optimizer": "adam",
    "base_lr": 0.0001,
    "scheduler": "cosine"
  }
}
```

### Command-Line Overrides

Most config parameters can be overridden via CLI:

| Flag | Description |
|------|-------------|
| `--epochs N` | Override num_epochs |
| `--batch_size N` | Override batch_size |
| `--feature_type T` | Set feature type (0-4 or comma-separated) |
| `--random_noise` | Enable augmentation |
| `--weight_avg` | Enable SWA |
| `--eval_best` | Evaluate on test when best dev found |

---

## Performance Optimizations

### PyTorch 2.x Optimizations

The codebase leverages modern PyTorch features:

| Optimization | Description | Speedup |
|--------------|-------------|---------|
| `torch.compile()` | JIT compilation with inductor | 30-50% |
| TF32 | TensorFloat-32 for matmul (Ampere+) | 10-20% |
| BF16 | BFloat16 mixed precision (Ampere+) | 20-30% |
| Channels-last | Memory format for CNNs | 10-30% |
| CuDNN benchmark | Auto-tune convolution algorithms | 5-15% |

### Hardware Detection

```python
# BF16 native support check (Ampere+ GPUs)
def check_bf16_native_support():
    major, minor = torch.cuda.get_device_capability(0)
    return major >= 8  # Ampere, Ada, Hopper

# TF32 configuration
torch.backends.cuda.matmul.fp32_precision = 'tf32'
```

### Mixed Precision Training

```python
with torch.amp.autocast('cuda', dtype=amp_dtype):
    _, batch_out = model(batch_x)
    loss = criterion(batch_out, batch_y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### DataLoader Optimizations

- `pin_memory=True` for faster GPU transfer
- `persistent_workers=True` for worker reuse
- `prefetch_factor=2` for async data loading
- `non_blocking=True` for async device transfer

---

## CLI Reference

### Main Training Script

```bash
python main.py \
  --config config/MODEL.conf \        # Required: model configuration
  --dataset 1 \                       # Dataset type (1=Fake-or-Real)
  --dataset_version 3 \               # Dataset version (3=2-sec clips)
  --feature_type 1 \                  # Feature type (0-4 or comma-separated)
  --epochs 20 \                       # Number of epochs
  --batch_size 32 \                   # Batch size
  --random_noise \                    # Enable augmentation
  --weight_avg \                      # Enable SWA
  --eval_best \                       # Evaluate on best dev model
  --seed 1234 \                       # Random seed
  --output_dir ./exp_result           # Output directory
```

### Evaluation Only

```bash
python main.py \
  --eval \
  --eval_model_weights path/to/weights.pth \
  --config config/MODEL.conf \
  --dataset 1 \
  --feature_type 1
```

### All Available Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | Required | Path to config file |
| `--dataset` | int | 1 | Dataset type |
| `--dataset_version` | int | 3 | Dataset version |
| `--feature_type` | str | "0" | Feature type(s) |
| `--epochs` | int | config | Number of epochs |
| `--batch_size` | int | config | Batch size |
| `--random_noise` | flag | False | Enable augmentation |
| `--weight_avg` | flag | False | Enable SWA |
| `--eval_best` | flag | False | Eval on best dev |
| `--eval` | flag | False | Evaluation mode |
| `--eval_model_weights` | str | None | Weights for eval |
| `--seed` | int | 1234 | Random seed |
| `--output_dir` | str | ./exp_result | Output directory |
| `--data_subset` | float | 1.0 | Data fraction for debugging |
| `--cpu` | flag | False | Force CPU mode |
| `--feature_analysis` | flag | False | Generate feature visualizations |
| `--comment` | str | None | Experiment comment |

---

## Developer Guide

### Adding a New Model

1. Create `models/YourModel.py`:
```python
class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        # Build architecture from d_args
    
    def forward(self, x, Freq_aug=False):
        # Return (embeddings, logits)
        return embeddings, output
```

2. Create `config/YourModel.conf`:
```json
{
  "model_config": {
    "architecture": "YourModel",
    "your_param": 123
  }
}
```

3. Run training:
```bash
python main.py --config config/YourModel.conf --feature_type 1
```

### Adding a New Augmentation

1. Add function to `data_utils.py`:
```python
def add_your_augmentation(waveform, param=1.0):
    # Apply augmentation
    return augmented_waveform
```

2. Register in `apply_augmentation()`:
```python
elif augmentation_type == 11:
    return add_your_augmentation(waveform)
```

3. Add to `apply_composed_augmentation()` aug_types list.

### Adding a New Dataset

1. Register provider in `dataset_factory.py`:
```python
register_dataset_provider(new_id, {
    'name': 'YourDataset',
    'expected_layout': ['train/', 'test/']
})
```

2. Create dataset classes extending `Dataset`.

### Reproducibility

Ensure deterministic behavior:

```python
from utils import set_seed
set_seed(1234, config)
```

Seeds: Python random, NumPy, PyTorch (CPU + CUDA), CuDNN.

---

## Examples & Quick Commands

### Quick Training Examples

```bash
# EfficientNet-B2 with attention on mel-spectrograms
python main.py --config config/EfficientNetB2_Attention.conf \
  --feature_type 1 --dataset 1 --epochs 20 --random_noise --weight_avg

# LCNN on LFCC features
python main.py --config config/LCNN.conf \
  --feature_type 2 --dataset 1 --epochs 20 --random_noise

# RawNet3 on raw waveform
python main.py --config config/RawNet3.conf \
  --feature_type 0 --dataset 1 --epochs 20 --random_noise

# Multimodal fusion (Mel + LFCC + CQT)
python main.py --config config/EfficientNetB2.conf \
  --feature_type 1,2,4 --dataset 1 --epochs 20 --random_noise

# Ensemble model
python main.py --config config/ensemble.conf \
  --feature_type 1 --dataset 1 --epochs 20 --random_noise
```

### Evaluation Examples

```bash
# Evaluate saved model
python main.py --eval --eval_model_weights exp_result/model/weights/best.pth \
  --config config/EfficientNetB2_Attention.conf --dataset 1 --feature_type 1

# Evaluate SWA weights
python main.py --eval --eval_model_weights exp_result/model/weights/swa.pth \
  --config config/EfficientNetB2_Attention.conf --dataset 1 --feature_type 1
```

### Visualization Examples

```bash
# Compare all experiments
python visualize_results.py --path "exp_result/*/metrics" --compare --output ./comparison_plots

# Single experiment visualization
python visualize_results.py --path exp_result/FakeorReal_audio_EfficientNetB2_Attention_rand_ep20_bs24_feat1/metrics
```

### Streamlit App

```bash
streamlit run app.py -- \
  --config config/EfficientNetB2_Attention.conf \
  --eval_model_weights exp_result/model/weights/best.pth
```

---

## License

See the [LICENSE](LICENSE) file for terms.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{ai_voice_detector,
  title = {AI Voice Detector: Deepfake Audio Detection Pipeline},
  year = {2026},
  url = {https://github.com/your-repo/ai-voice-detector}
}
```

---

*Documentation generated: January 2026*
