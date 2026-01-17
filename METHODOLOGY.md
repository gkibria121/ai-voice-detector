# AI Voice Detector — Methodology Documentation

> **Version:** 1.0 | **Last Updated:** January 2026

This document provides an in-depth technical explanation of every step in the AI Voice Detector pipeline, from raw audio input to final classification output.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Data Loading & Preprocessing](#data-loading--preprocessing)
3. [Feature Extraction](#feature-extraction)
4. [Data Augmentation](#data-augmentation)
5. [Model Architecture Selection](#model-architecture-selection)
6. [Training Process](#training-process)
7. [Optimization Techniques](#optimization-techniques)
8. [Evaluation & Metrics](#evaluation--metrics)
9. [Inference Pipeline](#inference-pipeline)
10. [Reproducibility](#reproducibility)

---

## 1. Pipeline Overview

The AI Voice Detector follows a systematic end-to-end pipeline for detecting synthetic/deepfake audio:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    │
│  │  Audio   │───▶│  Feature    │───▶│    Data     │───▶│    Model     │    │
│  │  Files   │    │  Extraction │    │ Augmentation│    │   Forward    │    │
│  └──────────┘    └─────────────┘    └─────────────┘    └──────────────┘    │
│       │                                                        │            │
│       │         ┌─────────────────────────────────────────────┘            │
│       │         ▼                                                           │
│       │    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│       │    │    Loss      │───▶│   Backward   │───▶│   Optimizer  │        │
│       │    │ Computation  │    │     Pass     │    │    Update    │        │
│       │    └──────────────┘    └──────────────┘    └──────────────┘        │
│       │                                                    │                │
│       │    ┌──────────────────────────────────────────────┘                │
│       │    ▼                                                                │
│       │ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│       │ │  Validation  │───▶│  EER / Acc   │───▶│ Checkpointing│           │
│       │ │   Epoch      │    │  Evaluation  │    │  Best Model  │           │
│       │ └──────────────┘    └──────────────┘    └──────────────┘           │
│       │                                                                     │
└───────┼─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Audio   │───▶│  Feature    │───▶│    Model     │───▶│  Prediction  │   │
│  │  Input   │    │  Extraction │    │   Inference  │    │  (Fake/Real) │   │
│  └──────────┘    └─────────────┘    └──────────────┘    └──────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Loading & Preprocessing

### 2.1 Audio File Loading

Audio files are loaded using `soundfile` library with automatic format detection:

```python
# Load audio file
waveform, sr = sf.read(audio_path)

# Convert to mono if stereo
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)

# Resample to 16kHz if needed
if sr != 16000:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
```

**Supported Formats:** WAV, FLAC, MP3 (via librosa)

### 2.2 Fixed-Length Processing

All audio is processed to a fixed length for batch training:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cut` | 64,600 samples | ~4 seconds at 16kHz |
| Sample Rate | 16,000 Hz | Standard for speech |

**Length Handling:**

```python
def pad(x, max_len=64600):
    """Pad short audio with repetition"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Repeat audio to fill length
    num_repeats = int(max_len / x_len) + 1
    padded = np.tile(x, num_repeats)[:max_len]
    return padded

def pad_random(x, max_len=64600):
    """Randomly crop or pad audio"""
    x_len = x.shape[0]
    if x_len > max_len:
        # Random crop
        start = np.random.randint(0, x_len - max_len)
        return x[start:start + max_len]
    # Pad with repetition
    return pad(x, max_len)
```

### 2.3 Dataset Classes

**Training Dataset (`Dataset_FakeOrReal_train`):**
- Loads audio files from `training/real/` and `training/fake/` directories
- Applies data augmentation when `random_noise=True`
- Returns `(feature_tensor, label)` where label: 1=real, 0=fake

**Validation/Test Dataset (`Dataset_FakeOrReal_devNeval`):**
- No augmentation applied
- Returns `(feature_tensor, file_path)` for score file generation

### 2.4 DataLoader Configuration

```python
DataLoader(
    dataset,
    batch_size=24,              # Configurable via --batch_size
    shuffle=True,               # True for training only
    drop_last=True,             # For training only
    pin_memory=True,            # Faster GPU transfer
    num_workers=4,              # Parallel data loading
    persistent_workers=True,    # Keep workers alive between epochs
    prefetch_factor=2,          # Prefetch 2 batches per worker
    worker_init_fn=seed_worker, # Reproducible worker seeding
)
```

---

## 3. Feature Extraction

### 3.1 Feature Type Overview

The system extracts features from raw audio waveforms. Feature type is selected via `--feature_type`:

| Type | Name | Shape | Best For |
|------|------|-------|----------|
| 0 | Raw Waveform | `(64600,)` | End-to-end models (RawNet3) |
| 1 | Mel-Spectrogram | `(128, T)` | CNN models (recommended) |
| 2 | LFCC | `(13, T)` | Spoofing detection |
| 3 | MFCC | `(13, T)` | Traditional speech |
| 4 | CQT | `(84, T)` | Harmonic analysis |
| 8 | Prosodic | `(7, T)` | TTS artifact detection |

### 3.2 Mel-Spectrogram Extraction (Type 1)

The recommended feature type for CNN-based models:

```python
def extract_mel_spectrogram(waveform, sr=16000):
    # Parameters
    n_mels = 128        # Number of mel bands
    n_fft = 512         # FFT window size
    hop_length = 160    # Hop between frames (10ms at 16kHz)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db  # Shape: (128, T)
```

**Processing Steps:**
1. Apply Short-Time Fourier Transform (STFT)
2. Map frequencies to mel scale (perceptually motivated)
3. Apply 128 triangular filterbanks
4. Take power spectrum
5. Convert to decibels (log compression)

### 3.3 LFCC Extraction (Type 2)

Linear Frequency Cepstral Coefficients - designed for spoofing detection:

```python
def extract_lfcc(waveform, sr=16000):
    n_fft = 512
    hop_length = 160
    n_filters = 20      # Linear filterbank size
    n_lfcc = 13         # Cepstral coefficients
    
    # 1. STFT → power spectrum
    S = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length)) ** 2
    
    # 2. Create LINEAR filterbank (not mel-scaled)
    freq_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, freq_bins)
    filter_freqs = np.linspace(0, sr / 2, n_filters + 2)
    filterbank = np.zeros((n_filters, freq_bins))
    
    for i in range(n_filters):
        left, center, right = filter_freqs[i:i+3]
        for j, freq in enumerate(fft_freqs):
            if left <= freq <= center:
                filterbank[i, j] = (freq - left) / (center - left)
            elif center <= freq <= right:
                filterbank[i, j] = (right - freq) / (right - center)
    
    # 3. Apply filterbank
    S_lin = np.dot(filterbank, S)
    
    # 4. Log compression
    log_S = np.log(S_lin + 1e-10)
    
    # 5. DCT → cepstral coefficients
    from scipy.fftpack import dct
    lfcc = dct(log_S, type=2, axis=0, norm='ortho')[:n_lfcc]
    
    return lfcc  # Shape: (13, T)
```

### 3.4 MFCC Extraction (Type 3)

Classic Mel-Frequency Cepstral Coefficients:

```python
def extract_mfcc(waveform, sr=16000):
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=13,      # 13 coefficients
        n_fft=512,
        hop_length=160
    )
    return mfcc  # Shape: (13, T)
```

### 3.5 CQT Extraction (Type 4)

Constant-Q Transform for harmonic analysis:

```python
def extract_cqt(waveform, sr=16000):
    cqt = librosa.cqt(
        y=waveform,
        sr=sr,
        hop_length=512,
        n_bins=84,          # 7 octaves × 12 bins
        bins_per_octave=12
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db  # Shape: (84, T)
```

### 3.6 Prosodic Features (Type 8)

High-level prosodic features:

```python
def extract_prosodic(waveform, sr=16000):
    # F0 (pitch) using PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform, fmin=50, fmax=500, sr=sr
    )
    
    # Energy (RMS)
    rms = librosa.feature.rms(y=waveform, frame_length=512, hop_length=160)
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=512, hop_length=160)
    
    # Stack features: (7, T)
    prosodic_features = np.stack([
        f0,              # Pitch
        rms,             # Energy
        zcr,             # Zero-crossing rate
        voiced_probs,    # Voiced probability
        delta_f0,        # Pitch delta
        delta_energy,    # Energy delta
        speaking_rate    # Local speaking rate
    ])
    
    return prosodic_features
```

### 3.7 Multimodal Feature Fusion

Multiple feature types can be combined:

```bash
python main.py --feature_type 1,2,4  # Mel + LFCC + CQT
```

**Fusion Process:**
1. Extract each feature type independently
2. Align time dimensions (crop to minimum)
3. Stack along channel dimension
4. Pass through `MultimodalFusionWrapper`

```python
class MultimodalFusionWrapper(nn.Module):
    def __init__(self, base_model, num_modalities):
        # Apply backbone separately per modality
        # Concatenate feature maps
        # Project back to expected channels
        # Continue with pooling → classifier
```

---

## 4. Data Augmentation

### 4.1 Augmentation Overview

Augmentation is enabled via `--random_noise` flag and applies during training only.

### 4.2 Waveform Augmentations

| ID | Augmentation | Parameters | Effect |
|----|--------------|------------|--------|
| 1 | Gaussian Noise | SNR: 10-25 dB | Adds random white noise |
| 2 | Background Noise | Factor: 0.01-0.05 | White/pink noise overlay |
| 3 | Reverberation | Delay: 50ms, Factor: 0.3-0.8 | Echo simulation |
| 4 | Pitch Shift | ±4 semitones | Alters pitch |
| 5 | Time Stretch | Rate: 0.85-1.15× | Speed perturbation |
| 6 | Gain | ±6 dB | Volume adjustment |
| 7 | Low-pass Filter | Cutoff: 2000-6000 Hz | High-frequency removal |
| 8 | High-pass Filter | Cutoff: 50-300 Hz | Low-frequency removal |
| 9 | RIR Simulation | RT60: 0.1-0.5s | Room acoustics |
| 10 | MUSAN-style Noise | SNR: 5-25 dB | Babble/music/ambient |

### 4.3 Composed Augmentation

Multiple augmentations are applied in sequence:

```python
def apply_composed_augmentation(waveform, sr=16000, num_augmentations=2, augment_prob=0.8):
    # 80% chance to apply any augmentation
    if np.random.random() > augment_prob:
        return waveform
    
    # Available augmentation types
    aug_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Randomly select 1-2 augmentations
    n_augs = np.random.randint(1, num_augmentations + 1)
    selected_augs = np.random.choice(aug_types, size=n_augs, replace=False)
    
    # Apply sequentially
    augmented = waveform.copy()
    for aug_type in selected_augs:
        augmented = apply_augmentation(augmented, aug_type, sr)
    
    return augmented
```

### 4.4 SpecAugment (Spectrogram Augmentation)

Applied after feature extraction for spectrogram-based features:

```python
def apply_spectrogram_augmentation(spectrogram, freq_mask_prob=0.5, time_mask_prob=0.5):
    spec = spectrogram.copy()
    n_freq, n_time = spec.shape
    
    # Frequency masking (50% probability)
    if np.random.random() < freq_mask_prob:
        f = np.random.randint(1, min(20, n_freq // 4) + 1)  # Max 20 bins
        f0 = np.random.randint(0, n_freq - f)
        spec[f0:f0 + f, :] = spec.mean()
    
    # Time masking (50% probability)
    if np.random.random() < time_mask_prob:
        t = np.random.randint(1, min(50, n_time // 4) + 1)  # Max 50 frames
        t0 = np.random.randint(0, n_time - t)
        spec[:, t0:t0 + t] = spec.mean()
    
    return spec
```

### 4.5 RIR Simulation Details

Room Impulse Response simulation for robust training:

```python
def add_rir_simulation(waveform, sr=16000):
    # Generate synthetic RIR
    rt60 = np.random.uniform(0.1, 0.5)  # Reverberation time
    rir_length = int(rt60 * sr)
    
    # Create exponentially decaying impulse response
    t = np.arange(rir_length) / sr
    decay = np.exp(-3 * t / rt60)
    noise = np.random.randn(rir_length)
    rir = noise * decay
    
    # Normalize and add direct path
    rir = rir / np.max(np.abs(rir))
    rir[0] = 1.0
    
    # Convolve with waveform
    from scipy.signal import fftconvolve
    convolved = fftconvolve(waveform, rir, mode='same')
    
    return convolved
```

---

## 5. Model Architecture Selection

### 5.1 Architecture Loading

Models are dynamically loaded based on configuration:

```python
def get_model(model_config, device):
    # Import model module dynamically
    module = import_module(f"models.{model_config['architecture']}")
    
    # Select variant
    if model_config.get("model_variant") == "attention":
        _model = getattr(module, "ModelWithAttention")
    elif model_config.get("model_variant") == "large":
        _model = getattr(module, "ModelLarge")
    else:
        _model = getattr(module, "Model")
    
    # Instantiate and move to device
    model = _model(model_config).to(device)
    return model
```

### 5.2 Model Forward Pass

All models follow a consistent interface:

```python
class Model(nn.Module):
    def forward(self, x, Freq_aug=False):
        # x: Input tensor (batch, channels, freq, time) or (batch, samples)
        
        # Feature extraction backbone
        features = self.backbone(x)
        
        # Pooling (attention-weighted or global average)
        pooled = self.pool(features)
        
        # Embedding layer
        embeddings = self.embedding(pooled)
        
        # Classification head
        output = self.classifier(embeddings)
        
        return embeddings, output  # (batch, emb_dim), (batch, 2)
```

### 5.3 Ensemble Models

Multiple models can be ensembled:

```python
class EnsembleModel(nn.Module):
    def __init__(self, model_list):
        self.models = nn.ModuleList(model_list)
        # Projection layers for heterogeneous embedding sizes
        self.projections = nn.ModuleList([...])
    
    def forward(self, x, Freq_aug=False):
        outs = []
        embs = []
        
        for m in self.models:
            emb, out = m(x, Freq_aug=Freq_aug)
            embs.append(emb)
            outs.append(out)
        
        # Average logits
        avg_out = torch.mean(torch.stack(outs), dim=0)
        
        # Project and average embeddings
        proj_embs = [self.projections[i](emb) for i, emb in enumerate(embs)]
        avg_emb = torch.mean(torch.stack(proj_embs), dim=0)
        
        return avg_emb, avg_out
```

---

## 6. Training Process

### 6.1 Training Loop

The main training loop (`train_epoch`):

```python
def train_epoch(trn_loader, model, optimizer, device, scheduler, config):
    model.train()
    running_loss = 0
    num_total = 0
    
    # Loss function with class weighting
    weight = torch.FloatTensor([0.1, 0.9]).to(device)  # [fake, real]
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    # AMP setup
    use_amp = config.get("use_amp", True)
    amp_dtype = torch.bfloat16 if BF16_SUPPORTED else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and not use_bf16))
    
    for batch_x, batch_y in trn_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        
        # Zero gradients efficiently
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            _, batch_out = model(batch_x)
            batch_loss = criterion(batch_out, batch_y)
        
        running_loss += batch_loss.item() * batch_x.size(0)
        
        # Backward pass with gradient scaling
        scaler.scale(batch_loss).backward()
        
        # Gradient clipping
        if config.get("grad_clip", 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        
        scaler.step(optimizer)
        scaler.update()
        
        # Scheduler step (for cosine/keras_decay)
        if scheduler is not None:
            scheduler.step()
        
        num_total += batch_x.size(0)
    
    return running_loss / num_total
```

### 6.2 Loss Function

**Weighted Cross-Entropy Loss:**

```python
weight = torch.FloatTensor([0.1, 0.9]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)
```

- Weight `[0.1, 0.9]` assigns higher importance to real (bonafide) class
- Addresses class imbalance in spoofing datasets
- Index 0 = fake/spoof, Index 1 = real/bonafide

### 6.3 Optimizer Configuration

**Adam Optimizer (Default):**

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,           # base_lr
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    amsgrad=False
)
```

**SGD Optimizer:**

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=False
)
```

### 6.4 Learning Rate Schedulers

**Cosine Annealing:**

```python
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(step, total_steps, 1, lr_min/base_lr)
)
```

**SGDR (Warm Restarts):**

```python
class SGDRScheduler:
    def __init__(self, optimizer, T0, T_mul, eta_min):
        self.Ti = T0       # Initial cycle length
        self.T_mul = T_mul # Cycle length multiplier
        self.eta_min = eta_min
```

### 6.5 Validation During Training

After each epoch:

```python
# Generate scores on dev set
produce_evaluation_file_simple(dev_loader, model, device, "dev_score.txt")

# Calculate metrics
dev_eer, dev_acc = calculate_simple_eer_accuracy("dev_score.txt")

# Save best model
if dev_eer < best_dev_eer:
    best_dev_eer = dev_eer
    torch.save(model.state_dict(), "best.pth")
    
    # Optional: evaluate on test set
    if config["eval_all_best"]:
        produce_evaluation_file_simple(eval_loader, model, device, "eval_score.txt")
```

### 6.6 Stochastic Weight Averaging (SWA)

Enabled via `--weight_avg`:

```python
from torchcontrib.optim import SWA

optimizer_swa = SWA(optimizer)

# On best models and late epochs
if is_best_model or epoch > num_epochs // 2:
    optimizer_swa.update_swa()
    n_swa_update += 1

# At end of training
optimizer_swa.swap_swa_sgd()              # Replace weights with averaged
optimizer_swa.bn_update(trn_loader, model) # Update BatchNorm statistics
torch.save(model.state_dict(), "swa.pth")
```

---

## 7. Optimization Techniques

### 7.1 Mixed Precision Training

**BFloat16 (Ampere+ GPUs):**
- Native hardware support on compute capability 8.0+
- Better dynamic range than FP16, no gradient scaling needed
- ~20-30% speedup

**Float16 (All CUDA GPUs):**
- Requires gradient scaling to prevent underflow
- ~20-30% speedup

```python
# Check BF16 support
def check_bf16_native_support():
    major, minor = torch.cuda.get_device_capability(0)
    return major >= 8  # Ampere, Ada, Hopper

# Mixed precision training
amp_dtype = torch.bfloat16 if BF16_SUPPORTED else torch.float16
with torch.amp.autocast('cuda', dtype=amp_dtype):
    _, output = model(input)
    loss = criterion(output, target)
```

### 7.2 TF32 Acceleration

TensorFloat-32 for matrix multiplication on Ampere+ GPUs:

```python
# PyTorch 2.9+ API
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

# Legacy API
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 7.3 Channels-Last Memory Format

Optimized memory layout for CNNs:

```python
# Convert model to channels_last
model = model.to(memory_format=torch.channels_last)

# Convert input tensors
if batch_x.dim() == 4:
    batch_x = batch_x.to(memory_format=torch.channels_last)
```

### 7.4 torch.compile (PyTorch 2.0+)

JIT compilation with inductor backend:

```python
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
    # modes: "default", "reduce-overhead", "max-autotune"
```

### 7.5 Gradient Clipping

Prevents gradient explosion:

```python
if config.get("grad_clip", 0) > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
```

---

## 8. Evaluation & Metrics

### 8.1 Score File Generation

```python
def produce_evaluation_file_simple(data_loader, model, device, save_path):
    model.eval()
    
    with torch.inference_mode():
        for batch_x, batch_info in data_loader:
            batch_x = batch_x.to(device)
            _, batch_out = model(batch_x)
            
            # Score = logit for real class
            batch_score = batch_out[:, 1].cpu().numpy()
            
            # Write: filename label score
            for fn, score in zip(batch_info, batch_score):
                label = "bonafide" if "/real/" in fn else "spoof"
                f.write(f"{fn} {label} {score}\n")
```

**Score File Format:**
```
training/real/audio_001.wav bonafide 2.3456
training/fake/audio_002.wav spoof -1.2345
```

### 8.2 Equal Error Rate (EER) Calculation

```python
def compute_eer(target_scores, nontarget_scores):
    """
    Compute EER where:
    - target_scores: scores for bonafide (real) samples
    - nontarget_scores: scores for spoof (fake) samples
    """
    # Compute DET curve
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    
    # Find intersection point
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    
    # EER = average of FRR and FAR at intersection
    eer = np.mean((frr[min_index], far[min_index]))
    threshold = thresholds[min_index]
    
    return eer, threshold
```

### 8.3 Accuracy Calculation

```python
def compute_accuracy(bona_cm, spoof_cm, threshold):
    # Bonafide should score >= threshold (predicted as real)
    correct_bona = np.sum(bona_cm >= threshold)
    
    # Spoof should score < threshold (predicted as fake)
    correct_spoof = np.sum(spoof_cm < threshold)
    
    total = len(bona_cm) + len(spoof_cm)
    accuracy = (correct_bona + correct_spoof) / total * 100
    
    return accuracy
```

### 8.4 Metrics Tracking

```python
class MetricsTracker:
    def __init__(self, save_dir):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'dev_eer': [],
            'dev_acc': [],
            'eval_eer': [],
            'eval_acc': [],
            'best_dev_eer': []
        }
    
    def add_epoch(self, epoch, train_loss, dev_eer, ...):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        # ...
    
    def save(self):
        # Save to metrics.json and metrics.csv
```

---

## 9. Inference Pipeline

### 9.1 Streamlit App Inference

```python
@st.cache_resource
def load_model(config_path, weights_path):
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
    
    # Build model
    model = get_model(config["model_config"], device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    return model, config

def classify_audio(audio_file, model, feature_type):
    # Load and preprocess
    waveform, sr = sf.read(audio_file)
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    
    # Pad to fixed length
    waveform = pad(waveform, max_len=64600)
    
    # Extract features
    features = extract_feature(waveform, feature_type)
    
    # To tensor
    x = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    
    # Inference
    with torch.inference_mode():
        _, output = model(x.to(device))
        probs = torch.softmax(output, dim=1)
        
    return {
        'prediction': 'Real' if probs[0, 1] > 0.5 else 'Fake',
        'confidence': max(probs[0, 0], probs[0, 1]).item() * 100
    }
```

### 9.2 Batch Inference

```python
def batch_inference(audio_files, model, feature_type, batch_size=32):
    results = []
    
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i+batch_size]
        
        # Process batch
        batch_features = []
        for f in batch_files:
            waveform, sr = sf.read(f)
            features = extract_feature(pad(waveform), feature_type)
            batch_features.append(features)
        
        # To tensor
        x = torch.stack([torch.FloatTensor(f) for f in batch_features])
        
        # Inference
        with torch.inference_mode():
            _, output = model(x.to(device))
            probs = torch.softmax(output, dim=1)
        
        # Collect results
        for j, f in enumerate(batch_files):
            results.append({
                'file': f,
                'prediction': 'Real' if probs[j, 1] > 0.5 else 'Fake',
                'score': probs[j, 1].item()
            })
    
    return results
```

---

## 10. Reproducibility

### 10.1 Seed Setting

```python
def set_seed(seed, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable for determinism
```

### 10.2 Worker Seeding

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# In DataLoader
DataLoader(..., worker_init_fn=seed_worker, generator=gen)
```

### 10.3 Experiment Reproducibility Checklist

- [ ] Set `--seed` (default: 1234)
- [ ] Use same `--feature_type`
- [ ] Use same `--dataset_version`
- [ ] Use same `--batch_size`
- [ ] Same augmentation settings (`--random_noise`)
- [ ] Same model configuration file
- [ ] Same PyTorch version
- [ ] Same CUDA/cuDNN versions

---

## Summary

The AI Voice Detector methodology encompasses:

1. **Data Loading**: Fixed-length audio processing with efficient DataLoader configuration
2. **Feature Extraction**: Multiple feature types (Mel, LFCC, MFCC, CQT, Prosodic) with fusion support
3. **Augmentation**: Comprehensive augmentation pipeline (RIR, MUSAN, SpecAugment)
4. **Training**: Mixed precision, gradient clipping, SWA, multiple schedulers
5. **Optimization**: TF32, BF16, channels-last, torch.compile
6. **Evaluation**: EER-based evaluation with accuracy at EER threshold
7. **Inference**: Cached model loading, batch processing, real-time classification

---

*Methodology documentation generated: January 2026*
