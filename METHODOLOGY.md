# AI Voice Detector — Methodology Documentation

> **Version:** 2.0 | **Last Updated:** January 17, 2026

This document provides a comprehensive methodology following academic journal and conference paper standards. It details the complete pipeline for detecting synthetic and deepfake audio, from data preparation through evaluation.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Preparation and Preprocessing](#2-data-preparation-and-preprocessing)
3. [Feature Extraction](#3-feature-extraction)
4. [Data Augmentation](#4-data-augmentation)
5. [Network Architectures](#5-network-architectures)
6. [Experimental Setup](#6-experimental-setup)
7. [Training Procedure](#7-training-procedure)
8. [Model Validation and Selection](#8-model-validation-and-selection)
9. [Ensemble Strategy](#9-ensemble-strategy)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Optimization Techniques](#12-optimization-techniques)
13. [Reproducibility](#13-reproducibility)
14. [Computational Resources](#14-computational-resources)

---

## 1. Introduction

 
---

## 2. Data Preparation and Preprocessing

## 2. Data Preparation and Preprocessing

### 2.1 Dataset Description

The system was developed and evaluated using the "Fake-or-Real" dataset, which consists of bonafide (genuine human) and spoofed (synthetically generated) audio recordings. The dataset is organized into training, validation, and testing partitions, with balanced representation of both classes to facilitate unbiased model learning.

### 2.2 Audio Loading and Standardization

### 2.2 Audio Loading and Standardization

All audio samples undergo standardized preprocessing to ensure consistency across the pipeline. Audio files are loaded using the `soundfile` library, which provides automatic format detection and efficient decoding for multiple audio formats including WAV, FLAC, and MP3.

The preprocessing pipeline consists of the following operations:

**Step 1: Format Conversion**
```python
waveform, sr = sf.read(audio_path)
```

**Step 2: Mono Conversion**
For stereo audio signals, channel reduction is performed via averaging:
```python
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)
```

**Step 3: Resampling**
All audio is resampled to a standardized sampling rate of 16 kHz to ensure consistency and computational efficiency:
```python
if sr != 16000:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
```

This sampling rate was selected based on the Nyquist-Shannon sampling theorem, which ensures adequate representation of speech frequencies (typically below 8 kHz) while maintaining computational tractability.

### 2.3 Fixed-Length Segmentation

### 2.3 Fixed-Length Segmentation

To facilitate batch processing and ensure consistent input dimensions for neural network training, all audio signals are processed to a fixed length of 64,600 samples, corresponding to approximately 4.04 seconds at 16 kHz sampling rate. This duration was empirically determined to capture sufficient contextual information for classification while maintaining computational efficiency.

**Length Normalization Strategy:**

For audio segments shorter than the target length ($L < 64600$), we employ a repetition-based padding technique:

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
```

For audio segments exceeding the target length ($L > 64600$), two strategies are employed:

1. **Training Phase:** Random cropping to introduce variability and prevent overfitting:
```python
start = np.random.randint(0, x_len - max_len)
return x[start:start + max_len]
```

2. **Inference Phase:** Center cropping for deterministic evaluation:
```python
start = (x_len - max_len) // 2
return x[start:start + max_len]
```

### 2.4 Batch Processing Configuration

The DataLoader configuration was optimized for both training efficiency and reproducibility:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 32 | Balanced GPU memory utilization and gradient stability |
| `num_workers` | 4 | Parallel data loading to prevent I/O bottlenecks |
| `pin_memory` | True | Accelerated CPU-to-GPU transfers |
| `persistent_workers` | True | Reduced worker initialization overhead |
| `prefetch_factor` | 2 | Overlapped data loading with computation |
| `drop_last` | True (train) | Consistent batch sizes for BatchNorm stability |

Worker initialization employs deterministic seeding to ensure reproducibility:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

---

## 3. Feature Extraction

## 3. Feature Extraction

### 3.1 Overview of Acoustic Representations

The system supports multiple acoustic feature representations, each capturing different aspects of audio characteristics relevant to deepfake detection. The feature extraction strategy is configurable via the `--feature_type` parameter, enabling comparative analysis and multi-modal fusion.

| Type | Feature Name | Dimensionality | Primary Application |
|------|-------------|----------------|---------------------|
| 0 | Raw Waveform | $(N,)$ | End-to-end learning |
| 1 | Log-Mel Spectrogram | $(128, T)$ | General-purpose CNN models |
| 2 | LFCC | $(13, T)$ | Anti-spoofing (linear frequency) |
| 3 | MFCC | $(13, T)$ | Traditional speech processing |
| 4 | CQT | $(84, T)$ | Harmonic and tonal analysis |
| 8 | Prosodic Features | $(7, T)$ | Naturalness assessment |

where $N$ denotes the number of samples (64,600) and $T$ represents the temporal dimension (varies by feature type).

### 3.2 Log-Mel Spectrogram (Type 1)

The Log-Mel Spectrogram serves as the primary feature representation for CNN-based architectures. This transformation converts the time-domain signal into a time-frequency representation that approximates human auditory perception.

**Mathematical Formulation:**

The extraction process involves five sequential transformations:

**1. Short-Time Fourier Transform (STFT):**
$$X(m, k) = \sum_{n=0}^{N-1} x(n + mH) w(n) e^{-j2\pi kn/N}$$

where $w(n)$ is the Hann window, $H$ is the hop length (160 samples), and $N$ is the FFT size (512).

**2. Power Spectrum:**
$$P(m, k) = |X(m, k)|^2$$

**3. Mel Filterbank Application:**
$$S_{\text{mel}}(m, b) = \sum_{k=0}^{N/2} P(m, k) \cdot H_b(k)$$

where $H_b(k)$ represents the $b$-th triangular Mel filter ($b = 1, \ldots, 128$).

**4. Logarithmic Compression:**
$$S_{\log}(m, b) = 10 \log_{10}(S_{\text{mel}}(m, b) + \epsilon)$$

where $\epsilon = 10^{-10}$ prevents numerical instability.

**Implementation:**
```python
def extract_mel_spectrogram(waveform, sr=16000):
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=128,         # Number of Mel bands
        n_fft=512,          # FFT window size
        hop_length=160,     # 10ms hop at 16kHz
        window='hann'
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db  # Shape: (128, T)
```

**Perceptual Motivation:** The Mel scale approximates the non-linear frequency resolution of human hearing, emphasizing perceptually relevant frequencies for speech (300-4000 Hz).

### 3.3 Linear Frequency Cepstral Coefficients (Type 2)

### 3.3 Linear Frequency Cepstral Coefficients (Type 2)

LFCC was specifically designed for spoofing detection tasks, as it employs a linear frequency scale rather than the perceptually-motivated Mel scale. This characteristic makes LFCC particularly sensitive to high-frequency artifacts often present in synthetic speech.

**Mathematical Formulation:**

**1. Linear Filterbank Construction:**
$$H_b^{\text{lin}}(k) = \begin{cases}
\frac{f(k) - f_b^{\text{left}}}{f_b^{\text{center}} - f_b^{\text{left}}} & f_b^{\text{left}} \leq f(k) \leq f_b^{\text{center}} \\
\frac{f_b^{\text{right}} - f(k)}{f_b^{\text{right}} - f_b^{\text{center}}} & f_b^{\text{center}} \leq f(k) \leq f_b^{\text{right}} \\
0 & \text{otherwise}
\end{cases}$$

where filter centers are linearly spaced: $f_b = b \cdot \frac{f_s/2}{B}$ for $b = 0, \ldots, B$ (20 filters).

**2. Filterbank Application:**
$$S_{\text{lin}}(m, b) = \sum_{k=0}^{N/2} P(m, k) \cdot H_b^{\text{lin}}(k)$$

**3. Logarithmic Compression:**
$$S_{\log}(m, b) = \log(S_{\text{lin}}(m, b) + \epsilon)$$

**4. Discrete Cosine Transform (DCT):**
$$\text{LFCC}(m, c) = \sum_{b=0}^{B-1} S_{\log}(m, b) \cos\left(\frac{\pi c(b + 0.5)}{B}\right)$$

The first 13 coefficients ($c = 0, \ldots, 12$) are retained, providing a compact representation.

**Implementation:**
```python
def extract_lfcc(waveform, sr=16000):
    from scipy.fftpack import dct
    
    # STFT and power spectrum
    S = np.abs(librosa.stft(y=waveform, n_fft=512, hop_length=160)) ** 2
    
    # Linear filterbank (20 filters, linearly spaced)
    freq_bins = 257  # (n_fft // 2 + 1)
    filterbank = create_linear_filterbank(freq_bins, n_filters=20, sr=sr)
    
    # Apply filterbank
    S_lin = np.dot(filterbank, S)
    
    # Log compression and DCT
    log_S = np.log(S_lin + 1e-10)
    lfcc = dct(log_S, type=2, axis=0, norm='ortho')[:13]
    
    return lfcc  # Shape: (13, T)
```

### 3.4 Prosodic Features (Type 8)

Prosodic features capture suprasegmental characteristics of speech that are challenging for text-to-speech (TTS) systems to replicate naturally. These include fundamental frequency ($F_0$), energy envelope, and temporal dynamics.

**Feature Components:**

1. **Fundamental Frequency ($F_0$):** Estimated using the probabilistic YIN (PYIN) algorithm, which provides robust pitch tracking in noisy conditions.

2. **Root Mean Square Energy (RMS):** Measures the energy envelope of the signal:
$$\text{RMS}(m) = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x^2(n + mH)}$$

3. **Zero-Crossing Rate (ZCR):** Indicates the rate of sign changes in the signal, correlating with noise content and voicing.

4. **$\Delta F_0$ and $\Delta$ Energy:** First-order temporal derivatives capturing dynamic contour information.

5. **Voiced Probability:** Confidence measure from the PYIN algorithm indicating voicing likelihood.

6. **Speaking Rate Proxy:** Local temporal density of voiced segments.

**Implementation:**
```python
def extract_prosodic(waveform, sr=16000):
    # F0 extraction using PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform, fmin=50, fmax=500, sr=sr, frame_length=2048
    )
    
    # Energy (RMS)
    rms = librosa.feature.rms(y=waveform, frame_length=512, hop_length=160)[0]
    
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=512, hop_length=160)[0]
    
    # Delta features
    delta_f0 = librosa.feature.delta(f0)
    delta_energy = librosa.feature.delta(rms)
    
    # Align temporal dimensions
    min_len = min(len(f0), len(rms), len(zcr), len(voiced_probs))
    
    # Stack features
    prosodic_features = np.stack([
        f0[:min_len],
        rms[:min_len],
        zcr[:min_len],
        voiced_probs[:min_len],
        delta_f0[:min_len],
        delta_energy[:min_len],
        compute_speaking_rate(voiced_flag)[:min_len]
    ])
    
    return prosodic_features  # Shape: (7, T)
```

### 3.5 Multi-Modal Feature Fusion

To leverage complementary information from multiple acoustic representations, the system supports multi-modal feature fusion. This is achieved by extracting multiple feature types and concatenating them along the channel dimension before feeding into the network.

**Fusion Strategy:**
```python
features = []
for feature_type in [1, 2, 4]:  # Mel + LFCC + CQT
    feat = extract_feature(waveform, feature_type)
    features.append(feat)

# Temporal alignment (crop to minimum length)
min_time = min(f.shape[-1] for f in features)
aligned_features = [f[..., :min_time] for f in features]

# Concatenate along channel dimension
fused_features = np.concatenate(aligned_features, axis=0)
```

---

## 4. Data Augmentation

## 4. Data Augmentation

### 4.1 Motivation and Strategy

Data augmentation serves two critical purposes in audio deepfake detection: (1) preventing overfitting to training set characteristics, and (2) improving generalization across diverse acoustic environments and recording conditions. Our augmentation pipeline applies probabilistic transformations at the waveform level during training only, with an application probability of $p = 0.8$.

### 4.2 Waveform-Level Augmentations

**4.2.1 Additive Noise**

**Gaussian White Noise:** Introduces random perturbations with controlled Signal-to-Noise Ratio (SNR):

$$y(t) = x(t) + \sqrt{\frac{P_x}{10^{\text{SNR}/10}}} \cdot \mathcal{N}(0, 1)$$

where $P_x = \mathbb{E}[x^2(t)]$ is the signal power. SNR is randomly sampled from the range [10, 25] dB.

**MUSAN-style Environmental Noise:** Simulates realistic recording conditions by mixing with ambient sound, music, or babble noise at randomly selected SNR levels (5-25 dB).

**4.2.2 Room Impulse Response (RIR) Simulation**

Acoustic reverberation is modeled by convolving the source signal with a synthetic room impulse response:

$$y(t) = x(t) * h(t)$$

where the impulse response is generated as:

$$h(t) = \delta(t) + \alpha \cdot \mathcal{N}(0, 1) \cdot e^{-3t/RT_{60}}$$

with reverberation time $RT_{60} \in [0.1, 0.5]$ seconds, and $\alpha$ controlling the wet/dry mix.

**Implementation:**
```python
def add_rir_simulation(waveform, sr=16000):
    rt60 = np.random.uniform(0.1, 0.5)
    rir_length = int(rt60 * sr)
    
    # Exponentially decaying noise
    t = np.arange(rir_length) / sr
    decay = np.exp(-3 * t / rt60)
    rir = np.random.randn(rir_length) * decay
    
    # Add direct path
    rir[0] = 1.0
    rir = rir / np.max(np.abs(rir))
    
    # Convolve
    from scipy.signal import fftconvolve
    return fftconvolve(waveform, rir, mode='same')
```

**4.2.3 Temporal Perturbations**

**Time Stretching:** Modifies the temporal characteristics without affecting pitch:
$$y = \text{TimeStretch}(x, \text{rate} \in [0.85, 1.15])$$

**Pitch Shifting:** Alters the fundamental frequency while preserving duration:
$$y = \text{PitchShift}(x, n_{\text{semitones}} \in [-4, +4])$$

**4.2.4 Spectral Filtering**

**Low-Pass Filter:** Attenuates high-frequency components (cutoff: 2000-6000 Hz).

**High-Pass Filter:** Removes low-frequency components (cutoff: 50-300 Hz).

These filters simulate bandwidth limitations in communication channels.

**4.2.5 Amplitude Perturbations**

**Gain Adjustment:** Random amplitude scaling in the range $[-6, +6]$ dB.

### 4.3 Spectrogram-Level Augmentation: SpecAugment

For spectro-temporal representations, we employ SpecAugment, which applies random masking along frequency and time axes.

**Frequency Masking:** Randomly selects a frequency band and masks it:
$$\tilde{S}(f, t) = \begin{cases}
\bar{S} & f \in [f_0, f_0 + F] \\
S(f, t) & \text{otherwise}
\end{cases}$$

where $F \sim \mathcal{U}(1, F_{\max})$ with $F_{\max} = 20$ bins, and $\bar{S}$ is the mean value.

**Time Masking:** Similarly masks contiguous time frames:
$$\tilde{S}(f, t) = \begin{cases}
\bar{S} & t \in [t_0, t_0 + T] \\
S(f, t) & \text{otherwise}
\end{cases}$$

with $T \sim \mathcal{U}(1, T_{\max})$ where $T_{\max} = 50$ frames.

Both masking operations are applied with probability 0.5 each.

### 4.4 Composition Strategy

During training, each audio sample undergoes a randomized composition of 1-2 augmentations selected from the available pool:

```python
def apply_composed_augmentation(waveform, sr=16000, num_augmentations=2, p=0.8):
    if np.random.random() > p:
        return waveform
    
    aug_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Available augmentations
    n_augs = np.random.randint(1, num_augmentations + 1)
    selected = np.random.choice(aug_types, size=n_augs, replace=False)
    
    augmented = waveform.copy()
    for aug_id in selected:
        augmented = apply_augmentation(augmented, aug_id, sr)
    
    return augmented
```

This compositional approach creates diverse acoustic variations while avoiding excessive distortion.

---

## 5. Network Architectures

## 5. Network Architectures

### 5.1 Overview

We employed a diverse set of deep neural network architectures, each offering distinct advantages for audio deepfake detection. The architectures range from lightweight models optimized for real-time inference to deep networks with high representational capacity.

### 5.2 EfficientNet-B2

**Architecture Description:**

EfficientNet-B2 is a compound-scaled convolutional neural network originally developed for image classification. We adapted it for audio forensics by modifying the input layer to accept single-channel spectrograms and replacing the classification head.

**Key Specifications:**
- **Input:** Log-Mel Spectrogram $(B, 1, 128, T)$
- **Backbone:** EfficientNet-B2 (depth=1.1, width=1.1, resolution scaling)
- **Parameters:** ~9.2M
- **Output Embedding:** 1408-dimensional feature vector

**Architectural Modifications:**

1. **Input Layer Adaptation:** The first convolutional layer was modified from Conv2d(3, 32) to Conv2d(1, 32) to accommodate single-channel input. Pre-trained weights from ImageNet were averaged across RGB channels:

$$W_{\text{new}}(1, :, :, :) = \frac{1}{3}\sum_{c=1}^{3} W_{\text{pretrained}}(c, :, :, :)$$

2. **Custom Classification Head:**
```
Input (1408) → Dropout(0.3) → Linear(512) → BatchNorm → ReLU 
→ Dropout(0.3) → Linear(256) → BatchNorm → ReLU 
→ Dropout(0.3) → Linear(2)
```

The progressive dimensionality reduction with interleaved regularization prevents overfitting while maintaining discriminative capacity.

**Transfer Learning Strategy:** Pre-training on ImageNet provides robust low-level feature extractors (edges, textures) that generalize well to spectro-temporal patterns in audio.

### 5.3 Light CNN (LCNN)

**Architecture Description:**

LCNN employs Max-Feature-Map (MFM) activation functions instead of traditional ReLU. MFM acts as a learnable feature selection mechanism, particularly effective for spoofing detection where discriminative artifact selection is crucial.

**MFM Activation:**

Given input $\mathbf{x} \in \mathbb{R}^{2C \times H \times W}$, MFM partitions channels and applies element-wise maximum:

$$\text{MFM}(\mathbf{x})_{c,h,w} = \max(\mathbf{x}_{c,h,w}, \mathbf{x}_{c+C,h,w})$$

This reduces the channel dimension by half while preserving the most salient features.

**Network Structure:**

```
Input (1, 128, T)
↓
Conv-MFM Block (32) → MaxPool(2×2)
↓
Conv-MFM Block (48) → Residual Block (48) → MaxPool(2×2)
↓
Conv-MFM Block (64) → Residual Block (64) → MaxPool(2×2)
↓
Conv-MFM Block (32) → Residual Block (32) → MaxPool(2×2)
↓
Attentive Statistics Pooling
↓
FC-MFM (256) → FC-MFM (128) → Linear(2)
```

**Attentive Statistics Pooling:**

Temporal aggregation is performed via attention-weighted statistics:

$$\alpha_t = \frac{\exp(w^T \phi(h_t))}{\sum_{t'} \exp(w^T \phi(h_{t'}))}$$

$$\mu = \sum_t \alpha_t h_t, \quad \sigma = \sqrt{\sum_t \alpha_t h_t^2 - \mu^2}$$

$$\mathbf{v} = [\mu; \sigma]$$

where $\phi$ is a learned transformation and $[\cdot; \cdot]$ denotes concatenation.

### 5.4 RawNet3

**Architecture Description:**

RawNet3 processes raw waveforms directly, eliminating handcrafted feature extraction. This end-to-end approach allows the network to learn optimal representations for the task.

**Key Components:**

1. **Sinc Convolution Layer:** Parameterized band-pass filters learn frequency band selection:

$$h[n] = 2f_c \text{sinc}(2\pi f_c n) \cdot w[n]$$

where $f_c$ is the learnable cutoff frequency and $w[n]$ is a Hamming window.

2. **Residual Blocks with Squeeze-and-Excitation:** Channel-wise attention recalibrates feature maps.

3. **Gated Recurrent Units (GRU):** Captures long-range temporal dependencies.

### 5.5 SE-ResNet

**Architecture Description:**

Squeeze-and-Excitation ResNet incorporates channel-wise attention mechanisms into the residual learning framework.

**SE Block:**

$$\tilde{\mathbf{F}}_c = \mathbf{F}_c \cdot \sigma(W_2 \delta(W_1 \mathbf{z}))$$

where $\mathbf{z} = \frac{1}{HW}\sum_{h,w} \mathbf{F}_{c,h,w}$ is global average pooling, $\delta$ is ReLU, $\sigma$ is sigmoid, and $W_1, W_2$ are learned projections.

---

## 6. Experimental Setup

### 6.1 Loss Function

A weighted Cross-Entropy loss addresses class imbalance:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} w_{y_i} \log p_{y_i}(\mathbf{x}_i)$$

where $w_0 = 0.1$ (spoof) and $w_1 = 0.9$ (bonafide), emphasizing correct classification of genuine audio.

### 6.2 Optimization

**Adam Optimizer:** Adaptive moment estimation with parameters:
- Learning rate: $\eta = 10^{-4}$
- $\beta_1 = 0.9$, $\beta_2 = 0.999$
- Weight decay: $\lambda = 10^{-4}$

**Learning Rate Scheduling:**

Cosine annealing without warm restarts:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

where $t$ is the current step, $T$ is the total training steps, $\eta_{\max} = 10^{-4}$, and $\eta_{\min} = 10^{-6}$.

### 6.3 Regularization

1. **Dropout:** Applied with $p = 0.3$ in fully connected layers
2. **Batch Normalization:** After each convolutional layer
3. **Gradient Clipping:** Maximum gradient norm of 1.0
4. **Early Stopping:** Based on validation EER with patience of 5 epochs

---

## 7. Training Procedure
## 7. Training Procedure

### 7.1 Training Configuration

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Epochs | 20 | Sufficient for convergence on dataset |
| Batch Size | 32 | Balanced GPU memory and gradient stability |
| Initial Learning Rate | $10^{-4}$ | Standard for Adam optimizer |
| Weight Decay | $10^{-4}$ | L2 regularization strength |
| Gradient Clip Norm | 1.0 | Prevents gradient explosion |

### 7.2 Mixed Precision Training

To accelerate training and reduce memory consumption, we employed Automatic Mixed Precision (AMP):

**Precision Selection:**
```python
# Ampere+ GPUs (compute capability ≥ 8.0)
if torch.cuda.get_device_capability()[0] >= 8:
    dtype = torch.bfloat16  # Superior dynamic range, no scaling
else:
    dtype = torch.float16   # Requires gradient scaling
```

**Training Loop with AMP:**
```python
with torch.amp.autocast('cuda', dtype=dtype):
    embeddings, logits = model(inputs)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### 7.3 Stochastic Weight Averaging (SWA)

SWA improves generalization by averaging model weights traversed during training:

$$\mathbf{w}_{\text{SWA}} = \frac{1}{K}\sum_{k=1}^{K} \mathbf{w}_{n_k}$$

where $\{n_k\}$ are epochs selected for averaging (typically the last 50% of training).

**Implementation:**
- Average weights from epochs 10-20
- Update BatchNorm statistics on training set before evaluation
- Often yields 0.5-1.0% improvement in EER

### 7.4 Epoch Training Loop

**Pseudocode:**
```
for epoch in 1 to num_epochs:
    # Training phase
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs = apply_augmentation(inputs)  # On-the-fly
        
        optimizer.zero_grad(set_to_none=True)  # Efficient gradient reset
        
        with autocast():
            embeddings, logits = model(inputs)
            loss = weighted_cross_entropy(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Per-step scheduling
    
    # Validation phase
    model.eval()
    with torch.inference_mode():
        dev_scores = evaluate(model, dev_loader)
        dev_eer = compute_eer(dev_scores)
    
    # Checkpoint best model
    if dev_eer < best_dev_eer:
        save_checkpoint(model, "best.pth")
        best_dev_eer = dev_eer
    
    # SWA update (if enabled and epoch > 10)
    if use_swa and epoch > num_epochs // 2:
        swa_optimizer.update_swa()
```

---

## 8. Model Validation and Selection

### 8.1 Score Generation

For each audio sample, the model produces a scalar score representing the bonafide likelihood:

$$s(\mathbf{x}) = \text{logit}_{\text{bonafide}}(\mathbf{x}) = \log\frac{p(y=1|\mathbf{x})}{p(y=0|\mathbf{x})}$$

Higher scores indicate higher confidence in bonafide classification.

### 8.2 Equal Error Rate (EER) Computation

EER is the primary evaluation metric, defined as the operating point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR):

$$\text{FAR}(\theta) = \frac{|\{i: s_i \geq \theta, y_i = 0\}|}{|\{i: y_i = 0\}|}$$

$$\text{FRR}(\theta) = \frac{|\{i: s_i < \theta, y_i = 1\}|}{|\{i: y_i = 1\}|}$$

$$\text{EER} = \text{FAR}(\theta^*) = \text{FRR}(\theta^*) \text{ where } |\text{FAR}(\theta^*) - \text{FRR}(\theta^*)|$$ is minimized.

**Algorithm:**
```
1. Sort all scores and compute FAR, FRR at each unique threshold
2. Find threshold θ* where |FAR(θ) - FRR(θ)| is minimized
3. EER = (FAR(θ*) + FRR(θ*)) / 2
```

### 8.3 Model Selection Criterion

The model checkpoint yielding the lowest EER on the validation set is selected for final evaluation on the test set. This prevents overfitting to validation data through early stopping.

---

## 9. Ensemble Strategy

### 9.1 Motivation

Ensemble methods reduce prediction variance and improve robustness by combining predictions from multiple models trained with different initializations, architectures, or hyperparameters.

### 9.2 Soft Voting

Given $M$ trained models $\{f_1, \ldots, f_M\}$, the ensemble prediction is computed via soft voting:

$$p_{\text{ensemble}}(y|\mathbf{x}) = \frac{1}{M}\sum_{m=1}^{M} p_m(y|\mathbf{x})$$

$$\hat{y} = \arg\max_y p_{\text{ensemble}}(y|\mathbf{x})$$

### 9.3 Score-Level Fusion

For EER evaluation, scores are averaged:

$$s_{\text{ensemble}}(\mathbf{x}) = \frac{1}{M}\sum_{m=1}^{M} s_m(\mathbf{x})$$

This provides a robust aggregate score for threshold-based decision-making.

### 9.4 Ensemble Configuration

Our best-performing ensemble consists of:
1. EfficientNet-B2 (Mel-spectrogram)
2. LCNN (LFCC)
3. RawNet3 (Raw waveform)

This combination leverages complementary feature representations and architectural inductive biases.

---

## 10. Inference Pipeline

## 10. Inference Pipeline

### 10.1 Pipeline Stages

The inference pipeline processes audio files through the following sequential stages:

**Stage 1: Audio Loading and Preprocessing**
```python
waveform, sr = sf.read(audio_path)
if waveform.ndim > 1:
    waveform = np.mean(waveform, axis=1)  # Mono conversion
if sr != 16000:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
```

**Stage 2: Fixed-Length Normalization**
```python
waveform = pad_center(waveform, target_length=64600)
```

**Stage 3: Feature Extraction**
```python
features = extract_feature(waveform, feature_type=1)  # Log-Mel
features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
```

**Stage 4: Model Inference**
```python
model.eval()
with torch.inference_mode():  # Disables autograd for speed
    embeddings, logits = model(features.to(device))
    probabilities = torch.softmax(logits, dim=1)
```

**Stage 5: Decision Making**
```python
score = logits[0, 1].item()  # Bonafide logit
prediction = "Bonafide" if score >= threshold else "Spoof"
confidence = probabilities[0, 1 if prediction == "Bonafide" else 0].item()
```

### 10.2 Batch Inference Optimization

For processing multiple files, batch inference significantly reduces overhead:

```python
def batch_inference(audio_paths, model, batch_size=32):
    results = []
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        
        # Parallel loading and preprocessing
        batch_features = []
        for path in batch_paths:
            waveform = load_and_preprocess(path)
            features = extract_feature(waveform)
            batch_features.append(features)
        
        # Stack and infer
        batch_tensor = torch.stack([torch.FloatTensor(f) for f in batch_features])
        
        with torch.inference_mode():
            _, logits = model(batch_tensor.to(device))
            scores = logits[:, 1].cpu().numpy()
        
        results.extend(scores)
    
    return results
```

### 10.3 Real-Time Processing

For real-time applications, streaming inference processes audio chunks:

```python
class StreamingDetector:
    def __init__(self, model, chunk_duration=4.0, overlap=0.5):
        self.model = model
        self.chunk_samples = int(chunk_duration * 16000)
        self.hop_samples = int(chunk_samples * (1 - overlap))
        self.buffer = []
    
    def process_chunk(self, audio_chunk):
        self.buffer.extend(audio_chunk)
        
        if len(self.buffer) >= self.chunk_samples:
            # Extract chunk
            chunk = np.array(self.buffer[:self.chunk_samples])
            self.buffer = self.buffer[self.hop_samples:]
            
            # Infer
            features = extract_feature(chunk)
            with torch.inference_mode():
                _, logits = self.model(features.to(device))
                score = logits[0, 1].item()
            
            return score
        return None
```

---

## 11. Evaluation Metrics

### 11.1 Primary Metric: Equal Error Rate (EER)

EER serves as the primary evaluation metric, providing a threshold-independent measure of system performance. It represents the operating point where the system makes equal proportions of false acceptances (spoof classified as bonafide) and false rejections (bonafide classified as spoof).

**Mathematical Definition:**

$$\text{EER} = \text{FAR}(\theta^*) = \text{FRR}(\theta^*))$$

where:
- $\text{FAR}(\theta) = P(s(\mathbf{x}) \geq \theta | y = 0)$ (False Acceptance Rate)
- $\text{FRR}(\theta) = P(s(\mathbf{x}) < \theta | y = 1)$ (False Rejection Rate)
- $\theta^*$ is the threshold satisfying $\text{FAR}(\theta^*) = \text{FRR}(\theta^*)$

**Advantages:**
- Threshold-independent comparison across systems
- Balanced assessment of both error types
- Standard metric in biometric and anti-spoofing research

### 11.2 Secondary Metric: Classification Accuracy

Accuracy at the EER threshold provides an intuitive performance measure:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

where decisions are made using the EER threshold $\theta^*$.

### 11.3 Detection Error Tradeoff (DET) Curve

The DET curve plots FRR against FAR across all possible thresholds on a normal deviate scale, providing a comprehensive visualization of system performance:

$$\text{DET}: \Phi^{-1}(\text{FRR}(\theta)) \text{ vs. } \Phi^{-1}(\text{FAR}(\theta))$$

where $\Phi^{-1}$ is the inverse standard normal CDF.

---

## 12. Optimization Techniques

### 12.1 Mixed Precision Training

Mixed precision training reduces memory footprint and accelerates computation by using lower precision (FP16/BF16) for forward and backward passes while maintaining FP32 master weights.

**BFloat16 (BF16):**
- 8-bit exponent, 7-bit mantissa
- Native support on Ampere (A100, RTX 30/40 series) and newer
- Superior dynamic range: $\approx 10^{-45}$ to $10^{38}$
- No gradient scaling required

**Float16 (FP16):**
- 5-bit exponent, 10-bit mantissa
- Supported on all CUDA GPUs (compute capability ≥ 7.0)
- Requires gradient scaling to prevent underflow
- Dynamic range: $\approx 10^{-8}$ to $65504$

**Performance Gains:** 
- Training speedup: 1.3-1.5×
- Memory reduction: ~40%
- No accuracy degradation

### 12.2 TensorFloat-32 (TF32)

TF32 is an intermediate precision format (19-bit) that accelerates FP32 matrix multiplications on Ampere+ GPUs:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Characteristics:**
- Automatic conversion from FP32
- 8× throughput improvement on Ampere
- Negligible accuracy impact (<0.1% typical)

### 12.3 Channels-Last Memory Layout

Channels-last format $(N, H, W, C)$ improves memory locality and enables hardware-specific optimizations:

$$\text{Speedup} \approx 1.2\text{-}1.3\times \text{ on modern GPUs}$$

### 12.4 Gradient Accumulation

For effective large batch training with limited memory:

```python
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    with autocast():
        loss = model(inputs, labels) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

Effective batch size: $B_{\text{eff}} = B \times A$ where $A$ is accumulation steps.

---

## 13. Reproducibility

### 13.1 Random Seed Control

Complete determinism requires seeding all random number generators:

```python
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Trade-off:** Deterministic mode may reduce performance by 10-20% due to disabled auto-tuning.

### 13.2 DataLoader Worker Seeding

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

DataLoader(..., worker_init_fn=seed_worker, generator=g)
```

### 13.3 Reproducibility Checklist

To reproduce results, ensure identical:

✓ Software versions (PyTorch, CUDA, cuDNN, Python)  
✓ Random seeds (NumPy, PyTorch, Python, CUDA)  
✓ Hyperparameters (learning rate, batch size, epochs)  
✓ Dataset preprocessing and augmentation settings  
✓ Model architecture and initialization  
✓ Hardware specifications (GPU model may affect numerical precision)  
✓ Operating system and driver versions  

### 13.4 Configuration Management

All experimental configurations are stored in JSON format:

```json
{
  "model_config": {
    "architecture": "EfficientNetB2",
    "pretrained": true,
    "dropout": 0.3
  },
  "training_config": {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "scheduler": "cosine"
  },
  "data_config": {
    "feature_type": 1,
    "sample_rate": 16000,
    "segment_length": 64600,
    "augmentation": true
  },
  "seed": 1234
}
```

---

## 14. Computational Resources

### 14.1 Hardware Specifications

**Recommended Configuration:**
- **GPU:** NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **CPU:** 8+ cores for data loading
- **RAM:** 32GB minimum
- **Storage:** SSD for dataset (recommended for I/O performance)

**Minimum Configuration:**
- **GPU:** NVIDIA GTX 1080 Ti (11GB VRAM)
- **CPU:** 4 cores
- **RAM:** 16GB

### 14.2 Training Time

| Model | Parameters | Training Time (20 epochs) | GPU Memory |
|-------|-----------|---------------------------|------------|
| EfficientNet-B2 | 9.2M | 3-4 hours | 8GB |
| LCNN | 3.5M | 2-3 hours | 6GB |
| RawNet3 | 8.1M | 4-5 hours | 10GB |
| SE-ResNet | 11.2M | 4-6 hours | 12GB |

*Measurements on NVIDIA RTX 3090 with batch size 32 and mixed precision.*

### 14.3 Inference Performance

**Single Sample Latency:**
- CPU (Intel i7-10700K): ~150-200ms
- GPU (RTX 3090): ~5-10ms

**Batch Inference Throughput (GPU):**
- Batch size 1: ~100-200 samples/sec
- Batch size 32: ~2000-3000 samples/sec
- Batch size 128: ~3500-4500 samples/sec

**Memory Usage (Inference):**
- Model weights: 30-50MB
- Single sample: <100MB
- Batch of 32: ~500MB-1GB

### 14.4 Storage Requirements

- **Dataset (Fake-or-Real):** ~2-5GB
- **Model checkpoints:** 30-50MB per model
- **Training artifacts:** ~500MB per experiment (logs, metrics, scores)
- **Augmentation cache (optional):** 10-20GB

---

## 15. Conclusion

This methodology presents a comprehensive deep learning framework for audio deepfake detection, incorporating state-of-the-art neural architectures, robust data augmentation, and advanced training techniques. The systematic approach—from feature extraction through ensemble inference—provides a reproducible pipeline achieving competitive performance on the Fake-or-Real dataset.

Key contributions include:
1. **Multi-modal feature extraction** supporting diverse acoustic representations
2. **Extensive augmentation pipeline** simulating realistic acoustic variability
3. **Optimized training procedures** leveraging mixed precision and weight averaging
4. **Ensemble strategies** combining complementary models for robust detection
5. **Production-ready inference** with real-time processing capabilities

The methodology establishes a foundation for future research in audio forensics, with extensibility to additional datasets, architectures, and evaluation protocols.

---

*Methodology Documentation v2.0 — January 17, 2026*
