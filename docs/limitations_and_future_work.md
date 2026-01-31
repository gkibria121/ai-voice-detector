# Limitations and Future Work

This document outlines the current limitations of our AI Voice Detection system and proposes directions for future research and development.

---

## Current Limitations

### 1. Dataset Constraints

#### Limited Dataset Scope
- **Single Dataset Training**: Models are trained and evaluated exclusively on the FoR-2sec (Fake or Real) dataset, which may limit generalization to other deepfake detection scenarios.
- **Constrained Audio Duration**: The dataset uses 2-second clips, which may not capture long-range temporal dependencies present in longer utterances.
- **Limited TTS Variety**: The dataset includes specific TTS systems (Deep Voice 3, Google WaveNet) but doesn't cover the full spectrum of modern voice cloning techniques (e.g., VITS, Tortoise-TTS, ElevenLabs, VALL-E).

#### Dataset Bias
- **Controlled Recording Conditions**: The dataset primarily contains clean audio recordings, which may not reflect real-world acoustic conditions.
- **Language Limitation**: English-only dataset limits applicability to multilingual scenarios.
- **Speaker Diversity**: Limited representation of diverse speaker demographics, accents, and speaking styles.

---

### 2. Model Architecture Limitations

#### MFCC Feature Underperformance
- MFCC features consistently underperformed across all architectures (best: 92.37% with SENet), indicating they may not capture sufficient discriminative information for deepfake detection.

#### Raw Waveform Model Weakness
- Raw waveform models (RawNet3: 78.12%, SimpleCNN: 64.52%) significantly underperformed compared to spectrogram-based approaches, suggesting time-frequency representations are crucial for this task.

#### Ensemble Component Instability
- **LCNN Volatility**: As identified in XAI analysis, LCNN generates high-confidence errors on both fake and real audio samples. This "hallucination" behavior can reduce ensemble reliability.
- **Voting Weight Sensitivity**: The ensemble's performance heavily depends on properly calibrated voting weights among component models.

---

### 3. Generalization Concerns

#### Cross-Dataset Performance Unknown
- No validation on other benchmark datasets (ASVspoof 2019/2021, WaveFake, In-the-Wild datasets).
- Potential overfitting to FoR-2sec-specific artifacts that may not generalize.

#### Unseen TTS/VC Systems
- Models may struggle to detect audio from TTS/VC systems not represented in training data.
- No evaluation against emerging diffusion-based voice synthesis methods.

#### Acoustic Environment Variability
- Limited robustness testing under real-world conditions:
  - Background noise
  - Compression artifacts (MP3, AAC, Opus)
  - Telephony transmission effects
  - Microphone quality variations
  - Reverberation

---

### 4. Performance Gap

#### Gap to State-of-the-Art
- There remains a **2.26% accuracy gap** between our best model (EfficientNet-B2 + Attention: 97.70%) and the best published result (VGG16 + STFT: 99.96%).
- STFT features, which achieved the highest accuracy in literature, were not explored in this work.

---

### 5. Technical Limitations

#### Computational Requirements
- EfficientNet-B2 models require substantial GPU memory (~9.2M parameters).
- No optimized variants for edge deployment (mobile, embedded systems).
- Real-time inference on resource-constrained devices not validated.

#### Explainability Depth
- Current Grad-CAM visualizations show *where* models focus but not *why* specific patterns indicate fake/real.
- Limited understanding of which specific acoustic artifacts trigger classification decisions.

---

### 6. Evaluation Metrics Limitations

#### Binary Classification Focus
- Current evaluation treats detection as binary (fake vs. real), not addressing:
  - Multi-class detection (identifying specific TTS/VC systems)
  - Detection confidence calibration
  - Partial spoofing scenarios (real audio with inserted fake segments)

---

## Future Work

### 1. Dataset and Training Improvements

#### Priority: HIGH

**Cross-Dataset Training and Validation**
- Incorporate additional datasets: ASVspoof 2019/2021, WaveFake, FakeAVCeleb, In-the-Wild.
- Implement domain adaptation techniques (CORAL, MMD, DANN) for cross-dataset generalization.
- Collect and annotate diverse real-world samples.

**Expanded TTS Coverage**
- Include modern synthesis systems: VITS, Tortoise-TTS, RVC, So-VITS-SVC, VALL-E.
- Regular retraining pipeline to incorporate newly released TTS technologies.
- Synthetic data augmentation using multiple TTS systems.

**Longer Audio Analysis**
- Extend support beyond 2-second clips to capture long-range temporal patterns.
- Implement hierarchical models for utterance-level and segment-level analysis.

---

### 2. Feature Representation Research

#### Priority: HIGH

**STFT Feature Exploration**
- Implement Short-Time Fourier Transform features given their 99.96% accuracy in literature.
- Experiment with different window sizes and hop lengths.
- Compare STFT vs. Mel Spectrogram performance across architectures.

**Multi-Feature Fusion**
- Feature-level fusion: Mel Spectrogram + CQT + STFT + LFCC.
- Learn optimal feature combination weights.
- Cross-attention mechanisms between feature modalities.

**Self-Supervised Representations**
- Integrate pre-trained audio embeddings (HuBERT, WavLM).
- Fine-tune SSL models for deepfake detection.
- Evaluate transfer learning from large-scale speech corpora.

---

### 3. Architecture Enhancements

#### Priority: HIGH

**Advanced Attention Mechanisms**
Given the 8.27% improvement from attention mechanisms, explore:
- Multi-head self-attention
- Cross-modal attention (for multi-feature inputs)
- CBAM (Convolutional Block Attention Module)
- Temporal attention for sequential patterns
- Non-local neural networks

**Ensemble Optimization**
- Replace or retrain LCNN component to address volatility issues.
- Implement learned ensemble weights instead of static averaging.
- Explore stacking with meta-learner for decision fusion.
- Confidence-aware weighting based on prediction uncertainty.

**VGG Architecture Investigation**
- Despite being older, VGG16/VGG19 achieved exceptional results (99.96%) in literature.
- Test VGG variants with modern training techniques (AdamW, cosine annealing, label smoothing).

---

### 4. Robustness and Adversarial Testing

#### Priority: MEDIUM

**Adversarial Robustness**
- Evaluate against adversarial attacks: FGSM, PGD, C&W.
- Audio-specific adversarial perturbations.
- Implement adversarial training for defense.
- Certified defense mechanisms.

**Real-World Condition Testing**
- Systematic evaluation under various noise conditions (SNR levels).
- Codec compression robustness (MP3, AAC, Opus at various bitrates).
- Telephony channel simulation (VoIP, cellular).
- Multi-microphone and cross-device testing.

**Adaptive Adversaries**
- Test against adversarial audio crafted to fool the detection system.
- Post-processing attacks (slight noise addition, pitch shifting).

---

### 5. Deployment and Optimization

#### Priority: MEDIUM

**Edge Deployment**
- Model pruning for lightweight inference.
- Knowledge distillation from EfficientNet-B2 to smaller architectures.
- Quantization (INT8, INT4) for efficient inference.
- Neural architecture search for optimal efficiency-accuracy trade-off.

**Target Platforms**
- Mobile deployment (iOS, Android).
- Edge TPU and embedded systems.
- WebAssembly for browser-based detection.
- Cloud API with batch processing capabilities.

**Real-Time Optimization**
- Streaming inference pipeline optimization.
- Latency profiling and reduction.
- Memory footprint optimization.

---

### 6. Explainability and Interpretability

#### Priority: MEDIUM

**Enhanced Visualization**
- Layer-wise relevance propagation (LRP).
- SHAP (SHapley Additive exPlanations) for feature importance.
- Attention weight visualization across layers.
- Temporal pattern highlighting.

**Artifact Characterization**
- Identify specific acoustic artifacts that distinguish fake from real audio.
- Document common patterns in different TTS/VC systems.
- Build interpretable feature sets for explainable decisions.

**Forensic Applications**
- Develop tools for audio forensic analysis.
- Generate detection reports with evidence highlighting.
- Confidence calibration for forensic use cases.

---

### 7. Multimodal Detection

#### Priority: LOW-MEDIUM

**Audio-Visual Fusion**
- Integrate with video-based deepfake detection.
- Lip-sync consistency analysis.
- Cross-modal attention between audio and visual streams.
- Face-voice association verification.

**Text-Audio Alignment**
- Detect inconsistencies between transcribed text and audio.
- Prosodic pattern analysis against expected linguistic patterns.

---

### 8. Continual Learning

#### Priority: LOW

**Adaptive Systems**
- Online learning to adapt to new deepfake techniques.
- Concept drift detection for evolving TTS methods.
- Active learning for efficient labeling of new samples.
- Federated learning for privacy-preserving model updates.

---

## Summary: Prioritized Roadmap

| Priority | Focus Area | Expected Impact |
|----------|------------|-----------------|
| **HIGH** | STFT feature implementation | Close 2.26% accuracy gap |
| **HIGH** | Cross-dataset validation | Verify generalization |
| **HIGH** | Ensemble optimization (fix LCNN) | Improve reliability |
| **HIGH** | VGG architecture testing | Potentially match SOTA |
| **MEDIUM** | Adversarial robustness | Production readiness |
| **MEDIUM** | Edge deployment optimization | Real-world applicability |
| **MEDIUM** | Enhanced explainability | Trust and transparency |
| **LOW** | Multimodal detection | Extended capabilities |
| **LOW** | Continual learning | Long-term adaptability |

---

## Conclusion

While our current system achieves competitive performance (97.70% accuracy, 2.39% EER), addressing these limitations is crucial for building a robust, generalizable, and production-ready audio deepfake detection system. The most pressing priorities are:

1. **Closing the accuracy gap** through STFT features and VGG architecture exploration
2. **Validating generalization** through cross-dataset evaluation
3. **Fixing ensemble reliability** by addressing LCNN component issues
4. **Ensuring robustness** against real-world conditions and adversarial attacks

These improvements will position the system for practical deployment in security-critical applications where detecting AI-generated voice is essential.
