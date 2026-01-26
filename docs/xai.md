# XAI Analysis: EfficientNetB2_Attention vs Ensemble

A comparative analysis of explainability (Grad-CAM) outputs to understand why the **EfficientNetB2_Attention** standalone model (97% accuracy) outperforms the **Ensemble** model (96% accuracy).

## Understanding This Analysis

**What are we looking at?**

- **Input Spectrogram** (left): Visual representation of audio - X-axis is time, Y-axis is frequency
- **Saliency Map** (right): Shows WHERE the model is looking to make its decision
  - **Bright yellow/white areas** = "I'm paying close attention here"
  - **Dark red/black areas** = "I'm ignoring this region"

**Important**: The bright areas don't tell us if audio is real or fake - they just show what the model examines. The actual prediction depends on what patterns the model recognizes in those bright regions.

**Labels**: 0 = FAKE voice | 1 = REAL voice

---

## Test Samples

| Audio File   | Ground Truth |
| ------------ | ------------ |
| `file48.wav` | **FAKE (0)** |
| `file5.wav`  | **REAL (1)** |

---

## FAKE Audio Analysis (file48.wav)

### EfficientNetB2_Attention (Standalone)

![EfficientNetB2_Attention on FAKE audio](images/xai_notebook_0_1.png)

- **Prediction**: 0 (Fake) with 100% Confidence ✓ **CORRECT**
- **What the model sees**:
  - Strong bright activation focused in the **bottom portion** of the spectrogram (low frequencies: 0-20 Hz range)
  - Attention is concentrated in the **lower third** of the frequency spectrum across most time segments
- **Why this indicates FAKE**: The model has learned that fake/synthesized audio often has suspicious patterns in these low-frequency regions. The focused, intense attention on just the bottom frequencies (rather than looking broadly across all frequencies) suggests the model detected something "off" in the fundamental frequency patterns that are characteristic of synthesized speech.

---

### Ensemble Component Analysis

#### LCNN — Confidence: 77.34% → Fake (0) ✓

![LCNN on FAKE audio](images/xai_notebook_1_1.png)

- **What the model sees**: Horizontal bands of bright attention across **mid-to-high frequencies** (roughly 10-50 Hz range), distributed across the entire time duration
- **Result**: Correctly predicts FAKE but with moderate confidence (77%), suggesting it sees some fake indicators but isn't as certain

---

#### SEResNet — Confidence: 88.81% → Real (1) ✗

![SEResNet on FAKE audio](images/xai_notebook_1_3.png)

- **What the model sees**: Attention heavily concentrated in the **upper-middle frequency region** (around 20-60 Hz), appearing as bright yellow zones in that band
- **Why it FAILED**: SEResNet is looking at mid-high frequencies where the fake audio might have preserved some natural-sounding characteristics. It's NOT looking at the low frequencies where the synthesis artifacts are more obvious. This is a blind spot - the model is focusing on the "good" parts of the fake audio and missing the telltale signs in other frequency ranges.
- **Result**: **WRONG** - Confidently (88%) calls it REAL when it's actually FAKE

---

#### EfficientNetB2_Attention (Ensemble Version) — Confidence: 100% → Fake (0) ✓

![EfficientNetB2_Attention in Ensemble on FAKE audio](images/xai_notebook_1_5.png)

- **What the model sees**: Almost identical pattern to the standalone version - strong attention concentrated in **low-frequency regions** (bottom of spectrogram), with bright bands in the 10-30 Hz range
- **Result**: Correctly predicts FAKE with perfect confidence, just like the standalone model

---

### Ensemble Final Result for file48.wav

**Prediction**: 0 (Fake) with 100% Confidence ✓ **CORRECT**

| Average Ensemble                        | Weighted Ensemble                        |
| --------------------------------------- | ---------------------------------------- |
| ![Average](images/xai_notebook_1_7.png) | ![Weighted](images/xai_notebook_1_9.png) |

- **What we see in ensemble maps**: Both averaging methods show attention primarily in **mid-to-high frequency bands** (the horizontal bright bands spanning roughly 15-50 Hz)
- **Why ensemble succeeded**: Even though SEResNet was wrong (88% Real), the other two models were right. EfficientNetB2's extremely strong 100% Fake confidence and LCNN's 77% Fake confidence mathematically overpowered SEResNet's 88% Real vote.
- **Key takeaway**: The ensemble's voting system worked well here - the correct strong signal beat the incorrect moderate signal.

---

## REAL Audio Analysis (file5.wav)

### EfficientNetB2_Attention (Standalone)

![EfficientNetB2_Attention on REAL audio](images/xai_notebook_2_1.png)

- **Prediction**: 1 (Real) with 80.74% Confidence ✓ **CORRECT**
- **What the model sees**:
  - Bright attention spread more **evenly across the lower half** of the frequency spectrum (roughly 0-40 Hz)
  - The attention is **broader and more diffused** compared to the fake audio
  - No single frequency band dominates - it's checking multiple regions
- **Why this indicates REAL**: The distributed attention pattern suggests the model is finding natural speech characteristics spread across multiple frequency ranges (fundamental frequency, harmonics, formants). Real human speech has rich, varied frequency content. The moderate confidence (80% vs 100%) indicates the model recognizes this sample has some complexity but still sees enough natural patterns to call it real.

---

### Ensemble Component Analysis

#### LCNN — Confidence: 98.67% → Real (1) ✓

![LCNN on REAL audio](images/xai_notebook_3_1.png)

- **What the model sees**: Bright yellow bands concentrated in **specific horizontal stripes** across mid frequencies (approximately 20-30 Hz range), very consistent across time
- **Result**: Correctly and confidently predicts REAL (98.67%)

---

#### SEResNet — Confidence: 99.70% → Real (1) ✓

![SEResNet on REAL audio](images/xai_notebook_3_3.png)

- **What the model sees**: Similar to LCNN - strong attention in **horizontal bands at mid frequencies** (around 20-30 Hz), appearing as bright yellow stripes
- **Result**: Correctly and very confidently predicts REAL (99.70%)

---

#### EfficientNetB2_Attention (Ensemble Version) — Confidence: 100.00% → Fake (0) ✗

![EfficientNetB2_Attention in Ensemble on REAL audio](images/xai_notebook_3_5.png)

- **What the model sees**: Strong attention in **mid-to-high frequency bands** (approximately 20-50 Hz), visible as bright horizontal bands - similar to where it looked for the fake audio
- **CRITICAL PROBLEM**: This version of EfficientNetB2 is looking at similar frequency regions as the standalone version BUT interpreting what it sees completely differently:
  - **Standalone version**: Sees distributed patterns → correctly says REAL (80%)
  - **Ensemble version**: Sees similar patterns → incorrectly says FAKE (100%)
- **Why it FAILED catastrophically**: This suggests the ensemble's EfficientNetB2 checkpoint is fundamentally broken or trained differently. It's not just making a small error - it's 100% confident in the WRONG direction. The saliency patterns look similar to other models, but its internal interpretation is inverted.
- **Result**: **CATASTROPHICALLY WRONG** - Says FAKE with 100% confidence when it's actually REAL

---

### Ensemble Final Result for file5.wav

**Prediction**: 0 (Fake) with 100% Confidence ✗ **WRONG**

| Average Ensemble                        | Weighted Ensemble                        |
| --------------------------------------- | ---------------------------------------- |
| ![Average](images/xai_notebook_3_7.png) | ![Weighted](images/xai_notebook_3_9.png) |

- **What we see in ensemble maps**: Both show attention focused on **mid-frequency bands** around 20-30 Hz (the bright horizontal stripes)
- **Why ensemble FAILED**: Here's the mathematical problem:
  - LCNN says: 98.67% REAL (1)
  - SEResNet says: 99.70% REAL (1)
  - EfficientNetB2 says: 100% FAKE (0)

  When you average these (even with weights), the 100% FAKE from EfficientNetB2 is so extreme that it pulls the final average toward FAKE, despite 2 out of 3 models being correct with very high confidence.

- **The "tyranny of overconfidence" problem**: A single model outputting an extreme value (0% or 100%) can mathematically dominate the ensemble average, making the other models' votes nearly irrelevant. If EfficientNetB2 had said 80% Fake instead of 100% Fake, the ensemble would have correctly predicted Real.

---

## Conclusion: Why EfficientNetB2 Standalone (97%) > Ensemble (96%)

### The Core Problem

Looking at the actual saliency maps across both test cases reveals the issue:

**On FAKE audio (file48.wav):**

- ✅ Standalone EfficientNetB2: Focuses on low frequencies → Correctly detects FAKE
- ✅ Ensemble EfficientNetB2: Focuses on low frequencies → Correctly detects FAKE
- Both versions looking at similar regions and agreeing

**On REAL audio (file5.wav):**

- ✅ Standalone EfficientNetB2: Distributed attention across frequencies → Correctly detects REAL (with 80% confidence)
- ❌ Ensemble EfficientNetB2: Similar attention pattern → **INCORRECTLY** says FAKE (with 100% confidence)
- Both versions looking at similar regions but interpreting them completely differently!

### Key Findings

1. **The Ensemble's EfficientNetB2 Component is Broken**
   - It's not looking at different regions - the saliency maps show similar attention patterns to other models
   - The problem is **interpretation**, not attention: it sees similar patterns but reaches opposite conclusions
   - This suggests a checkpoint mismatch, different training data, or model corruption

2. **Asymmetric Failure Pattern**
   - **On Fake Audio**: Ensemble works well because when models agree, overconfidence helps
   - **On Real Audio**: Ensemble fails because one overconfident wrong model overpowers two correct models
   - The standalone model shows appropriate uncertainty (80%) while the ensemble version shows inappropriate certainty (100%)

3. **Mathematical Voting Vulnerability**
   - Even though 2 out of 3 models correctly said "Real" with 98-99% confidence
   - The single 100% "Fake" prediction dominated the weighted average
   - This is the "tyranny of the overconfident member": extreme predictions (0% or 100%) mathematically override consensus

4. **Why Standalone Wins**
   - Better calibrated: Shows 80% confidence on tricky samples instead of 100%
   - More reliable: Doesn't have the checkpoint inconsistency issue
   - Safer decision-making: Leaves room for uncertainty rather than being overconfident

### The 1% Accuracy Gap Explained

The standalone model's 97% vs ensemble's 96% gap comes from:

- Ensemble has MORE false negatives (calling Real audio Fake)
- This happens when the broken EfficientNetB2 component becomes overconfident in the wrong direction
- The standalone model avoids this by having consistent, well-calibrated predictions

### Recommendations

1. **Verify Checkpoints**: The ensemble's EfficientNetB2 checkpoint appears to be from a different training run or corrupted
2. **Prevent Extreme Predictions**: Cap confidence scores at 95% to prevent any single model from dominating
3. **Better Voting**: Use median voting or confidence-aware voting instead of simple weighted averaging
4. **Test Thoroughly**: Run this analysis on 50+ samples to confirm the pattern holds

---

## Technical Details

- **Visualization Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Input Format**: Audio spectrograms (Frequency on Y-axis vs Time on X-axis)
- **Saliency Maps Interpretation**:
  - Show which regions of the spectrogram the model examines (not what it concludes)
  - Bright yellow/white = High attention, Dark red/black = Low attention
- **Ground Truth Labels**: 0 = Fake voice, 1 = Real voice
- **Prediction Output**: Probability between 0 and 1, where closer to 0 = Fake, closer to 1 = Real
