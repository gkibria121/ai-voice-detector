# XAI Analysis: EfficientNetB2_Attention vs Ensemble

A comparative analysis of explainability (Grad-CAM) outputs between the standalone **EfficientNetB2_Attention** model (97% accuracy) and the **Ensemble** model (96% accuracy).

---

## Test Samples

| Audio File   | Ground Truth |
| ------------ | ------------ |
| `file48.wav` | **FAKE**     |
| `file5.wav`  | **REAL**     |

---

## FAKE Audio Analysis (file48.wav)

### EfficientNetB2_Attention (Standalone)

![EfficientNetB2_Attention on FAKE audio](images/xai_notebook_0_1.png)

- **Prediction**: Fake ✅
- **Focus**: Strong attention on mid-frequencies (~20-40 range) with time-evolving patterns
- The attention mechanism concentrates on temporal transitions that indicate synthetic artifacts

---

### Ensemble Component Analysis

#### LCNN — Confidence: 77.34% → Fake ✅

![LCNN on FAKE audio](images/xai_notebook_1_1.png)

Horizontal band focus across all time steps, using frequency-based detection.

---

#### SEResNet — Confidence: 88.81% → Real ❌

![SEResNet on FAKE audio](images/xai_notebook_1_3.png)

**Key Issue**: SEResNet confidently misclassifies this fake audio as real, focusing on mid-high frequencies with time-concentrated attention.

---

#### EfficientNetB2_Attention — Confidence: 100% → Fake ✅

![EfficientNetB2_Attention in Ensemble on FAKE audio](images/xai_notebook_1_5.png)

Identical focus pattern to standalone — concentrated on temporal artifacts in mid frequencies.

---

### Ensemble Composites

| Average                                 | Weighted                                 |
| --------------------------------------- | ---------------------------------------- |
| ![Average](images/xai_notebook_1_7.png) | ![Weighted](images/xai_notebook_1_9.png) |

---

## REAL Audio Analysis (file5.wav)

### EfficientNetB2_Attention (Standalone)

![EfficientNetB2_Attention on REAL audio](images/xai_notebook_2_1.png)

- **Prediction**: Fake ❌
- **Issue**: Similar mid-frequency attention but misinterprets natural speech characteristics as fake artifacts

---

### Ensemble Component Analysis

#### LCNN — Confidence: 98.67% → Real ✅

![LCNN on REAL audio](images/xai_notebook_3_1.png)

---

#### SEResNet — Confidence: 99.70% → Real ✅

![SEResNet on REAL audio](images/xai_notebook_3_3.png)

---

#### EfficientNetB2_Attention — Confidence: 100% → Fake ❌

![EfficientNetB2_Attention in Ensemble on REAL audio](images/xai_notebook_3_5.png)

---

### Ensemble Composites

| Average                                 | Weighted                                 |
| --------------------------------------- | ---------------------------------------- |
| ![Average](images/xai_notebook_3_7.png) | ![Weighted](images/xai_notebook_3_9.png) |

---

## Confidence Summary

| Audio | Model                    | Prediction | Confidence | Correct? |
| ----- | ------------------------ | ---------- | ---------- | -------- |
| FAKE  | LCNN                     | Fake       | 77.34%     | ✅       |
| FAKE  | SEResNet                 | Real       | 88.81%     | ❌       |
| FAKE  | EfficientNetB2_Attention | Fake       | 100%       | ✅       |
| REAL  | LCNN                     | Real       | 98.67%     | ✅       |
| REAL  | SEResNet                 | Real       | 99.70%     | ✅       |
| REAL  | EfficientNetB2_Attention | Fake       | 100%       | ❌       |

---

## Why EfficientNetB2_Attention (97%) > Ensemble (96%)

### 1. SEResNet's Systematic Bias

SEResNet consistently misclassifies fake audio as real with high confidence (88.81%), diluting correct predictions in the ensemble.

### 2. Asymmetric Error Correction

- **Real audio**: LCNN + SEResNet correct EfficientNetB2's mistakes (2 vs 1)
- **Fake audio**: SEResNet's confident wrong predictions still pull the ensemble toward errors

### 3. Conflicting Saliency Patterns

| Model                    | Focus Pattern                                    |
| ------------------------ | ------------------------------------------------ |
| LCNN                     | Horizontal frequency bands (uniform across time) |
| SEResNet                 | Mid-high frequencies, time-concentrated          |
| EfficientNetB2_Attention | Time-evolving patterns in mid frequencies        |

Averaging these fundamentally different detection strategies doesn't improve accuracy.

---

## Conclusion

The 1% accuracy gap exists because **SEResNet's confident misclassifications on fake audio** harm the ensemble more than **EfficientNetB2's misclassifications on real audio**. The standalone model maintains its focused detection approach without dilution from conflicting ensemble members.
