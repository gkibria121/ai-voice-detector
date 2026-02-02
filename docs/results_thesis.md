# Chapter 5: Results and Discussion

## 5.1 Introduction

This chapter provides a comprehensive analysis of the experimental results obtained from evaluating several deep learning architectures for AI-generated voice detection. The primary objective was to identify a model that balances high detection accuracy with computational efficiency. All experiments were conducted using the **FakeOrReal (FoR) V3** dataset, and performance was measured across multiple acoustic feature representations.

## 5.2 Experimental Setup and Metrics

Models were trained for 20 epochs using Stochastic Weight Averaging (SWA) and random noise augmentation to improve generalization. The performance is evaluated based on three standard metrics:

- **Equal Error Rate (EER)**: The threshold where the probability of false acceptance and false rejection are equal (lower is preferred).
- **Receiver Operating Characteristic Area Under Curve (ROC AUC)**: Measures the model's ability to distinguish between classes (higher is preferred).
- **Accuracy**: The percentage of correctly classified real and fake samples.

## 5.3 Comparative Performance Analysis

### 5.3.1 Individual Model Performance

The following table summarizes the performance of various architectures across different feature types on the evaluation set.

| Model Architecture             | Feature Representation  | Eval Accuracy (%) | Eval EER (%) | ROC AUC   |
| :----------------------------- | :---------------------- | :---------------- | :----------- | :-------- |
| **EfficientNetB2 + Attention** | **Log Mel-Spectrogram** | **97.70%**        | **2.39%**    | **0.998** |
| EfficientNetB2                 | Log Mel-Spectrogram     | 96.97%            | 3.13%        | 0.994     |
| EfficientNetB2                 | CQT                     | 96.78%            | 3.31%        | 0.993     |
| LCNN Large                     | CQT                     | 95.50%            | 3.86%        | 0.991     |
| SEResNet                       | CQT                     | 96.05%            | 3.86%        | 0.991     |
| LCNN                           | CQT                     | 95.50%            | 4.78%        | 0.990     |
| **Ensemble (Top-3)**           | **Log Mel-Spectrogram** | **96.05%**        | **4.04%**    | **0.994** |

### 5.3.2 Identifying the Optimal Model

The **EfficientNetB2 with Attention mechanism** using **Log Mel-Spectrogram** features emerged as the superior model, achieving a state-of-the-art accuracy of **97.70%** and an EER of **2.39%**. This configuration demonstrates exceptional discrimination capability, correctly classifying 1,062 out of 1,088 evaluation samples.

## 5.4 Efficiency and Practicality

A core emphasis of this research was the development of an **efficient detector**. While some studies in the literature achieve slightly higher accuracies using massive architectures, our proposed solution offers a superior balance of performance and resource consumption.

| Model / Source               | Accuracy (%) | Parameters (Approx.) | Feature      |
| :--------------------------- | :----------- | :------------------- | :----------- |
| **Proposed (EffNetB2 Attn)** | **97.70%**   | **~9.2M**            | **Mel-Spec** |
| VGG16 (Ref [10])             | 99.96%       | ~138M                | STFT         |
| VGG19 (Ref [2])              | 98.00%       | ~144M                | Mel-Spec     |
| Deep-Sonar (Ref [12])        | 98.10%       | N/A                  | -            |

**Key Finding**: The **EfficientNetB2 Attention** model matches the performance of heavy-weight models like VGG19 while requiring significantly fewer parameters (~9.2M vs. ~144M). This ~15x reduction in model size without substantial loss in accuracy makes this approach highly suitable for real-world applications and edge deployment.

## 5.5 Analysis of Ensemble Performance

Contrary to initial expectations, the **Ensemble model (96.05%)** did not outperform the best individual model (EfficientNetB2 Attention, 97.70%).

### 5.5.1 The "Weak Link" Phenomenon

Explainable AI (XAI) analysis using Grad-CAM revealed that the ensemble’s sub-components—specifically **LCNN** and **SEResNet**—exhibited higher volatility. While EfficientNetB2 was consistently stable, LCNN occasionally generated **high-confidence errors** (e.g., confidently misclassifying real audio as fake due to over-indexing on mid-frequency patterns).

When these predictions were averaged, the "hallucinations" of the weaker models diluted the high-quality signals from the EfficientNetB2 component. The individual EfficientNetB2 Attention model is robust enough to capture the most discriminative artifacts independently, making the additional complexity of an ensemble unnecessary for this specific task.

## 5.6 Feature Representation Insights

- **Log Mel-Spectrogram**: Consistently the most effective feature, capturing the necessary spectral-temporal artifacts of voice synthesis systems.
- **Constant-Q Transform (CQT)**: A robust alternative, particularly effective for the LCNN architecture.
- **LFCC**: Showed significant overfitting (high training/dev performance but poor evaluation scores), suggesting it is less reliable for the FoR V3 dataset.
- **Raw Waveform**: Models like RawNet3 struggled to generalize, confirming that time-frequency representations are critical for deepfake detection.

## 5.7 Explainability (XAI) Findings

Grad-CAM visualizations confirmed that the EfficientNetB2 Attention model focuses on:

1. **Low-frequency regions (0-20 Hz)** for identifying synthetic fundamental frequency artifacts in fake audio.
2. **Broad frequency bands (0-40 Hz)** when analyzing real audio, indicating the detection of natural harmonic richness.

## 5.8 Conclusion

The experimental results demonstrate that **EfficientNetB2 with Attention** is the optimal model for AI voice detection in this study. Its ability to achieve **97.70% accuracy** with only **9.2M parameters** represents a significant contribution toward making deepfake detection both highly accurate and computationally accessible. Future work will explore if transitioning to STFT features can further bridge the gap to the highest reported literature benchmarks while maintaining this efficiency.
