# AI Voice Detection Experiment Results - Evaluation Set

This document presents the **Evaluation** set results for various deep learning models trained to distinguish between fake (AI-generated) and real human speech on the FakeOrReal V3 dataset.

---

## EfficientNet-B2 with Attention

### Feature Type 1 (Mel Spectrogram)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 97.70% |
| ROC AUC  | 0.9978 |
| EER      | 2.39%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9779    | 0.9761 | 0.9770   |
| Real/Bonafide | 0.9761    | 0.9779 | 0.9770   |

<p float="left" align="center">
  <img src="images/results_1_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_1_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_1_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 2 (MFCC)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 76.75% |
| ROC AUC  | 0.8357 |
| EER      | 23.35% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.7680    | 0.7665 | 0.7672   |
| Real/Bonafide | 0.7670    | 0.7684 | 0.7677   |

<p float="left" align="center">
  <img src="images/results_2_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_2_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_2_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 4 (CQT)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 96.05% |
| ROC AUC  | 0.9905 |
| EER      | 3.68%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9613    | 0.9596 | 0.9604   |
| Real/Bonafide | 0.9596    | 0.9614 | 0.9605   |

<p float="left" align="center">
  <img src="images/results_3_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_3_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_3_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## EfficientNet-B2 (Without Attention)

### Feature Type 1 (Mel Spectrogram)

#### Evaluation Set Metrics

| Metric   | Value    |
| -------- | -------- |
| Accuracy | 89.43%   |
| ROC AUC  | 0.9668   |
| EER      | 10.6618% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.8950    | 0.8934 | 0.8942   |
| Real/Bonafide | 0.8936    | 0.8952 | 0.8944   |

<p float="left" align="center">
  <img src="images/results_5_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_5_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_5_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 2 (MFCC)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 72.52% |
| ROC AUC  | 0.8100 |
| EER      | 27.57% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.7256    | 0.7243 | 0.7249   |
| Real/Bonafide | 0.7248    | 0.7261 | 0.7254   |

<p float="left" align="center">
  <img src="images/results_6_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_6_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_6_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 4 (CQT)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 92.00% |
| ROC AUC  | 0.9760 |
| EER      | 8.09%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9208    | 0.9191 | 0.9200   |
| Real/Bonafide | 0.9193    | 0.9210 | 0.9201   |

<p float="left" align="center">
  <img src="images/results_7_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_7_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_7_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## LCNN Large

### Feature Type 1 (Mel Spectrogram)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 91.82% |
| ROC AUC  | 0.9771 |
| EER      | 8.27%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9190    | 0.9173 | 0.9181   |
| Real/Bonafide | 0.9174    | 0.9191 | 0.9183   |

<p float="left" align="center">
  <img src="images/results_9_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_9_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_9_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 2 (MFCC)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 84.01% |
| ROC AUC  | 0.9244 |
| EER      | 15.99% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.8401    | 0.8382 | 0.8391   |
| Real/Bonafide | 0.8384    | 0.8401 | 0.8392   |

<p float="left" align="center">
  <img src="images/results_10_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_10_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_10_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 4 (CQT)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 94.12% |
| ROC AUC  | 0.9867 |
| EER      | 5.88%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9412    | 0.9393 | 0.9402   |
| Real/Bonafide | 0.9394    | 0.9412 | 0.9403   |

<p float="left" align="center">
  <img src="images/results_11_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_11_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_11_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## LCNN

### Feature Type 1 (Mel Spectrogram)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 92.56% |
| ROC AUC  | 0.9791 |
| EER      | 7.44%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9256    | 0.9239 | 0.9247   |
| Real/Bonafide | 0.9240    | 0.9257 | 0.9248   |

<p float="left" align="center">
  <img src="images/results_13_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_13_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_13_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 2 (MFCC)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 91.08% |
| ROC AUC  | 0.9653 |
| EER      | 8.82%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9116    | 0.9099 | 0.9108   |
| Real/Bonafide | 0.9101    | 0.9118 | 0.9109   |

<p float="left" align="center">
  <img src="images/results_14_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_14_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_14_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 4 (CQT)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 95.50% |
| ROC AUC  | 0.9903 |
| EER      | 4.78%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9558    | 0.9540 | 0.9549   |
| Real/Bonafide | 0.9541    | 0.9559 | 0.9550   |

<p float="left" align="center">
  <img src="images/results_15_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_15_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_15_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## SEResNet

### Feature Type 1 (Mel Spectrogram)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 89.43% |
| ROC AUC  | 0.9674 |
| EER      | 10.66% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.8950    | 0.8934 | 0.8942   |
| Real/Bonafide | 0.8936    | 0.8952 | 0.8944   |

<p float="left" align="center">
  <img src="images/results_17_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_17_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_17_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 2 (MFCC)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 92.00% |
| ROC AUC  | 0.9750 |
| EER      | 8.09%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9208    | 0.9191 | 0.9200   |
| Real/Bonafide | 0.9193    | 0.9210 | 0.9201   |

<p float="left" align="center">
  <img src="images/results_18_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_18_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_18_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 4 (CQT)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 93.20% |
| ROC AUC  | 0.9815 |
| EER      | 6.80%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9320    | 0.9301 | 0.9311   |
| Real/Bonafide | 0.9302    | 0.9320 | 0.9311   |

<p float="left" align="center">
  <img src="images/results_19_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_19_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_19_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## SENet

### Feature Type 1 (Mel Spectrogram)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 89.71% |
| ROC AUC  | 0.9655 |
| EER      | 10.29% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.8978    | 0.8960 | 0.8969   |
| Real/Bonafide | 0.8962    | 0.8978 | 0.8970   |

<p float="left" align="center">
  <img src="images/results_21_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_21_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_21_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 2 (MFCC)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 92.37% |
| ROC AUC  | 0.9750 |
| EER      | 7.72%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9244    | 0.9228 | 0.9236   |
| Real/Bonafide | 0.9229    | 0.9246 | 0.9237   |

<p float="left" align="center">
  <img src="images/results_22_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_22_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_22_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 4 (CQT)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 93.20% |
| ROC AUC  | 0.9802 |
| EER      | 6.80%  |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.9320    | 0.9301 | 0.9311   |
| Real/Bonafide | 0.9302    | 0.9320 | 0.9311   |

<p float="left" align="center">
  <img src="images/results_23_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_23_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_23_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## Additional Feature Experiments (EfficientNet-B2 with Attention)

### Feature Type 5

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 81.89% |
| ROC AUC  | 0.8956 |
| EER      | 18.20% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.8195    | 0.8180 | 0.8188   |
| Real/Bonafide | 0.8183    | 0.8199 | 0.8191   |

<p float="left" align="center">
  <img src="images/results_28_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_28_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_28_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

### Feature Type 6

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 62.96% |
| ROC AUC  | 0.6773 |
| EER      | 37.13% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.6298    | 0.6287 | 0.6293   |
| Real/Bonafide | 0.6294    | 0.6305 | 0.6299   |

<p float="left" align="center">
  <img src="images/results_29_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_29_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_29_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## RawNet3 (Raw Waveform)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 78.12% |
| ROC AUC  | 0.8729 |
| EER      | 21.88% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.7812    | 0.7812 | 0.7812   |
| Real/Bonafide | 0.7812    | 0.7812 | 0.7812   |

<p float="left" align="center">
  <img src="images/results_31_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_31_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_31_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## SimpleCNN (Raw Waveform)

#### Evaluation Set Metrics

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 64.52% |
| ROC AUC  | 0.6868 |
| EER      | 35.66% |

| Class         | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Fake/Spoof    | 0.6458    | 0.6434 | 0.6446   |
| Real/Bonafide | 0.6447    | 0.6471 | 0.6459   |

<p float="left" align="center">
  <img src="images/results_33_9.png" width="45%" title="Confusion Matrix (Eval)" />
  <img src="images/results_33_11.png" width="45%" title="ROC Curve (Eval)" /> 
</p>
<p align="center">
  <img src="images/results_33_13.png" width="95%" title="Accuracy Comparison" />
</p>

---

## Summary Table

# AI Voice Detection - Experiment Results

## Feature Type Legend

| Feature Type | Name                  | Description                            |
| ------------ | --------------------- | -------------------------------------- |
| 0            | Raw Audio             | Raw waveform input                     |
| 1            | Mel Spectrogram       | Log-mel spectrogram features           |
| 2            | LFCC                  | Linear Frequency Cepstral Coefficients |
| 4            | CQT                   | Constant-Q Transform                   |
| 5            | Chroma                | Chromagram features                    |
| 6            | Spectral Contrast     | Spectral contrast features             |
| 1,2,4        | Fusion (Mel+LFCC+CQT) | Multi-feature fusion                   |

---

## Single-Feature Results

| Model                    | Feature Type | Feature Name      | Epochs | Dev EER (%) | Dev Accuracy (%) | Eval EER (%) | Eval ROC AUC | Eval Accuracy (%) |
| ------------------------ | ------------ | ----------------- | ------ | ----------- | ---------------- | ------------ | ------------ | ----------------- |
| EfficientNetB2 Attention | 1            | Mel Spectrogram   | 20     | 0.0         | 99.96            | 2.3897       | 0.9978       | 97.70             |
| EfficientNetB2 Attention | 2            | LFCC              | 20     | 0.1415      | 99.82            | 23.3456      | 0.8357       | 76.75             |
| EfficientNetB2 Attention | 4            | CQT               | 20     | 0.2123      | 99.82            | 3.6765       | 0.9905       | 96.05             |
| EfficientNetB2 Attention | 5            | Chroma            | 20     | 4.3878      | 95.65            | 18.1985      | 0.8956       | 81.89             |
| EfficientNetB2 Attention | 6            | Spectral Contrast | 20     | 2.4062      | 97.63            | 37.1324      | 0.6773       | 62.96             |
| EfficientNetB2           | 1            | Mel Spectrogram   | 20     | 0.0         | 99.96            | 3.125        | 0.9944       | 96.97             |
| EfficientNetB2           | 2            | LFCC              | 20     | 0.4246      | 99.61            | 27.5735      | 0.8100       | 72.52             |
| EfficientNetB2           | 4            | CQT               | 20     | 0.2123      | 99.82            | 3.3088       | 0.9928       | 96.78             |
| LCNN Large               | 1            | Mel Spectrogram   | 20     | 0.0708      | 99.89            | 8.2721       | 0.9771       | 91.82             |
| LCNN Large               | 2            | LFCC              | 20     | 0.2831      | 99.75            | 5.1471       | 0.9890       | 94.94             |
| LCNN Large               | 4            | CQT               | 20     | 0.3539      | 99.61            | 3.8603       | 0.9907       | 95.50             |
| LCNN                     | 1            | Mel Spectrogram   | 20     | 0.0708      | 99.89            | 4.7794       | 0.9913       | 95.31             |
| LCNN                     | 2            | LFCC              | 20     | 0.2831      | 99.68            | 8.8235       | 0.9653       | 91.08             |
| LCNN                     | 4            | CQT               | 20     | 0.2831      | 99.75            | 4.7794       | 0.9903       | 95.50             |
| SEResNet                 | 1            | Mel Spectrogram   | 20     | 0.0708      | 99.89            | 10.6618      | 0.9674       | 89.43             |
| SEResNet                 | 2            | LFCC              | 20     | 0.1415      | 99.82            | 8.0882       | 0.9750       | 92.00             |
| SEResNet                 | 4            | CQT               | 20     | 0.2831      | 99.68            | 3.8603       | 0.9908       | 96.05             |
| RawNet3                  | 0            | Raw Audio         | 15     | 1.6277      | 98.41            | 21.8750      | 0.8729       | 78.12             |
| SimpleCNN                | 0            | Raw Audio         | 25     | 4.1047      | 95.93            | 35.6618      | 0.6868       | 64.52             |

---

## Feature Fusion Results

| Model                    | Feature Type | Feature Name          | Epochs | Dev EER (%) | Dev Accuracy (%) | Eval EER (%) | Eval ROC AUC | Eval Accuracy (%) |
| ------------------------ | ------------ | --------------------- | ------ | ----------- | ---------------- | ------------ | ------------ | ----------------- |
| EfficientNetB2 Attention | 1,2,4        | Fusion (Mel+LFCC+CQT) | 20     | 0.0708      | 99.89            | 5.3309       | 0.9911       | 94.58             |
| EfficientNetB2           | 1,2,4        | Fusion (Mel+LFCC+CQT) | 20     | 0.0708      | 99.96            | 5.8824       | 0.9864       | 94.03             |
| SEResNet                 | 1,2,4        | Fusion (Mel+LFCC+CQT) | 20     | 0.2831      | 99.68            | 6.2500       | 0.9821       | 92.74             |

---

## Ensemble Results

| Model                                   | Feature Type | Feature Name    | Epochs | Dev EER (%) | Dev Accuracy (%) | Eval EER (%) | Eval ROC AUC | Eval Accuracy (%) |
| --------------------------------------- | ------------ | --------------- | ------ | ----------- | ---------------- | ------------ | ------------ | ----------------- |
| Ensemble (EfficientNetB2+SEResNet+LCNN) | 1            | Mel Spectrogram | 20     | 0.0708      | 99.89            | 4.0441       | 0.9944       | 96.05             |

---

## Key Observations

### Best Performers (by Eval Accuracy):

1. **EfficientNetB2 Attention + Mel (feat 1)**: 97.70% accuracy, 2.39% EER
2. **EfficientNetB2 + Mel (feat 1)**: 96.97% accuracy, 3.12% EER
3. **EfficientNetB2 + CQT (feat 4)**: 96.78% accuracy, 3.31% EER
4. **EfficientNetB2 Attention + CQT (feat 4)**: 96.05% accuracy, 3.68% EER
5. **Ensemble + Mel (feat 1)**: 96.05% accuracy, 4.04% EER

### Feature Performance Summary:

- **Mel Spectrogram (1)**: Best overall performance across most models
- **CQT (4)**: Second best, consistent performance
- **LFCC (2)**: Poor generalization to eval set despite good dev performance
- **Chroma (5) & Spectral Contrast (6)**: Experimental features with limited success
- **Raw Audio (0)**: Significant overfitting issues

**Best performing model:** EfficientNet-B2 with Attention using Mel Spectrogram features achieved the highest evaluation accuracy (97.70%) and lowest EER (2.39%).
