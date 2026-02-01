# Comparative Analysis with Related Work

This document provides a detailed comparison between the proposed models (EfficientNetB2 Attention, Ensemble) and state-of-the-art methods found in related literature for the Fake-or-Real (FoR) dataset.

## Comparison Table

| Study                   | Method / Model          | Features        | Accuracy (%) | Parameters (Approx.) | Notes                     |
| :---------------------- | :---------------------- | :-------------- | :----------- | :------------------- | :------------------------ |
| **Proposed (Best)**     | **EfficientNetB2 Attn** | **Mel-Spec**    | **97.70%**   | **~9.2M**            | **Eval Set (FoR V3)**     |
| **Proposed (Ensemble)** | **Ensemble**            | **Mel-Spec**    | **96.05%**   | **~22M**             | **EffNetAttn+LCNN+SERes** |
| Ref [10] (2021)         | VGG16                   | STFT            | 99.96%       | ~138M                | Clean Dataset             |
| Ref [1] (2022)          | SVM                     | MFCC            | 98.83%       | N/A                  | For-rerec Dataset         |
| Ref [12] (2023)         | Deep-Sonar              | -               | 98.10%       | N/A                  | -                         |
| Ref [2] (2024)          | VGG19                   | Mel-Spec        | 98.00%       | ~144M                | Audio Modality            |
| Ref [12] (2023)         | RES-EfficientCNN        | -               | 97.61%\*     | ~5M                  | \*F1-Score reported       |
| Ref [1] (2022)          | VGG-16                  | Mel-Spectrogram | 93.00%       | ~138M                | For-original              |
| Ref [12] (2023)         | CNN                     | -               | 88.00%       | < 1M                 | Basic CNN                 |

## Visualizations

### Accuracy Comparison

![Accuracy Comparison](images/comparison_accuracy_bar.png)

### Efficiency Analysis (Accuracy vs. Model Size)

![Efficiency Analysis](images/comparison_efficiency_scatter.png)

## Analysis & Discussion

### Competitive Performance

Our proposed **EfficientNetB2 with Attention (97.70%)** demonstrates performance comparable to the top-tier state-of-the-art models. While slightly lower than the 99.96% reported for VGG16 on a "Clean" dataset version (Ref [10]), our model operates on the V3 dataset and likely faces more challenging conditions.

### Efficiency vs. Accuracy

A key advantage of our approach is efficiency. **EfficientNetB2 (~9.2M parameters)** is significantly lighter than the heavy VGG models (~138M-144M parameters) used in previous top-performing studies (Ref [1], [2], [10]), while achieving very similar accuracy. This makes our solution more practical for real-world deployment.

### Dataset Considerations

It is crucial to note that our experiments were conducted on the **FakeOrReal V3** dataset. Discrepancies in performance (e.g., Ref [10] reporting 99.96%) may be attributed to:

- **Dataset Version**: Earlier papers likely utilized the original V1 dataset. V3 may introduce more challenging samples or stricter evaluation protocols.
- **Testing Conditions**: Extremely high accuracies (>99%) often correspond to specific subsets (e.g., "Clean"), whereas our results reflect performance on a rigorous evaluation set where we observed a generalization gap (Dev vs. Eval).

### Ensemble vs. Single Model

Our **Ensemble model (96.05%)**, comprising **EfficientNetB2 Attention, LCNN, and SEResNet**, achieves robust performance but slightly underperforms the best single model. In our case, the **EfficientNetB2 Attention** model individually captures the most discriminative features (Mel-Spectrograms), and ensembling with weaker models (like LCNN) may dilute the final prediction slightly.

## Conclusion

The **EfficientNetB2 Attention** model proposed in this thesis is a highly effective and efficient solution for deepfake audio detection. It matches the accuracy of much larger state-of-the-art models (like VGG19) while requiring a fraction of the computational resources (~9.2M vs ~144M parameters).
