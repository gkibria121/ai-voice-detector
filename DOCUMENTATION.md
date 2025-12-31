# AI Voice Detector — Comprehensive Documentation

## Project Overview

AI Voice Detector is a research and engineering codebase for detecting synthetic / deepfake audio. This trimmed version of the project supports only the Fake-or-Real dataset (2‑second clips), and provides model implementations, feature extraction, augmentation pipelines, training/evaluation scripts, and visualization utilities. The goal is an end-to-end pipeline to train and evaluate countermeasure models on the Fake-or-Real benchmark.

## Repository Layout

- `main.py` — Main training/evaluation entrypoint and experiment runner.
- `app.py`, `cli.py`, `realtime.py` — Entrypoints and helpers for different modes (some are utility wrappers).
- `data_utils.py` — Feature extraction, augmentation functions, padding helpers.
- `dataset_factory.py` — Dataset loader factory and dataset classes for supported datasets.
- `models/` — Model definitions (EfficientNetB2, SEResNet, LCNN, RawNet3, SimpleCNN, FusionNet, etc.).
- `config/` — Example configuration files for models and experiments.
- `metrics.py` — Plotting, metrics collection, and utilities for EER/t-DCF and other visualizations.
- `evaluation.py` — t-DCF/EER computation and helper functions for evaluation.
- `visualize_results.py` and `notebook.ipynb` — Scripts and notebook to visualize and compare experiment results.
- `requirements.txt` — Python package requirements.


## Key Concepts

- Dataset: Fake-or-Real (use `--dataset 1` to select; codebase limited to this dataset).
- Feature types: `0` = Raw waveform, `1` = Mel-spectrogram (128 mel bins), `2` = LFCC, `3` = MFCC, `4` = CQT (Constant-Q Transform).
- Metrics: EER (Equal Error Rate) and Accuracy (t-DCF and ASV tandem evaluation were removed in this trimmed repository).

## Installation

1. Create a Python environment (Python 3.8+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If using GPU, ensure appropriate `torch` build and CUDA drivers.

## Configuration and Running Experiments

Model and training hyperparameters are stored in `config/*.conf`. Example configs include:

- `config/LCNN.conf`
- `config/SEResNet.conf`
- `config/EfficientNetB2.conf`
- `config/EfficientNetB2_Attention.conf`
- `config/RawNet3.conf`
- `config/ensemble.conf`

Typical command-line usage (from notebook examples):

```bash
python main.py \
  --config config/EfficientNetB2_Attention.conf \
  --feature_type 1 \
  --dataset 1 \
  --epochs 20 \
  --batch_size 32 \
  --random_noise \
  --weight_avg \
  --eval_best \
  --dataset_version 3
```

For evaluation-only mode (load weights and evaluate):

```bash
python main.py --eval --eval_model_weights path/to/weights.pth --dataset 1 --feature_type 1
```

See `notebook.ipynb` for many runnable `%run main.py ...` examples.


## Dataset Handling

This repository is focused on the Fake-or-Real dataset. Use `dataset_factory.create_dataset_loaders` to produce PyTorch DataLoaders for Fake-or-Real. The expected local layout for Fake-or-Real is:

```
training/real/
training/fake/
validation/real/
validation/fake/
testing/real/
testing/fake/
```

`dataset_factory.py` provides dataset loader classes for Fake-or-Real, including `Dataset_FakeOrReal_train` and `Dataset_FakeOrReal_devNeval` that handle feature extraction and optional augmentations.

## Features

- Raw (`feature_type = 0`): Raw waveform input; typically used for RawNet3/AASIST.
- Mel-spectrogram (`feature_type = 1`): 128 mel bins recommended (time-frequency image). Widely used by CNN-based models.
- LFCC (`feature_type = 2`): Linear-frequency cepstral coefficients, usually 13 coefficients.
- MFCC (`feature_type = 3`): Mel-frequency cepstral coefficients, commonly 13 coefficients.
- CQT (`feature_type = 4`): Constant-Q Transform (84 bins), useful for harmonic/artifact detection.

Feature extraction is handled centrally by `data_utils.extract_feature` — this ensures consistent shapes, normalization, and pre-processing across datasets and models.


## Data Augmentation (`--random_noise`)

When `--random_noise` is enabled the training data pipeline applies a set of random augmentations to increase robustness for Fake-or-Real. These augmentations include:

- **RIR Simulation (Room Impulse Response)**: Convolves audio with simulated or recorded RIR to emulate reverberant rooms.
- **MUSAN-style Noise**: Adds noise categories from MUSAN-like collections — typically **babble**, **music**, and **ambient** noise.
- **Gaussian Noise**: Additive white Gaussian noise applied with random SNR values (typical range: 10–25 dB) to emulate low-level recording noise.
- **Reverberation**: Adds reverberation / echo effects.
- **Pitch Shift**: Random pitch shifts in the range of approximately ±4 semitones to emulate pitch alteration and synthesis artifacts.
- **Time Stretch**: Speed perturbation in a small range (e.g., 0.85× to 1.15×) to emulate speed variations.
- **Gain**: Random gain adjustments (±6 dB) to emulate recording level differences.
- **Filters**: Low-pass and high-pass filters applied randomly to simulate different devices or recording quality.
- **SpecAugment**: Frequency and time masking applied to spectrograms (when feature is spectrogram-based), used to regularize models.

Notes about augmentation behavior:

- Augmentations are applied stochastically during training; exact selection and ordering are defined in `data_utils.apply_augmentation` and `dataset_factory` wrappers.
- Augmentations are not applied to dev/eval splits by default.
- For reproducibility, the code supports seeding (`set_seed`) and `seed_worker` for DataLoader workers.

## Models

Model definitions are under the `models/` directory. Notable models included:

- `EfficientNetB2` and `EfficientNetB2_Attention` — spectrogram-based CNN backbones.
- `SEResNet` — residual architectures adapted for audio.
- `LCNN` and `LCNN_Large` — lightweight CNN designs commonly used for spoofing detection.
- `RawNet3` — raw waveform end-to-end model.
- `SimpleCNN` — compact spectrogram CNN for fast experiments.
- `FusionNet.py`, `AASIST` variants (if available) — ensemble and specialized architectures.

Use `main.get_model(model_config, device)` to instantiate the configured model with weights and device placement.

## Training and Optimization

Training is orchestrated in `main.py` with support for:

- Mixed precision training (AMP / torch.amp)
- PyTorch 2.x optimizations: `torch.compile`, TF32 handling, channels-last memory format where beneficial
- Optimizers: SGD, Adam (configured in `config/*.conf`). Use `utils.create_optimizer`.
- Learning rate schedulers: MultiStep, SGDR (restarts), cosine annealing, or custom Keras-style decay.
- Optional Stochastic Weight Averaging (`--weight_avg`).

Typical training loop features:

- per-epoch training and validation, metrics logged to JSON via `MetricsTracker`.
- checkpointing of best models by dev EER or dev t-DCF.


## Evaluation & Metrics

Evaluation and metric computation functions are implemented in `evaluation.py` and `metrics.py` with a focus on the Fake-or-Real benchmark.

- `compute_eer(target_scores, nontarget_scores)` computes EER and associated threshold.
- `calculate_simple_eer_accuracy(cm_scores_file, ...)` computes EER and accuracy for Fake-or-Real.
- `MetricsTracker` collects training loss, dev EER, and accuracy and saves artifacts under experiment folders.

Output/evaluation formats:

- Score files: text files with columns (filename, label, score) — labels like `real` / `fake` are expected for Fake-or-Real.

## Visualization

- `visualize_results.py` — load saved `metrics.json` files and create training/evaluation plots: loss curves, dev vs eval comparisons, accuracy plots, and model comparison charts.
- `notebook.ipynb` — an interactive notebook with many ready-to-run `%run main.py ...` examples and a comparison pipeline that aggregates multiple experiment metrics.
- `feature_analysis.py` — utilities to analyze and visualize waveform and feature representations (waveform, spectrogram, feature stats, histograms, temporal dynamics, textual descriptions).

Example usage from the notebook to generate comparison plots:

```bash
%run visualize_results.py --path "exp_result/*/metrics" --compare --show-summary --output ./comparison_plots
```

And to display the created plot in a notebook cell:

```python
from IPython.display import Image, display
display(Image(filename='./comparison_plots/model_comparison.png'))
```

## Reproducibility & Seeding

- Use `--seed <int>` or set seeds via `utils.set_seed(seed, config)` to ensure deterministic behavior where possible.
- `seed_worker` is used with DataLoader worker init to produce consistent augmentation across workers.

## Notes on Performance Optimizations

- The code contains optional optimizations for PyTorch 2.x, including `torch.compile()` and TF32 control where supported.
- BF16 support detection is implemented; native BF16 is enabled only when underlying hardware supports it.

## Developer Notes

- To add a new dataset, register a provider with `dataset_factory.register_dataset_provider` or extend `get_dataset_info` and add new Dataset classes.
- To add a new augmentation, implement it in `data_utils.py` and hook it into `apply_augmentation` / composed augmentation pipeline.
- To add a model, place the module in `models/` and reference its config in `config/`.

## Examples & Quick Commands

- Train EfficientNet-B2 on Fake-or-Real (mel-spec):

```bash
python main.py --config config/EfficientNetB2.conf --feature_type 1 --dataset 1 --epochs 20 --random_noise --weight_avg
```

- Evaluate a saved model on the eval split:

```bash
python main.py --eval --eval_model_weights exp_result/my_model/best.pth --dataset 1 --feature_type 1
```

- Run full experiment and compare multiple runs (notebook example):

```
%run notebook.ipynb
```

## Contributing

- Fork the repo and submit pull requests for bug fixes and features.
- Add unit tests where possible and follow the repository's coding style.

## License

This repository includes a `LICENSE` file in the root. Follow the stated license terms for reuse and distribution.

## Where to find things

- Configs: [config](config/)
- Models: [models](models/)
- Notebook examples: [notebook.ipynb](notebook.ipynb)
- Main runner: [main.py](main.py)
- Feature/augmentation code: [data_utils.py](data_utils.py)
- Datasets and DataLoaders: [dataset_factory.py](dataset_factory.py)
- Metrics/plots: [metrics.py](metrics.py)
- Evaluation functions (EER/t-DCF): [evaluation.py](evaluation.py)

## Workflow / Pipeline (Beginning → End)

This section describes a recommended end-to-end workflow — from repository clone to model deployment — with concrete commands and checkpoints.

1. Clone repository and set up environment

  - Clone the repo and create a virtual environment:

  ```bash
  git clone <repo_url>
  cd ai-voice-detector
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

  - (GPU) Install a compatible `torch` build and CUDA drivers following PyTorch instructions.

2. Prepare datasets

  - Download or place datasets in local directories matching expected layout. For Fake-or-Real, ensure:

    ```text
    training/real/   training/fake/
    validation/real/ validation/fake/
    testing/real/    testing/fake/
    ```

  - For ASVspoof2019 follow the organizer file layout or use dataset scripts if available.

3. Inspect and preprocess (optional)

  - Use `dataset_factory.load_fake_or_real_data` (or the equivalent helper) to print dataset statistics and check file counts.
  - Run `feature_analysis.analyze_and_visualize_features()` on a few examples to verify feature extraction and choose `feature_type`.

4. Configure experiment

  - Choose a config from `config/` (e.g., `config/EfficientNetB2.conf`) or create a new one describing:
    - `feature_type`, optimizer, scheduler, batch size, learning rate, epochs, augmentation toggles, misc options (seed, bf16).

5. Quick smoke test (small subset)

  - Run a quick training session on a small subset to validate pipeline:

  ```bash
  python main.py --config config/LCNN.conf --dataset 1 --feature_type 1 --epochs 2 --batch_size 8 --data_subset 0.01
  ```

6. Train full model

  - Full training with augmentations and optional SWA:

  ```bash
  python main.py \
    --config config/EfficientNetB2_Attention.conf \
    --dataset 1 \
    --feature_type 1 \
    --epochs 20 \
    --batch_size 32 \
    --random_noise \
    --weight_avg \
    --seed 42
  ```

  - The training run creates an experiment folder (e.g., `exp_result/<name>/`) with checkpoints, `metrics.json`, and plots.

7. Monitor training

  - Use saved `metrics.json` and `visualize_results.py` to generate plots and summaries:

  ```bash
  python visualize_results.py --path exp_result/*/metrics --compare --output ./comparison_plots
  ```

8. Select best checkpoint

  - The training script checkpoints the best model by dev EER or dev t-DCF. Identify `best.pth` (or the configured name) inside the run folder.

9. Evaluate

  - For Fake-or-Real (simple binary): generate CM score file and run `calculate_simple_eer_accuracy` or use `main.py --eval`:

  ```bash
  python main.py --eval --eval_model_weights exp_result/<run>/best.pth --dataset 1 --feature_type 1
  ```

  - For ASVspoof (tandem evaluation), ensure ASV score file is provided to compute `t-DCF` along with CM scores:

  ```bash
  python -c "from evaluation import calculate_tDCF_EER; calculate_tDCF_EER('cm_scores.txt','asv_scores.txt','output.txt')"
  ```

10. Visualize evaluation outputs

   - Generate ROC, confusion matrices, and summary metrics with `metrics.py` utilities or `visualize_results.py`.

11. Model ensemble / fusion (optional)

   - Use the ensemble config (`config/ensemble.conf`) or `FusionNet` to combine multiple feature types or model outputs. Train and evaluate using the same pipeline.

12. Deployment (inference)

   - For low-latency inference, export model weights and create a small wrapper that:
    - loads audio,
    - extracts features via `data_utils.extract_feature`,
    - runs model forward to obtain scores,
    - applies decision threshold (EER threshold or tuned operating point).

   - `realtime.py` may be used as a starting point for live inference or streaming scenarios.

13. Reproducibility and best practices

   - Always log experiment configuration, seed, and platform information in the run folder.
   - Use `--seed` and `utils.set_seed` to reduce run-to-run variance.
   - Keep dev set strictly held-out; do not tune on test/eval sets.

14. Troubleshooting tips

   - If training is unstable: reduce batch size, lower learning rate, or disable `torch.compile()`.
   - If GPU memory is insufficient: use `--batch_size` smaller values or enable gradient accumulation.
   - For noisy augmentations causing poor dev performance: lower augmentation intensity (reduce SNR range, disable extreme pitch shifts/time-stretch).

15. Example end-to-end quick commands

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Download dataset (example)
# %run download_dataset.py --dataset 1

# 3) Train
python main.py --config config/LCNN.conf --dataset 1 --feature_type 1 --epochs 15 --batch_size 32 --random_noise

# 4) Visualize
python visualize_results.py --path exp_result/*/metrics --output ./comparison_plots

# 5) Evaluate
python main.py --eval --eval_model_weights exp_result/<run>/best.pth --dataset 1 --feature_type 1
```

---

If you want, I can also:

- Convert this to an expanded repository `README.md` or to a `docs/` folder with separate pages.
- Add short examples per-model with recommended hyperparameters.
- Generate an HTML version of this document for sharing.

## Methodology

This section explains the methodological principles, design choices, and experimental protocols used in the project. It is written so other researchers and engineers can reproduce experiments, extend the pipeline, or adapt it to new data.

- Objective: Train robust countermeasure (CM) models to distinguish bona fide (real) audio from spoofed / synthetic audio, and evaluate them under single-system and tandem (ASV + CM) scenarios.

- Data preparation:
  - Use the provided dataset loaders (`dataset_factory`) which read audio files, build train/dev/eval splits, and produce file lists and labels.
  - Standardize sampling rate (default 16 kHz) and ensure audio length normalization via padding or cropping. For visualization, limit samples to 4s.
  - Keep dev set strictly for model selection and test/eval set for final reporting.

- Feature engineering:
  - Raw waveform (`feature_type=0`) is used for end-to-end models (RawNet3, AASIST) to let the network learn low-level representations.
  - Spectrogram-based features (Mel, LFCC, MFCC, CQT) are computed with consistent parameters (e.g., 128 mel bins for mel-spec, 13 cepstral coefficients for MFCC/LFCC) via `data_utils.extract_feature`.
  - Feature normalization (per-utterance or global mean-variance) is applied consistently to stabilize training.

- Augmentation rationale (when `--random_noise` enabled):
  - Use multiple stochastic augmentations to improve robustness to real-world recording conditions and synthetic artifact variability.
  - RIR & Reverberation: simulate room acoustics to improve generalization across environments.
  - MUSAN-style noise (babble/music/ambient): simulate background noises encountered in real recordings.
  - Gaussian noise (SNR 10–25 dB): simulate low-level sensor noise.
  - Pitch shift & Time-stretch: emulate common synthesis and recording distortions and help models avoid overfitting to exact pitch or tempo characteristics.
  - Gain and filters: emulate recording level and device frequency-response differences.
  - SpecAugment: frequency/time masking on spectrograms to regularize and force models to learn more robust patterns.

- Model selection and architecture decisions:
  - Start with lightweight models (LCNN, SimpleCNN) for quick experimentation and move to larger backbones (SEResNet, EfficientNetB2) for best performance on spectrograms.
  - Choose raw-waveform models (RawNet3, AASIST) when letting the network learn feature front-ends is desired or when spectrogram pre-processing is undesirable.
  - Consider attention variants (EfficientNetB2_Attention) to better focus on discriminative time-frequency regions.

- Training protocol:
  - Optimizers: SGD with momentum for stable convergence on large models; Adam for quick prototyping. Configured in `config/*.conf` and created via `utils.create_optimizer`.
  - Learning rate scheduling: MultiStepLR, SGDR (restarts), or cosine annealing to balance exploration and fine-tuning.
  - Mixed precision: use torch.amp for faster training and reduced GPU memory when available.
  - Checkpointing: save best models by dev EER or dev t-DCF; use `--weight_avg` (SWA) optionally to improve generalization.
  - Early stopping and model selection: monitor dev metrics and choose operating point (threshold) via EER or a tuned threshold for specific operating points.

- Hyperparameter search and ablation:
  - Perform small grid or random searches over learning rate, weight decay, batch size, and augmentation intensities.
  - Ablation studies: turn augmentation types on/off, compare feature types, compare raw vs spectrogram backends, and test attention/ensemble variants.

- Evaluation metrics and protocols:
  - EER: primary metric for many anti-spoofing benchmarks; report as percentage.
  - t-DCF: tandem detection cost function for ASVspoof — compute using both CM and ASV scores to judge impact on ASV system.
  - Accuracy: reported for simple binary benchmarks (Fake-or-Real) alongside EER.
  - Use `calculate_tDCF_EER` for tandem evaluation and `calculate_simple_eer_accuracy` for simple datasets.

- Reproducibility and logging:
  - Log full config, seed, environment, and dependencies inside each experiment folder; save `metrics.json` and model checkpoints.
  - Use `utils.set_seed` and `seed_worker` to control randomness across runs and worker processes.

- Ethical considerations:
  - When using or publishing models, clearly state dataset licenses and respect privacy and consent for audio data.
  - Avoid releasing models that could be easily misused for large-scale synthetic audio production without appropriate safeguards.

- Deployment considerations:
  - For inference, prefer small, optimized models or use quantization/pruning. Export a lightweight wrapper that extracts features via `data_utils.extract_feature` and applies a saved checkpoint.
  - For live or streaming usage, start from `realtime.py` and adapt buffering strategies and latency trade-offs.

---

If you want, I can also:

- Convert this to an expanded repository `README.md` or to a `docs/` folder with separate pages.
- Add short examples per-model with recommended hyperparameters.
- Generate an HTML version of this document for sharing.

Requested file created: DOCUMENTATION.md
