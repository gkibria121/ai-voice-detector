# AI Voice Detector — Quick Overview


This repository implements an end-to-end pipeline for detecting synthetic / deepfake audio and audio spoofing. It has been trimmed to support only the Fake-or-Real dataset (2-second clips). The project still provides model implementations, augmentation pipelines, and evaluation/visualization tools.

For full, detailed documentation see: [DOCUMENTATION.md](DOCUMENTATION.md)

## Highlights

- Models: LCNN, LCNN Large, RawNet3, EfficientNet-B2 (+ attention), SEResNet, SimpleCNN, FusionNet, and more under `models/`.
- Config-driven experiments: all hyperparameters and training options live in `config/`.
- Entry points: `main.py` (train/eval), `visualize_results.py`, `realtime.py` (inference demo), and `notebook.ipynb` for example experiments.
- Outputs: experiment artifacts (checkpoints, metrics, plots) are saved under `exp_result/` per-run.

## Quick Start

Install dependencies and create a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download and prepare a dataset (example):

```bash
# Example helper (adjust args for chosen dataset)
python download_dataset.py --dataset 1
```

Train a model with a provided config:

```bash
python main.py --config config/EfficientNetB2.conf --dataset 1 --feature_type 1 --epochs 20 --batch_size 32 --random_noise
```

Evaluate a saved model:

```bash
python main.py --eval --eval_model_weights exp_result/<run>/weights/best.pth --dataset 1 --feature_type 1
```

View and compare metrics across runs:

```bash
python visualize_results.py --path "exp_result/*/metrics" --compare --output ./comparison_plots
```


## Features & Augmentations (Fake-or-Real focused)

- Feature types: `0`=raw waveform, `1`=mel-spec (128), `2`=LFCC, `3`=MFCC, `4`=CQT.
- Augmentations (enable via `--random_noise`): RIR/reverb, MUSAN-style noise (babble/music/ambient), Gaussian noise (SNR 10–25 dB), pitch shift (±4 semitones), time stretch (0.85–1.15x), gain (±6 dB), low/high-pass filters, and SpecAugment.

The pipeline and documentation are focused on training and evaluating models on the Fake-or-Real dataset. Other dataset-specific evaluations (t-DCF / ASV tandem evaluation) have been removed.

## Workflow Summary

1. Prepare dataset and verify file layout.
2. Choose feature type and model config from `config/`.
3. Run a quick smoke test (`--data_subset`) to verify configuration.
4. Train with `--random_noise` for robustness and enable `--weight_avg` for SWA if desired.
5. Monitor training via saved `metrics.json` and visualize with `visualize_results.py`.
6. Evaluate final model(s) using `main.py --eval` or `evaluation.py` utilities for t-DCF/EER.

Full workflow and best practices are in [DOCUMENTATION.md](DOCUMENTATION.md).

## Outputs & Metrics

- Checkpoints: saved under experiment `weights/` (e.g., `best.pth`, `swa.pth`).
- Metrics: saved as JSON under the run's `metrics/` folder (epochs, dev/eval EER, t-DCF, accuracy).
- Primary metrics: EER (primary), t-DCF (tandem evaluation for ASVspoof), accuracy (simple binary datasets).

## Contributing

Add models under `models/`, configs under `config/`, and open PRs describing experiments. Include unit tests where possible and document config changes in the experiment folder.

## License

See the project's `LICENSE` file for terms.

---

If you'd like, I can replace `README.md` with the full `DOCUMENTATION.md` content, split docs into `docs/` pages, or add per-model recommended configs. Which would you prefer?
