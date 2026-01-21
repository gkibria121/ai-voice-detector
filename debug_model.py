#!/usr/bin/env python3
"""
Debug script to investigate model predictions
"""
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from data_utils import extract_feature, pad


def load_model(config_path: str, device: torch.device):
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config = config.get("model_config", {})
    arch = model_config.get("architecture")
    
    module = import_module(f"models.{arch}")
    model_variant = model_config.get("model_variant", None)
    if model_variant == "attention":
        _model = getattr(module, "ModelWithAttention")
    elif model_variant == "large":
        _model = getattr(module, "ModelLarge")
    else:
        _model = getattr(module, "Model")

    model = _model(model_config).to(device)
    
    # Load weights
    weights_path = config.get("model_path", None)
    if weights_path and Path(weights_path).exists():
        print(f"Loading weights from: {weights_path}")
        state = torch.load(str(weights_path), map_location=device)
        try:
            model.load_state_dict(state)
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
    else:
        print(f"WARNING: Weights not found at {weights_path}")
        
    model.eval()
    return model, config


def preprocess_audio(path: str, feature_type: int = 0, sample_rate: int = 16000):
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    
    X_feat = extract_feature(y, feature_type=feature_type, sr=sr)
    
    cut = 64600
    if feature_type == 0:
        X_pad = pad(X_feat, cut)
        x = torch.from_numpy(X_pad).float().unsqueeze(0).unsqueeze(0)
    else:
        hop_length = 160
        target_steps = int(cut / hop_length) + 1
        
        time_steps = X_feat.shape[1]
        if time_steps >= target_steps:
            X_pad = X_feat[:, :target_steps]
        else:
            num_repeats = int(target_steps / time_steps) + 1
            X_pad = np.tile(X_feat, (1, num_repeats))[:, :target_steps]
        
        x = torch.from_numpy(X_pad).float().unsqueeze(0).unsqueeze(0)
        
    return x


def test_audio(model, audio_path: str, label: str, feature_type: int, device: torch.device):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Path: {audio_path}")
    print(f"{'='*60}")
    
    x = preprocess_audio(audio_path, feature_type=feature_type)
    x = x.to(device)
    
    print(f"Input shape: {x.shape}")
    
    with torch.inference_mode():
        out = model(x)
        
        # Check if model returns tuple
        if isinstance(out, tuple) or isinstance(out, list):
            embeddings, logits = out
            print(f"Model returns tuple: (embeddings: {embeddings.shape}, logits: {logits.shape})")
        else:
            logits = out
            print(f"Model returns logits only: {logits.shape}")
        
        print(f"\nRaw logits: {logits.cpu().numpy()}")
        
        # Calculate probabilities
        if logits.dim() == 1 or (logits.dim() > 1 and logits.size(1) == 1):
            # Single output
            scores = logits.view(-1).cpu().numpy()
            probs_real = 1.0 / (1.0 + np.exp(-scores))
            print(f"Sigmoid probability (real): {probs_real}")
        else:
            # Two outputs (fake, real)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            print(f"Softmax probabilities: {probs}")
            print(f"  - P(fake) = {probs[0, 0]:.4f}")
            print(f"  - P(real) = {probs[0, 1]:.4f}")
            probs_real = probs[:, 1]
        
        prediction = "Real" if probs_real[0] >= 0.5 else "Fake"
        print(f"\nPrediction: {prediction} (score: {probs_real[0]:.4f})")
        print(f"Expected: {label}")
        print(f"Correct: {prediction == label}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config_path = "config/EfficientNetB2_Attention.conf"
    model, config = load_model(config_path, device)
    
    feature_type = config.get("feature_type", 1)  # Default to mel-spectrogram
    print(f"Feature type: {feature_type}")
    
    # Test files
    fake_audio = "./fake_or_real/for-2sec/for-2seconds/validation/fake/file48.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"
    real_audio = "./fake_or_real/for-2sec/for-2seconds/validation/real/file5.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav"
    
    test_audio(model, fake_audio, "Fake", feature_type, device)
    test_audio(model, real_audio, "Real", feature_type, device)
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
