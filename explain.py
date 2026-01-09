#!/usr/bin/env python3
"""
CLI tool for Explainable AI (XAI) analysis of the Voice Detector models.
Supports Attention Maps, Gradient Saliency, and Integrated Gradients.
"""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explainability import (
    AttentionExtractor, 
    GradientSaliency, 
    IntegratedGradients,
    visualize_attention,
    visualize_saliency
)

def load_model(config_path: str, weights_path: str, device: torch.device):
    """Load model from config and weights."""
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config = config.get("model_config", {})
    arch = model_config.get("architecture")
    if arch is None:
        raise RuntimeError("model_config.architecture missing in config file")

    module = import_module(f"models.{arch}")
    model_variant = model_config.get("model_variant", None)
    
    # Instantiate model
    if model_variant == "attention":
        if hasattr(module, "ModelWithAttention"):
            _model = getattr(module, "ModelWithAttention")
        else:
            print(f"Warning: ModelWithAttention not found in {arch}, using standard Model")
            _model = getattr(module, "Model")
    elif model_variant == "large":
        if hasattr(module, "ModelLarge"):
            _model = getattr(module, "ModelLarge")
        else:
             _model = getattr(module, "Model")
    else:
        _model = getattr(module, "Model")

    model = _model(model_config).to(device)

    # Load weights
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        state = torch.load(weights_path, map_location=device)
        
        # Handle different state dict formats
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
        
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Attempting strict=False...")
            model.load_state_dict(state, strict=False)
    else:
        print(f"Warning: Weights file not found or not provided: {weights_path}")
        print("Using random initialization (results will be meaningless for explanation)")

    model.eval()
    return model, config

def preprocess_audio(path: str, feature_type: int = 0, sample_rate: int = 16000):
    """Preprocess audio file to model input tensor."""
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    
    # Ensure at least 2 seconds (common for Fake-or-Real / ASVspoof clips)
    target_seconds = 2
    target_len = sample_rate * target_seconds
    
    if y.shape[0] < target_len:
        pad = target_len - y.shape[0]
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
    elif y.shape[0] > target_len:
        y = y[:target_len]
        
    # Generate features based on type
    # This logic mirrors app.py/data_utils.py. Ideally should import from data_utils but avoiding complex dependencies for this script.
    if feature_type == 0:
        # raw waveform -> (1,1,L)
        x = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        spectrogram = None # No spectrogram for raw audio
    elif feature_type == 1:
        # log-mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64) # Using app.py defaults
        logmel = librosa.power_to_db(mel, ref=np.max)
        x = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0)
        spectrogram = logmel
    else:
        # Fallback to raw/custom
        print(f"Warning: Feature type {feature_type} simple preprocessing. May need adjustment.")
        x = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        spectrogram = None

    return x, spectrogram

def main():
    parser = argparse.ArgumentParser(description="Explainability CLI for Voice Detector")
    parser.add_argument("--config", required=True, help="Path to model configuration file")
    parser.add_argument("--audio_file", required=True, help="Path to input audio file")
    parser.add_argument("--model_path", help="Path to model weights (.pth)")
    parser.add_argument("--method", choices=["attention", "saliency", "integrated_gradients", "occlusion", "gradcam"], 
                        default="saliency", help="Explainability method to use")
    parser.add_argument("--output_dir", default="explained_outputs", help="Directory to save visualizations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--target_class", type=int, default=None, help="Target class index (0=fake, 1=real). Default: predicted class")
    parser.add_argument("--feature_type", type=int, default=None, help="Input feature type: 0=Raw, 1=Spectrogram. Overrides config.")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.config, args.model_path, device)
    
    # Prepare input
    if args.feature_type is not None:
        feature_type = args.feature_type
    else:
        feature_type = config.get("feature_type", 0)
        
    input_tensor, spectrogram_vis = preprocess_audio(args.audio_file, feature_type=feature_type)
    input_tensor = input_tensor.to(device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(args.audio_file).stem
    
    print(f"Running {args.method} on {base_name}...")

    if args.method == "attention":
        extractor = AttentionExtractor(model, device)
        # Check if model has attention layers hooked
        if not extractor.hooks:
            print("Error: No attention layers found in model. Is this a transformer/attention-based model?")
            sys.exit(1)
            
        maps = extractor.get_attention_maps(input_tensor)
        print(f"Extracted {len(maps)} attention maps")
        
        for i, attn_map in enumerate(maps):
            save_path = os.path.join(args.output_dir, f"{base_name}_attn_layer{i}.png")
            visualize_attention(attn_map, save_path=save_path)
            print(f"Saved attention map to {save_path}")
            
        extractor.cleanup()

    elif args.method == "saliency":
        saliency = GradientSaliency(model, device)
        # Compute saliency
        saliency_map = saliency.compute_saliency(input_tensor, target_class=None if args.target_class is None else torch.tensor([args.target_class], device=device))
        
        save_path = os.path.join(args.output_dir, f"{base_name}_saliency.png")
        
        # If we have a spectrogram, visualize over it. If raw audio, just show the 1D saliency or dummy spectrogram
        vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
        
        visualize_saliency(saliency_map, input_spectrogram=vis_input, save_path=save_path)
        print(f"Saved saliency map to {save_path}")

    elif args.method == "integrated_gradients":
        ig = IntegratedGradients(model, device)
        attr = ig.compute(input_tensor, target_class=None if args.target_class is None else torch.tensor([args.target_class], device=device))
        
        save_path = os.path.join(args.output_dir, f"{base_name}_integrated_gradients.png")
        
        vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
        
        visualize_saliency(attr, input_spectrogram=vis_input, save_path=save_path)
        print(f"Saved Integrated Gradients attribution to {save_path}")

    elif args.method == "occlusion":
        from explainability import OcclusionSensitivity
        occlusion = OcclusionSensitivity(model, device)
        
        # Default window settings (heuristic)
        if feature_type == 1: # Spectrogram
            window_shape = (8, 8) 
            stride = 4
        else: # Raw
            window_shape = 2000 # ~125ms at 16k
            stride = 1000
            
        print(f"Computing Occlusion Sensitivity (Window: {window_shape}, Stride: {stride})...")
        print("This may take a while...")
        
        heatmap = occlusion.compute(input_tensor, target_class=args.target_class, 
                                   window_shape=window_shape, stride=stride)
        
        save_path = os.path.join(args.output_dir, f"{base_name}_occlusion.png")
        vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
        
        visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=save_path)
        print(f"Saved Occlusion Sensitivity map to {save_path}")

    elif args.method == "gradcam":
        from explainability import GradCAM
        
        # Determine target layer based on model architecture
        target_layer = None
        
        # Heuristics for finding the last convolutional layer
        try:
            if hasattr(model, 'backbone'):
                # EfficientNet style
                if hasattr(model.backbone, 'features'):
                    # The last block in features
                    target_layer = model.backbone.features[-1]
                # RawNet3 style
                elif hasattr(model.backbone, 'encoder'):
                     target_layer = model.backbone.encoder[-1]
                # Default generic
                elif hasattr(model.backbone, 'layer4'): # ResNet style
                    target_layer = model.backbone.layer4
            
            # SimpleCNN style or others
            if target_layer is None:
                # Try to find the last Conv2d or Conv1d in the whole model
                layers = [m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d))]
                if layers:
                    target_layer = layers[-1]
        except Exception as e:
            print(f"Error trying to simplify target layer: {e}")
            
        if target_layer is None:
            print("Error: Could not automatically identify a target CA layer for Grad-CAM.")
            print("Grad-CAM requires hooking into a specific convolutional layer.")
            sys.exit(1)
            
        print(f"Using target layer for Grad-CAM: {target_layer}")
        
        gradcam = GradCAM(model, target_layer, device)
        heatmap = gradcam.compute(input_tensor, target_class=args.target_class)
        
        # Resize heatmap to match input size for visualization
        if heatmap is not None:
            # Upsample
            if heatmap.ndim == 2: # 2D Spectrogram
                # Add dims for interpolate coordinates
                h = heatmap.unsqueeze(0).unsqueeze(0)
                # Target size
                target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
                h = torch.nn.functional.interpolate(h, size=(target_h, target_w), mode='bilinear', align_corners=False)
                heatmap = h.squeeze()
            elif heatmap.ndim == 1: # 1D Raw
                h = heatmap.unsqueeze(0).unsqueeze(0)
                target_l = input_tensor.shape[2]
                h = torch.nn.functional.interpolate(h, size=(target_l), mode='linear', align_corners=False)
                heatmap = h.squeeze()
        
        save_path = os.path.join(args.output_dir, f"{base_name}_gradcam.png")
        vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
        
        visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=save_path)
        print(f"Saved Grad-CAM heatmap to {save_path}")

if __name__ == "__main__":
    main()

