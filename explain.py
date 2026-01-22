#!/usr/bin/env python3
"""
CLI tool for Explainable AI (XAI) analysis of the Voice Detector models.
Supports only Grad-CAM.
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

from explainability import GradCAM, visualize_saliency

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
    if feature_type == 0:
        # raw waveform -> (1,1,L)
        x = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        spectrogram = None
    elif feature_type == 1:
        # log-mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
        logmel = librosa.power_to_db(mel, ref=np.max)
        x = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0)
        spectrogram = logmel
    else:
        # Fallback to raw/custom
        print(f"Warning: Feature type {feature_type} simple preprocessing. May need adjustment.")
        x = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        spectrogram = None

    return x, spectrogram

def find_gradcam_target_layer(model):
    """Automatically find the best layer for Grad-CAM."""
    target_layer = None
    
    try:
        if hasattr(model, 'backbone'):
            # EfficientNet style
            if hasattr(model.backbone, 'features'):
                target_layer = model.backbone.features[-1]
            # RawNet3 style
            elif hasattr(model.backbone, 'encoder'):
                 target_layer = model.backbone.encoder[-1]
            # ResNet style
            elif hasattr(model.backbone, 'layer4'):
                target_layer = model.backbone.layer4
        
        # Try to find the last Conv2d or Conv1d in the whole model
        if target_layer is None:
            layers = [m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d))]
            if layers:
                target_layer = layers[-1]
    except Exception as e:
        print(f"Error finding target layer: {e}")
        
    return target_layer



def run_saliency(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, show_plots, save_plots):
    """Run gradient saliency."""
    print("\n[2/6] Running Gradient Saliency...")
    saliency = GradientSaliency(model, device)
    target = None if target_class is None else torch.tensor([target_class], device=device)
    
    saliency_map = saliency.compute_saliency(input_tensor, target_class=target)
    
    vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
    
    if save_plots:
        save_path = os.path.join(output_dir, f"{base_name}_saliency.png")
        visualize_saliency(saliency_map, input_spectrogram=vis_input, save_path=save_path)
        print(f"  ✓ Saved: {save_path}")
    
    if show_plots:
        visualize_saliency(saliency_map, input_spectrogram=vis_input, save_path=None)

def run_smoothgrad(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, show_plots, save_plots):
    """Run SmoothGrad."""
    print("\n[3/6] Running SmoothGrad (this may take a moment)...")
    smoothgrad = SmoothGrad(model, device, n_samples=50, noise_level=0.15)
    target = None if target_class is None else torch.tensor([target_class], device=device)
    
    smooth_map = smoothgrad.compute_saliency(input_tensor, target_class=target)
    
    vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
    
    if save_plots:
        save_path = os.path.join(output_dir, f"{base_name}_smoothgrad.png")
        visualize_saliency(smooth_map, input_spectrogram=vis_input, save_path=save_path)
        print(f"  ✓ Saved: {save_path}")
    
    if show_plots:
        visualize_saliency(smooth_map, input_spectrogram=vis_input, save_path=None)

def run_integrated_gradients(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, show_plots, save_plots):
    """Run Integrated Gradients."""
    print("\n[4/6] Running Integrated Gradients...")
    ig = IntegratedGradients(model, device)
    target = None if target_class is None else torch.tensor([target_class], device=device)
    
    attr = ig.compute(input_tensor, target_class=target)
    
    vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
    
    if save_plots:
        save_path = os.path.join(output_dir, f"{base_name}_integrated_gradients.png")
        visualize_saliency(attr, input_spectrogram=vis_input, save_path=save_path)
        print(f"  ✓ Saved: {save_path}")
    
    if show_plots:
        visualize_saliency(attr, input_spectrogram=vis_input, save_path=None)

def run_occlusion(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, feature_type, show_plots, save_plots):
    """Run Occlusion Sensitivity."""
    print("\n[5/6] Running Occlusion Sensitivity (this will take a while)...")
    occlusion = OcclusionSensitivity(model, device)
    
    # Default window settings
    if feature_type == 1:  # Spectrogram
        window_shape = (8, 8) 
        stride = 4
    else:  # Raw
        window_shape = 2000  # ~125ms at 16k
        stride = 1000
        
    print(f"  Window: {window_shape}, Stride: {stride}")
    
    heatmap = occlusion.compute(input_tensor, target_class=target_class, 
                               window_shape=window_shape, stride=stride)
    
    vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
    
    if save_plots:
        save_path = os.path.join(output_dir, f"{base_name}_occlusion.png")
        visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=save_path)
        print(f"  ✓ Saved: {save_path}")
    
    if show_plots:
        visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=None)

def run_gradcam(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, show_plots, save_plots):
    """Run Grad-CAM."""
    print("\n[6/6] Running Grad-CAM...")
    
    target_layer = find_gradcam_target_layer(model)
    
    if target_layer is None:
        print("  ⚠ Could not find suitable convolutional layer - skipping")
        return
        
    print(f"  Using target layer: {target_layer}")
    
    gradcam = GradCAM(model, target_layer, device)
    heatmap = gradcam.compute(input_tensor, target_class=target_class)
    
    if heatmap is not None:
        # Resize heatmap to match input size
        if heatmap.ndim == 2:  # 2D Spectrogram
            h = heatmap.unsqueeze(0).unsqueeze(0)
            target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
            h = torch.nn.functional.interpolate(h, size=(target_h, target_w), mode='bilinear', align_corners=False)
            heatmap = h.squeeze()
        elif heatmap.ndim == 1:  # 1D Raw
            h = heatmap.unsqueeze(0).unsqueeze(0)
            target_l = input_tensor.shape[2]
            h = torch.nn.functional.interpolate(h, size=(target_l,), mode='linear', align_corners=False)
            heatmap = h.squeeze()
    
        vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
        
        if save_plots:
            save_path = os.path.join(output_dir, f"{base_name}_gradcam.png")
            visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=save_path)
            print(f"  ✓ Saved: {save_path}")
        
        if show_plots:
            visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=None)

def main():
    parser = argparse.ArgumentParser(
        description="Explainability CLI for Voice Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ALL XAI methods and save to files (default)
  python explain.py --config model_config.json --audio_file sample.wav --model_path model.pth
  
  # Run ALL methods and display in matplotlib windows
  python explain.py --config model_config.json --audio_file sample.wav --model_path model.pth --show
  
  # Run specific method only
  python explain.py --config model_config.json --audio_file sample.wav --model_path model.pth --method saliency
  
  # Run multiple specific methods with display
  python explain.py --config model_config.json --audio_file sample.wav --model_path model.pth --method saliency smoothgrad --show
  
  # Display only (don't save files)
  python explain.py --config model_config.json --audio_file sample.wav --model_path model.pth --show --no-save
        """
    )
    
    parser.add_argument("--config", required=True, help="Path to model configuration file")
    parser.add_argument("--audio_file", required=True, help="Path to input audio file")
    parser.add_argument("--model_path", help="Path to model weights (.pth)")
    parser.add_argument("--method", nargs='+', 
                        choices=["gradcam", "all"], 
                        default=None,
                        help="XAI method(s) to use. Only Grad-CAM is supported.")
    parser.add_argument("--output_dir", default="explained_outputs", help="Directory to save visualizations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--target_class", type=int, default=None, 
                        help="Target class index (0=fake, 1=real). Default: predicted class")
    parser.add_argument("--feature_type", type=int, default=None, 
                        help="Input feature type: 0=Raw, 1=Spectrogram. Overrides config.")
    parser.add_argument("--show", action="store_true", 
                        help="Display plots interactively in matplotlib windows (in addition to saving)")
    parser.add_argument("--no-save", action="store_true", dest="no_save",
                        help="Don't save plots to files (only display if --show is used)")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load config to get model_path if not provided
    with open(args.config, "r") as f:
        config_json = json.load(f)
    default_model_path = config_json.get("model_path")
    model_path = args.model_path if args.model_path is not None else default_model_path
    model, config = load_model(args.config, model_path, device)
    
    # Prepare input
    if args.feature_type is not None:
        feature_type = args.feature_type
    else:
        feature_type = config.get("feature_type", 0)
        
    input_tensor, spectrogram_vis = preprocess_audio(args.audio_file, feature_type=feature_type)
    input_tensor = input_tensor.to(device)

    # Create output directory
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(args.audio_file).stem
    
    # Determine which methods to run
    if args.method is None or "all" in args.method:
        methods = ["gradcam"]
    else:
        methods = args.method 
    
    print(f"\n{'='*60}")
    print(f"Running XAI Analysis on: {base_name}")
    print(f"Methods: {', '.join(methods)}")
    if not args.no_save:
        print(f"Output directory: {args.output_dir}")
    if args.show:
        print(f"Display mode: Interactive matplotlib windows")
    print(f"{'='*60}")
    
    # Run each method
    save_plots = not args.no_save
    
    for method in methods:
        try:
            if method == "gradcam":
                run_gradcam(model, input_tensor, spectrogram_vis, base_name, 
                          args.output_dir, device, args.target_class, args.show, save_plots)
        except Exception as e:
            print(f"  ✗ Error running {method}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✓ XAI Analysis Complete!")
    if not args.no_save:
        print(f"Results saved to: {args.output_dir}")
    if args.show:
        print(f"Close matplotlib windows to exit.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()