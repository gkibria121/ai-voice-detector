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
    
    # Check if this is an ensemble configuration (list of model configs)
    if isinstance(model_config, (list, tuple)):
        # Build ensemble model with multiple sub-models
        import torch.nn as nn
        
        models = []
        for idx, mconf in enumerate(model_config):
            arch = mconf.get("architecture")
            if arch is None:
                raise RuntimeError(f"model_config[{idx}].architecture missing in config file")
            
            module = import_module(f"models.{arch}")
            model_variant = mconf.get("model_variant", None)
            
            # Instantiate sub-model
            if model_variant == "attention":
                if hasattr(module, "ModelWithAttention"):
                    _model_cls = getattr(module, "ModelWithAttention")
                else:
                    print(f"Warning: ModelWithAttention not found in {arch}, using standard Model")
                    _model_cls = getattr(module, "Model")
            elif model_variant == "large":
                if hasattr(module, "ModelLarge"):
                    _model_cls = getattr(module, "ModelLarge")
                else:
                    _model_cls = getattr(module, "Model")
            else:
                _model_cls = getattr(module, "Model")
            
            submodel = _model_cls(mconf).to(device)
            submodel.config_architecture = arch
            submodel.config_variant = model_variant
            print(f"Loaded sub-model[{idx}]: {arch} (variant: {model_variant})")
            models.append(submodel)
        
        # Create ensemble wrapper (same as in main.py)
        class EnsembleModel(nn.Module):
            """Ensemble wrapper that averages logits and projects embeddings."""
            def __init__(self, model_list):
                super().__init__()
                self.models = nn.ModuleList(model_list)
                self.is_ensemble = True

                # Infer embedding sizes for each submodel
                emb_dims = []
                for m in self.models:
                    dim = None
                    # Prefer BatchNorm1d feature size if present
                    if hasattr(m, 'embedding'):
                        bn_found = None
                        linear_found = None
                        has_mfm = False
                        for mod in m.embedding.modules():
                            if isinstance(mod, nn.BatchNorm1d):
                                bn_found = mod.num_features
                            if isinstance(mod, nn.Linear) and linear_found is None:
                                linear_found = getattr(mod, 'out_features', None)
                            if mod.__class__.__name__.startswith('MaxFeatureMap'):
                                has_mfm = True
                        if bn_found is not None:
                            dim = bn_found
                        elif linear_found is not None:
                            if has_mfm and linear_found % 2 == 0:
                                dim = linear_found // 2
                            else:
                                dim = linear_found
                    # Fallback: infer from classifier first linear input
                    if dim is None and hasattr(m, 'classifier'):
                        for mod in m.classifier.modules():
                            if isinstance(mod, nn.Linear):
                                dim = getattr(mod, 'in_features', None)
                                break
                    emb_dims.append(dim)

                # Choose a target embedding dimension (max of detected dims)
                valid_dims = [d for d in emb_dims if d is not None]
                self.target_emb_dim = max(valid_dims) if valid_dims else 128

                # Create projection layers to map each embedding to target dim
                projs = []
                for d in emb_dims:
                    if d is None or d == self.target_emb_dim:
                        projs.append(nn.Identity())
                    else:
                        lin = nn.Linear(d, self.target_emb_dim)
                        nn.init.xavier_uniform_(lin.weight)
                        if lin.bias is not None:
                            nn.init.constant_(lin.bias, 0.0)
                        projs.append(lin)
                self.projections = nn.ModuleList(projs)

            def forward(self, x, Freq_aug=False):
                outs = []
                embs = []
                for m in self.models:
                    emb, out = m(x, Freq_aug=Freq_aug)
                    embs.append(emb)
                    outs.append(out)

                # Average logits
                stacked_outs = torch.stack(outs, dim=0)  # (M, B, C)
                avg_out = torch.mean(stacked_outs, dim=0)

                # Project embeddings to common size then average
                proj_embs = []
                for i, emb in enumerate(embs):
                    proj = self.projections[i]
                    if isinstance(proj, nn.Identity):
                        proj_embs.append(emb)
                    else:
                        proj_embs.append(proj(emb))

                stacked_embs = torch.stack(proj_embs, dim=0)
                avg_emb = torch.mean(stacked_embs, dim=0)

                return avg_emb, avg_out
        
        model = EnsembleModel(models).to(device)
        print(f"Created ensemble with {len(models)} models")
    else:
        # Single model configuration
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
        
        # For ensemble models, filter out projection weights if sizes mismatch
        # (projections are architecture-dependent and can be reinitialized)
        if isinstance(model_config, (list, tuple)) and hasattr(model, 'is_ensemble'):
            # Remove projection weights from state dict to avoid size mismatches
            state_filtered = {k: v for k, v in state.items() if not k.startswith('projections.')}
            missing_keys, unexpected_keys = model.load_state_dict(state_filtered, strict=False)
            if missing_keys:
                print(f"Note: {len([k for k in missing_keys if k.startswith('projections.')])} projection parameters initialized randomly (architecture-dependent)")
        else:
            # Single model: use standard loading with fallback to strict=False
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
        # Special handling for EnsembleModel
        if hasattr(model, 'is_ensemble') and model.is_ensemble:
            print("  Ensemble detected, searching for best sub-model target...")
            # Prefer EfficientNet sub-model if available
            effnet_model = None
            for m in model.models:
                if "EfficientNet" in m.__class__.__name__ or (hasattr(m, 'backbone') and "EfficientNet" in m.backbone.__class__.__name__):
                    effnet_model = m
                    break
            
            # If no EfficientNet found, use the last model
            search_model = effnet_model if effnet_model is not None else model.models[-1]
            print(f"  Using sub-model: {search_model.__class__.__name__}")
            # Recursively find target in the chosen sub-model
            return find_gradcam_target_layer(search_model)

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
    """Run Grad-CAM. Handles both single models and ensembles."""
    print("\n[6/6] Running Grad-CAM...")
    
    # Check for ensemble
    if hasattr(model, 'is_ensemble') and model.is_ensemble:
        print("  Ensemble detected - generating visualizations for all components...")
        
        all_heatmaps = []
        all_confidences = []
        model_names = []
        
        # 1. Process each sub-model
        for i, submodel in enumerate(model.models):
            # Determine name
            arch = getattr(submodel, 'config_architecture', f"SubModel{i}")
            variant = getattr(submodel, 'config_variant', "")
            name_suffix = f"model{i}_{arch}"
            if variant:
                name_suffix += f"_{variant}"
            
            model_names.append(name_suffix)
            print(f"\n  -- Processing {name_suffix} --")
            
            # Compute confidence (prob of target class)
            # We need to run forward pass on this sub-model
            submodel.eval()
            with torch.no_grad():
                # Submodels return (emb, logits) usually, or just logits depending on arch
                # The EnsembleModel wrapper handles the call signature: emb, out = m(x)
                try:
                    _, logits = submodel(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    
                    # Determine target class if not set
                    current_target = target_class
                    if current_target is None:
                        current_target = logits.argmax(dim=1).item()
                        
                    confidence = probs[0, current_target].item()
                    all_confidences.append(confidence)
                    print(f"     Confidence (Class {current_target}): {confidence:.4f}")
                except Exception as e:
                    print(f"     Could not compute confidence: {e}")
                    all_confidences.append(1.0) # Default weight
            
            # Find target layer for this sub-model
            target_layer = find_gradcam_target_layer(submodel)
            
            if target_layer is None:
                print(f"     ⚠ Could not find suitable layer for {name_suffix}")
                all_heatmaps.append(None)
                continue
                
            # Run Grad-CAM
            gradcam = GradCAM(submodel, target_layer, device)
            heatmap = gradcam.compute(input_tensor, target_class=target_class)
            
            if heatmap is not None:
                # Resize to match input dimensions
                if input_tensor.ndim == 4: # Spectrogram (B, C, H, W)
                    target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
                    
                    if heatmap.ndim == 1:
                        # 1D Heatmap on 2D Input (e.g. LCNN might return Frequency-only map [H])
                        if heatmap.shape[0] == target_h:
                            # Assume Frequency axis, expand Time
                            heatmap = heatmap.unsqueeze(1).expand(-1, target_w)
                        elif heatmap.shape[0] == target_w:
                            # Assume Time axis, expand Frequency
                            heatmap = heatmap.unsqueeze(0).expand(target_h, -1)
                        else:
                            # Mismatch - treat as [L, 1] and stretch
                            h = heatmap.view(heatmap.shape[0], 1).unsqueeze(0).unsqueeze(0)
                            h = torch.nn.functional.interpolate(h, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            heatmap = h.squeeze()
                            
                    elif heatmap.ndim == 2:
                        h = heatmap.unsqueeze(0).unsqueeze(0)
                        h = torch.nn.functional.interpolate(h, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        heatmap = h.squeeze()
                        
                elif input_tensor.ndim == 3: # Raw Audio (B, C, L)
                    target_l = input_tensor.shape[2]
                    
                    h = heatmap
                    if h.ndim == 2:
                        h = h.view(1, 1, -1) # Flatten? Or handle 2D->1D? 
                        # If 2D on 1D input, likely F,T -> L. But messy. 
                        # Assuming 1D-ish.
                        pass 
                        
                    if h.ndim == 1:
                         h = h.unsqueeze(0).unsqueeze(0)
                    
                    # Interpolate
                    h = torch.nn.functional.interpolate(h, size=(target_l,), mode='linear', align_corners=False)
                    heatmap = h.squeeze()

                all_heatmaps.append(heatmap)
                
                # Save individual plot
                vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
                
                if save_plots:
                    save_path = os.path.join(output_dir, f"{base_name}_{name_suffix}_gradcam.png")
                    visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=save_path)
                    print(f"     ✓ Saved: {save_path}")
                
                if show_plots:
                    print(f"     Displaying {name_suffix}...")
                    visualize_saliency(heatmap, input_spectrogram=vis_input, save_path=None)
            else:
                all_heatmaps.append(None)

        # 2. Composite Visualizations
        print("\n  -- Generating Composite Visualizations --")
        
        valid_heatmaps = [h for h in all_heatmaps if h is not None]
        if not valid_heatmaps:
            print("  No valid heatmaps generated.")
            return

        vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()

        # A. Simple Average
        avg_heatmap = torch.stack(valid_heatmaps).mean(dim=0)
        
        if save_plots:
            save_path = os.path.join(output_dir, f"{base_name}_ensemble_average_gradcam.png")
            visualize_saliency(avg_heatmap, input_spectrogram=vis_input, save_path=save_path)
            print(f"  ✓ Saved Average: {save_path}")
            
        if show_plots:
            visualize_saliency(avg_heatmap, input_spectrogram=vis_input, save_path=None)

        # B. Weighted Fusion (by confidence)
        # Filter confidences to match valid heatmaps
        valid_indices = [i for i, h in enumerate(all_heatmaps) if h is not None]
        valid_confidences = [all_confidences[i] for i in valid_indices]
        
        if sum(valid_confidences) > 0:
            weights = torch.tensor(valid_confidences, device=avg_heatmap.device)
            weights = weights / weights.sum() # Normalize
            
            weighted_heatmap = torch.zeros_like(avg_heatmap)
            for i, h in enumerate(valid_heatmaps):
                weighted_heatmap += h * weights[i]
                
            if save_plots:
                save_path = os.path.join(output_dir, f"{base_name}_ensemble_weighted_gradcam.png")
                visualize_saliency(weighted_heatmap, input_spectrogram=vis_input, save_path=save_path)
                print(f"  ✓ Saved Weighted: {save_path}")
                
            if show_plots:
                visualize_saliency(weighted_heatmap, input_spectrogram=vis_input, save_path=None)
        else:
            print("  Skipping weighted fusion (sum of confidences is 0)")

    else:
        # Single Model (Original Logic)
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
    
    # Set seed for reproducibility (crucial for random projections in ensemble)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    
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
    
    # Extract config name and include it in output filename
    config_name = Path(args.config).stem  # e.g., "ensemble" from "config/ensemble.conf"
    audio_name = Path(args.audio_file).stem
    base_name = f"{config_name}_{audio_name}"
    
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