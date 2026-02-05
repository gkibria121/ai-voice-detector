#!/usr/bin/env python3
"""
CLI tool for Explainable AI (XAI) analysis of the Voice Detector models.
Supports Grad-CAM, SHAP, TCAV, and frequency band analysis.
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

# Force reload explainability module to pick up changes
import importlib
import explainability
importlib.reload(explainability)

from explainability import (GradCAM, visualize_saliency, AudioSHAP, 
                            SpectrogramRegionAnalysis, visualize_shap_values,
                            visualize_band_importance, SHAP_AVAILABLE,
                            TCAV, visualize_tcav_results)

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
        # Use weights_only=False for compatibility with older saved models
        state = torch.load(weights_path, map_location=device, weights_only=False)
        
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

        model_name = model.__class__.__name__
        
        if hasattr(model, 'backbone'):
            backbone = model.backbone
            backbone_name = backbone.__class__.__name__
            
            # EfficientNet style
            if hasattr(backbone, 'features'):
                target_layer = backbone.features[-1]
                print(f"     Selected EfficientNet features[-1]")
            # RawNet3 style
            elif hasattr(backbone, 'encoder'):
                target_layer = backbone.encoder[-1]
                print(f"     Selected encoder[-1]")
            # SEResNet/ResNet style - use layer3 for better spatial resolution (layer4 is often 2x2 or 1x1)
            elif hasattr(backbone, 'layer3'):
                layer3 = backbone.layer3
                # Find the last Conv2d in layer3
                conv_layers = [m for m in layer3.modules() if isinstance(m, torch.nn.Conv2d)]
                if conv_layers:
                    target_layer = conv_layers[-1]
                    print(f"     Selected last Conv2d in layer3 (ResNet): {target_layer}")
                else:
                    target_layer = layer3
                    print(f"     Selected layer3 container (ResNet)")
            # LCNN style - use conv4 (the last conv block in backbone)
            elif hasattr(backbone, 'conv4'):
                # conv4 is an LCNNBlock, get the conv layer inside it
                if hasattr(backbone.conv4, 'conv'):
                    target_layer = backbone.conv4.conv
                    print(f"     Selected LCNN backbone.conv4.conv")
                else:
                    target_layer = backbone.conv4
                    print(f"     Selected LCNN backbone.conv4")
            # Try to find any conv4 attribute
            elif hasattr(backbone, 'conv3'):
                if hasattr(backbone.conv3, 'conv'):
                    target_layer = backbone.conv3.conv
                    print(f"     Selected backbone.conv3.conv")
        
        # If we still have a Sequential, dig deeper to find the last Conv layer
        if target_layer is not None and isinstance(target_layer, torch.nn.Sequential):
            conv_layers = [m for m in target_layer.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d))]
            if conv_layers:
                target_layer = conv_layers[-1]
                print(f"     Refined Sequential to last Conv layer: {target_layer}")
        
        # Fallback: Try to find the last Conv2d or Conv1d in the whole model
        if target_layer is None:
            layers = [m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d))]
            if layers:
                target_layer = layers[-1]
                print(f"     Fallback to last Conv in model: {target_layer}")
    except Exception as e:
        print(f"Error finding target layer: {e}")
        
    return target_layer


def run_shap(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, show_plots, save_plots):
    """Run SHAP analysis for model interpretation."""
    print("\n[SHAP] Running SHAP Analysis...")
    
    if not SHAP_AVAILABLE:
        print("  [!] SHAP not available. Install with: pip install shap")
        return
    
    # Check if this is an ensemble model
    is_ensemble = hasattr(model, 'is_ensemble') and model.is_ensemble
    
    if is_ensemble:
        # For ensemble models, run SHAP on each sub-model individually
        print("  Ensemble detected - running SHAP analysis on each sub-model...")
        
        # Get ensemble prediction first
        model.eval()
        with torch.no_grad():
            _, ensemble_output = model(input_tensor)
            ensemble_probs = torch.softmax(ensemble_output, dim=1)
            ensemble_pred = ensemble_output.argmax(dim=1).item()
            ensemble_conf = ensemble_probs[0, ensemble_pred].item()
        
        ensemble_label = "REAL" if ensemble_pred == 1 else "FAKE"
        print(f"\n  Ensemble Overall Prediction: {ensemble_label} (Class {ensemble_pred})")
        print(f"  Ensemble Confidence: {ensemble_conf*100:.2f}%")
        
        current_target = target_class if target_class is not None else ensemble_pred
        
        all_shap_values = []
        all_confidences = []
        
        for idx, submodel in enumerate(model.models):
            arch_name = getattr(submodel, 'config_architecture', submodel.__class__.__name__)
            variant = getattr(submodel, 'config_variant', None)
            name_suffix = f"model{idx}_{arch_name}"
            if variant:
                name_suffix += f"_{variant}"
            
            print(f"\n  -- Processing {name_suffix} --")
            
            try:
                # Get sub-model prediction
                submodel.eval()
                with torch.no_grad():
                    _, sub_output = submodel(input_tensor)
                    sub_probs = torch.softmax(sub_output, dim=1)
                    sub_pred = sub_output.argmax(dim=1).item()
                    sub_conf = sub_probs[0, sub_pred].item()
                
                sub_label = "REAL" if sub_pred == 1 else "FAKE"
                print(f"     Prediction: {sub_label} (confidence: {sub_conf:.1%})")
                
                all_confidences.append(sub_conf)
                
                # Create SHAP explainer for this sub-model
                shap_explainer = AudioSHAP(submodel, device=device)
                
                # Compute SHAP values
                print(f"     Computing SHAP values...")
                shap_values = shap_explainer.compute_shap_values(input_tensor, target_class=current_target)
                
                if shap_values is not None:
                    all_shap_values.append(shap_values)
                    vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
                    
                    if save_plots:
                        save_path = os.path.join(output_dir, f"{base_name}_{name_suffix}_shap.png")
                        visualize_shap_values(shap_values, input_spectrogram=vis_input, save_path=save_path,
                                             title=f"SHAP Values - {name_suffix} ({sub_label}, conf={sub_conf:.1%})")
                        print(f"     ✓ Saved: {save_path}")
                    
                    if show_plots:
                        print(f"     Displaying {name_suffix}...")
                        visualize_shap_values(shap_values, input_spectrogram=vis_input, save_path=None,
                                             title=f"SHAP Values - {name_suffix} ({sub_label}, conf={sub_conf:.1%})")
                else:
                    print(f"     [!] Could not compute SHAP values for {name_suffix}")
                    all_shap_values.append(None)
                    
            except Exception as e:
                print(f"     [X] SHAP failed for {name_suffix}: {e}")
                all_shap_values.append(None)
                all_confidences.append(0.0)
        
        # Generate composite SHAP visualizations
        valid_shap = [s for s in all_shap_values if s is not None]
        if valid_shap:
            print("\n  -- Generating Composite SHAP Visualizations --")
            vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
            
            # Average SHAP values
            avg_shap = torch.stack(valid_shap).mean(dim=0)
            
            if save_plots:
                save_path = os.path.join(output_dir, f"{base_name}_ensemble_average_shap.png")
                visualize_shap_values(avg_shap, input_spectrogram=vis_input, save_path=save_path,
                                     title=f"Average SHAP Values ({ensemble_label}, conf={ensemble_conf:.1%})")
                print(f"  ✓ Saved Average: {save_path}")
            
            if show_plots:
                visualize_shap_values(avg_shap, input_spectrogram=vis_input, save_path=None,
                                     title=f"Average SHAP Values ({ensemble_label}, conf={ensemble_conf:.1%})")
            
            # Weighted SHAP values by confidence
            valid_indices = [i for i, s in enumerate(all_shap_values) if s is not None]
            valid_confs = [all_confidences[i] for i in valid_indices]
            
            if sum(valid_confs) > 0:
                weights = torch.tensor(valid_confs, device=device)
                weights = weights / weights.sum()
                
                weighted_shap = torch.zeros_like(valid_shap[0])
                for i, s in enumerate(valid_shap):
                    weighted_shap += s.to(device) * weights[i]
                
                if save_plots:
                    save_path = os.path.join(output_dir, f"{base_name}_ensemble_weighted_shap.png")
                    visualize_shap_values(weighted_shap.cpu(), input_spectrogram=vis_input, save_path=save_path,
                                         title=f"Weighted SHAP Values ({ensemble_label}, conf={ensemble_conf:.1%})")
                    print(f"  ✓ Saved Weighted: {save_path}")
                
                if show_plots:
                    visualize_shap_values(weighted_shap.cpu(), input_spectrogram=vis_input, save_path=None,
                                         title=f"Weighted SHAP Values ({ensemble_label}, conf={ensemble_conf:.1%})")
    else:
        # Single model SHAP analysis (original logic)
        try:
            # Get model prediction
            model.eval()
            with torch.no_grad():
                _, output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred_class = output.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            
            current_target = target_class if target_class is not None else pred_class
            pred_label = "REAL" if pred_class == 1 else "FAKE"
            print(f"  Prediction: {pred_label} (confidence: {confidence:.1%})")
            print(f"  Explaining class: {current_target}")
            
            # Create SHAP explainer
            shap_explainer = AudioSHAP(model, device=device)
            
            # Compute SHAP values
            print("  Computing SHAP values (this may take a moment)...")
            shap_values = shap_explainer.compute_shap_values(input_tensor, target_class=current_target)
            
            if shap_values is not None:
                vis_input = spectrogram_vis if spectrogram_vis is not None else input_tensor.detach().cpu().numpy()
                
                if save_plots:
                    save_path = os.path.join(output_dir, f"{base_name}_shap_values.png")
                    visualize_shap_values(shap_values, input_spectrogram=vis_input, save_path=save_path,
                                         title=f"SHAP Values ({pred_label}, conf={confidence:.1%})")
                    print(f"  [OK] Saved: {save_path}")
                
                if show_plots:
                    visualize_shap_values(shap_values, input_spectrogram=vis_input, save_path=None,
                                         title=f"SHAP Values ({pred_label}, conf={confidence:.1%})")
            else:
                print("  [!] Could not compute SHAP values")
                
        except Exception as e:
            print(f"  [X] SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()


def run_analysis(model, input_tensor, spectrogram_vis, base_name, output_dir, device, show_plots, save_plots):
    """Run frequency band and temporal analysis."""
    print("\n[ANALYSIS] Running Audio Region Analysis...")
    
    try:
        # Get model prediction
        model.eval()
        with torch.no_grad():
            _, output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        pred_label = "REAL" if pred_class == 1 else "FAKE"
        
        # First, get an importance map using a simple gradient method
        print("  Computing importance map...")
        
        # Use gradient saliency as the importance source
        from explainability import GradientSaliency
        saliency = GradientSaliency(model, device=device)
        importance_map = saliency.compute_saliency(input_tensor, target_class=pred_class)
        
        if importance_map is not None:
            # Create analyzer
            analyzer = SpectrogramRegionAnalysis(model, device=device)
            
            # Analyze frequency bands
            print("  Analyzing frequency band importance...")
            band_scores = analyzer.analyze_importance_by_band(importance_map)
            
            # Analyze temporal patterns
            print("  Analyzing temporal patterns...")
            temporal = analyzer.analyze_temporal_pattern(importance_map)
            
            # Generate and print report
            report = analyzer.generate_report(importance_map, pred_class, confidence)
            print(report)
            
            # Save report
            if save_plots:
                report_path = os.path.join(output_dir, f"{base_name}_analysis_report.txt")
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"  [OK] Saved report: {report_path}")
                
                # Visualize band importance
                band_path = os.path.join(output_dir, f"{base_name}_band_importance.png")
                visualize_band_importance(band_scores, save_path=band_path, 
                                         title=f"Frequency Band Importance ({pred_label})")
                print(f"  [OK] Saved: {band_path}")
            
            if show_plots:
                visualize_band_importance(band_scores, save_path=None,
                                         title=f"Frequency Band Importance ({pred_label})")
        else:
            print("  [!] Could not compute importance map")
            
    except Exception as e:
        print(f"  [X] Analysis failed: {e}")
        import traceback
        traceback.print_exc()


def run_tcav(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, show_plots, save_plots):
    """Run TCAV (Testing with Concept Activation Vectors) analysis."""
    print("\n[TCAV] Running Concept-Based Analysis...")
    
    try:
        # Get model prediction
        model.eval()
        with torch.no_grad():
            _, output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        pred_label = "REAL" if pred_class == 1 else "FAKE"
        current_target = target_class if target_class is not None else pred_class
        
        print(f"  Prediction: {pred_label} (confidence: {confidence:.1%})")
        print(f"  Explaining class: {current_target} ({'REAL' if current_target == 1 else 'FAKE'})")
        
        # Find target layer for TCAV
        target_layer = find_gradcam_target_layer(model)
        
        if target_layer is None:
            print("  [!] Could not find suitable layer for TCAV")
            return
        
        print(f"  Using layer: {target_layer.__class__.__name__}")
        
        # Create TCAV explainer
        tcav = TCAV(model, target_layer, device=device)
        
        # Define concepts to test
        concepts = [
            'high_freq_artifacts',
            'low_freq_energy', 
            'temporal_discontinuity',
            'noise_floor',
            'harmonic_structure',
            'spectral_flatness'
        ]
        
        print(f"  Testing {len(concepts)} audio concepts...")
        
        # Run TCAV analysis
        results = tcav.explain_with_concepts(
            input_tensor, 
            concepts=concepts,
            n_samples=30,
            target_class=current_target
        )
        
        # Cleanup
        tcav.cleanup()
        
        if results:
            # Generate ranking
            ranking = tcav.generate_concept_importance_ranking(results)
            
            print("\n  --- Concept Importance Ranking ---")
            for concept, importance in ranking:
                print(f"    {concept.replace('_', ' ').title()}: {importance:.3f}")
            
            # Print interpretation
            print("\n  --- Interpretation ---")
            if ranking:
                top_concept = ranking[0][0]
                interpretations = {
                    'high_freq_artifacts': "Model focuses on high-frequency patterns, often associated with synthesis artifacts",
                    'low_freq_energy': "Model attends to low-frequency content (fundamental frequency, bass)",
                    'temporal_discontinuity': "Model is sensitive to temporal irregularities or glitches",
                    'noise_floor': "Model considers background noise characteristics",
                    'harmonic_structure': "Model analyzes harmonic patterns typical in natural speech",
                    'spectral_flatness': "Model distinguishes between tonal and noise-like segments"
                }
                print(f"    Top concept: {top_concept.replace('_', ' ').title()}")
                if top_concept in interpretations:
                    print(f"    → {interpretations[top_concept]}")
            
            # Save results
            if save_plots:
                # Save visualization
                viz_path = os.path.join(output_dir, f"{base_name}_tcav.png")
                visualize_tcav_results(results, pred_label, save_path=viz_path,
                                       title=f"TCAV Concept Analysis")
                print(f"\n  ✓ Saved: {viz_path}")
                
                # Save detailed report
                report_path = os.path.join(output_dir, f"{base_name}_tcav_report.txt")
                with open(report_path, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write("TCAV CONCEPT ANALYSIS REPORT\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Prediction: {pred_label} (Confidence: {confidence:.1%})\n")
                    f.write(f"Target class explained: {current_target}\n\n")
                    
                    f.write("--- Concept Scores ---\n")
                    for concept, data in results.items():
                        f.write(f"\n{concept.replace('_', ' ').title()}:\n")
                        f.write(f"  TCAV Score: {data['tcav_score']:.3f}\n")
                        f.write(f"  CAV Accuracy: {data['classifier_accuracy']:.1%}\n")
                        f.write(f"  Significant: {'Yes' if data['is_significant'] else 'No'}\n")
                    
                    f.write("\n--- Ranking ---\n")
                    for i, (concept, importance) in enumerate(ranking, 1):
                        f.write(f"{i}. {concept.replace('_', ' ').title()}: {importance:.3f}\n")
                
                print(f"  ✓ Saved: {report_path}")
            
            if show_plots:
                visualize_tcav_results(results, pred_label, save_path=None,
                                       title=f"TCAV Concept Analysis")
        else:
            print("  [!] No concept results generated")
            
    except Exception as e:
        print(f"  [X] TCAV analysis failed: {e}")
        import traceback
        traceback.print_exc()


def run_gradcam(model, input_tensor, spectrogram_vis, base_name, output_dir, device, target_class, show_plots, save_plots):
    """Run Grad-CAM. Handles both single models and ensembles."""
    print("\n[6/6] Running Grad-CAM...")
    
    # Check for ensemble
    if hasattr(model, 'is_ensemble') and model.is_ensemble:
        print("  Ensemble detected - generating visualizations for all components...")
        
        all_heatmaps = []
        all_confidences = []
        model_names = []
        
        # FIRST: Get the ensemble's overall prediction to use as target
        model.eval()
        with torch.no_grad():
            try:
                _, ensemble_logits = model(input_tensor)
                ensemble_probs = torch.nn.functional.softmax(ensemble_logits, dim=1)
                
                # Determine target class from ensemble if not specified
                ensemble_target = target_class
                if ensemble_target is None:
                    ensemble_target = ensemble_logits.argmax(dim=1).item()
                
                ensemble_label = "Real" if ensemble_target == 1 else "Fake"
                ensemble_confidence = ensemble_probs[0, ensemble_target].item()
                
                print(f"\n  Ensemble Overall Prediction: {ensemble_label} (Class {ensemble_target})")
                print(f"  Ensemble Confidence: {ensemble_confidence*100:.2f}%\n")
            except Exception as e:
                print(f"  Warning: Could not compute ensemble prediction: {e}")
                ensemble_target = target_class if target_class is not None else 0
        
        # NOW: Process each sub-model
        for i, submodel in enumerate(model.models):
            # Determine name
            arch = getattr(submodel, 'config_architecture', f"SubModel{i}")
            variant = getattr(submodel, 'config_variant', "")
            name_suffix = f"model{i}_{arch}"
            if variant:
                name_suffix += f"_{variant}"
            
            model_names.append(name_suffix)
            print(f"  -- Processing {name_suffix} --")
            
            # Get this sub-model's individual prediction
            submodel.eval()
            with torch.no_grad():
                try:
                    _, sub_logits = submodel(input_tensor)
                    sub_probs = torch.nn.functional.softmax(sub_logits, dim=1)
                    
                    # Get prediction for ENSEMBLE's target class
                    sub_pred_class = sub_logits.argmax(dim=1).item()
                    sub_pred_label = "Real" if sub_pred_class == 1 else "Fake"
                    sub_confidence = sub_probs[0, sub_pred_class].item()
                    
                    # Store confidence for weighted fusion (using ensemble target)
                    target_confidence = sub_probs[0, ensemble_target].item()
                    all_confidences.append(target_confidence)
                    
                    print(f"     Prediction: {sub_pred_label} (Class {sub_pred_class})")
                    print(f"     Confidence: {sub_confidence*100:.2f}%")
                    
                except Exception as e:
                    print(f"     Could not compute confidence: {e}")
                    all_confidences.append(1.0)  # Default weight
            
            # Find target layer for this sub-model
            target_layer = find_gradcam_target_layer(submodel)
            
            if target_layer is None:
                print(f"     ⚠ Could not find suitable layer for {name_suffix}")
                all_heatmaps.append(None)
                continue
            
            # Run Grad-CAM using ENSEMBLE's target class for consistency
            gradcam = GradCAM(submodel, target_layer, device)
            heatmap = gradcam.compute(input_tensor, target_class=ensemble_target)
            
            if heatmap is not None:
                # Resize to match input dimensions
                if input_tensor.ndim == 4:  # Spectrogram (B, C, H, W)
                    target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
                    
                    if heatmap.ndim == 1:
                        # 1D Heatmap on 2D Input
                        if heatmap.shape[0] == target_h:
                            # Frequency axis, expand Time
                            heatmap = heatmap.unsqueeze(1).expand(-1, target_w)
                        elif heatmap.shape[0] == target_w:
                            # Time axis, expand Frequency
                            heatmap = heatmap.unsqueeze(0).expand(target_h, -1)
                        else:
                            # Mismatch - interpolate
                            h = heatmap.view(heatmap.shape[0], 1).unsqueeze(0).unsqueeze(0)
                            h = torch.nn.functional.interpolate(h, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            heatmap = h.squeeze()
                            
                    elif heatmap.ndim == 2:
                        h = heatmap.unsqueeze(0).unsqueeze(0)
                        h = torch.nn.functional.interpolate(h, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        heatmap = h.squeeze()
                        
                elif input_tensor.ndim == 3:  # Raw Audio (B, C, L)
                    target_l = input_tensor.shape[2]
                    
                    if heatmap.ndim == 2:
                        # Flatten to 1D
                        heatmap = heatmap.flatten()
                        
                    if heatmap.ndim == 1:
                        h = heatmap.unsqueeze(0).unsqueeze(0)
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

        # B. Weighted Fusion (by confidence on ensemble target)
        valid_indices = [i for i, h in enumerate(all_heatmaps) if h is not None]
        valid_confidences = [all_confidences[i] for i in valid_indices]
        
        if sum(valid_confidences) > 0:
            weights = torch.tensor(valid_confidences, device=avg_heatmap.device)
            weights = weights / weights.sum()  # Normalize
            
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
        print("  Single model detected")
        
        # Compute prediction and confidence
        model.eval()
        with torch.no_grad():
            try:
                _, logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Determine target class if not set
                current_target = target_class
                if current_target is None:
                    current_target = logits.argmax(dim=1).item()
                
                # Map class to label
                class_label = "Real" if current_target == 1 else "Fake"
                confidence = probs[0, current_target].item()
                
                print(f"  Prediction: {class_label} (Class {current_target})")
                print(f"  Confidence: {confidence*100:.2f}%")
            except Exception as e:
                print(f"  Could not compute prediction: {e}")
                current_target = target_class if target_class is not None else 0
        
        target_layer = find_gradcam_target_layer(model)
        
        if target_layer is None:
            print("  ⚠ Could not find suitable convolutional layer - skipping")
            return
            
        print(f"  Using target layer: {target_layer}")
        
        gradcam = GradCAM(model, target_layer, device)
        heatmap = gradcam.compute(input_tensor, target_class=current_target)
        
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
  
  # Run TCAV concept-based analysis
  python explain.py --config model_config.json --audio_file sample.wav --model_path model.pth --method tcav
  
  # Display only (don't save files)
  python explain.py --config model_config.json --audio_file sample.wav --model_path model.pth --show --no-save
        """
    )
    
    parser.add_argument("--config", required=True, help="Path to model configuration file")
    parser.add_argument("--audio_file", required=True, help="Path to input audio file")
    parser.add_argument("--model_path", help="Path to model weights (.pth)")
    parser.add_argument("--method", nargs='+', 
                        choices=["gradcam", "shap", "tcav", "analysis", "all"], 
                        default=None,
                        help="XAI method(s) to use: gradcam, shap, tcav, analysis, or all")
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
        methods = ["gradcam", "shap", "tcav", "analysis"]
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
            elif method == "shap":
                run_shap(model, input_tensor, spectrogram_vis, base_name,
                        args.output_dir, device, args.target_class, args.show, save_plots)
            elif method == "tcav":
                run_tcav(model, input_tensor, spectrogram_vis, base_name,
                        args.output_dir, device, args.target_class, args.show, save_plots)
            elif method == "analysis":
                run_analysis(model, input_tensor, spectrogram_vis, base_name,
                           args.output_dir, device, args.show, save_plots)
        except Exception as e:
            print(f"  [X] Error running {method}: {e}")
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
 