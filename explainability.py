"""
Explainability Utilities for AI Voice Detector

This module provides tools for:
1. Attention map extraction and visualization
2. Feature importance analysis (SHAP/LIME style)
3. Gradient-based saliency maps
"""

import numpy as np
import torch
import torch.nn.functional as F


class AttentionExtractor:
    """
    Extracts and visualizes attention weights from transformer-based models.
    Supports AudioViT and Conformer architectures.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on attention layers to capture attention weights."""
        self.hooks = []
        
        for name, module in self.model.named_modules():
            # Hook into attention layers (works for both AudioViT and Conformer)
            if 'attn' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)
    
    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights."""
        # For nn.MultiheadAttention, output is (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]
            if attn_weights is not None:
                self.attention_maps.append(attn_weights.detach().cpu())
    
    def get_attention_maps(self, x):
        """
        Run forward pass and return attention maps.
        
        Args:
            x: Input tensor of shape (batch, *input_shape)
            
        Returns:
            List of attention maps from each attention layer
        """
        self.attention_maps = []
        self.model.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            _ = self.model(x)
        
        return self.attention_maps
    
    def cleanup(self):
        """Remove hooks when done."""
        for hook in self.hooks:
            hook.remove()


class GradientSaliency:
    """
    Compute gradient-based saliency maps showing which input regions
    are most important for the model's prediction.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def compute_saliency(self, x, target_class=None):
        """
        Compute gradient-based saliency map.
        
        Args:
            x: Input tensor (batch, *input_shape)
            target_class: Class to compute gradients for. If None, uses predicted class.
            
        Returns:
            Saliency map of same shape as input
        """
        self.model.eval()
        x = x.to(self.device)
        x.requires_grad_(True)
        
        _, output = self.model(x)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Compute gradients
        self.model.zero_grad()
        target = output.gather(1, target_class.view(-1, 1))
        target.sum().backward()
        
        # Get absolute gradients
        saliency = x.grad.abs()
        
        # If input is spectrogram (2D), sum over frequency to get temporal saliency
        # or keep full 2D saliency
        return saliency.detach().cpu()


class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    Provides more accurate attributions than simple gradients.
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks"
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def compute(self, x, target_class=None, baseline=None, steps=50):
        """
        Compute integrated gradients.
        
        Args:
            x: Input tensor
            target_class: Target class for attribution
            baseline: Baseline input (default: zeros)
            steps: Number of interpolation steps
            
        Returns:
            Attribution map
        """
        self.model.eval()
        x = x.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)
        
        # Interpolate between baseline and input
        scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)
        
        # Forward pass
        _, output = self.model(scaled_inputs)
        
        if target_class is None:
            # Use the prediction for the original input
            with torch.no_grad():
                _, orig_out = self.model(x)
            target_class = orig_out.argmax(dim=1)
        
        # Expand target_class to match batch size
        expanded_target = target_class.repeat(steps + 1)
        
        # Compute gradients
        self.model.zero_grad()
        target = output.gather(1, expanded_target.view(-1, 1))
        target.sum().backward()
        
        grads = scaled_inputs.grad  # (steps+1, ...)
        
        # Average gradients
        avg_grads = grads.mean(dim=0, keepdim=True)
        
        # Integrated gradients = (input - baseline) * avg_grads
        integrated_grads = (x - baseline) * avg_grads
        
        return integrated_grads.detach().cpu()


def visualize_attention(attention_map, save_path=None):
    """
    Visualize attention map as a heatmap.
    
    Args:
        attention_map: 2D attention weights (query x key)
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.numpy()
        
        # Average over heads if multi-head
        if attention_map.ndim == 3:
            attention_map = attention_map.mean(axis=0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_map, cmap='viridis', xticklabels=False, yticklabels=False)
        plt.title('Attention Weights')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib and seaborn required for visualization")


def visualize_saliency(saliency_map, input_spectrogram=None, save_path=None):
    """
    Visualize saliency map overlaid on input spectrogram.
    
    Args:
        saliency_map: Gradient-based saliency (freq x time)
        input_spectrogram: Original input for overlay
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        
        if isinstance(saliency_map, torch.Tensor):
            saliency_map = saliency_map.numpy()
        
        # Handle batch and channel dimensions
        if saliency_map.ndim == 4:
            saliency_map = saliency_map[0, 0]  # Take first sample, first channel
        elif saliency_map.ndim == 3:
            saliency_map = saliency_map[0]
        
        fig, axes = plt.subplots(1, 2 if input_spectrogram is not None else 1, figsize=(12, 4))
        
        if input_spectrogram is not None:
            if isinstance(input_spectrogram, torch.Tensor):
                input_spectrogram = input_spectrogram.numpy()
            if input_spectrogram.ndim == 4:
                input_spectrogram = input_spectrogram[0, 0]
            elif input_spectrogram.ndim == 3:
                input_spectrogram = input_spectrogram[0]
            
            axes[0].imshow(input_spectrogram, aspect='auto', origin='lower', cmap='magma')
            axes[0].set_title('Input Spectrogram')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Frequency')
            
            axes[1].imshow(saliency_map, aspect='auto', origin='lower', cmap='hot')
            axes[1].set_title('Saliency Map')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Frequency')
        else:
            ax = axes if not hasattr(axes, '__iter__') else axes[0]
            ax.imshow(saliency_map, aspect='auto', origin='lower', cmap='hot')
            ax.set_title('Saliency Map')
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib required for visualization")
