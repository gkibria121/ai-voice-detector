"""
Explainability Utilities for AI Voice Detector

This module provides GradCAM, SHAP, and other XAI tools for model interpretability.
"""

import numpy as np
import torch
import torch.nn.functional as F


# Check for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


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
        x = x.detach().to(self.device)
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


class SmoothGrad:
    """
    SmoothGrad: Reduces noise in gradient-based saliency maps by averaging
    gradients over multiple noisy samples of the input.
    
    This significantly improves visualization quality and removes high-frequency
    noise that often appears in vanilla gradient saliency maps.
    
    Reference: Smilkov et al., "SmoothGrad: removing noise by adding noise"
    """
    def __init__(self, model, device='cpu', n_samples=50, noise_level=0.15):
        """
        Args:
            model: The model to explain
            device: Device to run on
            n_samples: Number of noisy samples to average over (default: 50)
            noise_level: Std dev of Gaussian noise as fraction of (max-min) (default: 0.15)
        """
        self.gradient_saliency = GradientSaliency(model, device)
        self.model = model
        self.device = device
        self.n_samples = n_samples
        self.noise_level = noise_level
    
    def compute_saliency(self, x, target_class=None):
        """
        Compute SmoothGrad saliency map.
        
        Args:
            x: Input tensor (batch, *input_shape)
            target_class: Class to compute gradients for. If None, uses predicted class.
            
        Returns:
            Smoothed saliency map of same shape as input
        """
        # Determine noise scale based on input range
        x_min = x.min()
        x_max = x.max()
        noise_scale = (x_max - x_min) * self.noise_level
        
        saliency_sum = None
        
        for i in range(self.n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(x) * noise_scale
            noisy_x = x + noise
            
            # Compute saliency for this noisy sample
            saliency = self.gradient_saliency.compute_saliency(noisy_x, target_class)
            
            if saliency_sum is None:
                saliency_sum = saliency
            else:
                saliency_sum += saliency
        
        # Average over all samples
        smooth_saliency = saliency_sum / self.n_samples
        
        return smooth_saliency


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
        x = x.detach().to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.detach().to(self.device)
        
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


class OcclusionSensitivity:
    """
    Compute Occlusion Sensitivity maps by masking parts of the input
    and measuring the drop in target class probability.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
    
    def compute(self, x, target_class=None, window_shape=(8, 8), stride=4, baseline=0.0):
        """
        Compute occlusion sensitivity map.
        
        Args:
            x: Input tensor (batch, 1, freqs, time) or (batch, 1, time)
            target_class: Target class index
            window_shape: Tuple (freq, time) for 2D or int for 1D
            stride: Stride for sliding window
            baseline: Value to replace occluded area with (default: 0)
            
        Returns:
            Sensitivity map
        """
        self.model.eval()
        x = x.to(self.device).clone()
        
        with torch.no_grad():
            # Get original probability
            _, orig_out = self.model(x)
            if target_class is None:
                target_class = orig_out.argmax(dim=1)
            
            orig_prob = F.softmax(orig_out, dim=1)[0, target_class].item()
        
        # Determine dimensionality
        if x.ndim == 4: # (B, C, H, W) -> Spectrogram
            h, w = x.shape[2], x.shape[3]
            if isinstance(window_shape, int):
                window_shape = (window_shape, window_shape)
            win_h, win_w = window_shape
            
            if isinstance(stride, int):
                stride_h = stride_w = stride
            else:
                stride_h, stride_w = stride
                
            heatmap = torch.zeros((h, w), device=self.device)
            # Counts to average overlapping windows
            counts = torch.zeros((h, w), device=self.device)
            
            # Sliding window
            for i in range(0, h - win_h + 1, stride_h):
                for j in range(0, w - win_w + 1, stride_w):
                    # Mask input
                    x_masked = x.clone()
                    x_masked[:, :, i:i+win_h, j:j+win_w] = baseline
                    
                    # Forward pass
                    with torch.no_grad():
                        _, out = self.model(x_masked)
                        prob = F.softmax(out, dim=1)[0, target_class].item()
                    
                    # Sensitivity = Drop in probability
                    score = orig_prob - prob
                    
                    heatmap[i:i+win_h, j:j+win_w] += score
                    counts[i:i+win_h, j:j+win_w] += 1
            
            # Average
            heatmap = heatmap / (counts + 1e-8)
            return heatmap.cpu()
            
        elif x.ndim == 3: # (B, C, L) -> Raw Audio
            l = x.shape[2]
            win_l = window_shape if isinstance(window_shape, int) else window_shape[0]
            stride_l = stride if isinstance(stride, int) else stride[0]
            
            heatmap = torch.zeros((l), device=self.device)
            counts = torch.zeros((l), device=self.device)
            
            for i in range(0, l - win_l + 1, stride_l):
                x_masked = x.clone()
                x_masked[:, :, i:i+win_l] = baseline
                
                with torch.no_grad():
                    _, out = self.model(x_masked)
                    prob = F.softmax(out, dim=1)[0, target_class].item()
                    
                score = orig_prob - prob
                heatmap[i:i+win_l] += score
                counts[i:i+win_l] += 1
                
            heatmap = heatmap / (counts + 1e-8)
            return heatmap.cpu()
            
        return None


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    Visualizes which parts of the image/spectrogram were relevant directly 
    from a specific convolutional layer.
    """
    def __init__(self, model, target_layer, device='cpu'):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def compute(self, x, target_class=None):
        """
        Compute Grad-CAM heatmap.
        """
        self.model.eval()
        
        # Reset gradients and activations from previous calls
        self.gradients = None
        self.activations = None
        
        x = x.detach().to(self.device).requires_grad_(True)
        self.model.zero_grad()
        
        # Forward
        _, output = self.model(x)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        elif isinstance(target_class, torch.Tensor):
            target_class = target_class.item()
            
        # Backward
        target = output[0, target_class]
        target.backward()
        
        # Check if hooks captured data
        if self.gradients is None or self.activations is None:
            print(f"    Warning: Gradients or activations not captured. Using activation fallback.")
            # Fallback: just use activation magnitude
            if self.activations is not None:
                activations = self.activations.detach().clone()
                if activations.ndim == 4:
                    heatmap = torch.mean(torch.abs(activations), dim=1).squeeze()
                elif activations.ndim == 3:
                    heatmap = torch.mean(torch.abs(activations), dim=1).squeeze()
                else:
                    return None
                if torch.max(heatmap) != 0:
                    heatmap /= torch.max(heatmap)
                return heatmap.cpu()
            return None
        
        # Generate heatmap
        # GAP of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])
        if self.gradients.ndim == 4: # (B, C, H, W)
             pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Check for zero gradients
        if torch.max(torch.abs(pooled_gradients)) == 0:
            print(f"    Warning: Zero gradients detected. Using activation magnitude fallback.")
            activations = self.activations.detach().clone()
            if activations.ndim == 4:
                heatmap = torch.mean(torch.abs(activations), dim=1).squeeze()
            elif activations.ndim == 3:
                heatmap = torch.mean(torch.abs(activations), dim=1).squeeze()
            else:
                return None
            if torch.max(heatmap) != 0:
                heatmap /= torch.max(heatmap)
            return heatmap.cpu()
        
        # Weight activations - clone to avoid in-place modification issues
        activations = self.activations.detach().clone() # (B, C, ...)
        
        if activations.ndim == 4: # Spectrogram (B, C, H, W)
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(activations, dim=1).squeeze()
            
            # Apply ReLU, but fall back to absolute values if all zeros
            heatmap_relu = F.relu(heatmap)
            if torch.max(heatmap_relu) == 0:
                # Model may have negative gradients - use absolute values instead
                heatmap = torch.abs(heatmap)
            else:
                heatmap = heatmap_relu
            
            # Normalize
            if torch.max(heatmap) != 0:
                heatmap /= torch.max(heatmap)
                
            # Resize
            # We return the small heatmap, visualization will handle scaling
            return heatmap.cpu()
            
        elif activations.ndim == 3: # Raw audio (B, C, L)
            for i in range(activations.shape[1]):
                activations[:, i, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(activations, dim=1).squeeze()
            
            # Apply ReLU, but fall back to absolute values if all zeros
            heatmap_relu = F.relu(heatmap)
            if torch.max(heatmap_relu) == 0:
                heatmap = torch.abs(heatmap)
            else:
                heatmap = heatmap_relu
            
            if torch.max(heatmap) != 0:
                heatmap /= torch.max(heatmap)
                
            return heatmap.cpu()
            
        return None


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
        
        # Handle 1D saliency (e.g. raw audio) by expanding to 2D strip
        if saliency_map.ndim == 1:
            saliency_map = saliency_map[None, :] # (1, L)
            
        fig, axes = plt.subplots(1, 2 if input_spectrogram is not None else 1, figsize=(12, 4))
        
        if input_spectrogram is not None:
            if isinstance(input_spectrogram, torch.Tensor):
                input_spectrogram = input_spectrogram.numpy()
            if input_spectrogram.ndim == 4:
                input_spectrogram = input_spectrogram[0, 0]
            elif input_spectrogram.ndim == 3:
                input_spectrogram = input_spectrogram[0]
            # Handle 1D input spectrogram (raw audio)
            if input_spectrogram.ndim == 1:
                 input_spectrogram = input_spectrogram[None, :]
            
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


class _ModelOutputWrapper(torch.nn.Module):
    """Wrapper that returns only logits from models that return (features, logits) tuples."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        # Handle tuple outputs (features, logits)
        if isinstance(output, tuple):
            return output[1]  # Return only logits
        return output


class AudioSHAP:
    """
    SHAP (SHapley Additive exPlanations) analysis for audio models.
    
    Uses DeepExplainer or GradientExplainer for efficient SHAP value computation
    on deep learning models. Provides feature attribution at the spectrogram level.
    
    Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"
    """
    def __init__(self, model, device='cpu', background_samples=None):
        """
        Args:
            model: The model to explain
            device: Device to run on
            background_samples: Background dataset for SHAP (tensor of shape (N, ...))
                               If None, will use zeros as baseline
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed. Install with: pip install shap")
        
        self.model = model
        self.device = device
        self.model.eval()
        
        # Create wrapper for SHAP that returns only logits
        self._model_for_shap = _ModelOutputWrapper(model).to(device)
        self._model_for_shap.eval()
        
        # Store background samples
        if background_samples is not None:
            self.background = background_samples.to(device)
        else:
            self.background = None
        
        self._explainer = None
    
    def _model_wrapper(self, x):
        """Wrapper to return only logits for SHAP."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            output = self._model_for_shap(x)
        return output.cpu().numpy()
    
    def _get_explainer(self, input_tensor):
        """Get or create SHAP explainer."""
        if self._explainer is not None:
            return self._explainer
        
        # Use background samples or create from input
        if self.background is not None:
            background = self.background
        else:
            # Use zeros as baseline (common for spectrograms)
            background = torch.zeros_like(input_tensor).to(self.device)
            # Add small noise to avoid numerical issues
            background = background + torch.randn_like(background) * 0.01
        
        # Try GradientExplainer first (works with gradients)
        try:
            self._explainer = shap.GradientExplainer(
                self._model_for_shap,
                background
            )
        except Exception:
            # Fallback to DeepExplainer
            try:
                self._explainer = shap.DeepExplainer(
                    self._model_for_shap,
                    background
                )
            except Exception as e:
                print(f"Warning: Could not create SHAP explainer: {e}")
                self._explainer = None
        
        return self._explainer
    
    def compute_shap_values(self, x, target_class=None, n_samples=100):
        """
        Compute SHAP values for input.
        
        Args:
            x: Input tensor (batch, channels, freq, time) or (batch, channels, time)
            target_class: Class to explain. If None, uses predicted class.
            n_samples: Number of samples for approximation
            
        Returns:
            SHAP values array of same shape as input
        """
        self.model.eval()
        x = x.to(self.device)
        
        # Get prediction if target_class not specified
        if target_class is None:
            with torch.no_grad():
                output = self._model_for_shap(x)
                target_class = output.argmax(dim=1).item()
        
        explainer = self._get_explainer(x)
        
        if explainer is None:
            # Fallback to perturbation-based SHAP
            return self._compute_perturbation_shap(x, target_class, n_samples)
        
        try:
            # Compute SHAP values
            shap_values = explainer.shap_values(x)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # Multi-class output - select target class
                shap_values = shap_values[target_class]
            
            if isinstance(shap_values, np.ndarray):
                return torch.from_numpy(shap_values)
            return shap_values
            
        except Exception as e:
            print(f"SHAP explainer failed: {e}, using perturbation method")
            return self._compute_perturbation_shap(x, target_class, n_samples)
    
    def _compute_perturbation_shap(self, x, target_class, n_samples=100):
        """
        Compute approximate SHAP values using perturbation sampling.
        Works when gradient-based methods fail.
        """
        self.model.eval()
        x = x.to(self.device)
        original_shape = x.shape
        
        with torch.no_grad():
            orig_output = self._model_for_shap(x)
            orig_prob = F.softmax(orig_output, dim=1)[0, target_class].item()
        
        # For spectrograms, compute importance per time-frequency region
        if x.ndim == 4:  # (B, C, H, W)
            h, w = x.shape[2], x.shape[3]
            # Use coarser grid for efficiency
            grid_h, grid_w = min(16, h), min(32, w)
            step_h, step_w = max(1, h // grid_h), max(1, w // grid_w)
            
            importance = torch.zeros((h, w), device=self.device)
            counts = torch.zeros((h, w), device=self.device)
            
            for _ in range(n_samples):
                # Random mask
                mask = torch.rand((grid_h, grid_w), device=self.device) > 0.5
                mask = mask.float()
                # Upsample mask to input size
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), 
                    size=(h, w), 
                    mode='nearest'
                ).squeeze()
                
                # Apply mask (multiply input by mask)
                x_masked = x.clone()
                x_masked = x_masked * mask.unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    output = self._model_for_shap(x_masked)
                    prob = F.softmax(output, dim=1)[0, target_class].item()
                
                # Contribution = difference when feature is present
                contrib = orig_prob - prob
                importance += mask * contrib
                counts += mask
            
            # Average contributions
            importance = importance / (counts + 1e-8)
            return importance.unsqueeze(0).unsqueeze(0).cpu()
            
        elif x.ndim == 3:  # (B, C, L) - raw audio
            l = x.shape[2]
            grid_l = min(64, l)
            
            importance = torch.zeros(l, device=self.device)
            counts = torch.zeros(l, device=self.device)
            
            for _ in range(n_samples):
                mask = torch.rand(grid_l, device=self.device) > 0.5
                mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=l,
                    mode='nearest'
                ).squeeze()
                
                x_masked = x.clone() * mask.unsqueeze(0).unsqueeze(0)
                
                with torch.no_grad():
                    output = self._model_for_shap(x_masked)
                    prob = F.softmax(output, dim=1)[0, target_class].item()
                
                contrib = orig_prob - prob
                importance += mask * contrib
                counts += mask
            
            importance = importance / (counts + 1e-8)
            return importance.unsqueeze(0).unsqueeze(0).cpu()
        
        return None


class SpectrogramRegionAnalysis:
    """
    Analyze which frequency bands and time segments are most important.
    Provides human-interpretable insights about what the model focuses on.
    """
    def __init__(self, model, device='cpu', sr=16000, n_mels=128, hop_length=512):
        self.model = model
        self.device = device
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        
        # Define frequency band names (approximate Hz ranges for mel scale)
        self.freq_bands = {
            'sub_bass': (0, 60),      # 0-60 Hz
            'bass': (60, 250),         # 60-250 Hz
            'low_mid': (250, 500),     # 250-500 Hz
            'mid': (500, 2000),        # 500-2000 Hz
            'high_mid': (2000, 4000),  # 2000-4000 Hz
            'presence': (4000, 6000),  # 4000-6000 Hz
            'brilliance': (6000, 8000) # 6000-8000+ Hz
        }
    
    def analyze_importance_by_band(self, importance_map, return_raw=False):
        """
        Analyze which frequency bands are most important.
        
        Args:
            importance_map: 2D importance map (freq x time)
            return_raw: If True, return raw band scores
            
        Returns:
            Dictionary with band names and their importance scores
        """
        if isinstance(importance_map, torch.Tensor):
            importance_map = importance_map.numpy()
        
        # Handle batch/channel dimensions
        while importance_map.ndim > 2:
            importance_map = importance_map[0]
        
        h, w = importance_map.shape  # freq x time
        
        # Convert mel bins to approximate frequency bands
        mel_to_hz = lambda m: 700 * (10**(m / 2595) - 1)
        
        band_scores = {}
        for band_name, (low_hz, high_hz) in self.freq_bands.items():
            # Approximate mel bin range
            low_mel = int(low_hz / (self.sr / 2) * h)
            high_mel = int(high_hz / (self.sr / 2) * h)
            low_mel = max(0, min(low_mel, h-1))
            high_mel = max(low_mel+1, min(high_mel, h))
            
            # Extract band importance
            band_importance = importance_map[low_mel:high_mel, :]
            band_scores[band_name] = float(np.mean(np.abs(band_importance)))
        
        # Normalize scores
        total = sum(band_scores.values()) + 1e-8
        normalized = {k: v / total * 100 for k, v in band_scores.items()}
        
        if return_raw:
            return band_scores, normalized
        return normalized
    
    def analyze_temporal_pattern(self, importance_map, n_segments=10):
        """
        Analyze temporal patterns in importance.
        
        Args:
            importance_map: 2D importance map (freq x time)
            n_segments: Number of time segments to analyze
            
        Returns:
            Dictionary with temporal analysis results
        """
        if isinstance(importance_map, torch.Tensor):
            importance_map = importance_map.numpy()
        
        while importance_map.ndim > 2:
            importance_map = importance_map[0]
        
        h, w = importance_map.shape
        segment_size = w // n_segments
        
        temporal_scores = []
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else w
            segment_importance = importance_map[:, start:end]
            temporal_scores.append(float(np.mean(np.abs(segment_importance))))
        
        # Find peak regions
        peak_segment = int(np.argmax(temporal_scores))
        
        return {
            'segment_scores': temporal_scores,
            'peak_segment': peak_segment,
            'peak_time_ratio': peak_segment / n_segments,
            'temporal_variance': float(np.var(temporal_scores)),
            'is_uniform': float(np.var(temporal_scores)) < 0.01
        }
    
    def generate_report(self, importance_map, prediction, confidence):
        """
        Generate a human-readable analysis report.
        
        Args:
            importance_map: 2D importance map
            prediction: Model prediction (0=fake, 1=real)
            confidence: Prediction confidence
            
        Returns:
            String report
        """
        band_scores = self.analyze_importance_by_band(importance_map)
        temporal = self.analyze_temporal_pattern(importance_map)
        
        # Sort bands by importance
        sorted_bands = sorted(band_scores.items(), key=lambda x: x[1], reverse=True)
        
        pred_label = "REAL" if prediction == 1 else "FAKE"
        
        report = []
        report.append("=" * 60)
        report.append("AUDIO ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nPrediction: {pred_label} (Confidence: {confidence:.1%})")
        report.append("\n--- Frequency Band Importance ---")
        
        for band, score in sorted_bands:
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            report.append(f"  {band:12s}: {bar} {score:.1f}%")
        
        report.append("\n--- Temporal Analysis ---")
        if temporal['is_uniform']:
            report.append("  Pattern: Uniform importance across time")
        else:
            peak_pct = temporal['peak_time_ratio'] * 100
            if peak_pct < 33:
                position = "beginning"
            elif peak_pct < 66:
                position = "middle"
            else:
                position = "end"
            report.append(f"  Peak importance: {position} of audio ({peak_pct:.0f}%)")
            report.append(f"  Temporal variance: {temporal['temporal_variance']:.4f}")
        
        report.append("\n--- Key Observations ---")
        top_band = sorted_bands[0][0]
        if top_band in ['sub_bass', 'bass']:
            report.append("  • Model focuses on low-frequency content")
            report.append("  • May indicate attention to voice fundamental frequency")
        elif top_band in ['mid', 'low_mid']:
            report.append("  • Model focuses on mid-frequency content")
            report.append("  • Typical for voice formant analysis")
        elif top_band in ['high_mid', 'presence', 'brilliance']:
            report.append("  • Model focuses on high-frequency content")
            report.append("  • May indicate attention to synthesis artifacts")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def visualize_shap_values(shap_values, input_spectrogram=None, save_path=None, title="SHAP Values"):
    """
    Visualize SHAP values as a heatmap.
    
    Args:
        shap_values: SHAP values tensor
        input_spectrogram: Original input for comparison
        save_path: Optional path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.numpy()
        
        # Handle batch/channel dimensions
        while shap_values.ndim > 2:
            shap_values = shap_values[0]
        
        # Handle 1D (raw audio)
        if shap_values.ndim == 1:
            shap_values = shap_values[None, :]
        
        fig, axes = plt.subplots(1, 2 if input_spectrogram is not None else 1, figsize=(14, 5))
        
        if input_spectrogram is not None:
            if isinstance(input_spectrogram, torch.Tensor):
                input_spectrogram = input_spectrogram.numpy()
            while input_spectrogram.ndim > 2:
                input_spectrogram = input_spectrogram[0]
            if input_spectrogram.ndim == 1:
                input_spectrogram = input_spectrogram[None, :]
            
            ax0 = axes[0] if hasattr(axes, '__iter__') else axes
            im0 = ax0.imshow(input_spectrogram, aspect='auto', origin='lower', cmap='magma')
            ax0.set_title('Input Spectrogram')
            ax0.set_xlabel('Time')
            ax0.set_ylabel('Frequency')
            plt.colorbar(im0, ax=ax0, label='Magnitude')
            
            ax1 = axes[1]
        else:
            ax1 = axes if not hasattr(axes, '__iter__') else axes[0]
        
        # Use diverging colormap for SHAP (red=positive, blue=negative)
        vmax = np.max(np.abs(shap_values))
        if vmax > 0:
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            norm = None
        
        im1 = ax1.imshow(shap_values, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm)
        ax1.set_title(title)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        plt.colorbar(im1, ax=ax1, label='SHAP Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib required for visualization")


def visualize_band_importance(band_scores, save_path=None, title="Frequency Band Importance"):
    """
    Visualize frequency band importance as a bar chart.
    
    Args:
        band_scores: Dictionary of band names to importance scores
        save_path: Optional path to save figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        bands = list(band_scores.keys())
        scores = list(band_scores.values())
        
        # Color by importance
        colors = plt.cm.RdYlGn_r(np.array(scores) / max(scores))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(bands, scores, color=colors)
        
        ax.set_xlabel('Importance (%)')
        ax.set_title(title)
        ax.set_xlim(0, max(scores) * 1.1)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib required for visualization")