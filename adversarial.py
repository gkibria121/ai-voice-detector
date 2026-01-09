"""
Adversarial Training Utilities for AI Voice Detector

This module provides tools for:
1. Generating adversarial audio examples
2. Adversarial training for improved robustness
3. Defense mechanisms against adversarial attacks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) for generating adversarial examples.
    
    Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
    """
    def __init__(self, model, epsilon=0.01):
        """
        Args:
            model: The target model
            epsilon: Maximum perturbation magnitude
        """
        self.model = model
        self.epsilon = epsilon
    
    def generate(self, x, y, criterion=None):
        """
        Generate adversarial examples using FGSM.
        
        Args:
            x: Input tensor (batch, ...)
            y: True labels
            criterion: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Adversarial examples
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        _, output = self.model(x_adv)
        loss = criterion(output, y)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Create adversarial example
        perturbation = self.epsilon * x_adv.grad.sign()
        x_adv = x_adv + perturbation
        
        # Clamp to valid range (assuming normalized input)
        x_adv = torch.clamp(x_adv, x.min(), x.max())
        
        return x_adv.detach()


class PGD:
    """
    Projected Gradient Descent (PGD) for generating stronger adversarial examples.
    
    Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
    """
    def __init__(self, model, epsilon=0.03, alpha=0.01, num_steps=10):
        """
        Args:
            model: The target model
            epsilon: Maximum perturbation magnitude (L-infinity bound)
            alpha: Step size for each iteration
            num_steps: Number of PGD iterations
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate(self, x, y, criterion=None, random_start=True):
        """
        Generate adversarial examples using PGD.
        
        Args:
            x: Input tensor
            y: True labels
            criterion: Loss function
            random_start: Whether to start from random perturbation
            
        Returns:
            Adversarial examples
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        x_adv = x.clone().detach()
        
        # Random initialization within epsilon ball
        if random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, x.min(), x.max())
        
        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            _, output = self.model(x_adv)
            loss = criterion(output, y)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update
            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                # Project back to epsilon ball around original input
                perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = x + perturbation
                x_adv = torch.clamp(x_adv, x.min(), x.max())
        
        return x_adv.detach()


class AudioPerturbation:
    """
    Audio-specific adversarial perturbations that are more imperceptible
    to human listeners while still affecting the model.
    """
    def __init__(self, model, epsilon=0.01, sr=16000):
        """
        Args:
            model: Target model
            epsilon: Perturbation strength
            sr: Sample rate
        """
        self.model = model
        self.epsilon = epsilon
        self.sr = sr
    
    def frequency_masking_attack(self, x, y, num_masks=3, max_width=20):
        """
        Create adversarial examples by selectively masking frequency bands.
        This attack is specific to spectrogram-based models.
        
        Args:
            x: Input spectrogram (batch, freq, time) or (batch, 1, freq, time)
            y: True labels
            num_masks: Number of frequency masks
            max_width: Maximum mask width in frequency bins
            
        Returns:
            Adversarial spectrogram
        """
        x_adv = x.clone()
        
        if x_adv.dim() == 4:
            _, _, freq_dim, time_dim = x_adv.shape
        else:
            _, freq_dim, time_dim = x_adv.shape
        
        for _ in range(num_masks):
            width = torch.randint(1, max_width, (1,)).item()
            start = torch.randint(0, freq_dim - width, (1,)).item()
            
            if x_adv.dim() == 4:
                x_adv[:, :, start:start + width, :] = x_adv[:, :, start:start + width, :].mean()
            else:
                x_adv[:, start:start + width, :] = x_adv[:, start:start + width, :].mean()
        
        return x_adv
    
    def time_domain_attack(self, waveform, y, criterion=None):
        """
        Adversarial attack in time domain for raw waveform models.
        Uses smooth perturbations that are less audible.
        
        Args:
            waveform: Raw audio waveform (batch, samples/channels)
            y: True labels
            criterion: Loss function
            
        Returns:
            Adversarial waveform
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        x_adv = waveform.clone().detach().requires_grad_(True)
        
        _, output = self.model(x_adv)
        loss = criterion(output, y)
        
        self.model.zero_grad()
        loss.backward()
        
        # Get gradient and apply smoothing for less audible perturbation
        grad = x_adv.grad
        
        # Simple smoothing via moving average (optional)
        kernel_size = 7
        if grad.dim() == 2:
            grad = F.avg_pool1d(grad.unsqueeze(1), kernel_size, stride=1, padding=kernel_size // 2).squeeze(1)
        
        perturbation = self.epsilon * grad.sign()
        x_adv = waveform + perturbation
        
        # Clamp to valid audio range
        x_adv = torch.clamp(x_adv, -1.0, 1.0)
        
        return x_adv.detach()


class AdversarialTrainer:
    """
    Wrapper for adversarial training, combining clean and adversarial examples.
    """
    def __init__(self, model, attack_method='pgd', epsilon=0.03, mix_ratio=0.5):
        """
        Args:
            model: The model to train
            attack_method: 'fgsm' or 'pgd'
            epsilon: Perturbation magnitude
            mix_ratio: Ratio of adversarial examples in each batch (0 to 1)
        """
        self.model = model
        self.epsilon = epsilon
        self.mix_ratio = mix_ratio
        
        if attack_method == 'fgsm':
            self.attack = FGSM(model, epsilon)
        elif attack_method == 'pgd':
            self.attack = PGD(model, epsilon)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
    
    def train_step(self, x, y, criterion, optimizer):
        """
        Perform one adversarial training step.
        
        Args:
            x: Input batch
            y: Labels
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Combined loss value
        """
        self.model.train()
        
        # Determine split
        batch_size = x.size(0)
        num_adv = int(batch_size * self.mix_ratio)
        
        # Generate adversarial examples for part of the batch
        if num_adv > 0:
            x_adv = self.attack.generate(x[:num_adv], y[:num_adv], criterion)
            x_combined = torch.cat([x_adv, x[num_adv:]], dim=0)
        else:
            x_combined = x
        
        # Forward pass
        optimizer.zero_grad()
        _, output = self.model(x_combined)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()


def apply_input_preprocessing(x, method='random_smoothing', sigma=0.1):
    """
    Apply input preprocessing as a defense mechanism.
    
    Args:
        x: Input tensor
        method: 'random_smoothing', 'jpeg_compression' (for spectrograms), or 'quantization'
        sigma: Noise level for random smoothing
        
    Returns:
        Preprocessed input
    """
    if method == 'random_smoothing':
        # Add Gaussian noise for certified defense
        noise = torch.randn_like(x) * sigma
        return x + noise
    
    elif method == 'quantization':
        # Reduce precision to remove small perturbations
        levels = 256
        x_min, x_max = x.min(), x.max()
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
        x_quantized = torch.round(x_normalized * (levels - 1)) / (levels - 1)
        return x_quantized * (x_max - x_min) + x_min
    
    else:
        return x
