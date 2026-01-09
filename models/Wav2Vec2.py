"""
Wav2Vec 2.0 Model Wrapper for Audio Deepfake Detection

This module provides a wrapper around the pre-trained Wav2Vec 2.0 model
from HuggingFace Transformers for self-supervised speech representation learning.

Wav2Vec 2.0 learns powerful audio representations from raw waveforms using
contrastive learning, making it highly effective for downstream tasks like
deepfake detection with minimal labeled data.

Reference:
    Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning
    of Speech Representations", NeurIPS 2020
"""

import torch
import torch.nn as nn

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")


class Wav2Vec2Classifier(nn.Module):
    """
    Wav2Vec 2.0 based classifier for audio deepfake detection.
    
    Uses a pre-trained Wav2Vec 2.0 model as feature extractor with a
    classification head on top. Supports both frozen and fine-tuned modes.
    """
    
    def __init__(self, d_args: dict):
        """
        Initialize Wav2Vec2 classifier.
        
        Args:
            d_args: Dictionary containing:
                - pretrained_model: HuggingFace model name (default: "facebook/wav2vec2-base")
                - num_classes: Number of output classes (default: 2)
                - freeze_encoder: Whether to freeze the encoder (default: False)
                - dropout: Dropout rate for classifier (default: 0.1)
                - pooling: Pooling method - 'mean', 'cls', or 'attention' (default: 'mean')
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers library is required for Wav2Vec2. "
                "Install with: pip install transformers"
            )
        
        # Configuration
        self.pretrained_model = d_args.get('pretrained_model', 'facebook/wav2vec2-base')
        self.num_classes = d_args.get('num_classes', 2)
        self.freeze_encoder = d_args.get('freeze_encoder', False)
        self.dropout_rate = d_args.get('dropout', 0.1)
        self.pooling = d_args.get('pooling', 'mean')
        
        # Load pre-trained model
        print(f"Loading Wav2Vec2 model: {self.pretrained_model}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(self.pretrained_model)
        
        # Get hidden size from the model
        self.hidden_size = self.wav2vec2.config.hidden_size
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            print("Freezing Wav2Vec2 encoder weights")
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Attention pooling (if used)
        if self.pooling == 'attention':
            self.attention_weights = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights with Xavier uniform."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pool hidden states to get a single vector representation.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) optional mask
            
        Returns:
            Pooled representation (batch, hidden_size)
        """
        if self.pooling == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)
        
        elif self.pooling == 'cls':
            # Use first token (CLS-like)
            return hidden_states[:, 0, :]
        
        elif self.pooling == 'attention':
            # Attention-weighted pooling
            attn_weights = self.attention_weights(hidden_states)  # (batch, seq, 1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = torch.sum(hidden_states * attn_weights, dim=1)
            return pooled
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def forward(self, x: torch.Tensor, Freq_aug: bool = False, attention_mask: torch.Tensor = None):
        """
        Forward pass.
        
        Args:
            x: Input waveform tensor (batch, samples) or (batch, 1, samples)
            Freq_aug: Ignored, for compatibility with training loop
            attention_mask: Optional attention mask
            
        Returns:
            (embeddings, logits): Tuple of pooled embeddings and classification logits
        """
        # Handle different input shapes
        if x.dim() == 3:
            # (batch, 1, samples) -> (batch, samples)
            x = x.squeeze(1)
        elif x.dim() == 1:
            # (samples,) -> (1, samples)
            x = x.unsqueeze(0)
        
        # Forward through Wav2Vec2
        outputs = self.wav2vec2(
            input_values=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Pool to get embeddings
        embeddings = self._pool_hidden_states(hidden_states, attention_mask)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return embeddings, logits
    
    def get_embeddings(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Extract embeddings without classification."""
        with torch.no_grad():
            embeddings, _ = self.forward(x, attention_mask=attention_mask)
            return embeddings


# For compatibility with main.py model loading
Model = Wav2Vec2Classifier


class Wav2Vec2FeatureExtractor(nn.Module):
    """
    Standalone Wav2Vec2 feature extractor for use with other classifiers.
    
    Extracts frame-level features from audio that can be used with
    downstream models like transformers or CNNs.
    """
    
    def __init__(self, pretrained_model: str = 'facebook/wav2vec2-base', freeze: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            pretrained_model: HuggingFace model name
            freeze: Whether to freeze weights
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library required")
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        self.hidden_size = self.wav2vec2.config.hidden_size
        
        if freeze:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frame-level features.
        
        Args:
            x: Input waveform (batch, samples)
            
        Returns:
            Frame-level features (batch, frames, hidden_size)
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        
        outputs = self.wav2vec2(input_values=x, return_dict=True)
        return outputs.last_hidden_state
