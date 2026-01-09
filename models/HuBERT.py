"""
HuBERT Model Wrapper for Audio Deepfake Detection

This module provides a wrapper around the pre-trained HuBERT model
from HuggingFace Transformers for self-supervised speech representation learning.

HuBERT (Hidden-Unit BERT) uses a BERT-like masked prediction objective on
discretized audio features, learning robust representations that capture
both phonetic and speaker-specific information.

Reference:
    Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by
    Masked Prediction of Hidden Units", IEEE/ACM TASLP 2021
"""

import torch
import torch.nn as nn

try:
    from transformers import HubertModel, HubertConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")


class HuBERTClassifier(nn.Module):
    """
    HuBERT-based classifier for audio deepfake detection.
    
    Uses a pre-trained HuBERT model as feature extractor with a
    classification head on top. Supports both frozen and fine-tuned modes.
    """
    
    def __init__(self, d_args: dict):
        """
        Initialize HuBERT classifier.
        
        Args:
            d_args: Dictionary containing:
                - pretrained_model: HuggingFace model name (default: "facebook/hubert-base-ls960")
                - num_classes: Number of output classes (default: 2)
                - freeze_encoder: Whether to freeze the encoder (default: False)
                - freeze_feature_extractor: Freeze only CNN feature extractor (default: True)
                - dropout: Dropout rate for classifier (default: 0.1)
                - pooling: Pooling method - 'mean', 'weighted', or 'attention' (default: 'mean')
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers library is required for HuBERT. "
                "Install with: pip install transformers"
            )
        
        # Configuration
        self.pretrained_model = d_args.get('pretrained_model', 'facebook/hubert-base-ls960')
        self.num_classes = d_args.get('num_classes', 2)
        self.freeze_encoder = d_args.get('freeze_encoder', False)
        self.freeze_feature_extractor = d_args.get('freeze_feature_extractor', True)
        self.dropout_rate = d_args.get('dropout', 0.1)
        self.pooling = d_args.get('pooling', 'mean')
        
        # Load pre-trained model
        print(f"Loading HuBERT model: {self.pretrained_model}")
        self.hubert = HubertModel.from_pretrained(self.pretrained_model)
        
        # Get hidden size from the model
        self.hidden_size = self.hubert.config.hidden_size
        
        # Freeze options
        if self.freeze_encoder:
            print("Freezing entire HuBERT encoder")
            for param in self.hubert.parameters():
                param.requires_grad = False
        elif self.freeze_feature_extractor:
            print("Freezing HuBERT feature extractor (CNN layers)")
            self.hubert.feature_extractor._freeze_parameters()
        
        # Weighted layer pooling (optional - combines all transformer layers)
        self.use_weighted_layer_sum = d_args.get('weighted_layer_sum', False)
        if self.use_weighted_layer_sum:
            num_layers = self.hubert.config.num_hidden_layers + 1  # +1 for embeddings
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Attention pooling (if used)
        if self.pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes),
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pool_hidden_states(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
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
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)
        
        elif self.pooling == 'attention':
            # Attention-weighted pooling
            attn_weights = self.attention(hidden_states)  # (batch, seq, 1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = torch.sum(hidden_states * attn_weights, dim=1)
            return pooled
        
        elif self.pooling == 'first':
            # Use first frame
            return hidden_states[:, 0, :]
        
        elif self.pooling == 'last':
            # Use last frame
            if attention_mask is not None:
                # Find last valid position
                seq_lens = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_indices, seq_lens]
            else:
                return hidden_states[:, -1, :]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def _combine_hidden_layers(self, all_hidden_states: tuple) -> torch.Tensor:
        """
        Combine all hidden layer outputs using learned weights.
        
        Args:
            all_hidden_states: Tuple of hidden states from all layers
            
        Returns:
            Weighted combination (batch, seq_len, hidden_size)
        """
        if not self.use_weighted_layer_sum:
            return all_hidden_states[-1]  # Just return last layer
        
        # Stack and weight
        stacked = torch.stack(all_hidden_states, dim=0)  # (layers, batch, seq, hidden)
        weights = torch.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted
    
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
            x = x.squeeze(1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward through HuBERT
        outputs = self.hubert(
            input_values=x,
            attention_mask=attention_mask,
            output_hidden_states=self.use_weighted_layer_sum,
            return_dict=True,
        )
        
        # Get hidden states (either last layer or weighted combination)
        if self.use_weighted_layer_sum:
            hidden_states = self._combine_hidden_layers(outputs.hidden_states)
        else:
            hidden_states = outputs.last_hidden_state
        
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
Model = HuBERTClassifier


class HuBERTFeatureExtractor(nn.Module):
    """
    Standalone HuBERT feature extractor for use with other classifiers.
    """
    
    def __init__(self, pretrained_model: str = 'facebook/hubert-base-ls960', freeze: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            pretrained_model: HuggingFace model name
            freeze: Whether to freeze weights
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library required")
        
        self.hubert = HubertModel.from_pretrained(pretrained_model)
        self.hidden_size = self.hubert.config.hidden_size
        
        if freeze:
            for param in self.hubert.parameters():
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
        
        outputs = self.hubert(input_values=x, return_dict=True)
        return outputs.last_hidden_state
