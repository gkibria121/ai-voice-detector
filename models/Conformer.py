"""
Conformer: Convolution-augmented Transformer for Audio Processing

The Conformer architecture combines CNN's local feature extraction with 
Transformer's global context modeling. It's particularly effective for 
speech/audio tasks as it can capture both local patterns (via convolutions)
and long-range dependencies (via self-attention).

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class DepthwiseConv1d(nn.Module):
    """Depthwise separable 1D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module
    
    Applies: LayerNorm -> Pointwise Conv -> GLU -> DepthwiseConv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """
    def __init__(self, input_dim, num_channels, kernel_size=31, dropout=0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Kernel size must be odd"
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.pointwise_conv1 = nn.Conv1d(input_dim, num_channels * 2, kernel_size=1)
        self.glu = GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            num_channels, num_channels, kernel_size, 
            padding=(kernel_size - 1) // 2, groups=num_channels
        )
        self.batch_norm = nn.BatchNorm1d(num_channels)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(num_channels, input_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, time, dim)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, dim, time)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x.transpose(1, 2)  # (batch, time, dim)


class FeedForwardModule(nn.Module):
    """
    Feed Forward Module with pre-norm
    
    Applies: LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    """
    Multi-Head Self-Attention Module with pre-norm
    """
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.layer_norm(x)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        return self.dropout(attn_out)


class ConformerBlock(nn.Module):
    """
    A single Conformer block combining:
    - Feed-forward module (first half)
    - Multi-head self-attention module
    - Convolution module
    - Feed-forward module (second half)
    
    Uses Macaron-Net style connections with 0.5 residual weighting for FF modules.
    """
    def __init__(
        self,
        encoder_dim=256,
        num_attention_heads=4,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        dropout=0.1
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(encoder_dim, encoder_dim * feed_forward_expansion_factor, dropout)
        self.attn = MultiHeadSelfAttentionModule(encoder_dim, num_attention_heads, dropout)
        self.conv = ConvolutionModule(encoder_dim, encoder_dim * conv_expansion_factor, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(encoder_dim, encoder_dim * feed_forward_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x, mask=None):
        # Macaron-Net style: half-step residual for first FFN
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, mask)
        x = x + self.conv(x)
        # Half-step residual for second FFN
        x = x + 0.5 * self.ff2(x)
        return self.layer_norm(x)


class ConformerEncoder(nn.Module):
    """
    Stack of Conformer blocks with input projection
    """
    def __init__(
        self,
        input_dim=128,
        encoder_dim=256,
        num_layers=4,
        num_attention_heads=4,
        feed_forward_expansion_factor=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, encoder_dim)
        self.pos_encoding = PositionalEncoding(encoder_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # x: (batch, freq, time) -> (batch, time, freq)
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x  # (batch, time, encoder_dim)


class Model(nn.Module):
    """
    Conformer model for audio classification (deepfake detection)
    """
    def __init__(self, d_args):
        super().__init__()
        
        input_dim = d_args.get("input_dim", 128)
        num_classes = d_args.get("num_classes", 2)
        encoder_dim = d_args.get("encoder_dim", 256)
        num_layers = d_args.get("num_encoder_layers", 4)
        num_heads = d_args.get("num_attention_heads", 4)
        ff_expansion = d_args.get("feed_forward_expansion_factor", 4)
        conv_expansion = d_args.get("conv_expansion_factor", 2)
        conv_kernel_size = d_args.get("conv_kernel_size", 31)
        dropout = d_args.get("dropout", 0.1)
        
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            feed_forward_expansion_factor=ff_expansion,
            conv_expansion_factor=conv_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        # Pooling and classification
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, Freq_aug=False):
        # x: (batch, freq, time) or (batch, 1, freq, time)
        if x.dim() == 4:
            x = x.squeeze(1)  # Remove channel dim if present
        
        # Encode
        encoded = self.encoder(x)  # (batch, time, encoder_dim)
        
        # Pool over time dimension
        encoded = encoded.transpose(1, 2)  # (batch, encoder_dim, time)
        pooled = self.pooling(encoded).squeeze(-1)  # (batch, encoder_dim)
        
        embeddings = pooled
        output = self.classifier(pooled)
        
        return embeddings, output
