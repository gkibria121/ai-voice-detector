"""
MobileViT: Lightweight Vision Transformer for Audio Classification

This module implements a MobileViT architecture adapted for audio spectrogram
classification. MobileViT combines the efficiency of MobileNetV2 with the
global context modeling of Vision Transformers.

Benefits for audio deepfake detection:
- ~2-3M parameters (vs ~25M for standard ViT)
- Real-time inference on CPU
- Suitable for edge/mobile deployment
- Good balance between accuracy and efficiency

Reference:
    Mehta & Rastegari, "MobileViT: Light-weight, General-purpose, and 
    Mobile-friendly Vision Transformer", ICLR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


def make_divisible(v: float, divisor: int = 8) -> int:
    """Make value divisible by divisor."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: nn.Module = nn.SiLU,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act(inplace=True) if act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual block.
    
    Expands channels, applies depthwise conv, then projects back.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 4.0,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = make_divisible(in_channels * expand_ratio)
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))
        
        # Depthwise
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        
        # Project
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head self-attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class MobileViTBlock(nn.Module):
    """
    MobileViT Block: combines local (CNN) and global (Transformer) processing.
    
    1. Local representation via 3x3 conv
    2. Unfold into patches
    3. Global representation via transformer
    4. Fold back and project
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        patch_size: Tuple[int, int] = (2, 2),
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.patch_h, self.patch_w = patch_size
        self.embed_dim = embed_dim
        
        # Local representation
        self.conv1 = ConvBNAct(in_channels, in_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        
        # Global representation (transformer)
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Fusion
        self.conv3 = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)
        self.conv4 = ConvBNAct(2 * in_channels, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Store for residual
        res = x
        
        # Local features
        x = self.conv1(x)
        x = self.conv2(x)  # (B, embed_dim, H, W)
        
        # Unfold into patches
        # Reshape: (B, embed_dim, H, W) -> (B, embed_dim, num_h, patch_h, num_w, patch_w)
        num_h = H // self.patch_h
        num_w = W // self.patch_w
        
        # Handle case where dimensions aren't divisible by patch size
        pad_h = (self.patch_h - H % self.patch_h) % self.patch_h
        pad_w = (self.patch_w - W % self.patch_w) % self.patch_w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            num_h = (H + pad_h) // self.patch_h
            num_w = (W + pad_w) // self.patch_w
        
        # Reshape for transformer: (B * num_h * num_w, patch_h * patch_w, embed_dim)
        x = x.reshape(B, self.embed_dim, num_h, self.patch_h, num_w, self.patch_w)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, num_h, num_w, patch_h, patch_w, embed_dim)
        x = x.reshape(B * num_h * num_w, self.patch_h * self.patch_w, self.embed_dim)
        
        # Transformer (global processing)
        x = self.transformer(x)
        
        # Fold back
        x = x.reshape(B, num_h, num_w, self.patch_h, self.patch_w, self.embed_dim)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, embed_dim, num_h, patch_h, num_w, patch_w)
        x = x.reshape(B, self.embed_dim, num_h * self.patch_h, num_w * self.patch_w)
        
        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        
        # Project and fuse with residual
        x = self.conv3(x)
        x = self.conv4(torch.cat([res, x], dim=1))
        
        return x


class MobileViT(nn.Module):
    """
    MobileViT model for audio spectrogram classification.
    
    Architecture:
    - Stem: Initial convolutions
    - Stage 1-2: MobileNetV2 blocks (local features)
    - Stage 3-5: MobileViT blocks (local + global features)
    - Head: Global pooling + classifier
    """
    
    def __init__(self, d_args: dict):
        """
        Initialize MobileViT.
        
        Args:
            d_args: Dictionary containing:
                - in_channels: Input channels (default: 1)
                - num_classes: Number of output classes (default: 2)
                - width_multiplier: Width scaling factor (default: 1.0)
                - img_size: Input size (H, W) (default: (128, 200))
                - dropout: Classifier dropout (default: 0.1)
        """
        super().__init__()
        
        # Configuration
        in_channels = d_args.get('in_channels', 1)
        num_classes = d_args.get('num_classes', 2)
        width_mult = d_args.get('width_multiplier', 1.0)
        self.dropout_rate = d_args.get('dropout', 0.1)
        
        # Base channel dimensions (scaled by width_multiplier)
        # Smaller than standard MobileViT for audio
        channels = {
            'stem': make_divisible(16 * width_mult),
            's1': make_divisible(32 * width_mult),
            's2': make_divisible(48 * width_mult),
            's3': make_divisible(64 * width_mult),
            's4': make_divisible(80 * width_mult),
            's5': make_divisible(96 * width_mult),
            'head': make_divisible(128 * width_mult),
        }
        
        embed_dims = {
            's3': make_divisible(48 * width_mult),
            's4': make_divisible(64 * width_mult),
            's5': make_divisible(80 * width_mult),
        }
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, channels['stem'], kernel_size=3, stride=2),
            InvertedResidual(channels['stem'], channels['s1'], stride=1),
        )
        
        # Stage 1: MobileNet blocks
        self.stage1 = nn.Sequential(
            InvertedResidual(channels['s1'], channels['s1'], stride=2),
            InvertedResidual(channels['s1'], channels['s1'], stride=1),
            InvertedResidual(channels['s1'], channels['s2'], stride=1),
        )
        
        # Stage 2: MobileNet blocks
        self.stage2 = nn.Sequential(
            InvertedResidual(channels['s2'], channels['s2'], stride=2),
            InvertedResidual(channels['s2'], channels['s3'], stride=1),
        )
        
        # Stage 3: MobileViT block
        self.stage3 = nn.Sequential(
            InvertedResidual(channels['s3'], channels['s3'], stride=2),
            MobileViTBlock(
                channels['s3'], channels['s4'],
                embed_dim=embed_dims['s3'],
                depth=2, num_heads=4, patch_size=(2, 2)
            ),
        )
        
        # Stage 4: MobileViT block
        self.stage4 = nn.Sequential(
            InvertedResidual(channels['s4'], channels['s4'], stride=2),
            MobileViTBlock(
                channels['s4'], channels['s5'],
                embed_dim=embed_dims['s4'],
                depth=2, num_heads=4, patch_size=(2, 2)
            ),
        )
        
        # Head
        self.head_conv = ConvBNAct(channels['s5'], channels['head'], kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(channels['head'], num_classes),
        )
        
        # Store embedding dimension
        self.embed_dim = channels['head']
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        return x
    
    def forward(self, x: torch.Tensor, Freq_aug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, freq, time) or (batch, freq, time)
            Freq_aug: Ignored, for compatibility
            
        Returns:
            (embeddings, logits)
        """
        # Handle input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Extract features
        embeddings = self.forward_features(x)
        
        # Classify
        logits = self.classifier(embeddings)
        
        return embeddings, logits


# For compatibility with main.py
Model = MobileViT


class MobileViTXS(MobileViT):
    """Extra-small MobileViT variant (~1.5M params)."""
    
    def __init__(self, d_args: dict):
        d_args = d_args.copy()
        d_args.setdefault('width_multiplier', 0.5)
        super().__init__(d_args)


class MobileViTS(MobileViT):
    """Small MobileViT variant (~2.5M params)."""
    
    def __init__(self, d_args: dict):
        d_args = d_args.copy()
        d_args.setdefault('width_multiplier', 0.75)
        super().__init__(d_args)
