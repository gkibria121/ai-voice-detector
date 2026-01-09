import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureCNN(nn.Module):
    """A small CNN for each feature type."""
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # Pool freq axis
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.cnn(x)  # (B, 64, 1, T)
        x = x.squeeze(2).transpose(1, 2)  # (B, T, 64)
        x = self.proj(x)  # (B, T, out_dim)
        return x

class AttentionFusion(nn.Module):
    """Self-attention fusion for multimodal embeddings."""
    def __init__(self, embed_dim, n_modalities):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.n_modalities = n_modalities

    def forward(self, feats):
        # feats: list of (B, T, D) for each modality
        x = torch.stack(feats, dim=2)  # (B, T, M, D)
        B, T, M, D = x.shape
        x = x.view(B * T, M, D)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.softmax((Q @ K.transpose(-2, -1)) / (D ** 0.5), dim=-1)  # (B*T, M, M)
        fused = (attn @ V).mean(dim=1)  # (B*T, D)
        fused = fused.view(B, T, D)
        return fused


# For compatibility with main.py, expose as Model
class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        # d_args should contain: in_shapes, embed_dim, n_classes
        in_shapes = d_args.get('in_shapes', [(1,128,200), (1,13,200), (1,84,200)])
        embed_dim = d_args.get('embed_dim', 128)
        n_classes = d_args.get('n_classes', 2)
        self.n_modalities = len(in_shapes)
        self.extractors = nn.ModuleList([
            FeatureCNN(ch, embed_dim) for (ch, _, _) in in_shapes
        ])
        self.fusion = AttentionFusion(embed_dim, self.n_modalities)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, xs, **kwargs):
        # xs: list of tensors, each (B, C, F, T)
        # Accept and ignore any extra keyword arguments for compatibility
        feats = [extractor(x) for extractor, x in zip(self.extractors, xs)]  # each (B, T, D)
        fused = self.fusion(feats)  # (B, T, D)
        pooled = self.pool(fused.transpose(1, 2)).squeeze(-1)  # (B, D)
        out = self.classifier(pooled)
        # For compatibility with main.py, return (embeddings, logits)
        return pooled, out

# Example usage:
# model = FusionNet([(1, 128, 200), (1, 13, 200), (1, 84, 200)])
# xs = [torch.randn(8, 1, 128, 200), torch.randn(8, 1, 13, 200), torch.randn(8, 1, 84, 200)]
# logits = model(xs)
