"""
Domain Adaptation Module for Cross-Dataset Generalization

This module provides domain adaptation techniques to improve model performance
when transferring from one dataset (source) to another (target).

Methods implemented:
1. CORAL (Correlation Alignment) - Aligns second-order statistics
2. MMD (Maximum Mean Discrepancy) - Kernel-based distribution matching
3. DANN (Domain-Adversarial Neural Network) - Adversarial domain confusion

These methods help the model generalize across different:
- Recording conditions
- Speaker populations
- Spoofing attack types
- Background noise environments

Reference:
    - Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
    - Long et al., "Learning Transferable Features with Deep Adaptation Networks"
    - Ganin et al., "Domain-Adversarial Training of Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CORALLoss(nn.Module):
    """
    CORAL (Correlation Alignment) Loss for domain adaptation.
    
    Aligns the second-order statistics (covariances) of source and target
    domain feature distributions.
    
    Reference:
        Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CORAL loss between source and target features.
        
        Args:
            source: Source domain features (batch_s, feature_dim)
            target: Target domain features (batch_t, feature_dim)
            
        Returns:
            CORAL loss (scalar)
        """
        d = source.size(1)  # Feature dimension
        
        # Compute covariance matrices
        source_cov = self._covariance(source)
        target_cov = self._covariance(target)
        
        # Frobenius norm of difference
        loss = torch.sum((source_cov - target_cov) ** 2)
        
        # Normalize by dimension squared
        loss = loss / (4 * d * d)
        
        return loss
    
    def _covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix."""
        n = x.size(0)
        if n == 1:
            return torch.zeros(x.size(1), x.size(1), device=x.device)
        
        # Center the features
        x_centered = x - x.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov = (x_centered.T @ x_centered) / (n - 1)
        
        return cov


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss for domain adaptation.
    
    Measures the distance between source and target distributions in a
    reproducing kernel Hilbert space (RKHS).
    
    Reference:
        Long et al., "Learning Transferable Features with Deep Adaptation Networks"
    """
    
    def __init__(self, kernel: str = 'rbf', gamma: Optional[float] = None):
        """
        Initialize MMD loss.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            gamma: RBF kernel bandwidth (if None, uses median heuristic)
        """
        super().__init__()
        self.kernel = kernel
        self.gamma = gamma
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between source and target features.
        
        Args:
            source: Source domain features (batch_s, feature_dim)
            target: Target domain features (batch_t, feature_dim)
            
        Returns:
            MMD loss (scalar)
        """
        # Compute kernel matrices
        K_ss = self._kernel_matrix(source, source)
        K_tt = self._kernel_matrix(target, target)
        K_st = self._kernel_matrix(source, target)
        
        n_s = source.size(0)
        n_t = target.size(0)
        
        # Unbiased MMD^2 estimator
        # Exclude diagonal for unbiased estimate
        mmd = (
            (K_ss.sum() - K_ss.diag().sum()) / (n_s * (n_s - 1)) +
            (K_tt.sum() - K_tt.diag().sum()) / (n_t * (n_t - 1)) -
            2 * K_st.mean()
        )
        
        return torch.clamp(mmd, min=0.0)  # MMD should be non-negative
    
    def _kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix between x and y."""
        if self.kernel == 'linear':
            return x @ y.T
        
        elif self.kernel == 'rbf':
            # Compute pairwise squared distances
            xx = (x ** 2).sum(dim=1, keepdim=True)
            yy = (y ** 2).sum(dim=1, keepdim=True)
            distances = xx + yy.T - 2 * (x @ y.T)
            
            # Compute gamma using median heuristic if not specified
            if self.gamma is None:
                gamma = 1.0 / (2 * torch.median(distances).clamp(min=1e-6))
            else:
                gamma = self.gamma
            
            return torch.exp(-gamma * distances)
        
        elif self.kernel == 'poly':
            return (x @ y.T + 1) ** 3
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")


class MultiKernelMMD(nn.Module):
    """
    Multi-Kernel MMD (MK-MMD) with multiple bandwidths for robust adaptation.
    """
    
    def __init__(self, gammas: List[float] = None):
        """
        Initialize MK-MMD.
        
        Args:
            gammas: List of RBF bandwidths (if None, uses default set)
        """
        super().__init__()
        
        if gammas is None:
            # Default bandwidth set (powers of 2)
            self.gammas = [2**i for i in range(-3, 4)]  # [0.125, 0.25, 0.5, 1, 2, 4, 8]
        else:
            self.gammas = gammas
        
        self.mmd_losses = nn.ModuleList([
            MMDLoss(kernel='rbf', gamma=g) for g in self.gammas
        ])
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-kernel MMD."""
        total_loss = 0
        for mmd_loss in self.mmd_losses:
            total_loss = total_loss + mmd_loss(source, target)
        return total_loss / len(self.mmd_losses)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial domain adaptation.
    
    During forward pass, acts as identity.
    During backward pass, reverses and scales the gradient.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Wraps GradientReversalFunction as a module.
    """
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation (DANN).
    
    Classifies whether features come from source or target domain.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        """
        Initialize domain discriminator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict domain (0 = source, 1 = target).
        
        Returns logits (use BCEWithLogitsLoss).
        """
        return self.discriminator(x)


class DANNModule(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN) module.
    
    Combines gradient reversal layer with domain discriminator for
    adversarial domain adaptation.
    
    Usage:
        dann = DANNModule(feature_dim=512)
        
        # In training loop:
        features = encoder(x)
        domain_logits = dann(features)
        domain_loss = dann.compute_loss(domain_logits, domain_labels)
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256, initial_lambda: float = 0.0):
        """
        Initialize DANN module.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for discriminator
            initial_lambda: Initial gradient reversal strength (typically 0)
        """
        super().__init__()
        
        self.grl = GradientReversalLayer(initial_lambda)
        self.discriminator = DomainDiscriminator(feature_dim, hidden_dim)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRL and discriminator."""
        reversed_features = self.grl(features)
        domain_logits = self.discriminator(reversed_features)
        return domain_logits
    
    def compute_loss(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain adversarial loss.
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            
        Returns:
            Domain classification loss
        """
        # Create domain labels
        source_labels = torch.zeros(source_features.size(0), 1, device=source_features.device)
        target_labels = torch.ones(target_features.size(0), 1, device=target_features.device)
        
        # Combine features and labels
        features = torch.cat([source_features, target_features], dim=0)
        labels = torch.cat([source_labels, target_labels], dim=0)
        
        # Forward through DANN
        logits = self.forward(features)
        
        return self.criterion(logits, labels)
    
    def update_lambda(self, progress: float, gamma: float = 10.0):
        """
        Update GRL lambda based on training progress.
        
        Uses schedule from original DANN paper:
        lambda = 2 / (1 + exp(-gamma * p)) - 1
        
        Args:
            progress: Training progress in [0, 1]
            gamma: Schedule steepness
        """
        import math
        lambda_ = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
        self.grl.set_lambda(lambda_)


class DomainAdaptationTrainer:
    """
    Training wrapper for domain adaptation.
    
    Combines task loss with domain adaptation loss for training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        adaptation_method: str = 'coral',
        lambda_domain: float = 0.1,
        feature_extractor_layer: str = 'embeddings',
    ):
        """
        Initialize domain adaptation trainer.
        
        Args:
            model: Base model to adapt
            adaptation_method: 'coral', 'mmd', 'mk_mmd', or 'dann'
            lambda_domain: Weight for domain adaptation loss
            feature_extractor_layer: Which layer to extract features from
        """
        self.model = model
        self.adaptation_method = adaptation_method
        self.lambda_domain = lambda_domain
        self.feature_extractor_layer = feature_extractor_layer
        
        # Initialize adaptation loss
        if adaptation_method == 'coral':
            self.domain_loss = CORALLoss()
        elif adaptation_method == 'mmd':
            self.domain_loss = MMDLoss()
        elif adaptation_method == 'mk_mmd':
            self.domain_loss = MultiKernelMMD()
        elif adaptation_method == 'dann':
            # DANN requires feature dimension - will be set on first forward
            self.domain_loss = None
            self.dann_initialized = False
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
    
    def _init_dann(self, feature_dim: int, device: torch.device):
        """Initialize DANN module with correct feature dimension."""
        self.domain_loss = DANNModule(feature_dim).to(device)
        self.dann_initialized = True
    
    def compute_adaptation_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        progress: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute domain adaptation loss.
        
        Args:
            source_features: Features from source domain
            target_features: Features from target domain
            progress: Training progress in [0, 1] (for DANN scheduling)
            
        Returns:
            Domain adaptation loss
        """
        if self.adaptation_method == 'dann':
            if not self.dann_initialized:
                self._init_dann(source_features.size(1), source_features.device)
            self.domain_loss.update_lambda(progress)
            return self.domain_loss.compute_loss(source_features, target_features)
        else:
            return self.domain_loss(source_features, target_features)
    
    def train_step(
        self,
        source_batch: Tuple[torch.Tensor, torch.Tensor],
        target_batch: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        progress: float = 0.5,
    ) -> dict:
        """
        Perform one training step with domain adaptation.
        
        Args:
            source_batch: (source_x, source_y) from labeled source domain
            target_batch: target_x from unlabeled target domain
            criterion: Task loss function
            optimizer: Optimizer
            progress: Training progress in [0, 1]
            
        Returns:
            Dictionary with loss values
        """
        source_x, source_y = source_batch
        target_x = target_batch
        
        optimizer.zero_grad()
        
        # Forward pass on source (with labels)
        source_embeddings, source_logits = self.model(source_x)
        task_loss = criterion(source_logits, source_y)
        
        # Forward pass on target (no labels, just for features)
        target_embeddings, _ = self.model(target_x)
        
        # Compute domain adaptation loss
        domain_loss = self.compute_adaptation_loss(
            source_embeddings, target_embeddings, progress
        )
        
        # Total loss
        total_loss = task_loss + self.lambda_domain * domain_loss
        
        # Backward and update
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'domain_loss': domain_loss.item(),
        }


class FewShotAdapter(nn.Module):
    """
    Few-shot domain adaptation using prototypical learning.
    
    Adapts a pre-trained model to a new domain using only a few
    labeled examples.
    """
    
    def __init__(self, model: nn.Module, feature_dim: int, num_classes: int = 2):
        """
        Initialize few-shot adapter.
        
        Args:
            model: Pre-trained feature extractor
            feature_dim: Dimension of extracted features
            num_classes: Number of classes
        """
        super().__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Class prototypes (will be computed from support set)
        self.prototypes = nn.Parameter(
            torch.zeros(num_classes, feature_dim), requires_grad=False
        )
    
    def compute_prototypes(self, support_x: torch.Tensor, support_y: torch.Tensor):
        """
        Compute class prototypes from support set.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
        """
        with torch.no_grad():
            embeddings, _ = self.model(support_x)
            
            for c in range(self.num_classes):
                mask = (support_y == c)
                if mask.sum() > 0:
                    self.prototypes.data[c] = embeddings[mask].mean(dim=0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify based on distance to prototypes.
        
        Args:
            x: Input tensor
            
        Returns:
            (embeddings, logits)
        """
        embeddings, _ = self.model(x)
        
        # Compute distances to prototypes
        distances = torch.cdist(embeddings, self.prototypes.unsqueeze(0)).squeeze(0)
        
        # Convert distances to logits (negative distance = higher score)
        logits = -distances
        
        return embeddings, logits
