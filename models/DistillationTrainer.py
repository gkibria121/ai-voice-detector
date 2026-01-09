"""
Knowledge Distillation Trainer for Model Compression

This module provides knowledge distillation utilities to train small, efficient
student models from larger, more accurate teacher models.

Benefits:
- Transfer knowledge from large models to lightweight ones
- Maintain accuracy while reducing model size and latency
- Enable deployment on edge devices

Reference:
    Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining hard and soft targets.
    
    L = α * L_hard + (1-α) * T² * L_soft
    
    where:
    - L_hard: Cross-entropy with true labels
    - L_soft: KL divergence with teacher soft labels
    - T: Temperature for softening probabilities
    - α: Weight for hard loss (1-α for soft loss)
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        reduction: str = 'mean',
    ):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Softmax temperature for soft targets (higher = softer)
            alpha: Weight for hard label loss (0-1)
            reduction: Loss reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.reduction = reduction
        
        self.hard_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model output logits
            teacher_logits: Teacher model output logits
            labels: Ground truth labels
            
        Returns:
            (total_loss, loss_dict) with individual loss components
        """
        # Hard loss (student vs true labels)
        hard_loss = self.hard_loss(student_logits, labels)
        
        # Soft loss (student vs teacher soft labels)
        # Use log_softmax for student, softmax for teacher (KL divergence input)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = self.kl_loss(student_soft, teacher_soft)
        # Scale by T² as per Hinton et al.
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
        }
        
        return total_loss, loss_dict


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss.
    
    Matches intermediate feature representations between teacher and student.
    Useful when teacher and student have different architectures.
    """
    
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        use_projector: bool = True,
    ):
        """
        Initialize feature distillation loss.
        
        Args:
            teacher_dim: Teacher feature dimension
            student_dim: Student feature dimension  
            use_projector: Whether to use projection layer to match dimensions
        """
        super().__init__()
        
        self.use_projector = use_projector
        if use_projector and teacher_dim != student_dim:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            student_features: Student intermediate features
            teacher_features: Teacher intermediate features
            
        Returns:
            MSE loss between features
        """
        # Project student features if needed
        student_proj = self.projector(student_features)
        
        # Normalize features
        student_norm = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_features, dim=-1)
        
        # MSE loss
        return F.mse_loss(student_norm, teacher_norm)


class DistillationTrainer:
    """
    Training wrapper for knowledge distillation.
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_weight: float = 0.0,
        teacher_dim: Optional[int] = None,
        student_dim: Optional[int] = None,
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher: Pre-trained teacher model (will be frozen)
            student: Student model to train
            temperature: Distillation temperature
            alpha: Hard loss weight
            feature_weight: Weight for feature distillation (0 = disabled)
            teacher_dim: Teacher embedding dimension (for feature distillation)
            student_dim: Student embedding dimension (for feature distillation)
        """
        self.teacher = teacher
        self.student = student
        self.feature_weight = feature_weight
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Losses
        self.distill_loss = DistillationLoss(temperature, alpha)
        
        if feature_weight > 0 and teacher_dim and student_dim:
            self.feature_loss = FeatureDistillationLoss(teacher_dim, student_dim)
        else:
            self.feature_loss = None
    
    @torch.no_grad()
    def _get_teacher_outputs(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get teacher embeddings and logits."""
        self.teacher.eval()
        embeddings, logits = self.teacher(x)
        return embeddings, logits
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform one distillation training step.
        
        Args:
            x: Input batch
            y: Labels
            optimizer: Student optimizer
            
        Returns:
            Dictionary with loss values
        """
        optimizer.zero_grad()
        
        # Get teacher outputs (no grad)
        teacher_embeddings, teacher_logits = self._get_teacher_outputs(x)
        
        # Get student outputs
        student_embeddings, student_logits = self.student(x)
        
        # Compute distillation loss
        total_loss, loss_dict = self.distill_loss(student_logits, teacher_logits, y)
        
        # Add feature distillation if enabled
        if self.feature_loss is not None and self.feature_weight > 0:
            feat_loss = self.feature_loss(student_embeddings, teacher_embeddings)
            total_loss = total_loss + self.feature_weight * feat_loss
            loss_dict['feature_loss'] = feat_loss.item()
        
        # Backward and update
        total_loss.backward()
        optimizer.step()
        
        return loss_dict
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Evaluate student model.
        
        Args:
            dataloader: Evaluation data loader
            device: Device for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.student.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                _, logits = self.student(batch_x)
                
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                
                preds = logits.argmax(dim=-1)
                total_correct += (preds == batch_y).sum().item()
                total_samples += batch_x.size(0)
        
        self.student.train()
        
        return {
            'accuracy': total_correct / total_samples,
            'loss': total_loss / total_samples,
        }


def distill_model(
    teacher: nn.Module,
    student: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = 20,
    lr: float = 1e-3,
    temperature: float = 4.0,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Complete distillation training loop.
    
    Args:
        teacher: Pre-trained teacher model
        student: Student model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Training device
        num_epochs: Number of training epochs
        lr: Learning rate
        temperature: Distillation temperature
        alpha: Hard loss weight
        save_path: Path to save best student weights
        
    Returns:
        Dictionary with final metrics
    """
    teacher = teacher.to(device)
    student = student.to(device)
    
    trainer = DistillationTrainer(teacher, student, temperature, alpha)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            loss_dict = trainer.train_step(batch_x, batch_y, optimizer)
            epoch_loss += loss_dict['total_loss']
        
        scheduler.step()
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            if save_path:
                torch.save(student.state_dict(), save_path)
    
    return {'best_accuracy': best_acc}
