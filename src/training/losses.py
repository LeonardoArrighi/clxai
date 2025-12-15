"""
Supervised Contrastive Learning loss implementation.
Based on: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    For each anchor, positives are samples with the same label,
    negatives are samples with different labels.
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature for softmax scaling
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SupCon loss.
        
        Args:
            features: Normalized embeddings of shape (2*B, D) where
                     first half are view1, second half are view2
            labels: Labels of shape (B,)
            mask: Optional contrastive mask of shape (B, B)
        
        Returns:
            Loss value
        """
        device = features.device
        batch_size = labels.shape[0]
        
        # Duplicate labels for both views
        labels = labels.contiguous().view(-1, 1)
        
        # Create mask: 1 where labels match, 0 otherwise
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Number of views (2 for two augmented views)
        contrast_count = features.shape[0] // batch_size
        
        # Split features into anchor and contrast
        contrast_feature = features
        anchor_feature = features
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask for multiple views
        mask = mask.repeat(contrast_count, contrast_count)
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # Avoid division by zero
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(
            mask_pos_pairs < 1e-6,
            torch.ones_like(mask_pos_pairs),
            mask_pos_pairs
        )
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss


class SupConLossV2(nn.Module):
    """
    Simplified Supervised Contrastive Loss implementation.
    More numerically stable version.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Concatenated features from both views [z1; z2], shape (2B, D)
            labels: Labels for one batch, shape (B,)
        """
        device = features.device
        batch_size = labels.shape[0]
        
        # Features are already L2 normalized
        # Compute all pairwise similarities
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for both views
        labels_full = labels.repeat(2)
        
        # Positive mask: same label (excluding self)
        pos_mask = torch.eq(labels_full.unsqueeze(0), labels_full.unsqueeze(1)).float()
        
        # Remove diagonal (self-similarity)
        self_mask = torch.eye(2 * batch_size, device=device)
        pos_mask = pos_mask - self_mask
        
        # For numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Compute log softmax
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # Mean log prob for positive pairs
        num_positives = pos_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / num_positives
        
        # Loss is negative mean log probability
        loss = -mean_log_prob_pos.mean()
        
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy).
    Used in SimCLR for self-supervised contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Embeddings from view 1, shape (B, D)
            z2: Embeddings from view 2, shape (B, D)
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # Create positive pair mask
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=device)
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim = sim.masked_fill(mask, -float('inf'))
        
        # Compute loss
        # For each sample, positive is the augmented version
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device)
        ])
        
        loss = F.cross_entropy(sim, labels)
        
        return loss


if __name__ == "__main__":
    # Test losses
    batch_size = 32
    dim = 128
    n_classes = 10
    
    # Random features and labels
    z1 = F.normalize(torch.randn(batch_size, dim), dim=1)
    z2 = F.normalize(torch.randn(batch_size, dim), dim=1)
    labels = torch.randint(0, n_classes, (batch_size,))
    
    # Test SupConLoss
    features = torch.cat([z1, z2], dim=0)
    supcon = SupConLoss(temperature=0.07)
    loss = supcon(features, labels)
    print(f"SupCon Loss: {loss.item():.4f}")
    
    # Test SupConLossV2
    supcon_v2 = SupConLossV2(temperature=0.07)
    loss_v2 = supcon_v2(features, labels)
    print(f"SupCon V2 Loss: {loss_v2.item():.4f}")
    
    # Test NT-Xent
    ntxent = NTXentLoss(temperature=0.5)
    loss_ntxent = ntxent(z1, z2)
    print(f"NT-Xent Loss: {loss_ntxent.item():.4f}")
