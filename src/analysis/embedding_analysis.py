"""
Embedding space analysis for SCL vs CE comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from tqdm import tqdm


class EmbeddingAnalyzer:
    """Analyze embedding space behavior under perturbation."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', n_classes: int = 10):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.n_classes = n_classes
        self.centroids = None
    
    @torch.no_grad()
    def compute_centroids(self, data_loader) -> np.ndarray:
        """Compute class centroids from data loader."""
        embeddings_by_class = {c: [] for c in range(self.n_classes)}
        
        for images, labels in tqdm(data_loader, desc="Computing centroids"):
            if isinstance(images, list):
                images = images[0]
            images = images.to(self.device)
            emb = self.model.get_embedding(images, normalize=True).cpu().numpy()
            
            for e, l in zip(emb, labels.numpy()):
                embeddings_by_class[int(l)].append(e)
        
        centroids = []
        for c in range(self.n_classes):
            if embeddings_by_class[c]:
                centroids.append(np.stack(embeddings_by_class[c]).mean(axis=0))
        
        self.centroids = np.stack(centroids)
        return self.centroids
    
    @torch.no_grad()
    def get_embedding(self, image: torch.Tensor) -> np.ndarray:
        """Get embedding for single image."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        return self.model.get_embedding(image, normalize=True).cpu().numpy()[0]
    
    def compute_embedding_trajectory(
        self, image: torch.Tensor, saliency: np.ndarray,
        n_steps: int = 20, perturbation_fn=None
    ) -> Dict[str, np.ndarray]:
        """Compute embedding trajectory under progressive perturbation."""
        if perturbation_fn is None:
            from src.xai.perturbations import MeanPerturbation
            perturbation_fn = MeanPerturbation()
        
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        original_emb = self.get_embedding(image)
        
        fractions = np.linspace(0, 1, n_steps + 1)
        embeddings = []
        
        for frac in fractions:
            n_pixels = saliency_norm.size
            n_perturb = int(frac * n_pixels)
            sorted_idx = np.argsort(saliency_norm.flatten())[::-1]
            
            mask = np.zeros(n_pixels, dtype=np.float32)
            if n_perturb > 0:
                mask[sorted_idx[:n_perturb]] = 1.0
            mask = torch.from_numpy(mask.reshape(saliency_norm.shape))
            
            perturbed = perturbation_fn(image, mask)
            if isinstance(perturbed, np.ndarray):
                perturbed = torch.from_numpy(perturbed)
            
            embeddings.append(self.get_embedding(perturbed))
        
        embeddings = np.stack(embeddings)
        drift = np.linalg.norm(embeddings - original_emb, axis=1)
        
        velocity = np.diff(embeddings, axis=0)
        accel = np.diff(velocity, axis=0)
        smoothness = np.linalg.norm(accel, axis=1)
        
        return {
            'fractions': fractions,
            'embeddings': embeddings,
            'original_embedding': original_emb,
            'embedding_drift': drift,
            'smoothness': smoothness,
            'mean_smoothness': smoothness.mean() if len(smoothness) > 0 else 0.0
        }
    
    def compute_cluster_distance(self, embeddings: np.ndarray, target_class: int) -> np.ndarray:
        """Compute distance from embeddings to target class centroid."""
        if self.centroids is None:
            raise RuntimeError("Call compute_centroids first")
        return np.linalg.norm(embeddings - self.centroids[target_class], axis=1)


def analyze_embedding_trajectory(
    model, data_loader, saliency_extractor,
    saliency_method='integrated_grad', perturbation='mean',
    n_samples=100, n_steps=20, device='cuda'
) -> Dict[str, np.ndarray]:
    """Analyze embedding trajectories for multiple samples."""
    from src.xai.perturbations import get_perturbation
    
    analyzer = EmbeddingAnalyzer(model, device=device)
    analyzer.compute_centroids(data_loader)
    pert_fn = get_perturbation(perturbation)
    
    all_drifts, all_smoothness = [], []
    count = 0
    
    for images, labels in tqdm(data_loader, desc="Analyzing"):
        for i in range(images.shape[0]):
            if count >= n_samples:
                break
            image, label = images[i], labels[i].item()
            saliency = saliency_extractor.extract(image, target_class=label, method=saliency_method)
            traj = analyzer.compute_embedding_trajectory(image, saliency, n_steps, pert_fn)
            all_drifts.append(traj['embedding_drift'])
            all_smoothness.append(traj['mean_smoothness'])
            count += 1
        if count >= n_samples:
            break
    
    drifts = np.stack(all_drifts)
    smoothness = np.array(all_smoothness)
    
    return {
        'embedding_drifts': drifts,
        'mean_drift_curve': drifts.mean(axis=0),
        'smoothness_scores': smoothness,
        'mean_smoothness': smoothness.mean(),
        'fractions': np.linspace(0, 1, n_steps + 1)
    }
