"""
Pixel flipping protocol for faithfulness evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm

from .saliency import SaliencyExtractor, get_importance_ranking, normalize_saliency
from .perturbations import BasePerturbation, MeanPerturbation, get_perturbation


class PixelFlipping:
    """
    Pixel flipping evaluation for saliency map faithfulness.
    
    Incrementally removes pixels in order of importance and measures
    the effect on model predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        perturbation: Union[str, BasePerturbation] = 'mean',
        n_steps: int = 20,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Model to evaluate
            perturbation: Perturbation strategy ('mean', 'noise', 'blur') or instance
            n_steps: Number of perturbation steps (0%, 5%, 10%, ..., 100%)
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.n_steps = n_steps
        
        if isinstance(perturbation, str):
            self.perturbation = get_perturbation(perturbation)
        else:
            self.perturbation = perturbation
    
    def _create_mask(
        self,
        saliency: np.ndarray,
        fraction: float,
        mode: str = 'deletion'
    ) -> torch.Tensor:
        """
        Create binary mask for given perturbation fraction.
        
        Args:
            saliency: Saliency map (H, W)
            fraction: Fraction of pixels to perturb (0-1)
            mode: 'deletion' (remove important first) or 'insertion' (add important first)
        
        Returns:
            Binary mask tensor (H, W)
        """
        n_pixels = saliency.size
        n_perturb = int(fraction * n_pixels)
        
        # Get importance ranking
        importance = saliency.flatten()
        sorted_indices = np.argsort(importance)
        
        if mode == 'deletion':
            # Most important pixels first
            indices_to_perturb = sorted_indices[-n_perturb:] if n_perturb > 0 else []
        else:  # insertion
            # Least important pixels first (will be revealed)
            indices_to_perturb = sorted_indices[:n_pixels - n_perturb] if n_perturb < n_pixels else []
        
        mask = np.zeros(n_pixels, dtype=np.float32)
        if len(indices_to_perturb) > 0:
            mask[indices_to_perturb] = 1.0
        
        return torch.from_numpy(mask.reshape(saliency.shape))
    
    def evaluate_deletion(
        self,
        image: torch.Tensor,
        saliency: np.ndarray,
        target_class: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate faithfulness using deletion metric.
        Remove pixels in order of importance (most important first).
        
        Args:
            image: Input image tensor (C, H, W)
            saliency: Saliency map (H, W)
            target_class: Target class for evaluation
        
        Returns:
            Dictionary with deletion curve and predictions
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Normalize saliency
        saliency = normalize_saliency(saliency, method='minmax')
        
        # Compute predictions at each perturbation level
        fractions = np.linspace(0, 1, self.n_steps + 1)
        predictions = []
        probabilities = []
        
        for frac in fractions:
            mask = self._create_mask(saliency, frac, mode='deletion')
            perturbed = self.perturbation(image.squeeze(0), mask)
            perturbed = perturbed.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(perturbed)
                probs = F.softmax(output, dim=1)
                predictions.append(output.argmax(dim=1).item())
                probabilities.append(probs[0, target_class].item())
        
        return {
            'fractions': fractions,
            'target_probs': np.array(probabilities),
            'predictions': np.array(predictions),
            'target_class': target_class
        }
    
    def evaluate_insertion(
        self,
        image: torch.Tensor,
        saliency: np.ndarray,
        target_class: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate faithfulness using insertion metric.
        Add pixels in order of importance (most important first).
        
        Args:
            image: Input image tensor (C, H, W)
            saliency: Saliency map (H, W)
            target_class: Target class for evaluation
        
        Returns:
            Dictionary with insertion curve and predictions
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        saliency = normalize_saliency(saliency, method='minmax')
        
        fractions = np.linspace(0, 1, self.n_steps + 1)
        predictions = []
        probabilities = []
        
        for frac in fractions:
            # For insertion, we start with all pixels perturbed and reveal important ones
            mask = self._create_mask(saliency, frac, mode='insertion')
            perturbed = self.perturbation(image.squeeze(0), mask)
            perturbed = perturbed.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(perturbed)
                probs = F.softmax(output, dim=1)
                predictions.append(output.argmax(dim=1).item())
                probabilities.append(probs[0, target_class].item())
        
        return {
            'fractions': fractions,
            'target_probs': np.array(probabilities),
            'predictions': np.array(predictions),
            'target_class': target_class
        }
    
    def evaluate_both(
        self,
        image: torch.Tensor,
        saliency: np.ndarray,
        target_class: Optional[int] = None
    ) -> Dict[str, Dict]:
        """Evaluate both deletion and insertion metrics."""
        deletion = self.evaluate_deletion(image, saliency, target_class)
        insertion = self.evaluate_insertion(image, saliency, deletion['target_class'])
        
        return {
            'deletion': deletion,
            'insertion': insertion
        }


def run_pixel_flipping_experiment(
    model: nn.Module,
    data_loader,
    saliency_extractor: SaliencyExtractor,
    perturbation: str = 'mean',
    saliency_method: str = 'integrated_grad',
    n_samples: Optional[int] = None,
    n_steps: int = 20,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    Run pixel flipping experiment on dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with test images
        saliency_extractor: SaliencyExtractor instance
        perturbation: Perturbation strategy
        saliency_method: Saliency method to use
        n_samples: Number of samples to evaluate (None = all)
        n_steps: Number of perturbation steps
        device: Device to use
    
    Returns:
        Dictionary with aggregated results
    """
    pixel_flipper = PixelFlipping(
        model=model,
        perturbation=perturbation,
        n_steps=n_steps,
        device=device
    )
    
    all_deletion_curves = []
    all_insertion_curves = []
    
    sample_count = 0
    
    for images, labels in tqdm(data_loader, desc="Pixel flipping"):
        for i in range(images.shape[0]):
            if n_samples is not None and sample_count >= n_samples:
                break
            
            image = images[i]
            label = labels[i].item()
            
            # Get saliency map
            saliency = saliency_extractor.extract(
                image, target_class=label, method=saliency_method
            )
            
            # Evaluate
            results = pixel_flipper.evaluate_both(image, saliency, target_class=label)
            
            all_deletion_curves.append(results['deletion']['target_probs'])
            all_insertion_curves.append(results['insertion']['target_probs'])
            
            sample_count += 1
        
        if n_samples is not None and sample_count >= n_samples:
            break
    
    deletion_curves = np.stack(all_deletion_curves)
    insertion_curves = np.stack(all_insertion_curves)
    
    # Compute AUC
    fractions = np.linspace(0, 1, n_steps + 1)
    deletion_auc = np.trapz(deletion_curves, fractions, axis=1)
    insertion_auc = np.trapz(insertion_curves, fractions, axis=1)
    
    return {
        'deletion_curves': deletion_curves,
        'insertion_curves': insertion_curves,
        'deletion_auc': deletion_auc,
        'insertion_auc': insertion_auc,
        'fractions': fractions,
        'mean_deletion_curve': deletion_curves.mean(axis=0),
        'mean_insertion_curve': insertion_curves.mean(axis=0),
        'mean_deletion_auc': deletion_auc.mean(),
        'mean_insertion_auc': insertion_auc.mean(),
        'std_deletion_auc': deletion_auc.std(),
        'std_insertion_auc': insertion_auc.std()
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.resnet import ResNet18
    
    # Test pixel flipping
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(num_classes=10).to(device)
    
    # Random input
    image = torch.randn(3, 32, 32)
    saliency = np.random.rand(32, 32)
    
    flipper = PixelFlipping(model, perturbation='mean', n_steps=10, device=device)
    
    results = flipper.evaluate_both(image, saliency)
    
    print("Deletion curve:", results['deletion']['target_probs'])
    print("Insertion curve:", results['insertion']['target_probs'])
