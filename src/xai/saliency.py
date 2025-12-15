"""
Saliency map extraction using Captum.
Implements multiple XAI methods for faithfulness comparison.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Callable, Union
from captum.attr import (
    Saliency,
    IntegratedGradients,
    GuidedGradCam,
    LayerGradCam,
    DeepLift,
    LRP
)


class SaliencyExtractor:
    """
    Unified interface for extracting saliency maps using different methods.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        target_layer: Optional[nn.Module] = None
    ):
        """
        Args:
            model: PyTorch model for explanation
            device: Device to use
            target_layer: Target layer for GradCAM (usually last conv layer)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_layer = target_layer
        
        # Initialize attribution methods
        self._init_methods()
    
    def _init_methods(self):
        """Initialize Captum attribution methods."""
        self.methods = {}
        
        # Gradient-based methods
        self.methods['vanilla_grad'] = Saliency(self.model)
        self.methods['integrated_grad'] = IntegratedGradients(self.model)
        
        # Deep learning specific
        self.methods['deeplift'] = DeepLift(self.model)
        
        # GradCAM requires target layer
        if self.target_layer is not None:
            self.methods['gradcam'] = LayerGradCam(self.model, self.target_layer)
    
    def get_available_methods(self) -> List[str]:
        """Return list of available saliency methods."""
        return list(self.methods.keys())
    
    def extract(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = 'integrated_grad',
        **kwargs
    ) -> np.ndarray:
        """
        Extract saliency map for given input.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W) or (C, H, W)
            target_class: Target class for attribution. If None, uses predicted class
            method: Attribution method to use
            **kwargs: Additional arguments for specific methods
        
        Returns:
            Saliency map as numpy array (H, W)
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Get attribution method
        if method not in self.methods:
            raise ValueError(f"Method '{method}' not available. Choose from {self.get_available_methods()}")
        
        attr_method = self.methods[method]
        
        # Compute attribution
        if method == 'vanilla_grad':
            attributions = attr_method.attribute(input_tensor, target=target_class)
        
        elif method == 'integrated_grad':
            n_steps = kwargs.get('n_steps', 50)
            baselines = kwargs.get('baselines', torch.zeros_like(input_tensor))
            attributions = attr_method.attribute(
                input_tensor,
                baselines=baselines.to(self.device),
                target=target_class,
                n_steps=n_steps
            )
        
        elif method == 'deeplift':
            baselines = kwargs.get('baselines', torch.zeros_like(input_tensor))
            attributions = attr_method.attribute(
                input_tensor,
                baselines=baselines.to(self.device),
                target=target_class
            )
        
        elif method == 'gradcam':
            attributions = attr_method.attribute(input_tensor, target=target_class)
            # GradCAM returns spatial attribution, need to upsample
            if attributions.shape[-2:] != input_tensor.shape[-2:]:
                attributions = torch.nn.functional.interpolate(
                    attributions,
                    size=input_tensor.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
        
        else:
            attributions = attr_method.attribute(input_tensor, target=target_class)
        
        # Convert to numpy and aggregate channels
        saliency = attributions.detach().cpu().numpy()
        
        # Sum over channels and take absolute value
        if saliency.ndim == 4:
            saliency = np.abs(saliency).sum(axis=1)  # (B, H, W)
        
        if saliency.ndim == 3:
            saliency = saliency[0]  # (H, W)
        
        return saliency
    
    def extract_all(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Extract saliency maps using all (or specified) methods.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class for attribution
            methods: List of methods to use. If None, uses all available
            **kwargs: Additional arguments
        
        Returns:
            Dictionary mapping method name to saliency map
        """
        if methods is None:
            methods = self.get_available_methods()
        
        saliency_maps = {}
        for method in methods:
            try:
                saliency_maps[method] = self.extract(
                    input_tensor, target_class, method, **kwargs
                )
            except Exception as e:
                print(f"Warning: Method {method} failed with error: {e}")
        
        return saliency_maps


def get_saliency_methods() -> List[str]:
    """Return list of supported saliency methods."""
    return ['vanilla_grad', 'integrated_grad', 'deeplift', 'gradcam']


def normalize_saliency(saliency: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize saliency map to [0, 1] range.
    
    Args:
        saliency: Raw saliency map
        method: Normalization method ('minmax', 'abs_max', 'percentile')
    
    Returns:
        Normalized saliency map
    """
    if method == 'minmax':
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            return (saliency - s_min) / (s_max - s_min)
        return np.zeros_like(saliency)
    
    elif method == 'abs_max':
        abs_max = np.abs(saliency).max()
        if abs_max > 1e-8:
            return saliency / abs_max
        return np.zeros_like(saliency)
    
    elif method == 'percentile':
        p1, p99 = np.percentile(saliency, [1, 99])
        saliency = np.clip(saliency, p1, p99)
        return (saliency - p1) / (p99 - p1 + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_importance_ranking(saliency: np.ndarray) -> np.ndarray:
    """
    Get pixel importance ranking from saliency map.
    
    Args:
        saliency: Saliency map (H, W)
    
    Returns:
        Indices sorted by importance (most important first)
    """
    flat = saliency.flatten()
    return np.argsort(flat)[::-1]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.resnet import ResNet18
    
    # Test saliency extraction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(num_classes=10)
    
    # Get target layer for GradCAM
    target_layer = model.encoder.layer4[-1].conv2
    
    extractor = SaliencyExtractor(model, device=device, target_layer=target_layer)
    
    print(f"Available methods: {extractor.get_available_methods()}")
    
    # Random input
    x = torch.randn(1, 3, 32, 32)
    
    # Extract saliency maps
    saliency_maps = extractor.extract_all(x)
    
    for method, smap in saliency_maps.items():
        print(f"{method}: shape={smap.shape}, range=[{smap.min():.4f}, {smap.max():.4f}]")
