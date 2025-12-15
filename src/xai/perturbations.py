"""
Perturbation strategies for pixel flipping experiments.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from abc import ABC, abstractmethod


class BasePerturbation(ABC):
    """Base class for perturbation strategies."""
    
    @abstractmethod
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Apply perturbation to masked pixels.
        
        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            mask: Binary mask where 1 indicates pixels to perturb (H, W) or (B, H, W)
        
        Returns:
            Perturbed image
        """
        pass


class MeanPerturbation(BasePerturbation):
    """
    Replace pixels with dataset mean.
    Standard baseline perturbation for pixel flipping.
    """
    
    def __init__(
        self,
        mean: tuple = (0.4914, 0.4822, 0.4465),
        normalized: bool = True
    ):
        """
        Args:
            mean: Dataset mean (per channel)
            normalized: Whether input is normalized (if True, use 0 as replacement)
        """
        self.mean = mean
        self.normalized = normalized
    
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        is_numpy = isinstance(image, np.ndarray)
        
        if is_numpy:
            image = torch.from_numpy(image.copy()).float()
            mask = torch.from_numpy(mask).float()
        else:
            image = image.clone()
        
        # Ensure mask has channel dimension
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)
        if mask.dim() == 3 and image.dim() == 4:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        
        # Replacement value
        if self.normalized:
            # For normalized inputs, mean is 0
            replacement = torch.zeros(3, 1, 1)
        else:
            replacement = torch.tensor(self.mean).view(3, 1, 1)
        
        replacement = replacement.to(image.device)
        
        # Apply perturbation
        if image.dim() == 4:  # Batched
            replacement = replacement.unsqueeze(0)
            mask = mask.expand_as(image)
        else:
            mask = mask.expand_as(image)
        
        perturbed = image * (1 - mask) + replacement * mask
        
        if is_numpy:
            return perturbed.numpy()
        return perturbed


class NoisePerturbation(BasePerturbation):
    """
    Replace pixels with random noise.
    Tests sensitivity to random perturbations.
    """
    
    def __init__(
        self,
        noise_type: str = 'uniform',
        noise_range: tuple = (-1.0, 1.0),
        seed: Optional[int] = None
    ):
        """
        Args:
            noise_type: Type of noise ('uniform', 'gaussian')
            noise_range: Range for uniform noise or (mean, std) for gaussian
            seed: Random seed for reproducibility
        """
        self.noise_type = noise_type
        self.noise_range = noise_range
        self.seed = seed
    
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        is_numpy = isinstance(image, np.ndarray)
        
        if is_numpy:
            image = torch.from_numpy(image.copy()).float()
            mask = torch.from_numpy(mask).float()
        else:
            image = image.clone()
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
        
        # Generate noise
        if self.noise_type == 'uniform':
            low, high = self.noise_range
            noise = torch.rand_like(image) * (high - low) + low
        elif self.noise_type == 'gaussian':
            mean, std = self.noise_range
            noise = torch.randn_like(image) * std + mean
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        # Ensure mask has correct shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 3 and image.dim() == 4:
            mask = mask.unsqueeze(1)
        
        mask = mask.expand_as(image)
        
        # Apply perturbation
        perturbed = image * (1 - mask) + noise * mask
        
        if is_numpy:
            return perturbed.numpy()
        return perturbed


class BlurPerturbation(BasePerturbation):
    """
    Replace pixels with blurred version.
    More realistic perturbation that maintains local statistics.
    """
    
    def __init__(self, kernel_size: int = 11, sigma: float = 5.0):
        """
        Args:
            kernel_size: Size of Gaussian blur kernel
            sigma: Standard deviation of Gaussian
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._kernel = None
    
    def _get_kernel(self, device: torch.device) -> torch.Tensor:
        """Create Gaussian kernel."""
        if self._kernel is None or self._kernel.device != device:
            x = torch.arange(self.kernel_size, dtype=torch.float32, device=device)
            x = x - self.kernel_size // 2
            kernel_1d = torch.exp(-x**2 / (2 * self.sigma**2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            self._kernel = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        return self._kernel
    
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        is_numpy = isinstance(image, np.ndarray)
        
        if is_numpy:
            image = torch.from_numpy(image.copy()).float()
            mask = torch.from_numpy(mask).float()
        else:
            image = image.clone()
        
        device = image.device
        kernel = self._get_kernel(device)
        
        # Handle batched input
        if image.dim() == 3:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply blur channel-wise
        padding = self.kernel_size // 2
        blurred = []
        for c in range(image.shape[1]):
            channel = image[:, c:c+1, :, :]
            blurred_channel = F.conv2d(channel, kernel, padding=padding)
            blurred.append(blurred_channel)
        blurred = torch.cat(blurred, dim=1)
        
        # Ensure mask has correct shape
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(image)
        
        # Apply perturbation
        perturbed = image * (1 - mask) + blurred * mask
        
        if squeeze_output:
            perturbed = perturbed.squeeze(0)
        
        if is_numpy:
            return perturbed.numpy()
        return perturbed


class InpaintingPerturbation(BasePerturbation):
    """
    Simple inpainting using average of surrounding pixels.
    Most realistic but computationally expensive.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: Size of neighborhood window for inpainting
        """
        self.window_size = window_size
    
    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray],
        mask: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        is_numpy = isinstance(image, np.ndarray)
        
        if is_numpy:
            image = torch.from_numpy(image.copy()).float()
            mask = torch.from_numpy(mask).float()
        else:
            image = image.clone()
        
        # Simple box blur as inpainting
        padding = self.window_size // 2
        kernel = torch.ones(1, 1, self.window_size, self.window_size, device=image.device)
        kernel = kernel / (self.window_size ** 2)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply box blur
        inpainted = []
        for c in range(image.shape[1]):
            channel = image[:, c:c+1, :, :]
            blurred = F.conv2d(channel, kernel, padding=padding)
            inpainted.append(blurred)
        inpainted = torch.cat(inpainted, dim=1)
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(image)
        
        perturbed = image * (1 - mask) + inpainted * mask
        
        if squeeze_output:
            perturbed = perturbed.squeeze(0)
        
        if is_numpy:
            return perturbed.numpy()
        return perturbed


def get_perturbation(name: str, **kwargs) -> BasePerturbation:
    """
    Factory function to get perturbation by name.
    
    Args:
        name: Perturbation name ('mean', 'noise', 'blur', 'inpainting')
        **kwargs: Additional arguments for perturbation
    
    Returns:
        Perturbation instance
    """
    perturbations = {
        'mean': MeanPerturbation,
        'noise': NoisePerturbation,
        'blur': BlurPerturbation,
        'inpainting': InpaintingPerturbation
    }
    
    if name not in perturbations:
        raise ValueError(f"Unknown perturbation: {name}. Choose from {list(perturbations.keys())}")
    
    return perturbations[name](**kwargs)


if __name__ == "__main__":
    # Test perturbations
    torch.manual_seed(42)
    
    # Create test image and mask
    image = torch.randn(3, 32, 32)
    mask = torch.zeros(32, 32)
    mask[10:20, 10:20] = 1.0
    
    print("Testing perturbations:")
    
    # Test mean perturbation
    mean_pert = MeanPerturbation()
    result = mean_pert(image, mask)
    print(f"Mean perturbation: shape={result.shape}, center_mean={result[:, 10:20, 10:20].mean():.4f}")
    
    # Test noise perturbation
    noise_pert = NoisePerturbation(noise_type='gaussian', noise_range=(0, 0.1))
    result = noise_pert(image, mask)
    print(f"Noise perturbation: shape={result.shape}")
    
    # Test blur perturbation
    blur_pert = BlurPerturbation(kernel_size=11, sigma=5.0)
    result = blur_pert(image, mask)
    print(f"Blur perturbation: shape={result.shape}")
    
    # Test with batched input
    batch_image = torch.randn(4, 3, 32, 32)
    batch_mask = torch.zeros(4, 32, 32)
    batch_mask[:, 10:20, 10:20] = 1.0
    
    result = mean_pert(batch_image, batch_mask)
    print(f"Batched mean perturbation: shape={result.shape}")
