"""
CIFAR-10 data loading utilities for CLXAI.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional


# CIFAR-10 statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_stats() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return CIFAR-10 mean and std for normalization."""
    return CIFAR10_MEAN, CIFAR10_STD


def get_train_transforms(augment: bool = True) -> transforms.Compose:
    """
    Get training transforms for CIFAR-10.
    
    Args:
        augment: Whether to apply data augmentation
    
    Returns:
        Composed transforms
    """
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_test_transforms() -> transforms.Compose:
    """Get test/evaluation transforms for CIFAR-10."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_contrastive_transforms() -> transforms.Compose:
    """
    Get transforms for contrastive learning.
    Returns two augmented views of each image.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class TwoCropTransform:
    """Create two augmented views of the same image for contrastive learning."""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    contrastive: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation (for CE training)
        contrastive: Whether to use contrastive transforms (for SCL training)
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        train_loader, test_loader
    """
    if contrastive:
        train_transform = TwoCropTransform(get_contrastive_transforms())
    else:
        train_transform = get_train_transforms(augment=augment)
    
    test_transform = get_test_transforms()
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_raw_cifar10_loader(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    train: bool = False,
) -> DataLoader:
    """
    Get CIFAR-10 loader WITHOUT normalization (for visualization).
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of workers
        train: Whether to load training set
    
    Returns:
        DataLoader with images in [0, 1] range
    """
    transform = transforms.ToTensor()
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize a CIFAR-10 tensor back to [0, 1] range.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    return tensor * std + mean


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test contrastive loader
    train_loader_cl, _ = get_cifar10_loaders(batch_size=64, contrastive=True)
    images, labels = next(iter(train_loader_cl))
    print(f"Contrastive batch: {len(images)} views, shape {images[0].shape}")
