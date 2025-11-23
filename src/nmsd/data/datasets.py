from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    root: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 28,
) -> Tuple[DataLoader, DataLoader]:
    """Get MNIST dataloaders with normalization to [-1, 1]."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])

    train_dataset = datasets.MNIST(root=str(root), train=True, transform=tfm, download=True)
    test_dataset = datasets.MNIST(root=str(root), train=False, transform=tfm, download=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, test_loader


def get_cifar10_dataloaders(
    root: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 dataloaders with normalization to [-1, 1]."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Data augmentation for training
    train_tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1] for RGB
    ])
    
    # No augmentation for test
    test_tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1] for RGB
    ])

    train_dataset = datasets.CIFAR10(root=str(root), train=True, transform=train_tfm, download=True)
    test_dataset = datasets.CIFAR10(root=str(root), train=False, transform=test_tfm, download=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, test_loader


def get_imagenet_dataloaders(
    root: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet dataloaders with normalization to [-1, 1].
    
    Expects the following directory structure:
    root/
        train/
            class1/
            class2/
            ...
        val/
            class1/
            class2/
            ...
    """
    root = Path(root)
    train_dir = root / "train"
    val_dir = root / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet dataset not found at {root}. "
            "Expected 'train' and 'val' subdirectories. "
            "Please download/prepare ImageNet dataset first."
        )

    # Data augmentation for training
    train_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Force resize to target resolution
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1] for RGB
    ])
    
    # No augmentation for test
    test_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1] for RGB
    ])

    print(f"Loading ImageNet from {root} (train: {train_dir}, val: {val_dir})...")
    # Using ImageFolder allows for standard ImageNet structure as well as other folder-based datasets
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_tfm)
    test_dataset = datasets.ImageFolder(root=str(val_dir), transform=test_tfm)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, test_loader


def get_dataloaders(
    dataset_name: str,
    root: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Generic function to get dataloaders for different datasets.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'cifar10', 'imagenet')
        root: Root directory for dataset storage
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloading
        image_size: Image size to resize to
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        if image_size is None:
            image_size = 28
        return get_mnist_dataloaders(root, batch_size, num_workers, image_size)
    
    elif dataset_name == "cifar10":
        if image_size is None:
            image_size = 32
        return get_cifar10_dataloaders(root, batch_size, num_workers, image_size)
    
    elif "imagenet" in dataset_name:
        # Handle 'imagenet', 'imagenet32', 'imagenet64', 'imagenet128'
        if image_size is None:
            # Try to deduce size from name
            if "32" in dataset_name:
                image_size = 32
            elif "64" in dataset_name:
                image_size = 64
            elif "128" in dataset_name:
                image_size = 128
            else:
                image_size = 64  # Default to 64x64 for ImageNet
                
        return get_imagenet_dataloaders(root, batch_size, num_workers, image_size)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'mnist', 'cifar10', 'imagenet'")


