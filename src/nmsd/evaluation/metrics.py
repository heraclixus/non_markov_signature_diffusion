"""
Evaluation metrics for diffusion models.

Implements FID (Fréchet Inception Distance) and IS (Inception Score) 
using torch-fidelity for consistent evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def compute_fid(
    generated_images_path: str | Path,
    real_images_dataset: str,
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """
    Compute FID (Fréchet Inception Distance) between generated and real images.
    
    Args:
        generated_images_path: Path to directory containing generated images
        real_images_dataset: Name of real dataset ('mnist' or 'cifar10')
        batch_size: Batch size for feature extraction
        device: Device to use
    
    Returns:
        FID score (lower is better)
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        raise ImportError(
            "torch-fidelity is required for FID computation. "
            "Install with: pip install torch-fidelity"
        )
    
    dataset_lower = real_images_dataset.lower()
    
    # torch-fidelity only has CIFAR-10 and STL-10 registered
    # For MNIST, we need to prepare the real images ourselves
    if dataset_lower == "mnist":
        print(f"Computing FID for MNIST (preparing reference images)...")
        # Prepare MNIST reference images
        mnist_ref_dir = _prepare_mnist_reference()
        real_images_input = str(mnist_ref_dir)
    elif dataset_lower == "cifar10":
        print(f"Computing FID between generated images and CIFAR-10 training set...")
        # Use registered CIFAR-10 dataset
        real_images_input = "cifar10-train"
    else:
        raise ValueError(f"Unknown dataset: {real_images_dataset}")
    
    # Fix for PyTorch 2.6+: torch-fidelity needs to load Inception weights
    # Since torch-fidelity is a trusted library, we temporarily allow weights_only=False
    # by monkey-patching torch.load
    import torch
    original_load = torch.load
    
    def load_with_weights_only_false(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    try:
        torch.load = load_with_weights_only_false
        
        metrics = calculate_metrics(
            input1=str(generated_images_path),
            input2=real_images_input,
            cuda=(device == "cuda"),
            isc=False,  # Don't compute IS here
            fid=True,
            kid=False,
            prc=False,
            batch_size=batch_size,
            verbose=True,
        )
    finally:
        # Restore original torch.load
        torch.load = original_load
    
    return metrics["frechet_inception_distance"]


def _prepare_mnist_reference(max_images: int = 10000) -> Path:
    """
    Prepare MNIST reference images for FID computation.
    Downloads MNIST if needed and saves images to a temporary directory.
    
    Args:
        max_images: Maximum number of reference images to use
    
    Returns:
        Path to directory containing reference images
    """
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import tempfile
    
    # Create a cache directory for MNIST reference images
    cache_dir = Path.home() / ".cache" / "nmsd_fid" / "mnist_reference"
    
    # Check if we already have prepared images
    if cache_dir.exists() and len(list(cache_dir.glob("*.png"))) >= max_images:
        print(f"Using cached MNIST reference images from {cache_dir}")
        return cache_dir
    
    print(f"Preparing MNIST reference images (this is a one-time setup)...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])
    
    mnist_dataset = datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )
    
    # Save reference images
    num_saved = 0
    for idx in range(min(len(mnist_dataset), max_images)):
        img, _ = mnist_dataset[idx]
        # Convert from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        save_path = cache_dir / f"mnist_{idx:05d}.png"
        save_image(img, save_path)
        num_saved += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"Saved {idx + 1}/{max_images} images...")
    
    print(f"✓ Saved {num_saved} MNIST reference images to {cache_dir}")
    return cache_dir


def compute_inception_score(
    generated_images_path: str | Path,
    batch_size: int = 64,
    splits: int = 10,
    device: str = "cuda",
) -> Tuple[float, float]:
    """
    Compute Inception Score for generated images.
    
    Args:
        generated_images_path: Path to directory containing generated images
        batch_size: Batch size for feature extraction
        splits: Number of splits for IS computation
        device: Device to use
    
    Returns:
        Tuple of (mean, std) of Inception Score
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        raise ImportError(
            "torch-fidelity is required for IS computation. "
            "Install with: pip install torch-fidelity"
        )
    
    print("Computing Inception Score...")
    
    # Fix for PyTorch 2.6+: torch-fidelity needs to load Inception weights
    # Since torch-fidelity is a trusted library, we temporarily allow weights_only=False
    import torch
    original_load = torch.load
    
    def load_with_weights_only_false(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    try:
        torch.load = load_with_weights_only_false
        
        metrics = calculate_metrics(
            input1=str(generated_images_path),
            cuda=(device == "cuda"),
            isc=True,
            fid=False,
            kid=False,
            prc=False,
            batch_size=batch_size,
            isc_splits=splits,
            verbose=True,
        )
    finally:
        # Restore original torch.load
        torch.load = original_load
    
    return metrics["inception_score_mean"], metrics["inception_score_std"]


def generate_samples_for_evaluation(
    model: nn.Module,
    encoder: Optional[nn.Module],
    scheduler,
    config: Dict,
    num_samples: int,
    output_dir: Path,
    device: str = "cuda",
    num_steps: int = 50,
    sampler_type: str = "ddim",
) -> Path:
    """
    Generate samples from a trained model and save them for evaluation.
    
    Args:
        model: Trained diffusion model
        encoder: Encoder for non-Markov models (None for Markov)
        scheduler: Noise scheduler
        config: Model configuration
        num_samples: Number of samples to generate
        output_dir: Directory to save generated images
        device: Device to use
        num_steps: Number of sampling steps
        sampler_type: Type of sampler ('ddim', 'nonmarkov', 'dart')
    
    Returns:
        Path to directory containing generated images
    """
    from torchvision.utils import save_image
    from nmsd.diffusion.sampler import (
        ddim_sample_loop,
        nonmarkov_ddim_sample_loop,
        dart_simple_sample_loop,
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    if encoder is not None:
        encoder.eval()
    
    in_channels = config["model"]["in_channels"]
    image_size = config["data"]["image_size"]
    batch_size = min(64, num_samples)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} samples ({num_batches} batches of {batch_size})...")
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            current_batch_size = min(batch_size, num_samples - sample_idx)
            shape = (current_batch_size, in_channels, image_size, image_size)
            
            # Sample based on model type
            if sampler_type == "ddim" or encoder is None:
                # Standard DDIM sampling for Markov models
                samples = ddim_sample_loop(
                    model, shape, scheduler, device,
                    num_steps=num_steps, eta=0.0
                )
            else:
                # Non-Markov sampling
                num_suffix_steps = config["diffusion"].get("num_suffix_steps", 5)
                loss_type = config["training"].get("loss_type", "epsilon")
                cfg_scale = config["sampling"].get("cfg_scale", 0.0)
                
                # Use DART sampler if CFG is enabled or loss type is DART
                if cfg_scale > 0 or loss_type == "dart":
                    # DART sampling with optional CFG
                    samples = dart_simple_sample_loop(
                        model, encoder, shape, scheduler, device,
                        num_steps=num_steps,
                        num_suffix_steps=num_suffix_steps,
                        noise_scale=1.0,
                        cfg_scale=cfg_scale,
                    )
                else:
                    # Standard non-Markov DDIM (epsilon prediction)
                    pred_type = "x0" if loss_type == "dart" else "epsilon"
                    samples = nonmarkov_ddim_sample_loop(
                        model, encoder, shape, scheduler, device,
                        num_steps=num_steps,
                        num_suffix_steps=num_suffix_steps,
                        eta=0.0,
                        prediction_type=pred_type,
                    )
            
            # Save individual images
            samples = (samples.clamp(-1, 1) + 1) / 2  # [-1, 1] -> [0, 1]
            
            for i in range(current_batch_size):
                img_path = output_dir / f"sample_{sample_idx:05d}.png"
                save_image(samples[i], img_path)
                sample_idx += 1
    
    print(f"Saved {sample_idx} samples to {output_dir}")
    return output_dir


def evaluate_model(
    model: nn.Module,
    encoder: Optional[nn.Module],
    scheduler,
    config: Dict,
    checkpoint_path: Path,
    dataset: str,
    num_samples: int = 10000,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    num_steps: int = 50,
    sampler_type: str = "ddim",
    skip_generation: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a trained diffusion model.
    
    Args:
        model: Trained diffusion model
        encoder: Encoder for non-Markov models
        scheduler: Noise scheduler
        config: Model configuration
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name ('mnist' or 'cifar10')
        num_samples: Number of samples to generate for evaluation
        output_dir: Directory for outputs (defaults to checkpoint_dir/evaluation/)
        device: Device to use
        num_steps: Number of sampling steps
        sampler_type: Type of sampler
        skip_generation: Skip generation if samples already exist
    
    Returns:
        Dictionary of metrics: {'fid': float, 'is_mean': float, 'is_std': float}
    """
    if output_dir is None:
        checkpoint_dir = checkpoint_path.parent.parent
        output_dir = checkpoint_dir / "evaluation"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_dir = output_dir / "samples"
    
    # Generate samples if needed
    if skip_generation and samples_dir.exists():
        print(f"Using existing samples from {samples_dir}")
    else:
        samples_dir = generate_samples_for_evaluation(
            model, encoder, scheduler, config,
            num_samples, samples_dir, device, num_steps, sampler_type
        )
    
    # Compute metrics
    metrics = {}
    
    # FID
    try:
        fid = compute_fid(samples_dir, dataset, device=device)
        metrics["fid"] = fid
        print(f"FID: {fid:.2f}")
    except Exception as e:
        print(f"Failed to compute FID: {e}")
        metrics["fid"] = float("nan")
    
    # Inception Score (only for CIFAR-10, MNIST IS is not meaningful)
    if dataset.lower() == "cifar10":
        try:
            is_mean, is_std = compute_inception_score(samples_dir, device=device)
            metrics["is_mean"] = is_mean
            metrics["is_std"] = is_std
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        except Exception as e:
            print(f"Failed to compute Inception Score: {e}")
            metrics["is_mean"] = float("nan")
            metrics["is_std"] = float("nan")
    else:
        metrics["is_mean"] = float("nan")
        metrics["is_std"] = float("nan")
    
    # Save metrics
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Num samples: {num_samples}\n")
        f.write(f"Sampling steps: {num_steps}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"FID: {metrics['fid']:.4f}\n")
        if dataset.lower() == "cifar10":
            f.write(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}\n")
    
    print(f"Metrics saved to {metrics_file}")
    
    return metrics

