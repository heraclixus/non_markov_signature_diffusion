"""
Script to evaluate a single model checkpoint.

Usage:
    python -m nmsd.evaluation.evaluate \
        --config configs/dart_cifar10.yaml \
        --checkpoint experiments/dart_cifar10/checkpoints/model_0050000.pt \
        --num-samples 10000 \
        --num-steps 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ..diffusion.schedulers import build_schedule
from ..models.unet import UNet
from ..models.unet_context import ContextUNet
from ..encoders.transformer_context import SuffixTransformerEncoder
from ..encoders.signature import SignatureEncoder, SignatureTransformerEncoder
from ..evaluation.metrics import evaluate_model


def load_model_from_checkpoint(config_path: Path, checkpoint_path: Path, device: str):
    """Load model and encoder from checkpoint."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Determine model type
    is_nonmarkov = "encoder" in config
    
    # Build scheduler
    scheduler = build_schedule(
        T=int(config["diffusion"]["T"]),
        schedule=config["diffusion"]["schedule"],
        cosine_s=float(config["diffusion"].get("cosine_s", 0.008)),
    ).to(device)
    
    # Build model
    if is_nonmarkov:
        # Non-Markov model (with context)
        model = ContextUNet(
            in_channels=int(config["model"]["in_channels"]),
            base_channels=int(config["model"]["base_channels"]),
            channel_mults=list(config["model"]["channel_mults"]),
            num_res_blocks=int(config["model"]["num_res_blocks"]),
            time_emb_dim=int(config["model"]["time_emb_dim"]),
            context_dim=int(config["model"]["context_dim"]),
            out_channels=int(config["model"]["in_channels"]),
        ).to(device)
        
        # Build encoder
        encoder_type = config["encoder"].get("type", "transformer")
        
        if encoder_type == "signature":
            encoder = SignatureEncoder(
                image_channels=int(config["model"]["in_channels"]),
                image_size=int(config["data"]["image_size"]),
                context_dim=int(config["model"]["context_dim"]),
                signature_degree=int(config["encoder"].get("signature_degree", 3)),
                pooling=config["encoder"].get("pooling", "spatial_mean"),
                time_augment=config["encoder"].get("time_augment", True),
                use_lead_lag=config["encoder"].get("use_lead_lag", False),
                hidden_dim=int(config["encoder"].get("hidden_dim", 256)),
            ).to(device)
        elif encoder_type == "signature_trans":
            # Hybrid: signature spatial features + transformer temporal modeling
            encoder = SignatureTransformerEncoder(
                image_channels=int(config["model"]["in_channels"]),
                image_size=int(config["data"]["image_size"]),
                context_dim=int(config["model"]["context_dim"]),
                hidden_dim=int(config["encoder"]["hidden_dim"]),
                num_heads=int(config["encoder"]["num_heads"]),
                num_layers=int(config["encoder"]["num_layers"]),
                pooling=config["encoder"]["pooling"],  # spatial pooling method
                transformer_pooling=config["encoder"].get("transformer_pooling", "mean"),
            ).to(device)
        else:  # transformer
            encoder = SuffixTransformerEncoder(
                image_channels=int(config["model"]["in_channels"]),
                image_size=int(config["data"]["image_size"]),
                context_dim=int(config["model"]["context_dim"]),
                hidden_dim=int(config["encoder"]["hidden_dim"]),
                num_heads=int(config["encoder"]["num_heads"]),
                num_layers=int(config["encoder"]["num_layers"]),
                pooling=config["encoder"]["pooling"],
            ).to(device)
    else:
        # Markov model (no context)
        model = UNet(
            in_channels=int(config["model"]["in_channels"]),
            base_channels=int(config["model"]["base_channels"]),
            channel_mults=list(config["model"]["channel_mults"]),
            num_res_blocks=int(config["model"]["num_res_blocks"]),
            time_emb_dim=int(config["model"]["time_emb_dim"]),
            out_channels=int(config["model"]["in_channels"]),
        ).to(device)
        encoder = None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights (try EMA first, then regular)
    if "ema_model" in checkpoint:
        print("Loading EMA model weights")
        model.load_state_dict(checkpoint["ema_model"], strict=False)
        if encoder is not None and "ema_encoder" in checkpoint:
            encoder.load_state_dict(checkpoint["ema_encoder"], strict=False)
    else:
        print("Loading model weights")
        model.load_state_dict(checkpoint["model"], strict=False)
        if encoder is not None and "encoder" in checkpoint:
            encoder.load_state_dict(checkpoint["encoder"], strict=False)
    
    model.eval()
    if encoder is not None:
        encoder.eval()
    
    # Determine sampler type
    if is_nonmarkov:
        loss_type = config["training"].get("loss_type", "epsilon")
        sampler_type = "dart" if loss_type == "dart" else "nonmarkov"
    else:
        sampler_type = "ddim"
    
    return model, encoder, scheduler, config, sampler_type


def main():
    parser = argparse.ArgumentParser(description="Evaluate a diffusion model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples to generate (default: 10000)")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of sampling steps (default: 50)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: checkpoint_dir/evaluation/)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation if samples already exist")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load config to get dataset name
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    dataset = config["data"]["dataset"]
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset}")
    print(f"Num samples: {args.num_samples}")
    print(f"Sampling steps: {args.num_steps}")
    print("=" * 60)
    print()
    
    # Load model
    print("Loading model...")
    model, encoder, scheduler, config, sampler_type = load_model_from_checkpoint(
        config_path, checkpoint_path, args.device
    )
    print(f"Sampler type: {sampler_type}")
    print()
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        encoder=encoder,
        scheduler=scheduler,
        config=config,
        checkpoint_path=checkpoint_path,
        dataset=dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device,
        num_steps=args.num_steps,
        sampler_type=sampler_type,
        skip_generation=args.skip_generation,
    )
    
    print()
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"FID: {metrics['fid']:.4f}")
    if dataset.lower() == "cifar10":
        print(f"IS: {metrics['is_mean']:.4f} ± {metrics['is_std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

