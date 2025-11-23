from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from ..models.unet_context import ContextUNet
from ..encoders.transformer_context import SuffixTransformerEncoder
from ..encoders.signature import SignatureEncoder
from ..diffusion.schedulers import build_schedule
from ..diffusion.sampler import dart_simple_sample_loop, nonmarkov_ddim_sample_loop, efficient_nonmarkov_sample_loop


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Sample from DART model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt")
    parser.add_argument("--num", type=int, default=16, help="Number of samples")
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--suffix-steps", type=int, default=5, help="Number of suffix steps for context")
    parser.add_argument("--sampler", type=str, default="dart", 
                       choices=["dart", "ddim"],
                       help="Sampling method: 'dart' (simple, direct) or 'ddim' (DDIM-based)")
    parser.add_argument("--noise-scale", type=float, default=1.0, 
                       help="Noise scale for DART sampler (default: 1.0)")
    parser.add_argument("--cfg-scale", type=float, default=0.0,
                       help="Classifier-free guidance scale (0=off, 1-2=typical, higher=stronger)")
    parser.add_argument("--eta", type=float, default=0.0, 
                       help="Stochasticity for DDIM sampler (default: 0.0)")
    parser.add_argument("--out", type=str, default="samples_dart.png", help="Output image path")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt["cfg"]

    # Load model
    model = ContextUNet(
        in_channels=int(cfg["model"]["in_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mults=list(cfg["model"]["channel_mults"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        time_emb_dim=int(cfg["model"]["time_emb_dim"]),
        context_dim=int(cfg["model"]["context_dim"]),
        out_channels=int(cfg["model"]["in_channels"]),
    ).to(device)
    model.load_state_dict(ckpt.get("ema_model", ckpt["model"]))
    model.eval()

    # Load encoder (Transformer or Signature)
    encoder_type = cfg["encoder"].get("type", "transformer")
    
    if encoder_type == "signature":
        encoder = SignatureEncoder(
            image_channels=int(cfg["model"]["in_channels"]),
            image_size=int(cfg["data"]["image_size"]),
            context_dim=int(cfg["model"]["context_dim"]),
            signature_degree=int(cfg["encoder"].get("signature_degree", 3)),
            pooling=cfg["encoder"].get("pooling", "spatial_mean"),
            time_augment=cfg["encoder"].get("time_augment", True),
            use_lead_lag=cfg["encoder"].get("use_lead_lag", False),
            hidden_dim=int(cfg["encoder"].get("hidden_dim", 256)),
        ).to(device)
    else:  # transformer
        encoder = SuffixTransformerEncoder(
            image_channels=int(cfg["model"]["in_channels"]),
            image_size=int(cfg["data"]["image_size"]),
            context_dim=int(cfg["model"]["context_dim"]),
            hidden_dim=int(cfg["encoder"]["hidden_dim"]),
            num_heads=int(cfg["encoder"]["num_heads"]),
            num_layers=int(cfg["encoder"]["num_layers"]),
            pooling=cfg["encoder"]["pooling"],
        ).to(device)
    
    encoder.load_state_dict(ckpt.get("ema_encoder", ckpt["encoder"]))
    encoder.eval()

    # Load schedule
    sch = build_schedule(
        T=int(cfg["diffusion"]["T"]),
        schedule=cfg["diffusion"]["schedule"],
        cosine_s=float(cfg["diffusion"].get("cosine_s", 0.008)),
    ).to(device)

    c = int(cfg["model"]["in_channels"])
    s = int(cfg["data"]["image_size"])
    
    # Auto-detect CFG scale from config if not specified via CLI
    cfg_scale = args.cfg_scale
    if cfg_scale == 0.0 and "sampling" in cfg and "cfg_scale" in cfg["sampling"]:
        cfg_scale = float(cfg["sampling"]["cfg_scale"])
        print(f"Using CFG scale from checkpoint config: {cfg_scale}")
    
    # Check if model was trained with CFG
    use_cfg_trained = cfg.get("training", {}).get("use_cfg", False)
    if cfg_scale > 0 and not use_cfg_trained:
        print(f"⚠ Warning: Model was NOT trained with CFG (use_cfg={use_cfg_trained})")
        print(f"  CFG may not work well. Consider training with use_cfg=true")

    # Choose sampler
    encoder_type = cfg.get("encoder", {}).get("type", "transformer")
    
    if args.sampler == "dart":
        print(f"Generating {args.num} samples with DART simple sampler ({args.steps} steps, CFG={cfg_scale})...")
        samples = dart_simple_sample_loop(
            model, encoder, (args.num, c, s, s), sch,
            device=device,
            num_steps=int(args.steps),
            num_suffix_steps=int(args.suffix_steps),
            noise_scale=float(args.noise_scale),
            cfg_scale=float(cfg_scale)
        )
    else:  # ddim
        print(f"Generating {args.num} samples with DDIM sampler ({args.steps} steps, eta={args.eta})...")
        if encoder_type == "signature":
             print("Using efficient incremental signature sampler...")
             samples = efficient_nonmarkov_sample_loop(
                model, encoder, (args.num, c, s, s), sch,
                device=device,
                num_steps=int(args.steps),
                eta=float(args.eta),
                prediction_type="x0"
             )
        else:
            samples = nonmarkov_ddim_sample_loop(
                model, encoder, (args.num, c, s, s), sch,
                device=device,
                num_steps=int(args.steps),
                num_suffix_steps=int(args.suffix_steps),
                eta=float(args.eta),
                prediction_type="x0"
            )

    # scale to [0,1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, str(out_path), nrow=int(max(1, round(args.num ** 0.5))))
    print(f"Saved {args.num} samples to {out_path}")


if __name__ == "__main__":
    main()

