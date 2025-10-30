from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from nmsd.models.unet_context import ContextUNet
from nmsd.encoders.transformer_context import SuffixTransformerEncoder
from nmsd.diffusion.schedulers import build_schedule
from nmsd.diffusion.sampler import nonmarkov_ddim_sample_loop


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt")
    parser.add_argument("--num", type=int, default=16, help="Number of samples")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--suffix-steps", type=int, default=5, help="Number of suffix steps for context")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity")
    parser.add_argument("--out", type=str, default="samples_nonmarkov.png", help="Output image path")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt["cfg"]
    
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

    # Load encoder
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
    
    # Determine prediction type from config
    prediction_type = cfg["model"].get("prediction_type", "epsilon")
    if "loss_type" in cfg.get("training", {}):
        # Infer from loss_type if prediction_type not explicitly set
        prediction_type = "x0" if cfg["training"]["loss_type"] == "dart" else "epsilon"

    print(f"Generating {args.num} samples with {args.steps} DDIM steps, {args.suffix_steps} suffix steps (eta={args.eta}, prediction_type={prediction_type})...")
    samples = nonmarkov_ddim_sample_loop(
        model, encoder, (args.num, c, s, s), sch,
        device=device,
        num_steps=int(args.steps),
        num_suffix_steps=int(args.suffix_steps),
        eta=float(args.eta),
        prediction_type=prediction_type
    )

    # scale to [0,1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, str(out_path), nrow=int(max(1, round(args.num ** 0.5))))
    print(f"Saved {args.num} samples to {out_path}")


if __name__ == "__main__":
    main()

