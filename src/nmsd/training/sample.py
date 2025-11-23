from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from ..models.unet import UNet
from ..diffusion.schedulers import build_schedule
from ..diffusion.sampler import ddim_sample_loop


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt")
    parser.add_argument("--num", type=int, default=16, help="Number of samples")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM sampling steps (default 50, faster than 1000)")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0.0=deterministic, 1.0=stochastic)")
    parser.add_argument("--out", type=str, default="samples.png", help="Output image path")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt["cfg"]

    model = UNet(
        in_channels=int(cfg["model"]["in_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mults=list(cfg["model"]["channel_mults"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        time_emb_dim=int(cfg["model"]["time_emb_dim"]),
        out_channels=int(cfg["model"]["in_channels"]),
    ).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()

    sch = build_schedule(
        T=int(cfg["diffusion"]["T"]),
        schedule=cfg["diffusion"]["schedule"],
        cosine_s=float(cfg["diffusion"].get("cosine_s", 0.008)),
    ).to(device)

    c = int(cfg["model"]["in_channels"]) 
    s = int(cfg["data"]["image_size"]) 
    
    print(f"Generating {args.num} samples with {args.steps} DDIM steps (eta={args.eta})...")
    samples = ddim_sample_loop(
        model, (args.num, c, s, s), sch, 
        device=device, 
        num_steps=int(args.steps), 
        eta=float(args.eta)
    )
    
    # scale to [0,1]
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, str(out_path), nrow=int(max(1, round(args.num ** 0.5))))
    print(f"Saved {args.num} samples to {out_path}")


if __name__ == "__main__":
    main()


