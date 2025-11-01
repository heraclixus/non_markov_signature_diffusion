from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from nmsd.diffusion.schedulers import build_schedule
from nmsd.diffusion.losses import ddpm_simple_loss
from nmsd.data.datasets import get_dataloaders
from nmsd.models.unet import UNet
from nmsd.diffusion.sampler import ddim_sample_loop
from nmsd.utils.logger import LossLogger, MemoryProfiler


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                # Handle newly added parameters (e.g., lazy initialization)
                if k not in self.shadow:
                    self.shadow[k] = v.detach().clone()
                else:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=False)


def save_image_grid(x: torch.Tensor, path: Path, nrow: int = 8):
    from torchvision.utils import save_image
    x = (x.clamp(-1, 1) + 1) / 2
    save_image(x, str(path), nrow=nrow)


def train(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.get("seed", 42)))

    # Data
    train_loader, _ = get_dataloaders(
        dataset_name=cfg["data"]["dataset"],
        root=cfg["data"]["root"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        image_size=int(cfg["data"]["image_size"]),
    )

    # Schedule
    sch = build_schedule(
        T=int(cfg["diffusion"]["T"]),
        schedule=cfg["diffusion"]["schedule"],
        cosine_s=float(cfg["diffusion"].get("cosine_s", 0.008)),
    )
    sch = sch.to(device)

    # Model
    model = UNet(
        in_channels=int(cfg["model"]["in_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mults=list(cfg["model"]["channel_mults"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        time_emb_dim=int(cfg["model"]["time_emb_dim"]),
        out_channels=int(cfg["model"]["in_channels"]),
    ).to(device)

    opt = AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    ema = EMA(model, decay=float(cfg["training"]["ema_decay"]))

    out_dir = Path(cfg["training"]["out_dir"]).expanduser()
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Initialize loss logger and memory profiler
    logger = LossLogger(out_dir / "logs", name="training_losses")
    memory_profiler = MemoryProfiler()

    global_step = 0
    epochs = int(cfg["training"]["epochs"])
    log_every = int(cfg["training"]["log_every"])
    sample_every = int(cfg["training"]["sample_every_steps"])
    save_every = int(cfg["training"]["save_every_steps"])

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, _ in pbar:
            model.train()
            x = x.to(device)
            b = x.shape[0]
            t = torch.randint(1, sch.T + 1, (b,), device=device)

            losses = ddpm_simple_loss(model, x, t, sch)
            loss = losses["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)

            global_step += 1
            
            # Update memory stats
            memory_profiler.update()
            
            # Log to CSV (include memory every log_every steps to reduce overhead)
            if global_step % log_every == 0:
                mem_stats = memory_profiler.get_stats()
                logger.log(step=global_step, loss=loss.item(), 
                          gpu_mb=mem_stats["gpu_mb"], 
                          cpu_mb=mem_stats["cpu_mb"])
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "gpu_mb": f"{mem_stats['gpu_mb']:.0f}" if mem_stats['gpu_mb'] > 0 else "N/A"
                })
            else:
                logger.log(step=global_step, loss=loss.item())

            if global_step % sample_every == 0:
                model.eval()
                ema_model = UNet(
                    in_channels=int(cfg["model"]["in_channels"]),
                    base_channels=int(cfg["model"]["base_channels"]),
                    channel_mults=list(cfg["model"]["channel_mults"]),
                    num_res_blocks=int(cfg["model"]["num_res_blocks"]),
                    time_emb_dim=int(cfg["model"]["time_emb_dim"]),
                    out_channels=int(cfg["model"]["in_channels"]),
                ).to(device)
                ema.copy_to(ema_model)
                with torch.no_grad():
                    # Use DDIM with 50 steps for fast sampling during training
                    samples = ddim_sample_loop(
                        ema_model, 
                        (16, cfg["model"]["in_channels"], cfg["data"]["image_size"], cfg["data"]["image_size"]), 
                        sch, device, num_steps=50, eta=0.0
                    )
                save_image_grid(samples, out_dir / "samples" / f"step_{global_step:07d}.png")
                
                # Plot losses
                logger.plot(title="Markov DDIM Training Loss")
                
                # Save memory summary
                memory_profiler.log_summary(out_dir / "logs" / "memory_usage.txt")

            if global_step % save_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "cfg": cfg,
                    "step": global_step,
                }, out_dir / "checkpoints" / f"model_{global_step:07d}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()


