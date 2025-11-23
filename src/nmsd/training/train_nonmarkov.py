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
from nmsd.diffusion.losses import nonmarkov_suffix_loss, dart_loss
from nmsd.data.datasets import get_dataloaders
from nmsd.models.unet_context import ContextUNet
from nmsd.encoders.transformer_context import SuffixTransformerEncoder
from nmsd.encoders.signature import SignatureEncoder, SignatureTransformerEncoder
from nmsd.diffusion.sampler import nonmarkov_ddim_sample_loop
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


def save_image_grid(x: torch.Tensor, path: Path, nrow: int = 4):
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

    # Model (context-conditioned UNet)
    model = ContextUNet(
        in_channels=int(cfg["model"]["in_channels"]),
        base_channels=int(cfg["model"]["base_channels"]),
        channel_mults=list(cfg["model"]["channel_mults"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        time_emb_dim=int(cfg["model"]["time_emb_dim"]),
        context_dim=int(cfg["model"]["context_dim"]),
        out_channels=int(cfg["model"]["in_channels"]),
    ).to(device)

    # Suffix encoder (Transformer, Signature, or SignatureTransformer hybrid)
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
    elif encoder_type == "signature_trans":
        # Hybrid: signature spatial features + transformer temporal modeling
        encoder = SignatureTransformerEncoder(
            image_channels=int(cfg["model"]["in_channels"]),
            image_size=int(cfg["data"]["image_size"]),
            context_dim=int(cfg["model"]["context_dim"]),
            hidden_dim=int(cfg["encoder"]["hidden_dim"]),
            num_heads=int(cfg["encoder"]["num_heads"]),
            num_layers=int(cfg["encoder"]["num_layers"]),
            pooling=cfg["encoder"]["pooling"],  # spatial pooling method
            transformer_pooling=cfg["encoder"].get("transformer_pooling", "mean"),
            use_signature_tokens=cfg["encoder"].get("use_signature_tokens", False),
            signature_degree=int(cfg["encoder"].get("signature_degree", 2)),
            window_size=int(cfg["encoder"].get("window_size", 3)),
            time_augment=cfg["encoder"].get("time_augment", True),
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

    # Optimizer for both model and encoder
    params = list(model.parameters()) + list(encoder.parameters())
    opt = AdamW(params, lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    
    # EMA for both
    ema_model = EMA(model, decay=float(cfg["training"]["ema_decay"]))
    ema_encoder = EMA(encoder, decay=float(cfg["training"]["ema_decay"]))

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
    num_suffix_steps = int(cfg["diffusion"]["num_suffix_steps"])
    loss_type = cfg["training"].get("loss_type", "epsilon")
    weighting = cfg["training"].get("weighting", "none")

    print(f"Training Non-Markov model (loss_type={loss_type}, encoder={encoder_type}, weighting={weighting}, num_suffix_steps={num_suffix_steps})")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder ({encoder_type}) parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, _ in pbar:
            model.train()
            encoder.train()
            x = x.to(device)
            b = x.shape[0]
            t = torch.randint(1, sch.T + 1 - num_suffix_steps, (b,), device=device)

            # Choose loss function
            if loss_type == "dart":
                losses = dart_loss(model, encoder, x, t, sch, 
                                 num_suffix_steps=num_suffix_steps,
                                 weighting=weighting)
            else:  # epsilon prediction
                losses = nonmarkov_suffix_loss(model, encoder, x, t, sch, 
                                              num_suffix_steps=num_suffix_steps)
            
            loss = losses["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # Clip gradients to prevent exploding gradients (helps with memory spikes)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            
            opt.step()
            ema_model.update(model)
            ema_encoder.update(encoder)
            
            # Clear cache periodically to prevent memory fragmentation
            if global_step % 50 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()

            global_step += 1
            
            # Update memory stats
            memory_profiler.update()
            
            # Log to CSV
            if global_step % log_every == 0:
                mem_stats = memory_profiler.get_stats()
                log_data = {"loss": loss.item(), 
                           "gpu_mb": mem_stats["gpu_mb"],
                           "cpu_mb": mem_stats["cpu_mb"]}
                if "loss_unweighted" in losses:
                    log_data["loss_unweighted"] = losses['loss_unweighted'].item()
                if "avg_weight" in losses:
                    log_data["avg_weight"] = losses['avg_weight']
                logger.log(step=global_step, **log_data)
                
                log_msg = {"loss": f"{loss.item():.4f}"}
                if "loss_unweighted" in losses:
                    log_msg["loss_unw"] = f"{losses['loss_unweighted'].item():.4f}"
                if "avg_weight" in losses:
                    log_msg["wgt"] = f"{losses['avg_weight']:.2f}"
                log_msg["gpu_mb"] = f"{mem_stats['gpu_mb']:.0f}" if mem_stats['gpu_mb'] > 0 else "N/A"
                pbar.set_postfix(log_msg)
            else:
                logger.log(step=global_step, loss=loss.item())

            if global_step % sample_every == 0:
                model.eval()
                encoder.eval()
                
                # Create EMA copies
                ema_model_copy = ContextUNet(
                    in_channels=int(cfg["model"]["in_channels"]),
                    base_channels=int(cfg["model"]["base_channels"]),
                    channel_mults=list(cfg["model"]["channel_mults"]),
                    num_res_blocks=int(cfg["model"]["num_res_blocks"]),
                    time_emb_dim=int(cfg["model"]["time_emb_dim"]),
                    context_dim=int(cfg["model"]["context_dim"]),
                    out_channels=int(cfg["model"]["in_channels"]),
                ).to(device)
                ema_model.copy_to(ema_model_copy)
                
                # Create EMA encoder copy with the same type as the original encoder
                if encoder_type == "signature":
                    ema_encoder_copy = SignatureEncoder(
                        image_channels=int(cfg["model"]["in_channels"]),
                        image_size=int(cfg["data"]["image_size"]),
                        context_dim=int(cfg["model"]["context_dim"]),
                        signature_degree=int(cfg["encoder"].get("signature_degree", 3)),
                        pooling=cfg["encoder"].get("pooling", "spatial_mean"),
                        time_augment=cfg["encoder"].get("time_augment", True),
                        use_lead_lag=cfg["encoder"].get("use_lead_lag", False),
                        hidden_dim=int(cfg["encoder"].get("hidden_dim", 256)),
                    ).to(device)
                elif encoder_type == "signature_trans":
                    # Hybrid: signature spatial features + transformer temporal modeling
                    ema_encoder_copy = SignatureTransformerEncoder(
                        image_channels=int(cfg["model"]["in_channels"]),
                        image_size=int(cfg["data"]["image_size"]),
                        context_dim=int(cfg["model"]["context_dim"]),
                        hidden_dim=int(cfg["encoder"]["hidden_dim"]),
                        num_heads=int(cfg["encoder"]["num_heads"]),
                        num_layers=int(cfg["encoder"]["num_layers"]),
                        pooling=cfg["encoder"]["pooling"],
                        transformer_pooling=cfg["encoder"].get("transformer_pooling", "mean"),
                        use_signature_tokens=cfg["encoder"].get("use_signature_tokens", False),
                        signature_degree=int(cfg["encoder"].get("signature_degree", 2)),
                        window_size=int(cfg["encoder"].get("window_size", 3)),
                        time_augment=cfg["encoder"].get("time_augment", True),
                    ).to(device)
                else:  # transformer
                    ema_encoder_copy = SuffixTransformerEncoder(
                        image_channels=int(cfg["model"]["in_channels"]),
                        image_size=int(cfg["data"]["image_size"]),
                        context_dim=int(cfg["model"]["context_dim"]),
                        hidden_dim=int(cfg["encoder"]["hidden_dim"]),
                        num_heads=int(cfg["encoder"]["num_heads"]),
                        num_layers=int(cfg["encoder"]["num_layers"]),
                        pooling=cfg["encoder"]["pooling"],
                    ).to(device)
                ema_encoder.copy_to(ema_encoder_copy)
                
                with torch.no_grad():
                    # Use non-Markov DDIM with 50 steps for fast sampling
                    # Determine prediction type from loss_type
                    pred_type = "x0" if loss_type == "dart" else "epsilon"
                    samples = nonmarkov_ddim_sample_loop(
                        ema_model_copy,
                        ema_encoder_copy,
                        (16, cfg["model"]["in_channels"], cfg["data"]["image_size"], cfg["data"]["image_size"]),
                        sch, device, num_steps=50, num_suffix_steps=5, eta=0.0,
                        prediction_type=pred_type
                    )
                save_image_grid(samples, out_dir / "samples" / f"step_{global_step:07d}.png")
                
                # Plot losses
                title = f"Non-Markov Training Loss (loss_type={loss_type})"
                logger.plot(title=title)
                
                # Save memory summary
                memory_profiler.log_summary(out_dir / "logs" / "memory_usage.txt")

            if global_step % save_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "encoder": encoder.state_dict(),
                    "ema_model": ema_model.shadow,
                    "ema_encoder": ema_encoder.shadow,
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

