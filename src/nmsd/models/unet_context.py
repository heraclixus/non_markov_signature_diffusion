from __future__ import annotations

from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = timesteps.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, context_dim: int = 0, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        # Context conditioning (optional)
        self.has_context = context_dim > 0
        if self.has_context:
            self.context_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(context_dim, out_ch)
            )

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Add context conditioning if available
        if self.has_context and context is not None:
            h = h + self.context_mlp(context).unsqueeze(-1).unsqueeze(-1)
        
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ContextUNet(nn.Module):
    """
    U-Net with optional context conditioning for non-Markov diffusion.
    
    When context_dim > 0, the model can condition on additional context 
    (e.g., encoded suffix sequence) via FiLM-style modulation in residual blocks.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] | None = None,
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        context_dim: int = 0,  # 0 means no context (Markov baseline)
        out_channels: int = 1,
    ):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4]

        self.context_dim = context_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down path
        downs = []
        in_ch = base_channels
        skip_channels = []  # track channels for each residual block output (skip)
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                downs.append(ResidualBlock(in_ch, out_ch, time_emb_dim, context_dim))
                skip_channels.append(out_ch)
                in_ch = out_ch
            if i != len(channel_mults) - 1:
                downs.append(Downsample(in_ch))
        self.downs = nn.ModuleList(downs)

        # Middle
        mid_ch = in_ch
        self.mid1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, context_dim)
        self.mid2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, context_dim)

        # Up path
        ups = []
        for i, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                ups.append(ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim, context_dim))
                in_ch = out_ch
            if i != 0:
                ups.append(Upsample(in_ch))
        self.ups = nn.ModuleList(ups)

        self.final_norm = nn.GroupNorm(8, in_ch)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - noisy input
            t: [B] - timestep
            context: [B, context_dim] - optional context (None for Markov baseline)
        
        Returns:
            noise prediction: [B, C, H, W]
        """
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)

        skips = []
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb, context)
                skips.append(x)
            else:
                x = layer(x)

        x = self.mid1(x, t_emb, context)
        x = self.mid2(x, t_emb, context)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb, context)
            else:
                x = layer(x)

        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x

