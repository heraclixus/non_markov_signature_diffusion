from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B, L] or [B] - timesteps
        Returns:
            embeddings: [B, L, dim] or [B, dim]
        """
        original_shape = timesteps.shape
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)  # [B, 1]
        
        half_dim = self.dim // 2
        device = timesteps.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)  # [B, L, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        
        if len(original_shape) == 1:
            emb = emb.squeeze(1)
        return emb


class SuffixTransformerEncoder(nn.Module):
    """
    Transformer-based encoder for the suffix sequence x_{t:T}.
    
    Takes a sequence of noisy images [B, L, C, H, W] and corresponding timesteps [B, L],
    and outputs a fixed-size context vector [B, context_dim] that summarizes the suffix.
    """
    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        context_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        pooling: str = "mean",  # "mean", "max", or "cls"
    ):
        super().__init__()
        self.image_channels = image_channels
        self.image_size = image_size
        self.context_dim = context_dim
        self.pooling = pooling
        
        # Project flattened images to hidden_dim
        input_dim = image_channels * image_size * image_size
        self.image_proj = nn.Linear(input_dim, hidden_dim)
        
        # Timestep embedding
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        
        # Optional CLS token for pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.LayerNorm(context_dim),
        )
    
    def forward(self, suffix: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            suffix: [B, L, C, H, W] - sequence of noisy images
            timesteps: [B, L] - corresponding timesteps
        
        Returns:
            context: [B, context_dim] - encoded suffix context
        """
        B, L, C, H, W = suffix.shape
        
        # Flatten images: [B, L, C*H*W]
        suffix_flat = suffix.view(B, L, -1)
        
        # Project to hidden_dim: [B, L, hidden_dim]
        x = self.image_proj(suffix_flat)
        
        # Add timestep embeddings: [B, L, hidden_dim]
        time_emb = self.time_emb(timesteps)  # [B, L, hidden_dim]
        x = x + time_emb
        
        # Add CLS token if using cls pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, hidden_dim]
        
        # Transformer encoding: [B, L(+1), hidden_dim]
        x = self.transformer(x)
        
        # Pool to fixed size: [B, hidden_dim]
        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "max":
            x = x.max(dim=1)[0]
        elif self.pooling == "cls":
            x = x[:, 0]  # Use CLS token
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Project to context_dim: [B, context_dim]
        context = self.output_proj(x)
        
        return context

