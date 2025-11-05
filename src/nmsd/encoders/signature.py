from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

import pysiglib.torch_api as pysiglib

class SignatureEncoder(nn.Module):
    """
    Path signature encoder for suffix sequences x_{t:T}.
    
    Uses truncated path signature (via pysiglib) to encode temporal sequences
    into fixed-size context vectors. The signature provides a principled,
    permutation-sensitive summary of the path.
    
    For a sequence of images [x_t, x_{t+1}, ..., x_{t+k}], we:
    1. Pool each image to a feature vector (e.g., spatial average)
    2. Create time-augmented path: [(t, feat_t), (t+1, feat_{t+1}), ...]
    3. Compute truncated signature up to degree m
    4. Project signature to context_dim
    """
    
    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        context_dim: int = 256,
        signature_degree: int = 2,
        pooling: str = "spatial_mean",  # "spatial_mean", "flatten", "conv_pool"
        time_augment: bool = True,
        use_lead_lag: bool = False,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.context_dim = context_dim
        self.signature_degree = signature_degree
        self.pooling = pooling
        self.time_augment = time_augment
        self.use_lead_lag = use_lead_lag
        
        # Feature extraction from images
        if pooling == "spatial_mean":
            # Simple spatial mean pooling
            self.feature_dim = image_channels
            self.pool = lambda x: x.mean(dim=[-2, -1])  # [B, L, C, H, W] -> [B, L, C]
        
        elif pooling == "flatten":
            # Flatten and project
            self.feature_dim = hidden_dim
            self.pool = nn.Sequential(
                nn.Flatten(start_dim=-3),  # [B, L, C*H*W]
                nn.Linear(image_channels * image_size * image_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        
        elif pooling == "conv_pool":
            # Convolutional feature extraction
            self.feature_dim = hidden_dim
            self.conv = nn.Sequential(
                nn.Conv2d(image_channels, 32, 3, stride=2, padding=1),  # 28->14
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14->7
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),  # 7->1
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
            )
            # Need to apply conv per-timestep
            self.pool = None  # Handled in forward
        
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Compute signature output dimension
        # For time-augmented path: feature_dim + 1 (time channel)
        path_dim = self.feature_dim + 1 if time_augment else self.feature_dim
        
        # Signature dimension calculation
        # pysiglib.signature returns the full signature tensor (including constant 1)
        # For a path of dimension d and degree N, the signature dimension is:
        # sig_dim = 1 + d + d^2 + ... + d^N = (d^(N+1) - 1) / (d - 1) for d > 1
        # For d = 1: sig_dim = 1 + N
        
        if path_dim == 1:
            sig_dim = 1 + signature_degree
        else:
            sig_dim = (path_dim ** (signature_degree + 1) - 1) // (path_dim - 1)
        
        # If using lead-lag, the effective dimension roughly doubles
        # lead-lag transforms [x_1, ..., x_n] to [(x_0,x_1), (x_1,x_1), (x_1,x_2), ...]
        # This approximately doubles the path dimension
        if use_lead_lag:
            # Re-calculate for doubled dimension
            path_dim_ll = 2 * path_dim
            if path_dim_ll == 1:
                sig_dim = 1 + signature_degree
            else:
                sig_dim = (path_dim_ll ** (signature_degree + 1) - 1) // (path_dim_ll - 1)
        
        print(f"SignatureEncoder: path_dim={path_dim}, degree={signature_degree}, sig_dim={sig_dim}")
        
        # Project signature to context_dim
        self.sig_proj = nn.Sequential(
            nn.Linear(sig_dim, context_dim * 2),
            nn.SiLU(),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
        )
    
    def forward(self, suffix: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            suffix: [B, L, C, H, W] - sequence of noisy images
            timesteps: [B, L] - corresponding timesteps
        
        Returns:
            context: [B, context_dim] - signature-encoded context
        """
        B, L, C, H, W = suffix.shape
        device = suffix.device
        
        # Extract features from images
        if self.pooling == "conv_pool":
            # Apply conv to each timestep
            # Reshape: [B, L, C, H, W] -> [B*L, C, H, W]
            suffix_flat = suffix.view(B * L, C, H, W)
            features = self.conv(suffix_flat)  # [B*L, feature_dim]
            features = features.view(B, L, self.feature_dim)  # [B, L, feature_dim]
        else:
            # Apply pooling
            features = self.pool(suffix)  # [B, L, feature_dim]
        
        # Time augmentation: append normalized time as a channel
        if self.time_augment:
            # Normalize timesteps to [0, 1]
            time_normalized = timesteps.float().unsqueeze(-1) / 1000.0  # [B, L, 1]
            path = torch.cat([time_normalized, features], dim=-1)  # [B, L, feature_dim+1]
        else:
            path = features  # [B, L, feature_dim]
        
        # Ensure path is float32 for pysiglib compatibility
        if path.dtype != torch.float32:
            path = path.float()
        
        # Add basepoint (zero) at the beginning of each path
        # Following utils_diff.py pattern: add_basepoint_zero
        # This is important for signature stability
        B, L, D = path.shape
        basepoint = torch.zeros(B, 1, D, dtype=path.dtype, device=path.device)
        path_with_basepoint = torch.cat([basepoint, path], dim=1)  # [B, L+1, D]
        
        # Compute signature for each path in the batch
        # pysiglib.signature expects: (batch_size, length, dimension)
        # Returns: (batch_size, sig_dim)
        signature = pysiglib.signature(
            path_with_basepoint,
            degree=self.signature_degree,
            time_aug=False,  # Already manually time-augmented if requested
            lead_lag=self.use_lead_lag,
        )
        
        # Convert to float32 if needed (pysiglib may return float64)
        if signature.dtype == torch.float64:
            signature = signature.float()
        
        # Project to context_dim
        context = self.sig_proj(signature)  # [B, context_dim]
        
        return context


class LogSignatureEncoder(nn.Module):
    """
    Log-signature encoder for suffix sequences.
    
    Similar to SignatureEncoder but uses log-signature, which is more compact
    and numerically stable for longer sequences.
    """
    
    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        context_dim: int = 256,
        signature_degree: int = 2,
        pooling: str = "spatial_mean",
        time_augment: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Use same feature extraction as SignatureEncoder
        self.signature_encoder = SignatureEncoder(
            image_channels=image_channels,
            image_size=image_size,
            context_dim=context_dim,
            signature_degree=signature_degree,
            pooling=pooling,
            time_augment=time_augment,
            use_lead_lag=False,  # Log-sig doesn't use lead-lag
            hidden_dim=hidden_dim,
        )
        
        print(f"LogSignatureEncoder initialized (wraps SignatureEncoder)")
    
    def forward(self, suffix: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Note: pysiglib doesn't have a separate log-signature function,
        so we use the regular signature. For true log-signature, you'd need
        to use signatory or iisignature libraries.
        
        For now, this is equivalent to SignatureEncoder.
        """
        # TODO: Implement proper log-signature when pysiglib adds support
        # For now, use regular signature
        return self.signature_encoder(suffix, timesteps)

