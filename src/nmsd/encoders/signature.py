from __future__ import annotations

from typing import Literal

import math
import torch
import torch.nn as nn

import pysiglib.torch_api as pysiglib

# --- Patch for pysiglib gradient mismatch ---
# The pysiglib.signature function seems to wrap an autograd.Function that
# expects 7 arguments in forward but returns 6 gradients in backward.
# We attempt to patch it here.
def _patch_pysiglib():
    try:
        func = pysiglib.signature
        SigClass = None
        
        # Method 1: Check function globals (most likely for a class used in the function)
        if hasattr(func, "__globals__") and "Signature" in func.__globals__:
            SigClass = func.__globals__["Signature"]
            
        # Method 2: Check closure (in case it was captured)
        elif hasattr(func, "__closure__") and func.__closure__:
            for cell in func.__closure__:
                content = cell.cell_contents
                if isinstance(content, type) and issubclass(content, torch.autograd.Function):
                    SigClass = content
                    break
        
        # Method 3: Check module directly
        elif hasattr(pysiglib, "Signature"):
             SigClass = getattr(pysiglib, "Signature")

        if SigClass and issubclass(SigClass, torch.autograd.Function):
            # Only patch if we can access the backward method
            if hasattr(SigClass, 'backward'):
                original_backward = SigClass.backward
                
                # Check if already patched to avoid recursion
                if getattr(original_backward, "_is_patched", False):
                    return

                @staticmethod
                def patched_backward(ctx, grad_output):
                    grads = original_backward(ctx, grad_output)
                    # Fix for "expected 7, got 6" error
                    if isinstance(grads, tuple) and len(grads) == 6:
                        # Append None for the missing 7th argument
                        return grads + (None,)
                    return grads
                
                patched_backward._is_patched = True
                SigClass.backward = patched_backward
                print(f"Pysiglib patched: {SigClass.__name__}.backward wrapped to fix gradient count (6->7).")
                return True
    except Exception as e:
        print(f"Warning: Failed to attempt pysiglib patch: {e}")
    return False

_patch_pysiglib()
# --------------------------------------------

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
        
        self.path_dim = path_dim
        self.sig_dim = sig_dim
        
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

    def _get_single_step_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Helper to extract features and time-augment for a single step."""
        # x: [B, C, H, W], t: [B]
        
        # Extract features
        if self.pooling == "conv_pool":
            features = self.conv(x)  # [B, feature_dim]
        else:
            # For pool lambda or sequential, inputs are usually expected as [..., C, H, W]
            # Since self.pool in __init__ handles [B, L, C, H, W], we might need to check dimensions
            # But the lambda x.mean(dim=[-2, -1]) works for any number of leading dims.
            # The Flatten/Linear in 'flatten' might expect [B, L, ...].
            if self.pooling == "flatten":
                # Reshape to behave like sequence of length 1 for the Sequential
                # Or just apply directly if layers support it.
                # Flatten(start_dim=-3) works on [B, C, H, W] -> [B, C*H*W]
                # Linear expects [..., in_features]
                features = self.pool(x)
            else:
                features = self.pool(x)
                
        # Time augmentation
        if self.time_augment:
            time_normalized = t.float().unsqueeze(-1) / 1000.0  # [B, 1]
            path = torch.cat([time_normalized, features], dim=-1)  # [B, feature_dim+1]
        else:
            path = features
            
        if path.dtype != torch.float32:
            path = path.float()
            
        return path

    def get_empty_signature(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns identity signature (for path of length 1)."""
        sig = torch.zeros(batch_size, self.sig_dim, device=device)
        sig[:, 0] = 1.0
        return sig

    def forward_incremental(
        self, 
        x_curr: torch.Tensor, 
        t_curr: torch.Tensor,
        x_next: torch.Tensor | None, 
        t_next: torch.Tensor | None,
        running_signature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Incremental signature update for efficient sampling.
        
        Updates the running signature of the suffix x_{t:T} by prepending x_curr.
        Also computes the full context signature for 0 -> x_curr -> ... -> x_T.
        
        Args:
            x_curr: [B, C, H, W] - current image at time t
            t_curr: [B] - current timestep t
            x_next: [B, C, H, W] or None - image at time t+1 (previous in loop)
            t_next: [B] or None - timestep t+1
            running_signature: [B, sig_dim] - signature of x_{t+1:T} (or identity if t=T)
            
        Returns:
            context: [B, context_dim] - encoded context for x_{t:T} with basepoint
            new_running_signature: [B, sig_dim] - updated signature for x_{t:T}
        """
        path_curr = self._get_single_step_features(x_curr, t_curr)  # [B, D]
        
        # 1. Update running signature to include segment x_curr -> x_next
        if x_next is not None:
            path_next = self._get_single_step_features(x_next, t_next)  # [B, D]
            
            # Create segment [x_curr, x_next]
            # pysiglib expects [B, L, D]
            segment = torch.stack([path_curr, path_next], dim=1)  # [B, 2, D]
            
            # Compute signature of this segment
            seg_sig = pysiglib.signature(
                segment,
                degree=self.signature_degree,
                time_aug=False,
                lead_lag=self.use_lead_lag
            )
            # pysiglib.sig_combine requires double inputs
            if seg_sig.dtype != torch.float64:
                seg_sig = seg_sig.double()
            
            running_sig_double = running_signature.double()
                
            # Combine: S(curr->next) * S(next->end)
            new_running_signature = pysiglib.sig_combine(
                seg_sig, 
                running_sig_double, 
                self.path_dim, 
                self.signature_degree
            )
            
            # Convert back to float for storage
            if new_running_signature.dtype == torch.float64:
                new_running_signature = new_running_signature.float()
        else:
            # At t=T, x_{t:T} is just x_T (length 1), so signature is identity
            # running_signature passed in should be identity
            new_running_signature = running_signature
            
        # 2. Compute full signature with basepoint: 0 -> x_curr -> ...
        # First compute segment 0 -> x_curr
        basepoint = torch.zeros_like(path_curr) # [B, D]
        prefix_segment = torch.stack([basepoint, path_curr], dim=1) # [B, 2, D]
        
        prefix_sig = pysiglib.signature(
            prefix_segment,
            degree=self.signature_degree,
            time_aug=False,
            lead_lag=self.use_lead_lag
        )
        if prefix_sig.dtype != torch.float64:
            prefix_sig = prefix_sig.double()
            
        new_running_signature_double = new_running_signature.double()
            
        # Combine: S(0->curr) * S(curr->end)
        full_sig = pysiglib.sig_combine(
            prefix_sig,
            new_running_signature_double,
            self.path_dim,
            self.signature_degree
        )
        
        if full_sig.dtype == torch.float64:
            full_sig = full_sig.float()
        
        # Project to context
        context = self.sig_proj(full_sig)
        
        return context, new_running_signature


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

    def forward_incremental(self, *args, **kwargs):
        return self.signature_encoder.forward_incremental(*args, **kwargs)

    def get_empty_signature(self, *args, **kwargs):
        return self.signature_encoder.get_empty_signature(*args, **kwargs)


class SignatureLinearEncoder(SignatureEncoder):
    """
    Signature encoder with a linear projection.
    
    This model implements: Context -> Signature Transform -> Learnable Linear Weight.
    It leverages the universal approximation property of linear functionals of the signature
    to encode the path context. Unlike SignatureEncoder which uses an MLP, this uses a
    single linear transformation.
    """
    
    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        context_dim: int = 256,
        signature_degree: int = 2,
        pooling: str = "spatial_mean",
        time_augment: bool = True,
        use_lead_lag: bool = False,
        hidden_dim: int = 256,
    ):
        super().__init__(
            image_channels=image_channels,
            image_size=image_size,
            context_dim=context_dim,
            signature_degree=signature_degree,
            pooling=pooling,
            time_augment=time_augment,
            use_lead_lag=use_lead_lag,
            hidden_dim=hidden_dim,
        )
        
        # Replace MLP with single Linear layer
        # "linear functional of signature"
        self.sig_proj = nn.Linear(self.sig_dim, context_dim)
        
        print(f"SignatureLinearEncoder: Linear projection (sig_dim={self.sig_dim} -> context_dim={context_dim})")




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


class SignatureTransformerEncoder(nn.Module):
    """
    Hybrid encoder that combines signature-based feature extraction with transformer temporal modeling.
    
    Architecture:
    1. Use signature encoder's spatial feature extraction (spatial_mean, flatten, or conv_pool)
    2. Apply transformer over the temporal dimension to model sequence dependencies
    3. Pool and project to context_dim
    
    This treats signature methods as sophisticated image feature extractors while using
    transformer attention to capture temporal relationships in the suffix sequence.
    
    Benefits:
    - Signature pooling methods provide better spatial features than simple flattening
    - Transformer captures complex temporal dependencies
    - Combines strengths of both approaches
    """
    
    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 28,
        context_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        pooling: str = "spatial_mean",  # "spatial_mean", "flatten", "conv_pool"
        transformer_pooling: str = "mean",  # "mean", "max", or "cls"
        use_signature_tokens: bool = False, # If True, compute signatures on sliding windows
        signature_degree: int = 2,
        window_size: int = 3,
        time_augment: bool = True,
    ):
        super().__init__()
        
        self.image_channels = image_channels
        self.image_size = image_size
        self.context_dim = context_dim
        self.pooling = pooling
        self.transformer_pooling = transformer_pooling
        self.use_signature_tokens = use_signature_tokens
        self.window_size = window_size
        self.signature_degree = signature_degree
        self.time_augment = time_augment
        
        # Feature extraction from images (borrowed from SignatureEncoder)
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
                nn.Conv2d(image_channels, 32, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_dim),
            )
            self.pool = None  # Handled in forward
        
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Project features to hidden_dim if needed (or if using signature tokens)
        if self.use_signature_tokens:
            # Calculate signature dimension
            path_dim = self.feature_dim + 1 if time_augment else self.feature_dim
            if path_dim == 1:
                sig_dim = 1 + signature_degree
            else:
                sig_dim = (path_dim ** (signature_degree + 1) - 1) // (path_dim - 1)
            
            self.signature_proj = nn.Linear(sig_dim, hidden_dim)
            print(f"Signature tokens enabled: window={window_size}, degree={signature_degree}, sig_dim={sig_dim}")
        
        elif self.feature_dim != hidden_dim:
            self.feature_proj = nn.Linear(self.feature_dim, hidden_dim)
        else:
            self.feature_proj = nn.Identity()
        
        # Timestep embedding
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        
        # Optional CLS token for transformer pooling
        if transformer_pooling == "cls":
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
        if transformer_pooling == "signature":
            # Calculate signature dimension for transformer output
            # Path dimension is hidden_dim (+1 if time_augment)
            path_dim = hidden_dim + 1 if time_augment else hidden_dim
            
            if path_dim == 1:
                sig_dim = 1 + signature_degree
            else:
                sig_dim = (path_dim ** (signature_degree + 1) - 1) // (path_dim - 1)
                
            self.output_proj = nn.Sequential(
                nn.Linear(sig_dim, context_dim * 2),
                nn.SiLU(),
                nn.Linear(context_dim * 2, context_dim),
                nn.LayerNorm(context_dim),
            )
            print(f"Transformer pooling: signature (path_dim={path_dim}, sig_dim={sig_dim})")
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, context_dim),
                nn.LayerNorm(context_dim),
            )
        
        print(f"SignatureTransformerEncoder: pooling={pooling}, feature_dim={self.feature_dim}, "
              f"hidden_dim={hidden_dim}, num_heads={num_heads}, num_layers={num_layers}, "
              f"sig_tokens={use_signature_tokens}, trans_pool={transformer_pooling}")
    
    def forward(self, suffix: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            suffix: [B, L, C, H, W] - sequence of noisy images
            timesteps: [B, L] - corresponding timesteps
        
        Returns:
            context: [B, context_dim] - encoded suffix context
        """
        B, L, C, H, W = suffix.shape
        device = suffix.device
        
        # Extract spatial features from images using signature-based methods
        if self.pooling == "conv_pool":
            # Apply conv to each timestep
            suffix_flat = suffix.view(B * L, C, H, W)
            features = self.conv(suffix_flat)  # [B*L, feature_dim]
            features = features.view(B, L, self.feature_dim)  # [B, L, feature_dim]
        else:
            # Apply pooling
            features = self.pool(suffix)  # [B, L, feature_dim]
        
        if self.use_signature_tokens:
            # Sliding window signatures
            # features: [B, L, D]
            if L < self.window_size:
                 # Pad if sequence is too short for window
                 pad_len = self.window_size - L
                 features = torch.cat([features[:, :1].expand(-1, pad_len, -1), features], dim=1)
                 timesteps = torch.cat([timesteps[:, :1].expand(-1, pad_len), timesteps], dim=1)
                 L = features.shape[1]

            # Create windows: [B, NumWindows, WindowSize, D]
            # unfold returns [B, NumWindows, D, WindowSize], so we permute
            windows = features.unfold(1, self.window_size, 1).permute(0, 1, 3, 2)
            NumWindows = windows.shape[1]
            
            # Prepare for signature computation: flatten batch and windows
            windows_flat = windows.reshape(B * NumWindows, self.window_size, self.feature_dim)
            
            # Handle timestamps for windows (use same unfolding)
            # timesteps: [B, L]
            time_windows = timesteps.unfold(1, self.window_size, 1) # [B, NumWindows, WindowSize]
            time_windows_flat = time_windows.reshape(B * NumWindows, self.window_size)
            
            # Use the last timestep of each window for the transformer position embedding later
            window_timesteps = time_windows[:, :, -1] # [B, NumWindows]
            
            # Time augmentation for signatures
            if self.time_augment:
                time_norm = time_windows_flat.float().unsqueeze(-1) / 1000.0 # [B*NW, WinSize, 1]
                path = torch.cat([time_norm, windows_flat], dim=-1)
            else:
                path = windows_flat
                
            if path.dtype != torch.float32:
                path = path.float()
                
            # Add basepoint zero for signature stability
            BNW, WS, PD = path.shape
            basepoint = torch.zeros(BNW, 1, PD, dtype=path.dtype, device=path.device)
            path_with_basepoint = torch.cat([basepoint, path], dim=1)
            
            # Compute signatures: [B*NumWindows, SigDim]
            sigs = pysiglib.signature(
                path_with_basepoint,
                degree=self.signature_degree,
                time_aug=False,
                lead_lag=False
            )
            
            if sigs.dtype == torch.float64:
                sigs = sigs.float()
                
            # Project to hidden_dim
            sigs_proj = self.signature_proj(sigs) # [B*NumWindows, hidden_dim]
            x = sigs_proj.view(B, NumWindows, -1) # [B, NumWindows, hidden_dim]
            
            # Use window timesteps for position embedding
            timesteps = window_timesteps
            
        else:
            # Standard path: Project to hidden_dim: [B, L, hidden_dim]
            x = self.feature_proj(features)
        
        # Add timestep embeddings: [B, L, hidden_dim]
        time_emb = self.time_emb(timesteps)  # [B, L, hidden_dim]
        x = x + time_emb
        
        # Add CLS token if using cls pooling
        if self.transformer_pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, hidden_dim]
        
        # Transformer encoding: [B, L(+1), hidden_dim]
        x = self.transformer(x)
        
        # Pool to fixed size: [B, hidden_dim]
        if self.transformer_pooling == "mean":
            x = x.mean(dim=1)
        elif self.transformer_pooling == "max":
            x = x.max(dim=1)[0]
        elif self.transformer_pooling == "cls":
            x = x[:, 0]  # Use CLS token
        elif self.transformer_pooling == "signature":
             # Compute signature of x [B, L, H]
             # Time augment if needed
             if self.time_augment:
                 # timesteps corresponds to the time of items in x
                 # Normalize timesteps to [0, 1]
                 time_normalized = timesteps.float().unsqueeze(-1) / 1000.0  # [B, L, 1]
                 path = torch.cat([time_normalized, x], dim=-1)  # [B, L, H+1]
             else:
                 path = x
                 
             if path.dtype != torch.float32:
                 path = path.float()
                 
             # Add basepoint (zero)
             B_sig, L_sig, D_sig = path.shape
             basepoint = torch.zeros(B_sig, 1, D_sig, dtype=path.dtype, device=path.device)
             path_with_basepoint = torch.cat([basepoint, path], dim=1)
             
             # Compute signature
             x = pysiglib.signature(
                 path_with_basepoint,
                 degree=self.signature_degree,
                 time_aug=False,
                 lead_lag=False
             )
             
             if x.dtype == torch.float64:
                 x = x.float()
                 
        else:
            raise ValueError(f"Unknown transformer_pooling: {self.transformer_pooling}")
        
        # Project to context_dim: [B, context_dim]
        context = self.output_proj(x)
        
        return context

