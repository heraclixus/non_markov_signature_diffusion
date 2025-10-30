from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedulers import DiffusionSchedule, q_sample, q_sample_suffix, extract_per_timestep


def ddpm_simple_loss(model, x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule) -> Dict[str, torch.Tensor]:
    """
    The "simple" DDPM objective: predict epsilon with MSE.
    """
    eps = torch.randn_like(x0)
    x_t = q_sample(x0, t, schedule, noise=eps)
    eps_pred = model(x_t, t)
    loss = F.mse_loss(eps_pred, eps)
    return {"loss": loss}


def nonmarkov_suffix_loss(
    model: nn.Module,
    encoder: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    num_suffix_steps: int = 10,
    context_dropout: float = 0.1,  # For CFG training
) -> Dict[str, torch.Tensor]:
    """
    Non-Markov training loss with suffix context (epsilon prediction).
    
    For each training sample:
    1. Sample the suffix x_{t:t+num_suffix_steps-1} in parallel (conditionally independent given x_0)
    2. Encode the suffix to get context c_t = Encoder(x_{t:T}, timesteps)
    3. Predict epsilon at time t: eps_pred = model(x_t, t, c_t)
    4. Minimize ||eps_t - eps_pred||^2
    
    Args:
        model: ContextUNet that accepts (x, t, context)
        encoder: SuffixEncoder that maps (suffix, timesteps) -> context
        x0: clean images [B, C, H, W]
        t: timestep for each batch element [B]
        schedule: diffusion schedule
        num_suffix_steps: number of steps in the suffix sequence
    
    Returns:
        dict with 'loss' key
    """
    device = x0.device
    B = x0.shape[0]
    
    # Sample suffix x_{t:t+num_suffix_steps-1} in parallel
    # suffix: [B, num_suffix_steps, C, H, W]
    # timesteps: [B, num_suffix_steps]
    suffix, suffix_timesteps = q_sample_suffix(x0, t, schedule, num_suffix_steps)
    
    # Encode suffix to get context
    # context: [B, context_dim]
    context = encoder(suffix, suffix_timesteps)
    
    # Classifier-free guidance training: randomly drop context
    if context_dropout > 0:
        mask = torch.rand(B, device=device) > context_dropout
        context = context * mask.unsqueeze(1).float()
    
    # Get x_t and target noise at timestep t
    # The first element of the suffix is x_t
    x_t = suffix[:, 0]  # [B, C, H, W]
    
    # We need the actual noise used to generate x_t
    # Recompute x_t with known noise
    eps_target = torch.randn_like(x0)
    x_t = q_sample(x0, t, schedule, noise=eps_target)
    
    # Predict noise with context
    eps_pred = model(x_t, t, context)
    
    # MSE loss
    loss = F.mse_loss(eps_pred, eps_target)
    
    return {"loss": loss}


def dart_loss(
    model: nn.Module,
    encoder: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    num_suffix_steps: int = 10,
    weighting: str = "snr_sum",  # "none", "simple", "snr_sum", "truncated_snr"
    context_dropout: float = 0.1,  # Probability of dropping context for CFG training
) -> Dict[str, torch.Tensor]:
    """
    DART (Denoising AutoRegressive Transformer) loss.
    
    Instead of predicting epsilon, directly predict x_0 from the suffix x_{t:T}.
    Following the DART paper:
        L_DART = E_{x_{1:T} ~ q(x_0)} [ Σ_t ω_t · ||x_θ(x_{t:T}) - x_0||^2 ]
    
    where:
        - x_θ(x_{t:T}) is the model's prediction of x_0 given the suffix
        - ω_t weighting depends on the chosen scheme
    
    For each training sample:
    1. Sample the suffix x_{t:t+k} in parallel (conditionally independent given x_0)
    2. Encode the suffix to get context c_t = Encoder(x_{t:T}, timesteps)
    3. Predict x_0: x0_pred = model(x_t, t, c_t)
    4. Minimize ω_t · ||x_0 - x0_pred||^2
    
    Args:
        model: ContextUNet that accepts (x, t, context) and predicts x_0
        encoder: SuffixEncoder that maps (suffix, timesteps) -> context
        x0: clean images [B, C, H, W]
        t: timestep for each batch element [B]
        schedule: diffusion schedule
        num_suffix_steps: number of steps in the suffix sequence
        weighting: weighting scheme
            - "none": no weighting (uniform)
            - "simple": ω_t = γ_{t-1} / (1 - γ_{t-1})  [single timestep SNR]
            - "snr_sum": ω_t = Σ_{τ=t}^T γ_τ / (1-γ_τ)  [cumulative SNR, DART paper]
            - "truncated_snr": ω_t = Σ_{τ=t}^{min(t+k, T)} γ_τ / (1-γ_τ)  [truncated sum]
    
    Returns:
        dict with 'loss', 'loss_unweighted', and 'avg_weight'
    """
    device = x0.device
    B = x0.shape[0]
    
    # Sample suffix x_{t:t+num_suffix_steps-1} in parallel
    # suffix: [B, num_suffix_steps, C, H, W]
    # timesteps: [B, num_suffix_steps]
    suffix, suffix_timesteps = q_sample_suffix(x0, t, schedule, num_suffix_steps)
    
    # Encode suffix to get context
    # context: [B, context_dim]
    context = encoder(suffix, suffix_timesteps)
    
    # Classifier-free guidance training: randomly drop context
    if context_dropout > 0:
        # Create dropout mask [B]
        mask = torch.rand(B, device=device) > context_dropout  # True = keep context
        # Zero out context for dropped samples
        context = context * mask.unsqueeze(1).float()  # [B, context_dim]
    
    # Get x_t (first element of the suffix)
    x_t = suffix[:, 0]  # [B, C, H, W]
    
    # Model directly predicts x_0 (not epsilon)
    x0_pred = model(x_t, t, context)
    
    # Compute unweighted loss
    loss_unweighted = F.mse_loss(x0_pred, x0, reduction='none').mean(dim=[1, 2, 3])  # [B]
    
    # Compute weights based on scheme
    if weighting == "none":
        weights = torch.ones(B, device=device)
    
    elif weighting == "simple":
        # ω_t = γ_{t-1} / (1 - γ_{t-1})  [single timestep SNR]
        gamma_prev = extract_per_timestep(schedule.gamma, (t - 1).clamp(min=0), x0.shape)  # [B, 1, 1, 1]
        weights = gamma_prev / (1.0 - gamma_prev + 1e-8)
        weights = weights.squeeze()  # [B]
    
    elif weighting == "snr_sum":
        # ω_t = Σ_{τ=t}^T γ_τ / (1-γ_τ)  [cumulative SNR from DART paper]
        # Vectorized implementation using cumsum for efficiency
        
        # Precompute SNR for all timesteps: [T+1]
        snr_all = schedule.gamma / (1.0 - schedule.gamma + 1e-8)  # [T+1]
        
        # Compute cumulative sum from the end: cumsum[t] = Σ_{τ=t}^T SNR(τ)
        # Reverse, cumsum, reverse again
        snr_cumsum = torch.flip(torch.cumsum(torch.flip(snr_all, [0]), dim=0), [0])  # [T+1]
        
        # Extract weight for each batch element's timestep
        weights = snr_cumsum[t]  # [B]
    
    elif weighting == "truncated_snr":
        # ω_t = Σ_{τ=t}^{min(t+k, T)} γ_τ / (1-γ_τ)  [truncated sum over suffix]
        # Vectorized implementation
        
        # Precompute SNR for all timesteps
        snr_all = schedule.gamma / (1.0 - schedule.gamma + 1e-8)  # [T+1]
        
        # For each batch element, sum from t to min(t+k, T)
        weights = torch.zeros(B, device=device)
        for i in range(B):
            t_val = int(t[i].item())
            end_tau = min(t_val + num_suffix_steps, schedule.T + 1)
            weights[i] = snr_all[t_val:end_tau].sum()
    
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting}")
    
    # Normalize weights to prevent explosion (optional but recommended)
    # Divide by mean to keep scale roughly similar to unweighted loss
    if weighting != "none":
        weights = weights / (weights.mean() + 1e-8)
    
    # Weighted loss
    loss = (weights * loss_unweighted).mean()
    
    return {
        "loss": loss,
        "loss_unweighted": loss_unweighted.mean(),
        "avg_weight": weights.mean().item(),
    }


