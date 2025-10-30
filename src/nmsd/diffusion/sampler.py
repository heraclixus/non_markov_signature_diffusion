from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .schedulers import DiffusionSchedule, extract_per_timestep, q_sample_suffix


@torch.no_grad()
def ddim_sample_step(
    model, 
    x_t: torch.Tensor, 
    t: torch.Tensor, 
    t_prev: torch.Tensor,
    schedule: DiffusionSchedule, 
    eta: float = 0.0
) -> torch.Tensor:
    """
    One DDIM reverse step from t -> t_prev.
    
    Args:
        model: noise prediction model
        x_t: current noisy sample at timestep t
        t: current timestep (can be any value in [1, T])
        t_prev: previous timestep to step to (can be any value in [0, t-1])
        schedule: diffusion schedule
        eta: stochasticity parameter (0 = deterministic DDIM, 1 = DDPM-like)
    
    Returns:
        x_{t_prev}: denoised sample at timestep t_prev
    """
    gamma_t = extract_per_timestep(schedule.gamma, t, x_t.shape)
    gamma_prev = extract_per_timestep(schedule.gamma, t_prev, x_t.shape)
    
    # Predict noise
    eps_pred = model(x_t, t)
    
    # Predict x0
    x0_pred = (x_t - torch.sqrt(1.0 - gamma_t) * eps_pred) / torch.sqrt(gamma_t)
    x0_pred = x0_pred.clamp(-1.0, 1.0)
    
    # DDIM formula with optional stochasticity
    # Direction pointing to x_t
    dir_xt = torch.sqrt(1.0 - gamma_prev - eta**2 * (1 - gamma_prev) / (1 - gamma_t) * (1 - gamma_t / gamma_prev + 1e-8)) * eps_pred
    
    # Deterministic part
    x_prev = torch.sqrt(gamma_prev) * x0_pred + dir_xt
    
    # Stochastic part (eta > 0)
    if eta > 0:
        sigma = eta * torch.sqrt((1 - gamma_prev) / (1 - gamma_t) * (1 - gamma_t / (gamma_prev + 1e-8)))
        noise = torch.randn_like(x_t)
        # Only add noise if not at the final step
        mask = (t_prev > 0).float().view(-1, 1, 1, 1)
        x_prev = x_prev + mask * sigma * noise
    
    return x_prev


@torch.no_grad()
def ddim_sample_loop(
    model, 
    shape, 
    schedule: DiffusionSchedule, 
    device: torch.device, 
    num_steps: int = 50,
    eta: float = 0.0
) -> torch.Tensor:
    """
    DDIM sampling loop with configurable number of steps.
    
    Args:
        model: noise prediction model
        shape: output shape (B, C, H, W)
        schedule: diffusion schedule (trained with T steps)
        device: torch device
        num_steps: number of sampling steps (can be much less than T, e.g., 50)
        eta: stochasticity (0 = deterministic, 1 = stochastic like DDPM)
    
    Returns:
        samples: generated samples
    """
    # Create sampling timestep schedule (uniform spacing)
    # Go from T -> 0 in num_steps steps
    timesteps = np.linspace(schedule.T, 0, num_steps + 1).astype(int)
    timesteps = torch.from_numpy(timesteps).to(device)
    
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    # Reverse diffusion
    for i in range(num_steps):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        t_prev_batch = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
        
        x = ddim_sample_step(model, x, t_batch, t_prev_batch, schedule, eta=eta)
    
    return x


@torch.no_grad()
def p_sample_loop(model, shape, schedule: DiffusionSchedule, device: torch.device, eta: float = 1.0) -> torch.Tensor:
    """
    Legacy DDPM sampling (slow, 1000 steps). Use ddim_sample_loop instead.
    """
    x = torch.randn(shape, device=device)
    for step in range(schedule.T, 0, -1):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        t_prev = torch.full((shape[0],), step - 1, device=device, dtype=torch.long)
        x = ddim_sample_step(model, x, t, t_prev, schedule, eta=eta)
    return x


@torch.no_grad()
def dart_simple_sample_loop(
    model: nn.Module,
    encoder: nn.Module,
    shape,
    schedule: DiffusionSchedule,
    device: torch.device,
    num_steps: int = 50,
    num_suffix_steps: int = 5,
    noise_scale: float = 1.0,
    cfg_scale: float = 0.0,  # Classifier-free guidance scale (0 = no CFG)
) -> torch.Tensor:
    """
    Simplified DART sampling as described in the DART paper.
    
    At each step:
    1. Build suffix x_{t:t+k}
    2. Predict mean: x_θ(x_{t:T}) with optional classifier-free guidance
    3. Add Gaussian noise: x_{t-1} = x_θ(x_{t:T}) + σ_t · ε
    
    Classifier-free guidance (CFG):
        x0_guided = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
    where:
        - x0_cond: prediction with context
        - x0_uncond: prediction with zero context
        - cfg_scale: guidance strength (0 = no guidance, higher = stronger)
    
    Args:
        model: ContextUNet that predicts x₀
        encoder: SuffixEncoder
        shape: output shape (B, C, H, W)
        schedule: diffusion schedule
        device: torch device
        num_steps: number of sampling steps
        num_suffix_steps: number of future steps to encode as context
        noise_scale: scale of added Gaussian noise (default: 1.0)
        cfg_scale: classifier-free guidance scale (0 = disabled, 1-2 = typical)
    
    Returns:
        samples: [B, C, H, W]
    """
    # Create sampling timestep schedule
    timesteps = np.linspace(schedule.T, 0, num_steps + 1).astype(int)
    timesteps = torch.from_numpy(timesteps).to(device)
    
    # Start from pure noise
    x = torch.randn(shape, device=device)
    B = shape[0]
    
    # Reverse diffusion
    for i in range(num_steps):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        # Build suffix context (same as before)
        gamma_t = extract_per_timestep(schedule.gamma, t_batch, x.shape)
        
        suffix_t_vals = torch.linspace(
            t.item(), 
            min(schedule.T, t.item() + num_suffix_steps * (schedule.T // num_steps)),
            num_suffix_steps
        ).long().to(device)
        
        suffix_list = []
        for j, t_future in enumerate(suffix_t_vals):
            gamma_future = schedule.gamma[t_future]
            noise_scale_future = torch.sqrt((gamma_t - gamma_future).clamp(min=0) / (gamma_t + 1e-8))
            x_future = x + noise_scale_future * torch.randn_like(x)
            suffix_list.append(x_future)
        
        suffix = torch.stack(suffix_list, dim=1)
        suffix_timesteps = suffix_t_vals.unsqueeze(0).expand(B, -1)
        
        # Encode suffix
        context = encoder(suffix, suffix_timesteps)
        
        # Predict x₀ with optional classifier-free guidance
        if cfg_scale > 0:
            # Conditional prediction (with context)
            x0_cond = model(x, t_batch, context)
            
            # Unconditional prediction (zero context)
            context_uncond = torch.zeros_like(context)
            x0_uncond = model(x, t_batch, context_uncond)
            
            # CFG: x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            x0_pred = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        else:
            # Standard prediction (no CFG)
            x0_pred = model(x, t_batch, context).clamp(-1.0, 1.0)
        
        # Simple update: x_{t-1} = x₀_pred + noise
        # Noise scale could be based on schedule or fixed
        if t_prev > 0:
            # Use a simple noise schedule (could be tuned)
            gamma_prev = schedule.gamma[t_prev]
            sigma = noise_scale * torch.sqrt(1.0 - gamma_prev)
            noise = torch.randn_like(x)
            x = x0_pred + sigma * noise
        else:
            # Final step: no noise
            x = x0_pred
    
    return x


@torch.no_grad()
def nonmarkov_ddim_sample_loop(
    model: nn.Module,
    encoder: nn.Module,
    shape,
    schedule: DiffusionSchedule,
    device: torch.device,
    num_steps: int = 50,
    num_suffix_steps: int = 5,
    eta: float = 0.0,
    prediction_type: str = "epsilon",  # "epsilon" or "x0"
) -> torch.Tensor:
    """
    Non-Markov DDIM sampling with suffix context.
    
    At each reverse step t, we:
    1. Generate a suffix x_{t:t+k} by adding noise to the current prediction
    2. Encode the suffix to get context
    3. Use context-conditioned model to predict epsilon or x0
    4. Take one DDIM step
    
    Args:
        model: ContextUNet that accepts (x, t, context)
        encoder: SuffixEncoder
        shape: output shape (B, C, H, W)
        schedule: diffusion schedule
        device: torch device
        num_steps: number of DDIM steps
        num_suffix_steps: number of future steps to encode as context
        eta: DDIM stochasticity
        prediction_type: what the model predicts - "epsilon" (noise) or "x0" (clean image)
                        Use "epsilon" for standard non-Markov, "x0" for DART
    
    Returns:
        samples: [B, C, H, W]
    """
    # Create sampling timestep schedule
    timesteps = np.linspace(schedule.T, 0, num_steps + 1).astype(int)
    timesteps = torch.from_numpy(timesteps).to(device)
    
    # Start from pure noise
    x = torch.randn(shape, device=device)
    B = shape[0]
    
    # Reverse diffusion
    for i in range(num_steps):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        t_prev_batch = torch.full((B,), t_prev, device=device, dtype=torch.long)
        
        # Build suffix context
        # For simplicity, we create a "pseudo-suffix" by predicting x0 and re-noising it
        # In practice, we approximate the suffix x_{t:t+k} 
        
        # First, get current x0 prediction without context (or use previous context)
        gamma_t = extract_per_timestep(schedule.gamma, t_batch, x.shape)
        
        # Simple heuristic: use uniform spacing for suffix timesteps
        suffix_t_vals = torch.linspace(
            t.item(), 
            min(schedule.T, t.item() + num_suffix_steps * (schedule.T // num_steps)),
            num_suffix_steps
        ).long().to(device)
        
        # Create a pseudo-suffix by adding progressively more noise
        # This is an approximation since we don't know the true future trajectory
        suffix_list = []
        for j, t_future in enumerate(suffix_t_vals):
            gamma_future = schedule.gamma[t_future]
            # Add more noise proportional to how far in the future
            noise_scale = torch.sqrt((gamma_t - gamma_future).clamp(min=0) / (gamma_t + 1e-8))
            x_future = x + noise_scale * torch.randn_like(x)
            suffix_list.append(x_future)
        
        suffix = torch.stack(suffix_list, dim=1)  # [B, num_suffix_steps, C, H, W]
        suffix_timesteps = suffix_t_vals.unsqueeze(0).expand(B, -1)  # [B, num_suffix_steps]
        
        # Encode suffix
        context = encoder(suffix, suffix_timesteps)
        
        # DDIM step with context
        gamma_prev = extract_per_timestep(schedule.gamma, t_prev_batch, x.shape)
        
        # Get model prediction (epsilon or x0 depending on training)
        model_output = model(x, t_batch, context)
        
        if prediction_type == "epsilon":
            # Model predicts noise, derive x0
            eps_pred = model_output
            x0_pred = (x - torch.sqrt(1.0 - gamma_t) * eps_pred) / torch.sqrt(gamma_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        elif prediction_type == "x0":
            # Model predicts x0 directly (DART)
            x0_pred = model_output.clamp(-1.0, 1.0)
            # Derive epsilon from x0 for DDIM formula
            eps_pred = (x - torch.sqrt(gamma_t) * x0_pred) / torch.sqrt(1.0 - gamma_t)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        
        # DDIM formula (same for both cases, uses x0_pred and eps_pred)
        dir_xt = torch.sqrt(1.0 - gamma_prev - eta**2 * (1 - gamma_prev) / (1 - gamma_t + 1e-8) * (1 - gamma_t / (gamma_prev + 1e-8))) * eps_pred
        x_prev = torch.sqrt(gamma_prev) * x0_pred + dir_xt
        
        if eta > 0 and t_prev > 0:
            sigma = eta * torch.sqrt((1 - gamma_prev) / (1 - gamma_t + 1e-8) * (1 - gamma_t / (gamma_prev + 1e-8)))
            noise = torch.randn_like(x)
            x_prev = x_prev + sigma * noise
        
        x = x_prev
    
    return x


