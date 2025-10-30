from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    T: int
    gamma: torch.Tensor  # shape [T+1], gamma[0]=1, gamma[T]~0
    alpha: torch.Tensor  # shape [T+1], alpha[0]=1 (dummy), alpha[t]=gamma[t]/gamma[t-1]
    beta: torch.Tensor   # shape [T+1], beta[0]=0 (dummy), beta[t]=1-alpha[t]

    def to(self, device: torch.device) -> DiffusionSchedule:
        self.gamma = self.gamma.to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        return self


def build_cosine_schedule(T: int, s: float = 0.008) -> DiffusionSchedule:
    """
    Cosine schedule from Nichol & Dhariwal (2021). We construct gamma_t (a.k.a. alpha_bar_t)
    over t=0..T with gamma_0=1 and gamma_T≈0.
    """
    steps = T
    t = torch.linspace(0, T, T + 1)
    f = torch.cos(((t / T + s) / (1 + s)) * torch.pi / 2) ** 2
    f = f / f[0]
    gamma = f.clamp(min=1e-8, max=1.0)
    alpha = torch.ones_like(gamma)
    alpha[1:] = gamma[1:] / gamma[:-1]
    beta = 1.0 - alpha
    beta[0] = 0.0
    alpha[0] = 1.0
    return DiffusionSchedule(T=T, gamma=gamma, alpha=alpha, beta=beta)


def build_schedule(T: int, schedule: str = "cosine", **kwargs) -> DiffusionSchedule:
    if schedule == "cosine":
        return build_cosine_schedule(T=T, s=float(kwargs.get("cosine_s", 0.008)))
    raise ValueError(f"Unknown schedule: {schedule}")


def extract_per_timestep(arr: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract index t from a 1D tensor arr and reshape to [B, 1, 1, 1] to match x dims.
    """
    out = arr.gather(-1, t.clamp(min=0, max=arr.shape[0] - 1))
    while out.dim() < len(x_shape):
        out = out.unsqueeze(-1)
    return out


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DiffusionSchedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    """
    x_t = sqrt(gamma_t) * x0 + sqrt(1 - gamma_t) * eps
    where gamma_t is the cumulative product (alpha_bar).
    """
    if noise is None:
        noise = torch.randn_like(x0)
    gamma_t = extract_per_timestep(schedule.gamma, t, x0.shape)
    return torch.sqrt(gamma_t) * x0 + torch.sqrt(1.0 - gamma_t) * noise


def q_sample_suffix(x0: torch.Tensor, t_start: torch.Tensor, schedule: DiffusionSchedule, num_suffix_steps: int = 10) -> torch.Tensor:
    """
    Non-Markov forward: sample suffix x_{t:T} in parallel.
    Given x_0 and starting timestep t, sample {x_t, x_{t+1}, ..., x_{t+num_suffix_steps-1}}.
    
    Each x_s is conditionally independent given x_0:
        x_s = sqrt(gamma_s) * x_0 + sqrt(1 - gamma_s) * eps_s
    
    Args:
        x0: clean data [B, C, H, W]
        t_start: starting timestep for each batch element [B]
        schedule: diffusion schedule
        num_suffix_steps: number of timesteps in the suffix
    
    Returns:
        suffix: [B, num_suffix_steps, C, H, W] - the suffix sequence x_{t:t+num_suffix_steps-1}
        timesteps: [B, num_suffix_steps] - corresponding timesteps
    """
    B, C, H, W = x0.shape
    device = x0.device
    
    # Create timestep sequence for each batch element
    # [B, num_suffix_steps]
    offsets = torch.arange(num_suffix_steps, device=device).unsqueeze(0)  # [1, num_suffix_steps]
    timesteps = (t_start.unsqueeze(1) + offsets).clamp(1, schedule.T)  # [B, num_suffix_steps]
    
    # Sample independent noise for each timestep in the suffix
    noise = torch.randn(B, num_suffix_steps, C, H, W, device=device)  # [B, num_suffix_steps, C, H, W]
    
    # Compute gamma for each timestep
    # Extract gamma values: [B, num_suffix_steps, 1, 1, 1]
    gamma = schedule.gamma[timesteps].view(B, num_suffix_steps, 1, 1, 1)
    
    # Broadcast x0: [B, 1, C, H, W]
    x0_expanded = x0.unsqueeze(1)  # [B, 1, C, H, W]
    
    # Parallel sampling: x_s = sqrt(gamma_s) * x_0 + sqrt(1 - gamma_s) * eps_s
    suffix = torch.sqrt(gamma) * x0_expanded + torch.sqrt(1.0 - gamma) * noise
    
    return suffix, timesteps


