"""
Diffusion Utilities for Distributional Models

This module contains common utilities for distributional diffusion models,
including noise schedules, GP noise generation, DDIM sampling, and a simple
transformer model variant.
"""

from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Diffusion Schedule
# ============================================================================

def get_betas(steps: int, beta_start: float = 1e-4, beta_end: float = 0.2, 
              device: torch.device = None) -> torch.Tensor:
    """
    Generate linear beta schedule for diffusion process.
    
    Args:
        steps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        device: Target device
        
    Returns:
        Beta values [steps] for each diffusion step
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind


def setup_diffusion_schedule(steps: int, device: torch.device) -> tuple:
    """
    Setup complete diffusion schedule with betas and alphas.
    
    Args:
        steps: Number of diffusion steps
        device: Target device
        
    Returns:
        Tuple of (betas, alphas) where alphas are cumulative products
    """
    betas = get_betas(steps, device=device)
    alphas = torch.cumprod(1 - betas, dim=0)
    return betas, alphas


# ============================================================================
# Gaussian Process Noise
# ============================================================================

def get_gp_covariance(t: torch.Tensor, gp_sigma: float = 0.05) -> torch.Tensor:
    """
    Get Gaussian Process covariance matrix with RBF kernel.
    
    Args:
        t: Time points [B, S, 1]
        gp_sigma: GP length scale parameter
        
    Returns:
        Covariance matrix [B, S, S]
    """
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) * 1e-5  # for numerical stability
    return torch.exp(-torch.square(s / gp_sigma)) + diag


def add_gp_noise(x: torch.Tensor, t: torch.Tensor, i: torch.Tensor, 
                 alphas: torch.Tensor, gp_sigma: float = 0.05) -> tuple:
    """
    Add Gaussian Process structured noise to clean data sample.
    
    This is the forward diffusion process: q(x_t | x_0)
    Uses GP with RBF kernel to maintain temporal smoothness.
    
    Args:
        x: Clean data sample [B, S, D]
        t: Times of observations [B, S, 1]
        i: Diffusion step [B, S, 1]
        alphas: Cumulative alpha values for diffusion
        gp_sigma: GP length scale parameter
        
    Returns:
        Tuple of (x_noisy, noise)
        - x_noisy: [B, S, D] noisy sample
        - noise: [B, S, D] the GP noise that was added
    """
    # Generate standard Gaussian noise
    noise_gaussian = torch.randn_like(x, dtype=torch.float32)
    
    # Apply GP covariance structure
    cov = get_gp_covariance(t, gp_sigma=gp_sigma)
    L = torch.linalg.cholesky(cov)
    noise = L @ noise_gaussian
    
    # Forward diffusion: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    return x_noisy, noise


# ============================================================================
# Data Generation
# ============================================================================

def generate_sinusoidal_data(N: int, T: int, freq: float = 10.0, 
                            phase_scale: float = 2 * np.pi,
                            normalize: bool = True,
                            device: torch.device = None) -> tuple:
    """
    Generate synthetic sinusoidal time series data with random phases.
    
    Args:
        N: Number of samples
        T: Number of time points
        freq: Frequency of sine wave
        phase_scale: Scale of random phase offset
        normalize: If True, normalize to [0, 1]
        device: Target device
        
    Returns:
        Tuple of (t, x) where:
        - t: [N, T, 1] time points
        - x: [N, T, 1] sinusoidal values
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    t = torch.linspace(0, 1, T, dtype=torch.float32).view(1, T, 1).expand(N, T, 1).to(device)
    x = torch.sin(freq * t + phase_scale * torch.rand(N, 1, 1, dtype=torch.float32).to(t))
    
    if normalize:
        x = (x + 1) / 2
    
    return t, x


# ============================================================================
# Simple Transformer Model (for distributional learning)
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        scale = torch.exp(
            -2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim
        )
        shift = torch.zeros(periodic_dim)
        shift[::2] = 0.5 * math.pi

        # Register as buffers so they get converted with .double()
        self.register_buffer('scale', scale)
        self.register_buffer('shift', shift)

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(
            t / torch.tensor(self.max_value, dtype=t.dtype, device=t.device)
        )
        return torch.cat([linear, periodic], -1)


class FeedForward(nn.Module):
    """Multi-layer feedforward network"""
    
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int,
                 activation: Callable = nn.ReLU(), final_activation: Callable = None):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleTransformerModel(nn.Module):
    """
    Simple transformer model for distributional diffusion with latent variable.
    
    This is a lightweight version without LayerNorm/dropout, suitable for quick
    experiments. For production use, consider the stabilized TransformerModel
    in signature_models.py.
    
    Forward signature: model(x_t, t, i, xi) -> x_0_sample
    where xi is a latent noise variable that induces stochasticity.
    """
    
    def __init__(self, dim: int, hidden_dim: int, max_i: int, num_layers: int = 8, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.t_enc = PositionalEncoding(hidden_dim, max_value=1)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)

        self.input_proj = FeedForward(dim, [], hidden_dim)
        self.latent_proj = FeedForward(dim, [], hidden_dim)  # for xi latent

        # Projection takes x, t, i, xi => 4 * hidden_dim
        self.proj = FeedForward(4 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())

        self.enc_att = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
            for _ in range(num_layers)
        ])
        self.i_proj = nn.ModuleList([
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = FeedForward(hidden_dim, [], dim)

    def forward(self, x, t, i, xi=None):
        """
        Forward pass with latent variable for distributional predictions.
        
        Args:
            x: [B, S, D] noisy input (x_t)
            t: [B, S, 1] timestamps
            i: [B, S, 1] diffusion step
            xi: [B, S, D] latent noise ~ N(0, I); if None, zeros (point-estimate)
            
        Returns:
            [B, S, D] predicted clean sample x_0
        """
        shape = x.shape
        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)
        
        if xi is None:
            xi = torch.zeros_like(x)
        else:
            xi = xi.view(-1, *shape[-2:])

        x_emb = self.input_proj(x)
        t_emb = self.t_enc(t)
        i_emb = self.i_enc(i)
        z_emb = self.latent_proj(xi)  # latent branch

        h = self.proj(torch.cat([x_emb, t_emb, i_emb, z_emb], dim=-1))

        for att_layer, _ in zip(self.enc_att, self.i_proj):
            y, _ = att_layer(query=h, key=h, value=h)
            h = h + torch.relu(y)

        out = self.output_proj(h)
        return out.view(*shape)


# ============================================================================
# DDIM Sampling for Distributional Models
# ============================================================================

@torch.no_grad()
def sample_ddim_distributional(model, t_grid: torch.Tensor, alphas: torch.Tensor,
                               diffusion_steps: int, gp_sigma: float = 0.05,
                               n_samples: int = 1) -> torch.Tensor:
    """
    Sample from distributional diffusion model using DDIM-style deterministic update.
    
    The model predicts samples from P(x_0 | x_t, t) by using latent noise xi.
    Each diffusion step samples xi to get x_0 prediction(s).
    
    Args:
        model: Distributional diffusion model
        t_grid: Time points [B, S, 1]
        alphas: Cumulative alphas from diffusion schedule
        diffusion_steps: Number of diffusion steps
        gp_sigma: GP length scale for initial noise
        n_samples: Number of samples to draw from generator and average for x0 estimate.
                   If n_samples=1, uses single stochastic sample.
                   If n_samples>1, averages multiple samples for better estimate.
        
    Returns:
        Generated samples [B, S, D]
    """
    device = t_grid.device
    
    # Initialize from GP prior
    cov = get_gp_covariance(t_grid, gp_sigma=gp_sigma)
    L = torch.linalg.cholesky(cov)
    x = L @ torch.randn_like(t_grid)

    # Reverse diffusion
    for diff_step in reversed(range(0, diffusion_steps)):
        alpha_t = alphas[diff_step]
        alpha_prev = (
            alphas[diff_step - 1]
            if diff_step > 0
            else torch.tensor(1.0, dtype=alpha_t.dtype, device=device)
        )
        i = torch.tensor([diff_step], dtype=t_grid.dtype, device=device).expand_as(x[..., :1])

        if n_samples == 1:
            # Single stochastic sample
            xi = torch.randn_like(x)
            x0_tilde = model(x, t_grid, i, xi)
        else:
            # Multiple samples averaged for better estimate
            # Shape: x is [B, S, D]
            B, S, D = x.shape
            
            # Expand inputs for multiple samples
            x_expanded = x.unsqueeze(1).expand(B, n_samples, S, D).reshape(B * n_samples, S, D)
            t_expanded = t_grid.unsqueeze(1).expand(B, n_samples, S, 1).reshape(B * n_samples, S, 1)
            i_expanded = i.unsqueeze(1).expand(B, n_samples, S, 1).reshape(B * n_samples, S, 1)
            
            # Sample multiple latents
            xi_samples = torch.randn(B, n_samples, S, D, device=device, dtype=x.dtype)
            xi_expanded = xi_samples.reshape(B * n_samples, S, D)
            
            # Get multiple x0 predictions
            x0_predictions = model(x_expanded, t_expanded, i_expanded, xi_expanded)
            x0_predictions = x0_predictions.reshape(B, n_samples, S, D)
            
            # Average the predictions
            x0_tilde = x0_predictions.mean(dim=1)  # [B, S, D]

        # Convert to epsilon_tilde and do DDIM step
        eps_tilde = (x - alpha_t.sqrt() * x0_tilde) / (1 - alpha_t).sqrt()
        x = alpha_prev.sqrt() * x0_tilde + (1 - alpha_prev).sqrt() * eps_tilde

    return x


# ============================================================================
# Signature-specific utilities
# ============================================================================

def add_basepoint_zero(paths: torch.Tensor) -> torch.Tensor:
    """
    Add zero basepoint to beginning of paths for signature computation.
    
    Signatures require a reference basepoint. This prepends zeros at t=0.
    
    Args:
        paths: Tensor of shape [N, L, D] or [B, M, L, D]
        
    Returns:
        Paths with basepoint: [N, L+1, D] or [B, M, L+1, D]
    """
    if paths.dim() == 3:  # [N, L, D]
        N, L, D = paths.shape
        return torch.cat([
            torch.zeros(N, 1, D, dtype=paths.dtype, device=paths.device), 
            paths
        ], dim=1)
    elif paths.dim() == 4:  # [B, M, L, D]
        B, M, L, D = paths.shape
        return torch.cat([
            torch.zeros(B, M, 1, D, dtype=paths.dtype, device=paths.device), 
            paths
        ], dim=2)
    else:
        raise ValueError("Expected paths of shape [N, L, D] or [B, M, L, D].")


# ============================================================================
# Loss Functions
# ============================================================================

def loss_distributional_rbf_mmd(x: torch.Tensor, t: torch.Tensor, model,
                                diffusion_steps: int, alphas: torch.Tensor,
                                M: int = 4, sigma2: float = 0.1,
                                gp_sigma: float = 0.05) -> torch.Tensor:
    """
    Distributional loss using RBF-MMD (Maximum Mean Discrepancy).
    
    For each clean sample x_0, corrupts to x_t, then generates M predictions
    from the model. Computes MMD² between true x_0 and the predicted distribution.
    
    Args:
        x: Clean data [B, S, D]
        t: Time grid [B, S, 1]
        model: Distributional model
        diffusion_steps: Number of diffusion steps
        alphas: Cumulative alphas
        M: Number of samples per condition
        sigma2: RBF kernel bandwidth (per-element variance)
        gp_sigma: GP covariance parameter
        
    Returns:
        Scalar loss (mean MMD² over batch)
    """
    B, S, D = x.shape
    SD = S * D

    # Sample diffusion index per item and corrupt
    i_idx = torch.randint(0, diffusion_steps, size=(B,), dtype=torch.int64, device=x.device)
    i = i_idx.view(-1, 1, 1).expand(B, S, 1).to(dtype=x.dtype)
    x_t, _ = add_gp_noise(x, t, i, alphas, gp_sigma=gp_sigma)

    # Draw M latents and get M samples per condition
    xi = torch.randn(B, M, S, D, dtype=x.dtype, device=x.device)
    x_t_r = x_t.unsqueeze(1).expand(B, M, S, D).reshape(B * M, S, D)
    t_r = t.unsqueeze(1).expand(B, M, S, 1).reshape(B * M, S, 1)
    i_r = i.unsqueeze(1).expand(B, M, S, 1).reshape(B * M, S, 1)
    xi_r = xi.reshape(B * M, S, D)
    x0_samps = model(x_t_r, t_r, i_r, xi_r).reshape(B, M, S, D)  # [B, M, S, D]

    # Flatten for MMD computation
    X = x.reshape(B, SD)              # [B, SD] true samples
    Y = x0_samps.reshape(B, M, SD)    # [B, M, SD] model samples

    # Data-model term: mean_m k(x0, y_m)
    d2_xm = (Y - X.unsqueeze(1)).pow(2).mean(-1)                      # [B, M]
    k_xm = torch.exp(-d2_xm / (2.0 * sigma2)).mean(dim=1)             # [B]

    # Model-model term: mean_{m,m'} k(y_m, y_m')
    d2_mm = (Y.unsqueeze(2) - Y.unsqueeze(1)).pow(2).mean(-1)        # [B, M, M]
    k_mm = torch.exp(-d2_mm / (2.0 * sigma2)).mean(dim=(1, 2))       # [B]

    # Data-data term is 1.0 for RBF kernel
    mmd2_per_b = 1.0 + k_mm - 2.0 * k_xm                              # [B]
    return mmd2_per_b.mean()


def loss_expected_sigscore(x: torch.Tensor, t: torch.Tensor, model,
                          diffusion_steps: int, alphas: torch.Tensor, M: int = 16,
                          gp_sigma: float = 0.05, dyadic_order: int = 2, 
                          lam: float = 1.0, time_aug: bool = True,
                          lead_lag: bool = False, end_time: float = 1.0,
                          n_jobs: int = 16, max_batch: int = -1) -> torch.Tensor:
    """
    Distributional loss using signature scoring function.
    
    Requires pySigLib to be installed: pip install pysiglib
    
    Args:
        x: Clean data [B, S, D]
        t: Time grid [B, S, 1]
        model: Distributional model
        diffusion_steps: Number of diffusion steps
        alphas: Cumulative alphas
        M: Number of samples per condition
        gp_sigma: GP covariance parameter
        dyadic_order: Signature truncation order
        lam: Regularization parameter
        time_aug: Whether to use time augmentation
        lead_lag: Whether to use lead-lag transformation
        end_time: End time for signature computation
        n_jobs: Number of parallel jobs for signature computation
        max_batch: Maximum batch size for signature computation
        
    Returns:
        Scalar loss (mean sig_score over batch)
    """
    from pysiglib.torch_api import sig_score
    
    B, S, D = x.shape

    # Sample diffusion index per item and corrupt
    i_idx = torch.randint(0, diffusion_steps, size=(B,), dtype=torch.int64, device=x.device)
    i = i_idx.view(-1, 1, 1).expand(B, S, 1).to(dtype=x.dtype)
    x_t, _ = add_gp_noise(x, t, i, alphas, gp_sigma=gp_sigma)

    # Draw M latents and get M samples per condition
    xi = torch.randn(B, M, S, D, dtype=x.dtype, device=x.device)
    x_t_r = x_t.unsqueeze(1).expand(B, M, S, D).reshape(B * M, S, D)
    t_r = t.unsqueeze(1).expand(B, M, S, 1).reshape(B * M, S, 1)
    i_r = i.unsqueeze(1).expand(B, M, S, 1).reshape(B * M, S, 1)
    xi_r = xi.reshape(B * M, S, D)
    y_samps = model(x_t_r, t_r, i_r, xi_r).reshape(B, M, S, D)

    # Add basepoint for signature computation
    Y = add_basepoint_zero(y_samps)
    y = add_basepoint_zero(x)

    # Compute signature score for each batch item
    scores = []
    for b in range(B):
        # sample: [M, L, D], target: [L, D]
        score_b = sig_score(
            Y[b], y[b],
            dyadic_order=dyadic_order, lam=lam,
            time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
            n_jobs=n_jobs, max_batch=max_batch,
        )
        scores.append(score_b)

    return torch.stack(scores).mean()


def loss_sig_mmd(x: torch.Tensor, t: torch.Tensor, model,
                 diffusion_steps: int, alphas: torch.Tensor, M: int = 16,
                 gp_sigma: float = 0.05, dyadic_order: int = 2,
                 time_aug: bool = True, lead_lag: bool = False,
                 end_time: float = 1.0, n_jobs: int = 16,
                 max_batch: int = -1) -> torch.Tensor:
    """
    Distributional loss using Signature MMD from pySigLib.
    
    Computes MMD between the distribution of model samples and the target path
    under the signature kernel.
    
    Notes:
        - Uses basepoint augmentation for stability (consistent with sig_score usage).
        - Ensures both inputs to sig_mmd have the same batch size by repeating
          the single target path M times, to avoid potential NaNs when sizes differ
          as observed in practice.
    
    Args:
        x: Clean data [B, S, D]
        t: Time grid [B, S, 1]
        model: Distributional model
        diffusion_steps: Number of diffusion steps
        alphas: Cumulative alphas
        M: Number of samples per condition
        gp_sigma: GP covariance parameter
        dyadic_order: Signature truncation order (or tuple for per-input refinement)
        time_aug: Whether to use time augmentation
        lead_lag: Whether to use lead-lag transformation
        end_time: End time for signature computation
        n_jobs: Number of parallel jobs for signature computation
        max_batch: Maximum batch size for signature computation
        
    Returns:
        Scalar loss (mean Signature MMD over batch)
    """
    from pysiglib.torch_api import sig_mmd
    
    B, S, D = x.shape
    
    # Sample diffusion index per item and corrupt
    i_idx = torch.randint(0, diffusion_steps, size=(B,), dtype=torch.int64, device=x.device)
    i = i_idx.view(-1, 1, 1).expand(B, S, 1).to(dtype=x.dtype)
    x_t, _ = add_gp_noise(x, t, i, alphas, gp_sigma=gp_sigma)
    
    # Draw M latents and get M samples per condition
    xi = torch.randn(B, M, S, D, dtype=x.dtype, device=x.device)
    x_t_r = x_t.unsqueeze(1).expand(B, M, S, D).reshape(B * M, S, D)
    t_r = t.unsqueeze(1).expand(B, M, S, 1).reshape(B * M, S, 1)
    i_r = i.unsqueeze(1).expand(B, M, S, 1).reshape(B * M, S, 1)
    xi_r = xi.reshape(B * M, S, D)
    y_samps = model(x_t_r, t_r, i_r, xi_r).reshape(B, M, S, D)
    
    # Add basepoint for signature computation
    Y = add_basepoint_zero(y_samps)  # [B, M, L+1, D]
    y = add_basepoint_zero(x)        # [B, L+1, D]
    
    # Compute Signature MMD per batch item
    mmd_vals = []
    for b in range(B):
        # Ensure equal batch sizes for stability: repeat target M times
        target_b = y[b].unsqueeze(0).expand(M, -1, -1)
        sample_b = Y[b]  # [M, L+1, D]
        mmd_b = sig_mmd(
            sample_b,
            target_b,
            dyadic_order=dyadic_order,
            time_aug=time_aug,
            lead_lag=lead_lag,
            end_time=end_time,
            n_jobs=n_jobs,
            max_batch=max_batch,
        )
        mmd_vals.append(mmd_b)
    
    return torch.stack(mmd_vals).mean()

# ============================================================================
# Evaluation and Visualization
# ============================================================================

def compute_metrics(samples: torch.Tensor, ground_truth: torch.Tensor, 
                   t_samples: torch.Tensor = None, t_gt: torch.Tensor = None) -> dict:
    """
    Compute quantitative metrics comparing generated samples with ground truth.
    
    Args:
        samples: Generated samples [N, S, D]
        ground_truth: Ground truth data [M, S, D]
        t_samples: Time points for samples [N, S, 1] (optional)
        t_gt: Time points for ground truth [M, S, 1] (optional)
        
    Returns:
        Dictionary of metrics
    """
    from scipy.stats import wasserstein_distance
    import numpy as np
    
    # Convert to numpy
    generated_samples = samples.detach().cpu().numpy()
    ground_truth_data = ground_truth.detach().cpu().numpy()
    
    # Flatten all samples for distributional comparison
    generated_flat = generated_samples.reshape(-1)
    ground_truth_flat = ground_truth_data.reshape(-1)
    
    # Compute Wasserstein distance
    wasserstein_dist = wasserstein_distance(generated_flat, ground_truth_flat)
    
    # Compute statistical metrics
    gen_mean, gen_std = np.mean(generated_flat), np.std(generated_flat)
    gt_mean, gt_std = np.mean(ground_truth_flat), np.std(ground_truth_flat)
    
    # Mean Squared Error of statistics
    mse_mean = (gen_mean - gt_mean) ** 2
    mse_std = (gen_std - gt_std) ** 2
    mse_stats = mse_mean + mse_std
    
    # Trajectory-wise metrics
    gen_traj_means = np.mean(generated_samples, axis=(1, 2))
    gen_traj_stds = np.std(generated_samples, axis=(1, 2))
    gt_traj_means = np.mean(ground_truth_data, axis=(1, 2))
    gt_traj_stds = np.std(ground_truth_data, axis=(1, 2))
    
    # Wasserstein distance between trajectory statistics
    wd_means = wasserstein_distance(gen_traj_means, gt_traj_means)
    wd_stds = wasserstein_distance(gen_traj_stds, gt_traj_stds)
    
    metrics = {
        'wasserstein_distance': wasserstein_dist,
        'mse_stats': mse_stats,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'gen_mean': gen_mean,
        'gen_std': gen_std,
        'gt_mean': gt_mean,
        'gt_std': gt_std,
        'wd_traj_means': wd_means,
        'wd_traj_stds': wd_stds,
        'gen_traj_means': gen_traj_means,
        'gen_traj_stds': gen_traj_stds,
        'gt_traj_means': gt_traj_means,
        'gt_traj_stds': gt_traj_stds,
        'generated_flat': generated_flat,
        'ground_truth_flat': ground_truth_flat,
    }
    
    return metrics


def plot_comparison(samples: torch.Tensor, ground_truth: torch.Tensor,
                   t_samples: torch.Tensor, t_gt: torch.Tensor,
                   save_path: str = None, title_prefix: str = ""):
    """
    Create comprehensive visualization comparing generated samples with ground truth.
    
    Args:
        samples: Generated samples [N, S, D]
        ground_truth: Ground truth data [M, S, D]
        t_samples: Time points for samples [N, S, 1]
        t_gt: Time points for ground truth [M, S, 1]
        save_path: Path to save the figure (if None, just display)
        title_prefix: Prefix for plot titles
    """
    import matplotlib.pyplot as plt
    
    # Compute metrics
    metrics = compute_metrics(samples, ground_truth, t_samples, t_gt)
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Sample trajectories (first 10)
    ax = axes[0, 0]
    for i in range(min(10, len(samples))):
        ax.plot(
            t_samples[i].squeeze().detach().cpu().numpy(),
            samples[i].squeeze().detach().cpu().numpy(),
            color="C0",
            alpha=0.7,
        )
    ax.set_title(f"{title_prefix}Generated Samples (First 10)")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Ground truth trajectories (first 10)
    ax = axes[0, 1]
    for i in range(min(10, len(ground_truth))):
        ax.plot(
            t_gt[i].squeeze().detach().cpu().numpy(),
            ground_truth[i].squeeze().detach().cpu().numpy(),
            color="C1",
            alpha=0.7
        )
    ax.set_title(f"{title_prefix}Ground Truth (First 10)")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution comparison (histograms)
    ax = axes[0, 2]
    ax.hist(
        metrics['generated_flat'], bins=50, alpha=0.6, 
        label="Generated", color="C0", density=True
    )
    ax.hist(
        metrics['ground_truth_flat'], bins=50, alpha=0.6,
        label="Ground Truth", color="C1", density=True
    )
    ax.set_title(f"{title_prefix}Value Distribution Comparison")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory means comparison
    ax = axes[1, 0]
    ax.hist(
        metrics['gen_traj_means'], bins=30, alpha=0.6,
        label="Generated", color="C0", density=True
    )
    ax.hist(
        metrics['gt_traj_means'], bins=30, alpha=0.6,
        label="Ground Truth", color="C1", density=True
    )
    ax.set_title(f"{title_prefix}Trajectory Means Distribution")
    ax.set_xlabel("Trajectory Mean")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Trajectory stds comparison
    ax = axes[1, 1]
    ax.hist(
        metrics['gen_traj_stds'], bins=30, alpha=0.6,
        label="Generated", color="C0", density=True
    )
    ax.hist(
        metrics['gt_traj_stds'], bins=30, alpha=0.6,
        label="Ground Truth", color="C1", density=True
    )
    ax.set_title(f"{title_prefix}Trajectory Stds Distribution")
    ax.set_xlabel("Trajectory Std")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Side-by-side sample comparison
    ax = axes[1, 2]
    # Plot a few generated samples
    for i in range(min(5, len(samples))):
        ax.plot(
            t_samples[i].squeeze().detach().cpu().numpy(),
            samples[i].squeeze().detach().cpu().numpy(),
            color="C0",
            alpha=0.8,
            linewidth=1.5,
            label="Generated" if i == 0 else "",
        )
    # Plot a few ground truth samples
    for i in range(min(5, len(ground_truth))):
        ax.plot(
            t_gt[i].squeeze().detach().cpu().numpy(),
            ground_truth[i].squeeze().detach().cpu().numpy(),
            color="C1",
            alpha=0.8,
            linewidth=1.5,
            linestyle="--",
            label="Ground Truth" if i == 0 else "",
        )
    ax.set_title(f"{title_prefix}Direct Comparison (5 samples each)")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    return fig, metrics


def print_metrics(metrics: dict):
    """
    Print formatted metrics from compute_metrics().
    
    Args:
        metrics: Dictionary returned by compute_metrics()
    """
    print("\n📊 QUANTITATIVE ANALYSIS")
    print("=" * 50)
    print(f"Wasserstein Distance:     {metrics['wasserstein_distance']:.6f}")
    print(f"MSE (Statistics):         {metrics['mse_stats']:.6f}")
    print(f"  - MSE (Mean):           {metrics['mse_mean']:.6f}")
    print(f"  - MSE (Std):            {metrics['mse_std']:.6f}")
    print()
    print("Statistical Comparison:")
    print(f"Generated - Mean: {metrics['gen_mean']:.6f}, Std: {metrics['gen_std']:.6f}")
    print(f"Ground Truth - Mean: {metrics['gt_mean']:.6f}, Std: {metrics['gt_std']:.6f}")
    print()
    print("Trajectory Analysis:")
    print(f"Wasserstein Distance (Trajectory Means): {metrics['wd_traj_means']:.6f}")
    print(f"Wasserstein Distance (Trajectory Stds):  {metrics['wd_traj_stds']:.6f}")

