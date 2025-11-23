import torch
import time
from nmsd.encoders.signature import SignatureEncoder

def test_incremental_equivalence():
    print("=== Testing Numerical Equivalence ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 1, 8, 8
    
    # Setup encoder
    encoder = SignatureEncoder(
        image_channels=C, image_size=H, context_dim=16, 
        signature_degree=2, pooling="spatial_mean", time_augment=True
    ).to(device).eval()
    
    # Simulate reverse path T -> 0
    timesteps = torch.tensor([1000, 800, 600, 400, 200, 0], device=device)
    xs = [torch.randn(B, C, H, W, device=device) for _ in range(len(timesteps))]
    
    # Incremental
    running_sig = encoder.get_empty_signature(B, device)
    x_next, t_next = None, None
    
    for i, (t, x) in enumerate(zip(timesteps, xs)):
        t_batch = torch.full((B,), t, device=device)
        t_next_batch = t_next if t_next is not None else None
        
        # 1. Incremental Update
        context_inc, running_sig = encoder.forward_incremental(
            x, t_batch, x_next, t_next_batch, running_sig
        )
        
        # 2. Batch Reference (construct suffix x_{t:T})
        suffix_imgs = torch.stack(xs[0:i+1][::-1], dim=1) # [B, L, C, H, W]
        suffix_ts = timesteps[0:i+1].flip(0).unsqueeze(0).expand(B, -1)
        context_ref = encoder(suffix_imgs, suffix_ts)
        
        # Compare
        diff = (context_inc - context_ref).abs().max()
        print(f"Step {i} (t={t.item()}): Max Diff = {diff.item():.6e}")
        
        x_next, t_next = x, t_batch
    print("Equivalence test passed (check diffs above).")
    print()

def measure_runtime():
    print("=== Profiling Runtime ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Settings for profiling
    B, C, H, W = 16, 1, 28, 28
    num_steps = 100  # Number of steps in the sequence
    
    # Setup encoder
    encoder = SignatureEncoder(
        image_channels=C, image_size=H, context_dim=64, 
        signature_degree=3, pooling="spatial_mean", time_augment=True
    ).to(device).eval()
    
    # Pre-generate data
    timesteps = torch.linspace(1000, 0, num_steps, device=device).long()
    xs = [torch.randn(B, C, H, W, device=device) for _ in range(num_steps)]
    
    # Warmup
    print("Warming up...")
    dummy_x = xs[0]
    dummy_t = torch.full((B,), timesteps[0], device=device)
    for _ in range(10):
        _ = encoder(dummy_x.unsqueeze(1), dummy_t.unsqueeze(1))
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # --- Profile Incremental ---
    print(f"Profiling Incremental Approach ({num_steps} steps)...")
    start_time = time.perf_counter()
    
    running_sig = encoder.get_empty_signature(B, device)
    x_next, t_next = None, None
    
    for i in range(num_steps):
        t = timesteps[i]
        x = xs[i]
        t_batch = torch.full((B,), t, device=device)
        t_next_batch = t_next if t_next is not None else None
        
        context_inc, running_sig = encoder.forward_incremental(
            x, t_batch, x_next, t_next_batch, running_sig
        )
        x_next, t_next = x, t_batch
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inc_duration = time.perf_counter() - start_time
    print(f"Incremental Time: {inc_duration:.4f}s")
    
    # --- Profile Batch ---
    print(f"Profiling Batch Approach ({num_steps} steps)...")
    start_time = time.perf_counter()
    
    for i in range(num_steps):
        # In batch mode, at each step we process the full suffix so far
        # suffix length grows from 1 to num_steps
        suffix_imgs = torch.stack(xs[0:i+1][::-1], dim=1) # [B, L, C, H, W]
        suffix_ts = timesteps[0:i+1].flip(0).unsqueeze(0).expand(B, -1)
        
        context_ref = encoder(suffix_imgs, suffix_ts)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
    batch_duration = time.perf_counter() - start_time
    print(f"Batch Time:       {batch_duration:.4f}s")
    
    print("-" * 30)
    print(f"Speedup: {batch_duration / inc_duration:.2f}x")

if __name__ == "__main__":
    test_incremental_equivalence()
    measure_runtime()
