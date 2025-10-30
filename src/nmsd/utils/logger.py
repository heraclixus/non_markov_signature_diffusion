from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List
import torch
import psutil

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class MemoryProfiler:
    """
    Track GPU and CPU memory usage during training.
    """
    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.process = psutil.Process()
        self.peak_gpu_mb = 0
        self.peak_cpu_mb = 0
        self.current_gpu_mb = 0
        self.current_cpu_mb = 0
    
    def update(self):
        """Update current and peak memory usage."""
        # CPU memory (RSS - Resident Set Size)
        self.current_cpu_mb = self.process.memory_info().rss / 1024 / 1024
        self.peak_cpu_mb = max(self.peak_cpu_mb, self.current_cpu_mb)
        
        # GPU memory
        if self.has_cuda:
            self.current_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            self.peak_gpu_mb = max(self.peak_gpu_mb, self.current_gpu_mb)
    
    def reset_gpu(self):
        """Reset GPU memory stats."""
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()
    
    def get_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        self.update()
        return {
            "cpu_mb": self.current_cpu_mb,
            "gpu_mb": self.current_gpu_mb,
            "peak_cpu_mb": self.peak_cpu_mb,
            "peak_gpu_mb": self.peak_gpu_mb,
        }
    
    def log_summary(self, log_path: Path):
        """Write memory summary to file."""
        with open(log_path, 'w') as f:
            f.write("Memory Usage Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Peak GPU Memory: {self.peak_gpu_mb:.2f} MB\n")
            f.write(f"Peak CPU Memory: {self.peak_cpu_mb:.2f} MB\n")
            f.write(f"Current GPU Memory: {self.current_gpu_mb:.2f} MB\n")
            f.write(f"Current CPU Memory: {self.current_cpu_mb:.2f} MB\n")
            if self.has_cuda:
                f.write(f"\nGPU Device: {torch.cuda.get_device_name(0)}\n")
                f.write(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.2f} MB\n")


class LossLogger:
    """
    Simple CSV logger for training losses with plotting capabilities.
    """
    def __init__(self, log_dir: Path, name: str = "losses"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / f"{name}.csv"
        self.plot_path = self.log_dir / f"{name}.png"
        
        self.fieldnames = ["step", "loss"]
        self.extra_fields = set()
        
        # Initialize CSV
        self.file_handle = open(self.csv_path, 'w', newline='')
        self.writer = None
        self.rows = []
    
    def log(self, step: int, loss: float, **kwargs):
        """
        Log a training step.
        
        Args:
            step: global training step
            loss: main loss value
            **kwargs: additional metrics (e.g., loss_unweighted, avg_weight)
        """
        row = {"step": step, "loss": loss}
        
        # Add extra fields
        for k, v in kwargs.items():
            if k not in self.extra_fields:
                self.extra_fields.add(k)
                # Update fieldnames
                if k not in self.fieldnames:
                    self.fieldnames.append(k)
            row[k] = v
        
        # Initialize writer if first time
        if self.writer is None:
            self.writer = csv.DictWriter(self.file_handle, fieldnames=self.fieldnames)
            self.writer.writeheader()
        
        # Write row
        self.writer.writerow(row)
        self.file_handle.flush()
        
        # Keep in memory for plotting
        self.rows.append(row)
    
    def plot(self, title: str = "Training Loss"):
        """
        Generate a plot of training losses.
        """
        if not self.rows:
            return
        
        steps = [r["step"] for r in self.rows]
        losses = [r["loss"] for r in self.rows]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, linewidth=2, label="Loss")
        
        # Plot additional metrics if available
        for field in self.extra_fields:
            if field in self.rows[0]:
                values = [r.get(field, None) for r in self.rows]
                # Filter out None values
                valid_steps = [s for s, v in zip(steps, values) if v is not None]
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    plt.plot(valid_steps, valid_values, linewidth=1.5, 
                            alpha=0.7, label=field, linestyle='--')
        
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150)
        plt.close()
    
    def close(self):
        """Close the CSV file."""
        if self.file_handle:
            self.file_handle.close()
    
    def __del__(self):
        self.close()

