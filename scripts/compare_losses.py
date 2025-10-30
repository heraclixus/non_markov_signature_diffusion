#!/usr/bin/env python3
"""
Compare training losses across all model variants.
Usage: python scripts/compare_losses.py [--output comparison.png]
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compare training losses")
    parser.add_argument("--output", type=str, default="loss_comparison.png",
                       help="Output plot filename")
    parser.add_argument("--metric", type=str, default="auto",
                       choices=["auto", "loss", "loss_unweighted"],
                       help="Which metric to plot")
    args = parser.parse_args()

    variants = {
        'Markov (ε)': 'experiments/markov_ddim/logs/training_losses.csv',
        'Non-Markov (ε, Trans)': 'experiments/nonmarkov_ddim/logs/training_losses.csv',
        'Non-Markov (x₀, Trans)': 'experiments/nonmarkov_x0/logs/training_losses.csv',
        'DART (Trans, CFG)': 'experiments/dart/logs/training_losses.csv',
        'DART (Trans, no CFG)': 'experiments/dart_no_cfg/logs/training_losses.csv',
        'Non-Markov (ε, Sig)': 'experiments/nonmarkov_signature/logs/training_losses.csv',
        'DART (Sig, CFG)': 'experiments/dart_signature/logs/training_losses.csv',
    }

    plt.figure(figsize=(12, 8))
    found_any = False

    for name, path in variants.items():
        csv_path = Path(path)
        if not csv_path.exists():
            print(f"⚠ Skipping {name}: {path} not found")
            continue
        
        try:
            df = pd.read_csv(path)
            
            # Choose metric
            if args.metric == "auto":
                # Use unweighted loss if available for fair comparison
                loss_col = 'loss_unweighted' if 'loss_unweighted' in df.columns else 'loss'
            else:
                loss_col = args.metric
                if loss_col not in df.columns:
                    print(f"⚠ {name}: column '{loss_col}' not found, using 'loss'")
                    loss_col = 'loss'
            
            plt.plot(df['step'], df[loss_col], label=name, linewidth=2, alpha=0.8)
            found_any = True
            print(f"✓ Loaded {name}: {len(df)} steps, final loss = {df[loss_col].iloc[-1]:.4f}")
        
        except Exception as e:
            print(f"⚠ Error loading {name}: {e}")

    if not found_any:
        print("\n❌ No training data found. Train some models first!")
        return

    metric_name = "Loss (Unweighted)" if args.metric == "auto" else args.metric.replace('_', ' ').title()
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title('Training Loss Comparison Across Variants', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(args.output)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {output_path}")
    
    # Also save a zoomed-in version (last 50% of training)
    if found_any:
        plt.xlim(left=df['step'].max() * 0.5)
        output_zoom = output_path.stem + "_zoomed" + output_path.suffix
        plt.savefig(output_zoom, dpi=150, bbox_inches='tight')
        print(f"✓ Saved zoomed plot to: {output_zoom}")


if __name__ == "__main__":
    main()

