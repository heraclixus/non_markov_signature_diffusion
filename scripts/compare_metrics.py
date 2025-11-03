"""
Compare evaluation metrics across all models.

Usage:
    python scripts/compare_metrics.py --dataset mnist
    python scripts/compare_metrics.py --dataset cifar10 --output results.csv
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def parse_metrics_file(filepath: Path) -> Dict[str, float]:
    """Parse a metrics.txt file and extract values."""
    metrics = {}
    
    with open(filepath) as f:
        content = f.read()
    
    # Extract FID
    fid_match = re.search(r"FID:\s+([\d.]+|nan)", content)
    if fid_match:
        try:
            metrics["FID"] = float(fid_match.group(1))
        except ValueError:
            metrics["FID"] = float("nan")
    
    # Extract IS (mean and std)
    is_match = re.search(r"IS:\s+([\d.]+|nan)\s+±\s+([\d.]+|nan)", content)
    if is_match:
        try:
            metrics["IS_mean"] = float(is_match.group(1))
            metrics["IS_std"] = float(is_match.group(2))
        except ValueError:
            metrics["IS_mean"] = float("nan")
            metrics["IS_std"] = float("nan")
    
    return metrics


def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """Load all metrics files from a results directory."""
    data = []
    
    for metrics_file in sorted(results_dir.glob("*_metrics.txt")):
        model_name = metrics_file.stem.replace("_metrics", "")
        metrics = parse_metrics_file(metrics_file)
        
        row = {"Model": model_name}
        row.update(metrics)
        data.append(row)
    
    return pd.DataFrame(data)


def categorize_model(model_name: str) -> Dict[str, str]:
    """Categorize model by type and encoder."""
    categories = {
        "Type": "Unknown",
        "Encoder": "N/A",
        "Dataset": "Unknown"
    }
    
    # Determine type
    if "markov" in model_name and "nonmarkov" not in model_name:
        categories["Type"] = "Markov DDIM"
    elif "nonmarkov" in model_name and "dart" not in model_name:
        categories["Type"] = "Non-Markov"
    elif "dart" in model_name:
        categories["Type"] = "DART"
    
    # Determine encoder
    if "signature" in model_name:
        categories["Encoder"] = "Signature"
    elif "transformer" in model_name or categories["Type"] in ["Non-Markov", "DART"]:
        categories["Encoder"] = "Transformer"
    else:
        categories["Encoder"] = "N/A"
    
    # Determine dataset
    if "cifar10" in model_name:
        categories["Dataset"] = "CIFAR-10"
    elif "mnist" in model_name:
        categories["Dataset"] = "MNIST"
    
    return categories


def format_model_name(model_name: str) -> str:
    """Format model name for display."""
    # Replace underscores with spaces
    formatted = model_name.replace("_", " ")
    
    # Capitalize words
    formatted = formatted.title()
    
    # Fix known acronyms
    formatted = formatted.replace("Mnist", "MNIST")
    formatted = formatted.replace("Cifar10", "CIFAR-10")
    formatted = formatted.replace("Dart", "DART")
    formatted = formatted.replace("X0", "x0")
    
    return formatted


def create_comparison_table(df: pd.DataFrame, output_path: Path = None) -> pd.DataFrame:
    """Create a formatted comparison table."""
    # Add categories
    categories = df["Model"].apply(categorize_model).apply(pd.Series)
    df = pd.concat([df, categories], axis=1)
    
    # Add formatted model names
    df["Model_Display"] = df["Model"].apply(format_model_name)
    
    # Sort by Type, then Encoder, then FID
    df = df.sort_values(["Type", "Encoder", "FID"])
    
    # Reorder columns
    cols = ["Model_Display", "Type", "Encoder", "FID"]
    if "IS_mean" in df.columns:
        cols.extend(["IS_mean", "IS_std"])
    
    display_df = df[cols].copy()
    display_df.columns = ["Model", "Type", "Encoder", "FID ↓"] + (
        ["IS ↑", "IS Std"] if "IS_mean" in df.columns else []
    )
    
    # Save to CSV if requested
    if output_path:
        display_df.to_csv(output_path, index=False, float_format="%.4f")
        print(f"\nSaved table to {output_path}")
    
    return display_df


def plot_metrics(df: pd.DataFrame, dataset: str, output_dir: Path):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add categories for plotting
    categories = df["Model"].apply(categorize_model).apply(pd.Series)
    df = pd.concat([df, categories], axis=1)
    
    # Plot FID comparison
    plt.figure(figsize=(12, 6))
    
    # Sort by FID for better visualization
    plot_df = df.sort_values("FID")
    
    # Create bar plot with different colors for different types
    type_colors = {
        "Markov DDIM": "#3498db",
        "Non-Markov": "#2ecc71",
        "DART": "#e74c3c"
    }
    
    colors = [type_colors.get(t, "#95a5a6") for t in plot_df["Type"]]
    
    bars = plt.bar(range(len(plot_df)), plot_df["FID"], color=colors, alpha=0.7, edgecolor="black")
    
    # Add model names
    plt.xticks(range(len(plot_df)), plot_df["Model"].apply(format_model_name), 
               rotation=45, ha="right")
    
    plt.ylabel("FID Score (lower is better)", fontsize=12)
    plt.title(f"FID Comparison - {dataset.upper()}", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor="black", label=label, alpha=0.7)
                      for label, color in type_colors.items()]
    plt.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    fid_plot_path = output_dir / f"fid_comparison_{dataset}.png"
    plt.savefig(fid_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved FID plot to {fid_plot_path}")
    plt.close()
    
    # Plot IS if available (CIFAR-10)
    if "IS_mean" in df.columns and not df["IS_mean"].isna().all():
        plt.figure(figsize=(12, 6))
        
        plot_df_is = df[~df["IS_mean"].isna()].sort_values("IS_mean", ascending=False)
        
        colors_is = [type_colors.get(t, "#95a5a6") for t in plot_df_is["Type"]]
        
        bars = plt.bar(range(len(plot_df_is)), plot_df_is["IS_mean"], 
                      yerr=plot_df_is["IS_std"], color=colors_is, alpha=0.7, 
                      edgecolor="black", capsize=5)
        
        plt.xticks(range(len(plot_df_is)), plot_df_is["Model"].apply(format_model_name),
                  rotation=45, ha="right")
        
        plt.ylabel("Inception Score (higher is better)", fontsize=12)
        plt.title(f"Inception Score Comparison - {dataset.upper()}", fontsize=14, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)
        
        plt.legend(handles=legend_elements, loc="upper right")
        
        plt.tight_layout()
        is_plot_path = output_dir / f"is_comparison_{dataset}.png"
        plt.savefig(is_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved IS plot to {is_plot_path}")
        plt.close()


def print_summary_statistics(df: pd.DataFrame, dataset: str):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print(f"SUMMARY STATISTICS - {dataset.upper()}")
    print("=" * 70)
    
    # Add categories
    categories = df["Model"].apply(categorize_model).apply(pd.Series)
    df = pd.concat([df, categories], axis=1)
    
    # Best models
    print("\n🏆 BEST MODELS:")
    print("-" * 70)
    
    best_fid_idx = df["FID"].idxmin()
    best_model = df.loc[best_fid_idx]
    print(f"  Best FID: {format_model_name(best_model['Model'])}")
    print(f"    FID: {best_model['FID']:.4f}")
    
    if "IS_mean" in df.columns and not df["IS_mean"].isna().all():
        best_is_idx = df["IS_mean"].idxmax()
        best_is_model = df.loc[best_is_idx]
        print(f"\n  Best IS: {format_model_name(best_is_model['Model'])}")
        print(f"    IS: {best_is_model['IS_mean']:.4f} ± {best_is_model['IS_std']:.4f}")
    
    # Comparison by type
    print("\n\n📊 AVERAGE BY TYPE:")
    print("-" * 70)
    
    type_stats = df.groupby("Type")["FID"].agg(["mean", "std", "min", "max"])
    for type_name in type_stats.index:
        stats = type_stats.loc[type_name]
        print(f"  {type_name}:")
        print(f"    Mean FID: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Comparison by encoder (for non-Markov models)
    print("\n\n🔧 ENCODER COMPARISON (Non-Markov & DART):")
    print("-" * 70)
    
    non_markov_df = df[df["Type"].isin(["Non-Markov", "DART"])]
    if len(non_markov_df) > 0:
        encoder_stats = non_markov_df.groupby("Encoder")["FID"].agg(["mean", "std", "count"])
        for encoder_name in encoder_stats.index:
            stats = encoder_stats.loc[encoder_name]
            print(f"  {encoder_name}:")
            print(f"    Mean FID: {stats['mean']:.4f} ± {stats['std']:.4f} (n={int(stats['count'])})")
    
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare metrics across models")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["mnist", "cifar10"],
                       help="Dataset to analyze")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Results directory (default: evaluation_results_<dataset>)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV file path (default: <results-dir>/comparison.csv)")
    parser.add_argument("--plots-dir", type=str, default=None,
                       help="Directory for plots (default: <results-dir>/plots)")
    
    args = parser.parse_args()
    
    # Set default paths
    if args.results_dir is None:
        results_dir = Path(f"evaluation_results_{args.dataset}")
    else:
        results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print(f"Run evaluation first: bash scripts/evaluate_all_models.sh {args.dataset}")
        return
    
    if args.output is None:
        output_csv = results_dir / "comparison.csv"
    else:
        output_csv = Path(args.output)
    
    if args.plots_dir is None:
        plots_dir = results_dir / "plots"
    else:
        plots_dir = Path(args.plots_dir)
    
    print("=" * 70)
    print(f"METRICS COMPARISON - {args.dataset.upper()}")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print()
    
    # Load metrics
    print("Loading metrics...")
    df = load_all_metrics(results_dir)
    
    if len(df) == 0:
        print("No metrics files found!")
        return
    
    print(f"Found {len(df)} models\n")
    
    # Create comparison table
    print("\nCOMPARISON TABLE:")
    print("-" * 70)
    table = create_comparison_table(df, output_csv)
    print(table.to_string(index=False))
    
    # Print summary statistics
    print_summary_statistics(df, args.dataset)
    
    # Create plots
    print("Creating plots...")
    plot_metrics(df, args.dataset, plots_dir)
    
    print("\n✓ Analysis complete!")
    print(f"\nFiles saved:")
    print(f"  - Table: {output_csv}")
    print(f"  - Plots: {plots_dir}/")


if __name__ == "__main__":
    main()

