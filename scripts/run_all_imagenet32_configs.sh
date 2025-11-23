#!/usr/bin/env bash
set -euo pipefail

# Script to run all ImageNet32 experiments sequentially on one GPU
# Each experiment logs to a separate file and runs with nohup

GPU=${1:-0}  # GPU to use (default: 0)

echo "Starting ImageNet32 experiments sequentially (GPU=$GPU)..."
echo "Logs will be written to logs/ directory"
echo ""

# Get the project root directory (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs

# Function to run an experiment with nohup and wait for completion
run_experiment() {
    local config=$1
    local script=$2
    local logfile=$3
    local name=$4
    
    echo "[$name] Starting..."
    echo "  Config: $config"
    echo "  Log: $logfile"
    
    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src nohup python -m nmsd.training.$script "$config" &> "$logfile" &
    local pid=$!
    
    echo "  PID: $pid"
    echo "  Waiting for completion..."
    
    # Wait for this job to complete
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $exit_code"
    fi
    echo ""
}

# Track overall timing
start_time=$(date +%s)

# 1. Markov DDIM (Baseline)
echo "[1/8] Markov DDIM ImageNet32"
run_experiment "configs/markov_imagenet32.yaml" "train" "logs/train_markov_imagenet32.log" "Markov-ImageNet32"

# 2. Non-Markov with Transformer Encoder
echo "[2/8] Non-Markov Transformer ImageNet32"
run_experiment "configs/nonmarkov_imagenet32.yaml" "train_nonmarkov" "logs/train_nonmarkov_imagenet32.log" "Non-Markov(Transformer)-ImageNet32"

# 3. Non-Markov with Signature Encoder (Balanced)
echo "[3/8] Non-Markov Signature Balanced ImageNet32"
run_experiment "configs/nonmarkov_imagenet32_signature_balanced.yaml" "train_nonmarkov" "logs/train_nonmarkov_sig_balanced_imagenet32.log" "Non-Markov(Signature-Balanced)-ImageNet32"

# 4. Non-Markov with Signature Encoder (Original - High Memory)
echo "[4/8] Non-Markov Signature ImageNet32"
run_experiment "configs/nonmarkov_imagenet32_signature.yaml" "train_nonmarkov" "logs/train_nonmarkov_sig_imagenet32.log" "Non-Markov(Signature)-ImageNet32"

# 5. DART (Baseline - Context)
echo "[5/8] DART ImageNet32 (Standard)"
run_experiment "configs/dart_imagenet32.yaml" "train_dart" "logs/train_dart_imagenet32.log" "DART-ImageNet32"

# 6. DART with Signature Encoder
echo "[6/8] DART Signature ImageNet32"
run_experiment "configs/dart_imagenet32_signature.yaml" "train_dart" "logs/train_dart_sig_imagenet32.log" "DART(Signature)-ImageNet32"

# 7. DART with Signature-Transformer Hybrid Encoder
echo "[7/8] DART Signature-Trans ImageNet32"
run_experiment "configs/dart_imagenet32_signature_trans.yaml" "train_dart" "logs/train_dart_sig_trans_imagenet32.log" "DART(Signature-Trans)-ImageNet32"

# 8. DART with Transformer -> Signature Encoder
echo "[8/8] DART Trans-Signature ImageNet32"
run_experiment "configs/dart_imagenet32_trans_signature.yaml" "train_dart" "logs/train_dart_trans_sig_imagenet32.log" "DART(Trans-Signature)-ImageNet32"

# Overall summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))
hours=$((total_duration / 3600))
minutes=$(((total_duration % 3600) / 60))
seconds=$((total_duration % 60))

echo "========================================"
echo "All ImageNet32 experiments complete!"
echo "========================================"
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "GPU used: $GPU"
echo ""
echo "Results saved in:"
echo "  - experiments/markov_ddim_imagenet32/"
echo "  - experiments/nonmarkov_transformer_imagenet32/"
echo "  - experiments/nonmarkov_signature_balanced_imagenet32/"
echo "  - experiments/nonmarkov_signature_imagenet32/"
echo "  - experiments/dart_imagenet32/"
echo "  - experiments/dart_signature_imagenet32/"
echo "  - experiments/dart_signature_trans_imagenet32/"
echo "  - experiments/dart_trans_signature_imagenet32/"
echo ""
echo "Compare results:"
echo "  python scripts/compare_losses.py"
echo "  python scripts/compare_memory.py"

