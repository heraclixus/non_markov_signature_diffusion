#!/usr/bin/env bash
set -euo pipefail

# Train all models sequentially on one GPU
# Order: MNIST first, then CIFAR-10; Markov, Non-Markov, DART; Transformer then Signature

GPU=${1:-0}

echo "=========================================="
echo "Training All Models Sequentially"
echo "=========================================="
echo "GPU: $GPU"
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs

# Track total time
start_time=$(date +%s)

# Function to train a model
train_model() {
    local config=$1
    local script=$2
    local name=$3
    
    echo "=========================================="
    echo "[$name]"
    echo "=========================================="
    echo "Config: $config"
    echo "Started: $(date)"
    
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src nohup python -m nmsd.training.$script "$config" &> "logs/train_${name}.log" &
    local pid=$!
    
    echo "PID: $pid"
    echo "Log: logs/train_${name}.log"
    
    wait $pid
    
    echo "Completed: $(date)"
    echo ""
}

echo "=========================================="
echo "MNIST MODELS (~22K steps each)"
echo "=========================================="
echo ""

# MNIST - Markov
train_model "configs/markov_mnist.yaml" "train" "mnist_markov"

# MNIST - Non-Markov Transformer
train_model "configs/nonmarkov_mnist.yaml" "train_nonmarkov" "mnist_nonmarkov_transformer"

# MNIST - Non-Markov x0
train_model "configs/nonmarkov_mnist_x0.yaml" "train_nonmarkov" "mnist_nonmarkov_x0"

# MNIST - Non-Markov Signature
train_model "configs/nonmarkov_mnist_signature.yaml" "train_nonmarkov" "mnist_nonmarkov_signature"

# MNIST - DART Transformer
train_model "configs/dart_mnist.yaml" "train_dart" "mnist_dart_transformer"

# MNIST - DART Signature
train_model "configs/dart_mnist_signature.yaml" "train_dart" "mnist_dart_signature"

# MNIST - DART SignatureLinear
train_model "configs/dart_mnist_signature_linear.yaml" "train_dart" "mnist_dart_signature_linear"

echo "=========================================="
echo "CIFAR-10 MODELS (~155K steps each)"
echo "=========================================="
echo ""

# CIFAR-10 - Markov
train_model "configs/markov_cifar10.yaml" "train" "cifar10_markov"

# CIFAR-10 - Non-Markov Transformer
train_model "configs/nonmarkov_cifar10.yaml" "train_nonmarkov" "cifar10_nonmarkov_transformer"

# CIFAR-10 - Non-Markov Signature (Balanced)
train_model "configs/nonmarkov_cifar10_signature_balanced.yaml" "train_nonmarkov" "cifar10_nonmarkov_signature_balanced"

# CIFAR-10 - DART Transformer
train_model "configs/dart_cifar10.yaml" "train_dart" "cifar10_dart_transformer"

# CIFAR-10 - DART Signature
train_model "configs/dart_cifar10_signature.yaml" "train_dart" "cifar10_dart_signature"

# CIFAR-10 - DART SignatureLinear
train_model "configs/dart_cifar10_signature_linear.yaml" "train_dart" "cifar10_dart_signature_linear"

# Summary
end_time=$(date +%s)
total_duration=$((end_time - start_time))
hours=$((total_duration / 3600))
minutes=$(((total_duration % 3600) / 60))

echo "=========================================="
echo "ALL TRAINING COMPLETE!"
echo "=========================================="
echo "Total time: ${hours}h ${minutes}m"
echo "GPU used: $GPU"
echo ""
echo "Logs saved in: logs/"
echo ""
echo "Next steps:"
echo "  1. Evaluate MNIST:    bash scripts/evaluate_at_step.sh mnist 22000 $GPU"
echo "  2. Evaluate CIFAR-10: bash scripts/evaluate_at_step.sh cifar10 155000 $GPU"
echo "  3. Compare results:   python scripts/compare_metrics.py --dataset mnist"
echo "                        python scripts/compare_metrics.py --dataset cifar10"

