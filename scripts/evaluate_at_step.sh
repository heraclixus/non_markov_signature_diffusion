#!/usr/bin/env bash
set -euo pipefail

# Script to evaluate all models at a specific training step for fair comparison
# Usage: bash scripts/evaluate_at_step.sh <dataset> <target_step> [gpu]

DATASET=$1
TARGET_STEP=${2:-155000}
GPU=${3:-0}

echo "========================================"
echo "Evaluating All ${DATASET} Models at ${TARGET_STEP} Steps"
echo "========================================"
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create results directory
RESULTS_DIR="evaluation_results_${DATASET}_${TARGET_STEP}steps"
mkdir -p "$RESULTS_DIR"

# Function to evaluate a model at specific step
evaluate_at_step() {
    local config=$1
    local experiment_dir=$2
    local model_name=$3
    
    echo "----------------------------------------"
    echo "Evaluating: $model_name"
    echo "----------------------------------------"
    
    # Find checkpoint closest to target step
    local checkpoint_dir="$experiment_dir/checkpoints"
    
    if [ ! -d "$checkpoint_dir" ]; then
        echo "⚠ No checkpoints found for $model_name"
        echo ""
        return 1
    fi
    
    # Find closest checkpoint to target step
    local closest_checkpoint=""
    local min_diff=999999999
    
    for ckpt in "$checkpoint_dir"/model_*.pt; do
        if [ -f "$ckpt" ]; then
            # Extract step number from filename
            local step=$(basename "$ckpt" .pt | sed 's/model_0*//')
            local diff=$(( $TARGET_STEP - $step ))
            # Get absolute difference
            [ $diff -lt 0 ] && diff=$(( -$diff ))
            
            if [ $diff -lt $min_diff ]; then
                min_diff=$diff
                closest_checkpoint="$ckpt"
            fi
        fi
    done
    
    if [ -z "$closest_checkpoint" ]; then
        echo "⚠ No checkpoint files found for $model_name"
        echo ""
        return 1
    fi
    
    local actual_step=$(basename "$closest_checkpoint" .pt | sed 's/model_0*//')
    
    echo "Config: $config"
    echo "Target step: $TARGET_STEP"
    echo "Actual checkpoint: $(basename "$closest_checkpoint") (step $actual_step)"
    echo "Difference: $min_diff steps"
    
    # Only evaluate if difference is reasonable (<20K steps)
    if [ $min_diff -gt 20000 ]; then
        echo "⚠ Checkpoint too far from target ($min_diff steps), skipping"
        echo ""
        return 1
    fi
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python -m nmsd.evaluation.evaluate \
        --config "$config" \
        --checkpoint "$closest_checkpoint" \
        --num-samples 10000 \
        --num-steps 50 \
        --output-dir "$experiment_dir/evaluation_${actual_step}"
    
    # Copy metrics to results directory
    if [ -f "$experiment_dir/evaluation_${actual_step}/metrics.txt" ]; then
        cp "$experiment_dir/evaluation_${actual_step}/metrics.txt" "$RESULTS_DIR/${model_name}_metrics.txt"
        echo "✓ Saved metrics to $RESULTS_DIR/${model_name}_metrics.txt"
    fi
    
    echo ""
}

# Evaluate models based on dataset
if [ "$DATASET" == "mnist" ]; then
    echo "⚠ For MNIST, use 14K steps (all models stop at 14K)"
    TARGET_STEP=14000
    RESULTS_DIR="evaluation_results_${DATASET}_${TARGET_STEP}steps"
    mkdir -p "$RESULTS_DIR"
    
    evaluate_at_step "configs/markov_mnist.yaml" "experiments/markov_ddim" "markov_mnist"
    evaluate_at_step "configs/nonmarkov_mnist.yaml" "experiments/nonmarkov_ddim" "nonmarkov_transformer_mnist"
    evaluate_at_step "configs/nonmarkov_mnist_x0.yaml" "experiments/nonmarkov_x0" "nonmarkov_x0_mnist"
    evaluate_at_step "configs/nonmarkov_mnist_signature.yaml" "experiments/nonmarkov_signature" "nonmarkov_signature_mnist"
    evaluate_at_step "configs/dart_mnist.yaml" "experiments/dart" "dart_transformer_mnist"
    evaluate_at_step "configs/dart_mnist_signature.yaml" "experiments/dart_signature" "dart_signature_mnist"
    
elif [ "$DATASET" == "cifar10" ]; then
    # CIFAR-10 Models at 155K steps
    evaluate_at_step "configs/markov_cifar10.yaml" "experiments/markov_ddim_cifar10" "markov_cifar10"
    evaluate_at_step "configs/nonmarkov_cifar10.yaml" "experiments/nonmarkov_ddim_cifar10" "nonmarkov_transformer_cifar10"
    evaluate_at_step "configs/nonmarkov_cifar10_signature.yaml" "experiments/nonmarkov_signature_cifar10" "nonmarkov_signature_cifar10"
    evaluate_at_step "configs/nonmarkov_cifar10_signature_lowmem.yaml" "experiments/nonmarkov_signature_cifar10_lowmem" "nonmarkov_signature_cifar10_lowmem"
    evaluate_at_step "configs/nonmarkov_cifar10_signature_balanced.yaml" "experiments/nonmarkov_signature_cifar10_balanced" "nonmarkov_signature_cifar10_balanced"
    evaluate_at_step "configs/dart_cifar10.yaml" "experiments/dart_cifar10" "dart_transformer_cifar10"
    evaluate_at_step "configs/dart_cifar10_signature.yaml" "experiments/dart_signature_cifar10" "dart_signature_cifar10"
    
else
    echo "Error: Unknown dataset '$DATASET'"
    echo "Usage: bash scripts/evaluate_at_step.sh <mnist|cifar10> [target_step] [gpu]"
    exit 1
fi

echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Compare results:"
echo "  python scripts/compare_metrics.py --results-dir $RESULTS_DIR --dataset $DATASET"

