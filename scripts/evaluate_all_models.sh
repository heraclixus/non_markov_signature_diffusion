#!/usr/bin/env bash
set -euo pipefail

# Script to evaluate all trained models at target checkpoint for fair comparison
# Generates samples and computes FID/IS for each checkpoint

DATASET=${1:-mnist}  # 'mnist' or 'cifar10'
NUM_SAMPLES=1000  # Number of samples per model
NUM_STEPS=${3:-50}  # Sampling steps
GPU=${2:-0}  # GPU to use

# Set target checkpoint based on dataset for fair comparison
if [ "$DATASET" == "mnist" ]; then
    TARGET_STEP=22000  # MNIST target: 22K steps
elif [ "$DATASET" == "cifar10" ]; then
    TARGET_STEP=40000  # CIFAR-10 target: 40K steps (updated from 155K for faster training)
else
    TARGET_STEP=0
fi

echo "========================================"
echo "Evaluating All ${DATASET} Models"
echo "========================================"
echo "Target checkpoint: ${TARGET_STEP} steps"
echo "Num samples: $NUM_SAMPLES"
echo "Sampling steps: $NUM_STEPS"
echo "GPU: $GPU"
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Create results directory
RESULTS_DIR="evaluation_results_${DATASET}"
mkdir -p "$RESULTS_DIR"

# Function to evaluate a model at target checkpoint
evaluate_model() {
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
    
    # Find closest checkpoint to target
    local closest_checkpoint=""
    local min_diff=999999999
    
    for ckpt in "$checkpoint_dir"/model_*.pt; do
        if [ -f "$ckpt" ]; then
            # Extract step number from filename
            local step=$(basename "$ckpt" .pt | sed 's/model_0*//')
            local diff=$(( TARGET_STEP - step ))
            # Get absolute difference
            [ $diff -lt 0 ] && diff=$(( -diff ))
            
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
    
    # Skip if checkpoint is too far from target
    if [ $min_diff -gt 15000 ]; then
        echo "⚠ Skipping: Checkpoint is $min_diff steps away from target (>15K)"
        echo ""
        return 1
    fi
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python -m nmsd.evaluation.evaluate \
        --config "$config" \
        --checkpoint "$closest_checkpoint" \
        --num-samples $NUM_SAMPLES \
        --num-steps $NUM_STEPS \
        --output-dir "$experiment_dir/evaluation"
    
    # Copy metrics to results directory
    if [ -f "$experiment_dir/evaluation/metrics.txt" ]; then
        cp "$experiment_dir/evaluation/metrics.txt" "$RESULTS_DIR/${model_name}_metrics.txt"
        echo "✓ Saved metrics to $RESULTS_DIR/${model_name}_metrics.txt"
    fi
    
    echo ""
}

# Evaluate models based on dataset
if [ "$DATASET" == "mnist" ]; then
    # MNIST Models
    evaluate_model "configs/markov_mnist.yaml" "experiments/markov_ddim" "markov_mnist"
    evaluate_model "configs/nonmarkov_mnist.yaml" "experiments/nonmarkov_ddim" "nonmarkov_transformer_mnist"
    evaluate_model "configs/nonmarkov_mnist_x0.yaml" "experiments/nonmarkov_x0" "nonmarkov_x0_mnist"
    evaluate_model "configs/nonmarkov_mnist_signature.yaml" "experiments/nonmarkov_signature" "nonmarkov_signature_mnist"
    evaluate_model "configs/nonmarkov_mnist_signature_trans.yaml" "experiments/nonmarkov_signature_trans" "nonmarkov_signature_trans_mnist"
    evaluate_model "configs/dart_mnist.yaml" "experiments/dart" "dart_transformer_mnist"
    evaluate_model "configs/dart_mnist_signature.yaml" "experiments/dart_signature" "dart_signature_mnist"
    evaluate_model "configs/dart_mnist_signature_trans.yaml" "experiments/dart_signature_trans" "dart_signature_trans_mnist"
    
elif [ "$DATASET" == "cifar10" ]; then
    # CIFAR-10 Models
    evaluate_model "configs/markov_cifar10.yaml" "experiments/markov_ddim_cifar10" "markov_cifar10"
    evaluate_model "configs/nonmarkov_cifar10.yaml" "experiments/nonmarkov_ddim_cifar10" "nonmarkov_transformer_cifar10"
    evaluate_model "configs/nonmarkov_cifar10_signature_lowmem.yaml" "experiments/nonmarkov_signature_cifar10_lowmem" "nonmarkov_signature_cifar10_lowmem"
    evaluate_model "configs/nonmarkov_cifar10_signature_balanced.yaml" "experiments/nonmarkov_signature_cifar10_balanced" "nonmarkov_signature_cifar10_balanced"
    evaluate_model "configs/nonmarkov_cifar10_signature_trans.yaml" "experiments/nonmarkov_signature_trans_cifar10_lowmem" "nonmarkov_signature_trans_cifar10_lowmem"
    evaluate_model "configs/nonmarkov_cifar10_signature_trans_full.yaml" "experiments/nonmarkov_signature_trans_cifar10_full" "nonmarkov_signature_trans_cifar10_full"
    evaluate_model "configs/dart_cifar10.yaml" "experiments/dart_cifar10" "dart_transformer_cifar10"
    evaluate_model "configs/dart_cifar10_signature.yaml" "experiments/dart_signature_cifar10" "dart_signature_cifar10"
    evaluate_model "configs/dart_cifar10_signature_trans.yaml" "experiments/dart_signature_trans_cifar10_lowmem" "dart_signature_trans_cifar10_lowmem"
    evaluate_model "configs/dart_cifar10_signature_trans_full.yaml" "experiments/dart_signature_trans_cifar10_full" "dart_signature_trans_cifar10_full"
    
else
    echo "Error: Unknown dataset '$DATASET'"
    echo "Usage: bash scripts/evaluate_all_models.sh <mnist|cifar10> [num_samples] [num_steps] [gpu]"
    exit 1
fi

echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "Next step: Compare results"
echo "  python scripts/compare_metrics.py --dataset $DATASET"

