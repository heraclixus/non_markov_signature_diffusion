#!/bin/bash
# Sample from all trained CIFAR-10 models
# This generates images from the latest checkpoint of each CIFAR-10 experiment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Sampling from CIFAR-10 Models${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Function to sample from a trained model
sample_model() {
    local experiment_dir=$1
    local config_file=$2
    local is_dart=$3 # "true" if dart model
    local experiment_name=$(basename "$experiment_dir")
    
    # Find the latest checkpoint
    local checkpoint_dir="$experiment_dir/checkpoints"
    
    if [ ! -d "$checkpoint_dir" ]; then
        echo -e "${YELLOW}⚠ No checkpoint directory found for $experiment_name${NC}"
        return 1
    fi
    
    local latest_checkpoint=$(ls -t "$checkpoint_dir"/model_*.pt 2>/dev/null | head -1)
    
    if [ -z "$latest_checkpoint" ]; then
        echo -e "${YELLOW}⚠ No checkpoints found for $experiment_name${NC}"
        return 1
    fi
    
    local step=$(basename "$latest_checkpoint" .pt | sed 's/model_//')
    
    echo -e "${GREEN}Sampling: $experiment_name${NC}"
    echo "  Checkpoint: $(basename "$latest_checkpoint")"
    echo "  Step: $step"
    
    # Determine which script to run
    local script_cmd=""
    
    if [[ "$experiment_name" == "markov"* ]]; then
        # Markov models
        script_cmd="python -m nmsd.training.sample $latest_checkpoint --num 16 --steps 50 --out $experiment_dir/samples/latest_sample.png"
    elif [[ "$is_dart" == "true" ]]; then
        # DART models
        # Use DDIM sampler for consistent comparison, or DART sampler if preferred
        # Let's use DDIM sampler for fair comparison across model types
        script_cmd="python -m nmsd.training.sample_dart $latest_checkpoint --num 16 --steps 50 --sampler ddim --out $experiment_dir/samples/latest_sample.png"
    else
        # Non-Markov models (epsilon prediction)
        script_cmd="python -m nmsd.training.sample_nonmarkov $latest_checkpoint --num 16 --steps 50 --out $experiment_dir/samples/latest_sample.png"
    fi
    
    echo "  Running: $script_cmd"
    # Execute the command with PYTHONPATH set
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" $script_cmd
    
    echo ""
}

# Sample from all CIFAR-10 models
echo -e "${BLUE}[1/5] Markov DDIM${NC}"
sample_model "experiments/markov_cifar10" "configs/markov_cifar10.yaml" "false"

echo -e "${BLUE}[2/5] Non-Markov Transformer${NC}"
sample_model "experiments/nonmarkov_cifar10" "configs/nonmarkov_cifar10.yaml" "false"

echo -e "${BLUE}[3/5] Non-Markov Signature${NC}"
sample_model "experiments/nonmarkov_cifar10_signature" "configs/nonmarkov_cifar10_signature.yaml" "false"

echo -e "${BLUE}[4/5] DART Transformer${NC}"
sample_model "experiments/dart_cifar10" "configs/dart_cifar10.yaml" "true"

echo -e "${BLUE}[5/5] DART Signature${NC}"
sample_model "experiments/dart_cifar10_signature" "configs/dart_cifar10_signature.yaml" "true"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Sampling Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
