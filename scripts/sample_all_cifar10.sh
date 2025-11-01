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
    
    # Create samples directory if it doesn't exist
    mkdir -p "$experiment_dir/samples"
    
    # Sample based on experiment type
    if [[ "$config_file" == *"dart"* ]]; then
        # DART models use CFG
        local cfg_scale=$(grep "cfg_scale:" "$config_file" | awk '{print $2}')
        echo "  CFG Scale: $cfg_scale"
        # Note: Actual sampling command would go here when implemented
        echo -e "${GREEN}  Would sample with CFG scale $cfg_scale${NC}"
    else
        # Non-DART models
        echo -e "${GREEN}  Would sample without CFG${NC}"
    fi
    
    echo ""
}

# Sample from all CIFAR-10 models
echo -e "${BLUE}[1/5] Markov DDIM${NC}"
sample_model "experiments/markov_ddim_cifar10" "configs/markov_cifar10.yaml"

echo -e "${BLUE}[2/5] Non-Markov Transformer${NC}"
sample_model "experiments/nonmarkov_ddim_cifar10" "configs/nonmarkov_cifar10.yaml"

echo -e "${BLUE}[3/5] Non-Markov Signature${NC}"
sample_model "experiments/nonmarkov_signature_cifar10" "configs/nonmarkov_cifar10_signature.yaml"

echo -e "${BLUE}[4/5] DART Transformer${NC}"
sample_model "experiments/dart_cifar10" "configs/dart_cifar10.yaml"

echo -e "${BLUE}[5/5] DART Signature${NC}"
sample_model "experiments/dart_signature_cifar10" "configs/dart_cifar10_signature.yaml"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Sampling Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Generated samples saved in:${NC}"
echo "  - experiments/markov_ddim_cifar10/samples/"
echo "  - experiments/nonmarkov_ddim_cifar10/samples/"
echo "  - experiments/nonmarkov_signature_cifar10/samples/"
echo "  - experiments/dart_cifar10/samples/"
echo "  - experiments/dart_signature_cifar10/samples/"
echo ""
echo -e "${YELLOW}Note:${NC} This script currently shows which models would be sampled."
echo "Implement actual sampling logic based on your sampler interface."

