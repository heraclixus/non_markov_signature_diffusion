#!/usr/bin/env bash
set -euo pipefail

# Script to sample from all trained models in parallel
# Usage: bash scripts/sample_all_models.sh <mnist|cifar10> [checkpoint_step] [num_samples] [num_ddim_steps]
# Example: bash scripts/sample_all_models.sh mnist 0022000 64 50
# Example: bash scripts/sample_all_models.sh cifar10 0155000 64 50

DATASET=${1:-mnist}
STEP=${2:-0014000}
NUM_SAMPLES=${3:-64}
NUM_DDIM_STEPS=${4:-50}

echo "=========================================="
echo "Sampling from all $DATASET models"
echo "=========================================="
echo "Checkpoint step: $STEP"
echo "Num samples: $NUM_SAMPLES"
echo "DDIM steps: $NUM_DDIM_STEPS"
echo ""

# Create logs directory
mkdir -p logs

# Define checkpoints based on dataset
if [ "$DATASET" == "mnist" ]; then
    CKPT_MARKOV="experiments/markov_ddim/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_EPS="experiments/nonmarkov_ddim/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_X0="experiments/nonmarkov_x0/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIG="experiments/nonmarkov_signature/checkpoints/model_${STEP}.pt"
    CKPT_DART="experiments/dart/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIG="experiments/dart_signature/checkpoints/model_${STEP}.pt"
    CKPT_DART_NO_CFG="experiments/dart_no_cfg/checkpoints/model_${STEP}.pt"
elif [ "$DATASET" == "cifar10" ]; then
    CKPT_MARKOV="experiments/markov_ddim_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_TRANS="experiments/nonmarkov_ddim_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIG="experiments/nonmarkov_signature_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIG_LOWMEM="experiments/nonmarkov_signature_cifar10_lowmem/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIG_BALANCED="experiments/nonmarkov_signature_cifar10_balanced/checkpoints/model_${STEP}.pt"
    CKPT_DART="experiments/dart_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIG="experiments/dart_signature_cifar10/checkpoints/model_${STEP}.pt"
else
    echo "Error: Unknown dataset '$DATASET'"
    echo "Usage: bash scripts/sample_all_models.sh <mnist|cifar10> [step] [num_samples] [ddim_steps]"
    exit 1
fi

if [ "$DATASET" == "mnist" ]; then
    # MNIST Models
    
    # Markov
    if [ -f "$CKPT_MARKOV" ]; then
        echo "[1] Sampling from Markov DDIM..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample.sh \
            "$CKPT_MARKOV" "experiments/markov_ddim/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 0.0 &> logs/sample_markov_${STEP}.log &
        PID_MARKOV=$!
    else
        echo "  [SKIP] $CKPT_MARKOV"
        PID_MARKOV=""
    fi

    # Non-Markov Transformer (epsilon)
    if [ -f "$CKPT_NONMARKOV_EPS" ]; then
        echo "[2] Sampling from Non-Markov Transformer..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_EPS" "experiments/nonmarkov_ddim/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_eps_${STEP}.log &
        wait $PID_MARKOV 2>/dev/null || true
        PID_NONMARKOV_EPS=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_EPS"
        PID_NONMARKOV_EPS=""
    fi

    # Non-Markov x0
    if [ -f "$CKPT_NONMARKOV_X0" ]; then
        echo "[3] Sampling from Non-Markov x0..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_X0" "experiments/nonmarkov_x0/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_x0_${STEP}.log &
        wait $PID_NONMARKOV_EPS 2>/dev/null || true
        PID_NONMARKOV_X0=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_X0"
        PID_NONMARKOV_X0=""
    fi

    # Non-Markov Signature
    if [ -f "$CKPT_NONMARKOV_SIG" ]; then
        echo "[4] Sampling from Non-Markov Signature..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIG" "experiments/nonmarkov_signature/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_sig_${STEP}.log &
        wait $PID_NONMARKOV_X0 2>/dev/null || true
        PID_NONMARKOV_SIG=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIG"
        PID_NONMARKOV_SIG=""
    fi

    # DART Transformer
    if [ -f "$CKPT_DART" ]; then
        echo "[5] Sampling from DART Transformer (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART" "experiments/dart/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_${STEP}.log &
        wait $PID_NONMARKOV_SIG 2>/dev/null || true
        PID_DART=$!
    else
        echo "  [SKIP] $CKPT_DART"
        PID_DART=""
    fi

    # DART Signature
    if [ -f "$CKPT_DART_SIG" ]; then
        echo "[6] Sampling from DART Signature (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIG" "experiments/dart_signature/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_sig_${STEP}.log &
        wait $PID_DART 2>/dev/null || true
        PID_DART_SIG=$!
    else
        echo "  [SKIP] $CKPT_DART_SIG"
        PID_DART_SIG=""
    fi
    
    # DART No CFG
    if [ -f "$CKPT_DART_NO_CFG" ]; then
        echo "[7] Sampling from DART No CFG..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_NO_CFG" "experiments/dart_no_cfg/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 0.0 &> logs/sample_dart_no_cfg_${STEP}.log &
        wait $PID_DART_SIG 2>/dev/null || true
        PID_DART_NO_CFG=$!
    else
        echo "  [SKIP] $CKPT_DART_NO_CFG"
        PID_DART_NO_CFG=""
    fi
    
    wait $PID_DART_NO_CFG 2>/dev/null || true

elif [ "$DATASET" == "cifar10" ]; then
    # CIFAR-10 Models
    
    # Markov
    if [ -f "$CKPT_MARKOV" ]; then
        echo "[1] Sampling from Markov DDIM..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample.sh \
            "$CKPT_MARKOV" "experiments/markov_ddim_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 0.0 &> logs/sample_markov_cifar10_${STEP}.log &
        PID_MARKOV=$!
    else
        echo "  [SKIP] $CKPT_MARKOV"
        PID_MARKOV=""
    fi

    # Non-Markov Transformer
    if [ -f "$CKPT_NONMARKOV_TRANS" ]; then
        echo "[2] Sampling from Non-Markov Transformer..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_TRANS" "experiments/nonmarkov_ddim_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_trans_cifar10_${STEP}.log &
        wait $PID_MARKOV 2>/dev/null || true
        PID_NONMARKOV_TRANS=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_TRANS"
        PID_NONMARKOV_TRANS=""
    fi

    # Non-Markov Signature (original)
    if [ -f "$CKPT_NONMARKOV_SIG" ]; then
        echo "[3] Sampling from Non-Markov Signature..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIG" "experiments/nonmarkov_signature_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_sig_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_TRANS 2>/dev/null || true
        PID_NONMARKOV_SIG=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIG"
        PID_NONMARKOV_SIG=""
    fi

    # Non-Markov Signature (lowmem)
    if [ -f "$CKPT_NONMARKOV_SIG_LOWMEM" ]; then
        echo "[4] Sampling from Non-Markov Signature (lowmem)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIG_LOWMEM" "experiments/nonmarkov_signature_cifar10_lowmem/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_sig_lowmem_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_SIG 2>/dev/null || true
        PID_NONMARKOV_SIG_LOWMEM=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIG_LOWMEM"
        PID_NONMARKOV_SIG_LOWMEM=""
    fi

    # Non-Markov Signature (balanced)
    if [ -f "$CKPT_NONMARKOV_SIG_BALANCED" ]; then
        echo "[5] Sampling from Non-Markov Signature (balanced)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIG_BALANCED" "experiments/nonmarkov_signature_cifar10_balanced/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 7 0.0 &> logs/sample_nonmarkov_sig_balanced_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_SIG_LOWMEM 2>/dev/null || true
        PID_NONMARKOV_SIG_BALANCED=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIG_BALANCED"
        PID_NONMARKOV_SIG_BALANCED=""
    fi

    # DART Transformer
    if [ -f "$CKPT_DART" ]; then
        echo "[6] Sampling from DART Transformer (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART" "experiments/dart_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_SIG_BALANCED 2>/dev/null || true
        PID_DART=$!
    else
        echo "  [SKIP] $CKPT_DART"
        PID_DART=""
    fi

    # DART Signature
    if [ -f "$CKPT_DART_SIG" ]; then
        echo "[7] Sampling from DART Signature (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIG" "experiments/dart_signature_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_sig_cifar10_${STEP}.log &
        wait $PID_DART 2>/dev/null || true
        PID_DART_SIG=$!
    else
        echo "  [SKIP] $CKPT_DART_SIG"
        PID_DART_SIG=""
    fi
    
    wait $PID_DART_SIG 2>/dev/null || true

fi

echo ""
echo "=========================================="
echo "All Sampling Complete!"
echo "=========================================="
echo ""
echo "Samples saved in experiments/*/samples/final_${STEP}.png"
echo "Logs saved in logs/sample_*_${STEP}.log"
echo ""
echo "View samples:"
if [ "$DATASET" == "mnist" ]; then
    echo "  open experiments/markov_ddim/samples/final_${STEP}.png"
    echo "  open experiments/nonmarkov_signature/samples/final_${STEP}.png"
    echo "  open experiments/dart/samples/final_${STEP}.png"
elif [ "$DATASET" == "cifar10" ]; then
    echo "  open experiments/markov_ddim_cifar10/samples/final_${STEP}.png"
    echo "  open experiments/nonmarkov_signature_cifar10_lowmem/samples/final_${STEP}.png"
    echo "  open experiments/nonmarkov_signature_cifar10_balanced/samples/final_${STEP}.png"
    echo "  open experiments/dart_cifar10/samples/final_${STEP}.png"
fi

