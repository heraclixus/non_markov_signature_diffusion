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
    CKPT_NONMARKOV_SIGTRANS="experiments/nonmarkov_signature_trans/checkpoints/model_${STEP}.pt"
    CKPT_DART="experiments/dart/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIG="experiments/dart_signature/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIGTRANS="experiments/dart_signature_trans/checkpoints/model_${STEP}.pt"
    CKPT_DART_NO_CFG="experiments/dart_no_cfg/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIGLINEAR="experiments/nonmarkov_signature_linear/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIGLINEAR="experiments/dart_signature_linear/checkpoints/model_${STEP}.pt"
elif [ "$DATASET" == "cifar10" ]; then
    CKPT_MARKOV="experiments/markov_ddim_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_TRANS="experiments/nonmarkov_ddim_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIG="experiments/nonmarkov_signature_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIG_LOWMEM="experiments/nonmarkov_signature_cifar10_lowmem/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIG_BALANCED="experiments/nonmarkov_signature_cifar10_balanced/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIGTRANS="experiments/nonmarkov_signature_trans_cifar10_lowmem/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIGTRANS_FULL="experiments/nonmarkov_signature_trans_cifar10_full/checkpoints/model_${STEP}.pt"
    CKPT_DART="experiments/dart_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIG="experiments/dart_signature_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIGTRANS="experiments/dart_signature_trans_cifar10_lowmem/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIGTRANS_FULL="experiments/dart_signature_trans_cifar10_full/checkpoints/model_${STEP}.pt"
    CKPT_NONMARKOV_SIGLINEAR="experiments/nonmarkov_signature_linear_cifar10/checkpoints/model_${STEP}.pt"
    CKPT_DART_SIGLINEAR="experiments/dart_signature_linear_cifar10/checkpoints/model_${STEP}.pt"
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

    # Non-Markov SignatureTransformer
    if [ -f "$CKPT_NONMARKOV_SIGTRANS" ]; then
        echo "[5] Sampling from Non-Markov SignatureTransformer..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIGTRANS" "experiments/nonmarkov_signature_trans/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_sigtrans_${STEP}.log &
        wait $PID_NONMARKOV_SIG 2>/dev/null || true
        PID_NONMARKOV_SIGTRANS=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIGTRANS"
        PID_NONMARKOV_SIGTRANS=""
    fi

    # DART Transformer
    if [ -f "$CKPT_DART" ]; then
        echo "[6] Sampling from DART Transformer (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART" "experiments/dart/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_${STEP}.log &
        wait $PID_NONMARKOV_SIGTRANS 2>/dev/null || true
        PID_DART=$!
    else
        echo "  [SKIP] $CKPT_DART"
        PID_DART=""
    fi

    # DART Signature
    if [ -f "$CKPT_DART_SIG" ]; then
        echo "[7] Sampling from DART Signature (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIG" "experiments/dart_signature/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_sig_${STEP}.log &
        wait $PID_DART 2>/dev/null || true
        PID_DART_SIG=$!
    else
        echo "  [SKIP] $CKPT_DART_SIG"
        PID_DART_SIG=""
    fi

    # DART SignatureTransformer
    if [ -f "$CKPT_DART_SIGTRANS" ]; then
        echo "[8] Sampling from DART SignatureTransformer (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIGTRANS" "experiments/dart_signature_trans/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_sigtrans_${STEP}.log &
        wait $PID_DART_SIG 2>/dev/null || true
        PID_DART_SIGTRANS=$!
    else
        echo "  [SKIP] $CKPT_DART_SIGTRANS"
        PID_DART_SIGTRANS=""
    fi
    
    # DART No CFG
    if [ -f "$CKPT_DART_NO_CFG" ]; then
        echo "[9] Sampling from DART No CFG..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_NO_CFG" "experiments/dart_no_cfg/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 0.0 &> logs/sample_dart_no_cfg_${STEP}.log &
        wait $PID_DART_SIGTRANS 2>/dev/null || true
        PID_DART_NO_CFG=$!
    else
        echo "  [SKIP] $CKPT_DART_NO_CFG"
        PID_DART_NO_CFG=""
    fi
    
    # Non-Markov SignatureLinear
    if [ -f "$CKPT_NONMARKOV_SIGLINEAR" ]; then
        echo "[10] Sampling from Non-Markov SignatureLinear..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIGLINEAR" "experiments/nonmarkov_signature_linear/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_siglinear_${STEP}.log &
        wait $PID_DART_NO_CFG 2>/dev/null || true
        PID_NONMARKOV_SIGLINEAR=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIGLINEAR"
        PID_NONMARKOV_SIGLINEAR=""
    fi

    # DART SignatureLinear
    if [ -f "$CKPT_DART_SIGLINEAR" ]; then
        echo "[11] Sampling from DART SignatureLinear (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIGLINEAR" "experiments/dart_signature_linear/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_siglinear_${STEP}.log &
        wait $PID_NONMARKOV_SIGLINEAR 2>/dev/null || true
        PID_DART_SIGLINEAR=$!
    else
        echo "  [SKIP] $CKPT_DART_SIGLINEAR"
        PID_DART_SIGLINEAR=""
    fi
    
    wait $PID_DART_SIGLINEAR 2>/dev/null || true

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

    # Non-Markov SignatureTransformer (lowmem)
    if [ -f "$CKPT_NONMARKOV_SIGTRANS" ]; then
        echo "[6] Sampling from Non-Markov SignatureTransformer (lowmem)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIGTRANS" "experiments/nonmarkov_signature_trans_cifar10_lowmem/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_sigtrans_lowmem_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_SIG_BALANCED 2>/dev/null || true
        PID_NONMARKOV_SIGTRANS=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIGTRANS"
        PID_NONMARKOV_SIGTRANS=""
    fi

    # Non-Markov SignatureTransformer (full)
    if [ -f "$CKPT_NONMARKOV_SIGTRANS_FULL" ]; then
        echo "[7] Sampling from Non-Markov SignatureTransformer (full)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIGTRANS_FULL" "experiments/nonmarkov_signature_trans_cifar10_full/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 10 0.0 &> logs/sample_nonmarkov_sigtrans_full_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_SIGTRANS 2>/dev/null || true
        PID_NONMARKOV_SIGTRANS_FULL=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIGTRANS_FULL"
        PID_NONMARKOV_SIGTRANS_FULL=""
    fi

    # DART Transformer
    if [ -f "$CKPT_DART" ]; then
        echo "[8] Sampling from DART Transformer (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART" "experiments/dart_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_SIGTRANS_FULL 2>/dev/null || true
        PID_DART=$!
    else
        echo "  [SKIP] $CKPT_DART"
        PID_DART=""
    fi

    # DART Signature
    if [ -f "$CKPT_DART_SIG" ]; then
        echo "[9] Sampling from DART Signature (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIG" "experiments/dart_signature_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_sig_cifar10_${STEP}.log &
        wait $PID_DART 2>/dev/null || true
        PID_DART_SIG=$!
    else
        echo "  [SKIP] $CKPT_DART_SIG"
        PID_DART_SIG=""
    fi

    # DART SignatureTransformer (lowmem)
    if [ -f "$CKPT_DART_SIGTRANS" ]; then
        echo "[10] Sampling from DART SignatureTransformer (lowmem, CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIGTRANS" "experiments/dart_signature_trans_cifar10_lowmem/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_sigtrans_lowmem_cifar10_${STEP}.log &
        wait $PID_DART_SIG 2>/dev/null || true
        PID_DART_SIGTRANS=$!
    else
        echo "  [SKIP] $CKPT_DART_SIGTRANS"
        PID_DART_SIGTRANS=""
    fi

    # DART SignatureTransformer (full)
    if [ -f "$CKPT_DART_SIGTRANS_FULL" ]; then
        echo "[11] Sampling from DART SignatureTransformer (full, CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIGTRANS_FULL" "experiments/dart_signature_trans_cifar10_full/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 10 dart 1.0 1.5 &> logs/sample_dart_sigtrans_full_cifar10_${STEP}.log &
        wait $PID_DART_SIGTRANS 2>/dev/null || true
        PID_DART_SIGTRANS_FULL=$!
    else
        echo "  [SKIP] $CKPT_DART_SIGTRANS_FULL"
        PID_DART_SIGTRANS_FULL=""
    fi
    
    # Non-Markov SignatureLinear
    if [ -f "$CKPT_NONMARKOV_SIGLINEAR" ]; then
        echo "[12] Sampling from Non-Markov SignatureLinear..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_nonmarkov.sh \
            "$CKPT_NONMARKOV_SIGLINEAR" "experiments/nonmarkov_signature_linear_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_siglinear_cifar10_${STEP}.log &
        wait $PID_DART_SIGTRANS_FULL 2>/dev/null || true
        PID_NONMARKOV_SIGLINEAR=$!
    else
        echo "  [SKIP] $CKPT_NONMARKOV_SIGLINEAR"
        PID_NONMARKOV_SIGLINEAR=""
    fi

    # DART SignatureLinear
    if [ -f "$CKPT_DART_SIGLINEAR" ]; then
        echo "[13] Sampling from DART SignatureLinear (CFG=1.5)..."
        CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample_dart.sh \
            "$CKPT_DART_SIGLINEAR" "experiments/dart_signature_linear_cifar10/samples/final_${STEP}.png" \
            "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_siglinear_cifar10_${STEP}.log &
        wait $PID_NONMARKOV_SIGLINEAR 2>/dev/null || true
        PID_DART_SIGLINEAR=$!
    else
        echo "  [SKIP] $CKPT_DART_SIGLINEAR"
        PID_DART_SIGLINEAR=""
    fi
    
    wait $PID_DART_SIGLINEAR 2>/dev/null || true

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
    echo "  open experiments/dart_signature_linear_cifar10/samples/final_${STEP}.png"
fi

