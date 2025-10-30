#!/usr/bin/env bash
set -euo pipefail

# Script to sample from all trained models in parallel
# Usage: bash scripts/sample_all_models.sh [checkpoint_step]
# Example: bash scripts/sample_all_models.sh 0004000

STEP=${1:-0004000}
NUM_SAMPLES=${2:-64}
NUM_DDIM_STEPS=${3:-50}

echo "Sampling from all models at step $STEP with $NUM_SAMPLES samples..."
echo ""

# Create logs directory
mkdir -p logs

# Check which checkpoints exist
CKPT_MARKOV="experiments/markov_ddim/checkpoints/model_${STEP}.pt"
CKPT_NONMARKOV_EPS="experiments/nonmarkov_ddim/checkpoints/model_${STEP}.pt"
CKPT_NONMARKOV_X0="experiments/nonmarkov_x0/checkpoints/model_${STEP}.pt"
CKPT_DART="experiments/dart/checkpoints/model_${STEP}.pt"
CKPT_NONMARKOV_SIG="experiments/nonmarkov_signature/checkpoints/model_${STEP}.pt"
CKPT_DART_SIG="experiments/dart_signature/checkpoints/model_${STEP}.pt"
CKPT_DART_NO_CFG="experiments/dart_no_cfg/checkpoints/model_${STEP}.pt"

# Sample from Markov model
if [ -f "$CKPT_MARKOV" ]; then
    echo "[GPU 0] Sampling from Markov model..."
    CUDA_VISIBLE_DEVICES=0 nohup bash scripts/sample.sh \
        "$CKPT_MARKOV" \
        "experiments/markov_ddim/samples/final_${STEP}.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 0.0 &> logs/sample_markov_${STEP}.log &
    PID_MARKOV=$!
    echo "  PID: $PID_MARKOV"
else
    echo "  [SKIP] Checkpoint not found: $CKPT_MARKOV"
    PID_MARKOV=""
fi

# Sample from Non-Markov (epsilon) model
if [ -f "$CKPT_NONMARKOV_EPS" ]; then
    echo "[GPU 1] Sampling from Non-Markov (ε) model..."
    CUDA_VISIBLE_DEVICES=1 nohup bash scripts/sample_nonmarkov.sh \
        "$CKPT_NONMARKOV_EPS" \
        "experiments/nonmarkov_ddim/samples/final_${STEP}.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_eps_${STEP}.log &
    PID_NONMARKOV_EPS=$!
    echo "  PID: $PID_NONMARKOV_EPS"
else
    echo "  [SKIP] Checkpoint not found: $CKPT_NONMARKOV_EPS"
    PID_NONMARKOV_EPS=""
fi

# Sample from Non-Markov (x0) model
if [ -f "$CKPT_NONMARKOV_X0" ]; then
    echo "[GPU 2] Sampling from Non-Markov (x₀) model..."
    CUDA_VISIBLE_DEVICES=2 nohup bash scripts/sample_nonmarkov.sh \
        "$CKPT_NONMARKOV_X0" \
        "experiments/nonmarkov_x0/samples/final_${STEP}.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_x0_${STEP}.log &
    PID_NONMARKOV_X0=$!
    echo "  PID: $PID_NONMARKOV_X0"
else
    echo "  [SKIP] Checkpoint not found: $CKPT_NONMARKOV_X0"
    PID_NONMARKOV_X0=""
fi

# Sample from DART model
if [ -f "$CKPT_DART" ]; then
    echo "[GPU 3] Sampling from DART model (with CFG=1.5)..."
    CUDA_VISIBLE_DEVICES=3 nohup bash scripts/sample_dart.sh \
        "$CKPT_DART" \
        "experiments/dart/samples/final_${STEP}.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_${STEP}.log &
    PID_DART=$!
    echo "  PID: $PID_DART"
    
    # Also generate without CFG for comparison
    echo "[GPU 3] Sampling from DART model (without CFG)..."
    CUDA_VISIBLE_DEVICES=3 nohup bash scripts/sample_dart.sh \
        "$CKPT_DART" \
        "experiments/dart/samples/final_${STEP}_no_cfg.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 0.0 &> logs/sample_dart_no_cfg_${STEP}.log &
else
    echo "  [SKIP] Checkpoint not found: $CKPT_DART"
    PID_DART=""
fi

# Sample from Non-Markov (Signature) model
if [ -f "$CKPT_NONMARKOV_SIG" ]; then
    echo "[GPU 4] Sampling from Non-Markov (ε, Signature) model..."
    CUDA_VISIBLE_DEVICES=4 nohup bash scripts/sample_nonmarkov.sh \
        "$CKPT_NONMARKOV_SIG" \
        "experiments/nonmarkov_signature/samples/final_${STEP}.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 0.0 &> logs/sample_nonmarkov_sig_${STEP}.log &
    PID_NONMARKOV_SIG=$!
    echo "  PID: $PID_NONMARKOV_SIG"
else
    echo "  [SKIP] Checkpoint not found: $CKPT_NONMARKOV_SIG"
    PID_NONMARKOV_SIG=""
fi

# Sample from DART (Signature) model
if [ -f "$CKPT_DART_SIG" ]; then
    echo "[GPU 5] Sampling from DART (Signature, CFG=1.5) model..."
    CUDA_VISIBLE_DEVICES=5 nohup bash scripts/sample_dart.sh \
        "$CKPT_DART_SIG" \
        "experiments/dart_signature/samples/final_${STEP}.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 1.5 &> logs/sample_dart_sig_${STEP}.log &
    PID_DART_SIG=$!
    echo "  PID: $PID_DART_SIG"
    
    # Also without CFG
    echo "[GPU 5] Sampling from DART (Signature, no CFG)..."
    CUDA_VISIBLE_DEVICES=5 nohup bash scripts/sample_dart.sh \
        "$CKPT_DART_SIG" \
        "experiments/dart_signature/samples/final_${STEP}_no_cfg.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 0.0 &> logs/sample_dart_sig_no_cfg_${STEP}.log &
else
    echo "  [SKIP] Checkpoint not found: $CKPT_DART_SIG"
    PID_DART_SIG=""
fi

# Sample from DART (no CFG) model
if [ -f "$CKPT_DART_NO_CFG" ]; then
    echo "[GPU 6] Sampling from DART (no CFG) model..."
    CUDA_VISIBLE_DEVICES=6 nohup bash scripts/sample_dart.sh \
        "$CKPT_DART_NO_CFG" \
        "experiments/dart_no_cfg/samples/final_${STEP}.png" \
        "$NUM_SAMPLES" "$NUM_DDIM_STEPS" 5 dart 1.0 0.0 &> logs/sample_dart_no_cfg_${STEP}.log &
    PID_DART_NO_CFG=$!
    echo "  PID: $PID_DART_NO_CFG"
else
    echo "  [SKIP] Checkpoint not found: $CKPT_DART_NO_CFG"
    PID_DART_NO_CFG=""
fi

echo ""
echo "All sampling jobs launched!"
echo ""

# Collect active PIDs
PIDS=""
[ -n "$PID_MARKOV" ] && PIDS="$PIDS $PID_MARKOV"
[ -n "$PID_NONMARKOV_EPS" ] && PIDS="$PIDS $PID_NONMARKOV_EPS"
[ -n "$PID_NONMARKOV_X0" ] && PIDS="$PIDS $PID_NONMARKOV_X0"
[ -n "$PID_DART" ] && PIDS="$PIDS $PID_DART"
[ -n "$PID_NONMARKOV_SIG" ] && PIDS="$PIDS $PID_NONMARKOV_SIG"
[ -n "$PID_DART_SIG" ] && PIDS="$PIDS $PID_DART_SIG"
[ -n "$PID_DART_NO_CFG" ] && PIDS="$PIDS $PID_DART_NO_CFG"

if [ -n "$PIDS" ]; then
    echo "Active PIDs:$PIDS"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f logs/sample_*_${STEP}.log"
    echo ""
    echo "Wait for completion:"
    echo "  wait$PIDS"
    echo ""
    
    # Optional: wait for all jobs
    # wait$PIDS
    # echo "All sampling complete!"
fi

