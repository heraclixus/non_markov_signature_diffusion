#!/usr/bin/env bash
set -euo pipefail

# Script to run all experiments in parallel across GPUs
# Each experiment uses a dedicated GPU and logs to a separate file

MODE=${1:-all}  # 'all', 'core', or 'signature'

echo "Starting experiments in parallel (mode=$MODE)..."
echo "Logs will be written to logs/ directory"
echo ""

# Create logs directory
mkdir -p logs

# GPU assignments
GPU_MARKOV=0
GPU_NONMARKOV_EPS=1
GPU_NONMARKOV_X0=2
GPU_DART=3
GPU_NONMARKOV_SIG=4
GPU_DART_SIG=5
GPU_DART_NO_CFG=6

PIDS=()
NAMES=()

# Core experiments (always run)
if [[ "$MODE" == "all" ]] || [[ "$MODE" == "core" ]]; then
    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU_MARKOV nohup bash scripts/train_markov.sh configs/markov_mnist.yaml &> logs/train_markov.log &
    PIDS+=($!)
    NAMES+=("Markov")

    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU_NONMARKOV_EPS nohup bash scripts/train_nonmarkov.sh configs/nonmarkov_mnist.yaml &> logs/train_nonmarkov_eps.log &
    PIDS+=($!)
    NAMES+=("Non-Markov(ε,Transformer)")

    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU_NONMARKOV_X0 nohup bash scripts/train_nonmarkov.sh configs/nonmarkov_mnist_x0.yaml &> logs/train_nonmarkov_x0.log &
    PIDS+=($!)
    NAMES+=("Non-Markov(x₀,Transformer)")

    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU_DART nohup bash scripts/train_dart.sh configs/dart_mnist.yaml &> logs/train_dart.log &
    PIDS+=($!)
    NAMES+=("DART(Transformer,CFG)")
fi

# Signature experiments
if [[ "$MODE" == "all" ]] || [[ "$MODE" == "signature" ]]; then
    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU_NONMARKOV_SIG nohup bash scripts/train_nonmarkov.sh configs/nonmarkov_mnist_signature.yaml &> logs/train_nonmarkov_sig.log &
    PIDS+=($!)
    NAMES+=("Non-Markov(ε,Signature)")

    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU_DART_SIG nohup bash scripts/train_dart.sh configs/dart_mnist_signature.yaml &> logs/train_dart_sig.log &
    PIDS+=($!)
    NAMES+=("DART(Signature,CFG)")
fi

# Optional: DART without CFG (ablation)
if [[ "$MODE" == "all" ]]; then
    # Run with nohup and capture PID (set PYTHONPATH to find nmsd module)
    PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$GPU_DART_NO_CFG nohup bash scripts/train_dart.sh configs/dart_mnist_no_cfg.yaml &> logs/train_dart_no_cfg.log &
    PIDS+=($!)
    NAMES+=("DART(Transformer,no-CFG)")
fi

echo ""
echo "All training jobs launched!"
echo ""
echo "Summary:"
echo "--------"
for i in "${!PIDS[@]}"; do
    echo "  ${NAMES[$i]}: PID ${PIDS[$i]}"
done
echo ""
echo "Mode: $MODE"
echo "  'core' = 4 core experiments (GPUs 0-3)"
echo "  'signature' = 2 signature experiments (GPUs 4-5)"
echo "  'all' = 7 experiments including ablations (GPUs 0-6)"
echo ""
echo "Monitor training:"
echo "  tail -f logs/train_*.log"
echo ""
echo "Check running jobs:"
echo "  ps -p $(IFS=,; echo "${PIDS[*]}")"
echo ""
echo "Compare results after training:"
echo "  python scripts/compare_losses.py"
echo "  python scripts/compare_memory.py"