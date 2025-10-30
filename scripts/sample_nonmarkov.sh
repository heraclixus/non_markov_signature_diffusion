#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-experiments/nonmarkov_ddim/checkpoints/model_0002000.pt}
OUT=${2:-experiments/nonmarkov_ddim/samples/samples.png}
NUM=${3:-16}
STEPS=${4:-50}
SUFFIX_STEPS=${5:-5}
ETA=${6:-0.0}

PYTHONPATH=src python -m nmsd.training.sample_nonmarkov "$CKPT" \
    --num "$NUM" \
    --steps "$STEPS" \
    --suffix-steps "$SUFFIX_STEPS" \
    --eta "$ETA" \
    --out "$OUT"

