#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-experiments/dart/checkpoints/model_0002000.pt}
OUT=${2:-experiments/dart/samples/samples.png}
NUM=${3:-16}
STEPS=${4:-50}
SUFFIX_STEPS=${5:-5}
SAMPLER=${6:-dart}  # 'dart' or 'ddim'
NOISE_SCALE=${7:-1.0}
CFG_SCALE=${8:-0.0}  # Classifier-free guidance scale

PYTHONPATH=src python -m nmsd.training.sample_dart "$CKPT" \
    --num "$NUM" \
    --steps "$STEPS" \
    --suffix-steps "$SUFFIX_STEPS" \
    --sampler "$SAMPLER" \
    --noise-scale "$NOISE_SCALE" \
    --cfg-scale "$CFG_SCALE" \
    --out "$OUT"

