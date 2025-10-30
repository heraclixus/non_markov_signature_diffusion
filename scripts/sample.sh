#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-experiments/markov_ddim/checkpoints/model_0002000.pt}
OUT=${2:-experiments/markov_ddim/samples/samples.png}
NUM=${3:-16}
STEPS=${4:-50}
ETA=${5:-0.0}

PYTHONPATH=src python -m nmsd.training.sample "$CKPT" --num "$NUM" --steps "$STEPS" --eta "$ETA" --out "$OUT"

