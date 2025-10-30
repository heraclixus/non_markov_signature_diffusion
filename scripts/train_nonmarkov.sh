#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/nonmarkov_mnist.yaml}

PYTHONPATH=src python -m nmsd.training.train_nonmarkov "$CONFIG"

