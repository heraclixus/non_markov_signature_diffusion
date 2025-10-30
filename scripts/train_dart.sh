#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/dart_mnist.yaml}

PYTHONPATH=src python -m nmsd.training.train_dart "$CONFIG"

