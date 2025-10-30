#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/markov_mnist.yaml}

PYTHONPATH=src python -m nmsd.training.train "$CONFIG"

