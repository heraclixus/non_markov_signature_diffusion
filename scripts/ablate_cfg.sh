#!/usr/bin/env bash
set -euo pipefail

# Script to ablate CFG scale for DART model
# Generates samples with different CFG scales for comparison
# Usage: bash scripts/ablate_cfg.sh <checkpoint> [output_dir]

CKPT=${1:-experiments/dart/checkpoints/model_0004000.pt}
OUT_DIR=${2:-cfg_ablation}
NUM_SAMPLES=${3:-36}
STEPS=${4:-50}

echo "CFG Ablation Study"
echo "===================="
echo "Checkpoint: $CKPT"
echo "Output dir: $OUT_DIR"
echo "Num samples: $NUM_SAMPLES"
echo "DDIM steps: $STEPS"
echo ""

mkdir -p "$OUT_DIR"

# CFG scales to test
CFG_SCALES=(0.0 0.5 1.0 1.5 2.0 2.5 3.0)

echo "Generating samples with different CFG scales..."
for cfg in "${CFG_SCALES[@]}"; do
    echo "  CFG scale = $cfg"
    PYTHONPATH=src python -m nmsd.training.sample_dart \
        "$CKPT" \
        --num "$NUM_SAMPLES" \
        --steps "$STEPS" \
        --suffix-steps 5 \
        --sampler dart \
        --cfg-scale "$cfg" \
        --out "$OUT_DIR/cfg_${cfg}.png"
done

echo ""
echo "Done! Results saved to $OUT_DIR/"
echo ""
echo "Files generated:"
ls -lh "$OUT_DIR"/*.png

echo ""
echo "View all results:"
echo "  open $OUT_DIR/*.png"
echo ""
echo "CFG Scale Guide:"
echo "  0.0  - No guidance (baseline)"
echo "  1.0  - Moderate guidance"
echo "  1.5  - Recommended (good quality/diversity balance)"
echo "  2.0  - Strong guidance (may reduce diversity)"
echo "  3.0+ - Very strong (may introduce artifacts)"

