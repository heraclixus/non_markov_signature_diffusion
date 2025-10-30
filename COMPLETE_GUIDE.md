# Complete Experimentation Guide

This guide provides a comprehensive overview of all implemented variants and how to run complete experiments.

## Quick Reference: All 7 Variants

| # | Variant | Config | Encoder | Predicts | Weighting | CFG | Directory |
|---|---------|--------|---------|----------|-----------|-----|-----------|
| 1 | **Markov** | `markov_mnist.yaml` | None | ε | Uniform | No | `markov_ddim/` |
| 2 | **Non-Markov (ε)** | `nonmarkov_mnist.yaml` | Transformer | ε | Uniform | No | `nonmarkov_ddim/` |
| 3 | **Non-Markov (x₀)** | `nonmarkov_mnist_x0.yaml` | Transformer | x₀ | Uniform | No | `nonmarkov_x0/` |
| 4 | **DART** | `dart_mnist.yaml` | Transformer | x₀ | SNR Sum | **Yes** | `dart/` |
| 5 | **DART (no CFG)** | `dart_mnist_no_cfg.yaml` | Transformer | x₀ | SNR Sum | No | `dart_no_cfg/` |
| 6 | **Non-Markov (Sig)** | `nonmarkov_mnist_signature.yaml` | **Signature** | ε | Uniform | No | `nonmarkov_signature/` |
| 7 | **DART (Sig)** | `dart_mnist_signature.yaml` | **Signature** | x₀ | SNR Sum | **Yes** | `dart_signature/` |

## Installation

```bash
# Basic requirements
pip install -r requirements.txt

# For signature encoding (variants 6-7)
pip install pysiglib
```

## Running Experiments

### Option 1: Run All (7 experiments on GPUs 0-6)

```bash
bash scripts/run_all_configs.sh all
```

**Requires**: 7 GPUs  
**Time**: ~90 minutes (parallel)  
**Output**: All 7 variants trained

### Option 2: Run Core Only (4 experiments on GPUs 0-3)

```bash
bash scripts/run_all_configs.sh core
```

**Requires**: 4 GPUs  
**Includes**: Markov, Non-Markov (ε), Non-Markov (x₀), DART (CFG)  
**Excludes**: Signature variants, DART (no CFG)

### Option 3: Run Signature Only (2 experiments on GPUs 4-5)

```bash
bash scripts/run_all_configs.sh signature
```

**Requires**: 2 GPUs  
**Includes**: Non-Markov (Signature), DART (Signature)

### Option 4: Run Individual

```bash
# Pick one
bash scripts/train_markov.sh configs/markov_mnist.yaml
bash scripts/train_nonmarkov.sh configs/nonmarkov_mnist.yaml
bash scripts/train_dart.sh configs/dart_mnist.yaml
bash scripts/train_dart.sh configs/dart_mnist_signature.yaml
```

## Monitoring

```bash
# Watch all training logs
tail -f logs/train_*.log

# Check running jobs
ps aux | grep train

# View loss plots (updated every 1000 steps)
open experiments/*/logs/training_losses.png

# Check memory usage
cat experiments/*/logs/memory_usage.txt
```

## Sampling

### After Training Completes

```bash
# Sample from all models
bash scripts/sample_all_models.sh 0004000 64 50

# This generates:
# - experiments/*/samples/final_0004000.png
# - experiments/dart*/samples/final_0004000_no_cfg.png (CFG comparison)
```

### Individual Sampling

```bash
# Markov
bash scripts/sample.sh experiments/markov_ddim/checkpoints/model_0004000.pt

# Non-Markov
bash scripts/sample_nonmarkov.sh experiments/nonmarkov_ddim/checkpoints/model_0004000.pt

# DART (with CFG, auto-detected)
bash scripts/sample_dart.sh experiments/dart/checkpoints/model_0004000.pt

# DART Signature (with CFG)
bash scripts/sample_dart.sh experiments/dart_signature/checkpoints/model_0004000.pt
```

### CFG Ablation

```bash
# Test different CFG scales for DART
bash scripts/ablate_cfg.sh experiments/dart/checkpoints/model_0004000.pt

# Or DART Signature
bash scripts/ablate_cfg.sh experiments/dart_signature/checkpoints/model_0004000.pt

# Output: cfg_ablation/{cfg_0.0.png, cfg_1.0.png, ..., cfg_3.0.png}
```

## Analysis

### Compare Training Losses

```bash
python scripts/compare_losses.py

# Output: loss_comparison.png, loss_comparison_zoomed.png
```

**Shows**: All variants on one plot (uses unweighted loss for fair DART comparison)

### Compare Memory Usage

```bash
python scripts/compare_memory.py

# Output: Text summary with overhead vs baseline
```

**Shows**: Peak GPU/CPU usage, overhead percentages

### View Individual Results

```bash
# Loss CSV
cat experiments/dart/logs/training_losses.csv

# Memory summary
cat experiments/dart/logs/memory_usage.txt

# Samples
ls experiments/dart/samples/
```

## Expected Results

### Training Time (10-30 epochs on MNIST)

- **Markov**: 25-35 min (GPU)
- **Non-Markov (Transformer)**: 30-40 min
- **Non-Markov (Signature)**: 25-35 min (fewer params, faster)
- **DART**: 30-40 min (same as Non-Markov)

### Memory Usage (batch_size=128)

- **Markov**: ~2.4 GB GPU
- **Non-Markov (Transformer)**: ~3.1 GB (+27%)
- **Non-Markov (Signature)**: ~2.7 GB (+12%, more efficient!)
- **DART**: Similar to Non-Markov

### Sample Quality (Expected Trends)

**Hypothesis to test**:
1. Non-Markov > Markov (suffix context helps)
2. x₀ ≈ ε (similar quality, different parameterization)
3. SNR weighting helps training stability
4. CFG improves quality, reduces diversity
5. Signature ≈ Transformer (compact, similar quality)

## Ablation Plan

### Core Ablations

1. **Context effect**: Markov vs Non-Markov
2. **Prediction type**: ε vs x₀
3. **Weighting**: Uniform vs SNR
4. **CFG**: With vs without
5. **Encoder**: Transformer vs Signature

### Recommended Ablation Order

```bash
# 1. Baseline
bash scripts/train_markov.sh configs/markov_mnist.yaml

# 2. Add context (Transformer)
bash scripts/train_nonmarkov.sh configs/nonmarkov_mnist.yaml

# 3. Full DART (Transformer + CFG)
bash scripts/train_dart.sh configs/dart_mnist.yaml

# 4. Signature encoding
bash scripts/train_dart.sh configs/dart_mnist_signature.yaml

# 5. Compare all
python scripts/compare_losses.py
python scripts/compare_memory.py
```

## Troubleshooting

### pysiglib Not Found

```bash
pip install pysiglib

# Or skip signature variants and run core only
bash scripts/run_all_configs.sh core
```

### CUDA Out of Memory

Reduce batch size in configs:
```yaml
data:
  batch_size: 64  # or 32
```

### Training Too Slow

Run fewer variants:
```bash
# Just run DART with signature
bash scripts/train_dart.sh configs/dart_mnist_signature.yaml
```

## Directory Structure (After All Experiments)

```
experiments/
├── markov_ddim/           # Variant 1
├── nonmarkov_ddim/        # Variant 2
├── nonmarkov_x0/          # Variant 3
├── dart/                  # Variant 4 ⭐
├── dart_no_cfg/           # Variant 5
├── nonmarkov_signature/   # Variant 6
└── dart_signature/        # Variant 7 ⭐

Each contains:
  ├── checkpoints/model_XXXXXXX.pt
  ├── samples/step_XXXXXXX.png, final_XXXXXXX.png
  └── logs/training_losses.csv, training_losses.png, memory_usage.txt

logs/  # Training logs from run_all_configs.sh
  ├── train_markov.log
  ├── train_nonmarkov_eps.log
  ├── train_dart.log
  └── ...
```

## Documentation Index

- **README.md**: Project overview and theory
- **QUICKSTART.md**: Basic usage guide
- **CONFIG_VARIANTS.md**: All config files explained
- **SIGNATURE_ENCODING.md**: Signature encoder details
- **CLASSIFIER_FREE_GUIDANCE.md**: CFG explanation
- **DART_LOSS.md**: DART loss and weighting
- **LOGGING_AND_DEBUGGING.md**: Loss/memory tracking
- **SAMPLING_SCRIPTS_GUIDE.md**: All sampling scripts
- **COMPLETE_GUIDE.md**: This file

## Summary

✅ **7 variants implemented** covering all combinations  
✅ **Parallel execution** across multiple GPUs  
✅ **Automatic logging** for losses and memory  
✅ **Easy comparison** tools for analysis  
✅ **Comprehensive documentation** for each component  
✅ **Signature encoding** with pysiglib integration  
✅ **CFG support** with auto-detection  

All three project phases (Markov → Non-Markov → Signature) are fully implemented and ready to run!

