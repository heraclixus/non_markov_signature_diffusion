# CIFAR-10 Dataset Guide

This guide explains how to use the CIFAR-10 dataset with the non-Markov signature diffusion pipeline.

## Overview

The pipeline now supports both MNIST and CIFAR-10 datasets. CIFAR-10 consists of 60,000 32x32 color images (RGB) in 10 classes, compared to MNIST's 70,000 28x28 grayscale images in 10 classes.

### Key Differences

| Dataset | Channels | Image Size | Classes | Samples |
|---------|----------|------------|---------|---------|
| MNIST   | 1 (grayscale) | 28x28 | 10 | 70,000 |
| CIFAR-10 | 3 (RGB) | 32x32 | 10 | 60,000 |

## Configuration Changes

When using CIFAR-10, the following configuration parameters are automatically adjusted:

### Data Section
```yaml
data:
  dataset: cifar10  # Changed from 'mnist'
  image_size: 32    # Changed from 28
  batch_size: 64    # Often reduced due to larger images
```

### Model Section
```yaml
model:
  in_channels: 3      # Changed from 1 (RGB vs grayscale)
  base_channels: 128  # Increased from 64 for more complex data
  channel_mults: [1, 2, 2, 2]  # More channels for better capacity
  time_emb_dim: 512   # Increased from 256
```

### Encoder Section (for Non-Markov models)
```yaml
encoder:
  hidden_dim: 512     # Increased from 256
  num_heads: 8        # Increased from 4 (for transformer)
  num_layers: 4       # Increased from 2 (for transformer)
  pooling: conv_pool  # Recommended for RGB (signature encoder)
```

### Training Section
```yaml
training:
  epochs: 100            # Increased from 30
  batch_size: 64         # Reduced from 128 (memory constraints)
  save_every_steps: 5000 # Increased from 2000
  ema_decay: 0.9999      # Slightly adjusted from 0.999
```

## Available Configurations

The following CIFAR-10 config files are provided:

### 1. Markov DDIM (Baseline)
```bash
configs/markov_cifar10.yaml
```
Standard DDIM model without non-Markov suffix context.

### 2. Non-Markov with Transformer Encoder
```bash
configs/nonmarkov_cifar10.yaml
```
Non-Markov model using transformer-based suffix encoder.

### 3. Non-Markov with Signature Encoder
```bash
configs/nonmarkov_cifar10_signature.yaml
```
Non-Markov model using path signature-based suffix encoder.

### 4. DART (Transformer Encoder)
```bash
configs/dart_cifar10.yaml
```
DART loss with transformer encoder and classifier-free guidance.

### 5. DART (Signature Encoder)
```bash
configs/dart_cifar10_signature.yaml
```
DART loss with signature encoder and classifier-free guidance.

## Training Commands

### Markov Baseline
```bash
python -m nmsd.training.train configs/markov_cifar10.yaml
```

### Non-Markov Models
```bash
# Transformer encoder
python -m nmsd.training.train_nonmarkov configs/nonmarkov_cifar10.yaml

# Signature encoder
python -m nmsd.training.train_nonmarkov configs/nonmarkov_cifar10_signature.yaml
```

### DART Models
```bash
# Transformer encoder
python -m nmsd.training.train_dart configs/dart_cifar10.yaml

# Signature encoder
python -m nmsd.training.train_dart configs/dart_cifar10_signature.yaml
```

## Data Augmentation

CIFAR-10 training includes random horizontal flipping for data augmentation:

```python
train_tfm = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

Test data does not include augmentation for consistent evaluation.

## Memory Considerations

CIFAR-10 requires more memory than MNIST due to:
- **3x more channels** (RGB vs grayscale)
- **Larger spatial dimensions** (32x32 vs 28x28)
- **Deeper models** (more channels and layers)

### Recommendations:
1. **Reduce batch size** from 128 to 64 (or 32 on smaller GPUs)
2. **Use gradient accumulation** if needed
3. **Monitor GPU memory** during training

Example with smaller batch size:
```yaml
data:
  batch_size: 32  # For GPUs with limited memory
```

## Expected Results

CIFAR-10 is more challenging than MNIST, so expect:
- **Longer training times** (100 epochs vs 30)
- **More training steps** for convergence
- **Higher loss values** initially
- **More nuanced improvements** from non-Markov/DART methods

## Signature Encoder Recommendations

For CIFAR-10's RGB images, the signature encoder configuration differs:

### Recommended Settings
```yaml
encoder:
  type: signature
  signature_degree: 3        # Good balance of expressiveness and efficiency
  pooling: conv_pool         # Better for RGB images (vs spatial_mean for grayscale)
  time_augment: true         # Include time information
  use_lead_lag: false        # Usually not needed
  hidden_dim: 512            # Match model capacity
```

### Pooling Options
- **`spatial_mean`**: Simple average pooling (good for MNIST)
- **`flatten`**: Flatten and project (memory intensive)
- **`conv_pool`**: Convolutional features (recommended for CIFAR-10)

## Sampling

Sampling uses the same commands as MNIST but with CIFAR-10 checkpoints:

```bash
# Sample from a trained model
python -m nmsd.diffusion.sampler \
    --config configs/dart_cifar10_signature.yaml \
    --checkpoint experiments/dart_signature_cifar10/checkpoints/model_0050000.pt \
    --num_samples 64 \
    --output samples_cifar10.png
```

## Switching Between Datasets

To switch from MNIST to CIFAR-10 in an existing config:

1. Change `data.dataset` from `mnist` to `cifar10`
2. Change `data.image_size` from `28` to `32`
3. Change `model.in_channels` from `1` to `3`
4. Increase model capacity (base_channels, time_emb_dim, etc.)
5. Adjust training epochs and batch size
6. Update output directory to avoid overwriting MNIST results

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size` to 32 or 16
- Reduce `base_channels` to 96 or 64
- Reduce `num_suffix_steps` from 10 to 5
- Use gradient checkpointing (if implemented)

### Slow Training
- Ensure GPU is being used (`torch.cuda.is_available()`)
- Reduce `num_workers` if CPU is bottleneck
- Consider mixed precision training (fp16)

### Poor Sample Quality
- Train for more epochs (100-200 for CIFAR-10)
- Increase model capacity
- Try different CFG scales during sampling
- Verify normalization is correct ([-1, 1] range)

## Code Implementation

The dataset switching is handled automatically in `src/nmsd/data/datasets.py`:

```python
def get_dataloaders(dataset_name, root, batch_size, num_workers, image_size=None):
    if dataset_name.lower() == "mnist":
        if image_size is None:
            image_size = 28
        return get_mnist_dataloaders(...)
    elif dataset_name.lower() == "cifar10":
        if image_size is None:
            image_size = 32
        return get_cifar10_dataloaders(...)
```

All training scripts (`train.py`, `train_nonmarkov.py`, `train_dart.py`) now use this generic function.

## Next Steps

1. **Start with baseline**: Train `markov_cifar10.yaml` first to verify setup
2. **Compare architectures**: Train both transformer and signature encoders
3. **Experiment with DART**: Try different weighting schemes
4. **Tune hyperparameters**: Adjust CFG scale, learning rate, etc.
5. **Evaluate quantitatively**: Use FID, IS, or other metrics for CIFAR-10

## References

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- torchvision.datasets.CIFAR10: https://pytorch.org/vision/stable/datasets.html#cifar

