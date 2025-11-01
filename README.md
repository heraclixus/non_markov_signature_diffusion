# Non Markov Diffusion Model with Signature Encoding

This repository explores a non‑Markov variant of variational diffusion models and the use of path signatures to encode non‑Markov reverse context. We proceed in three phases:

- Baseline: reproduce a standard Markov diffusion model (DDPM‑style).
- Non‑Markov: train a reverse model that conditions on the entire future suffix x_{t:T} instead of only x_t.
- Signature context: replace generic sequence encoders with a signature/log‑signature encoder of the suffix x_{t:T}.

The goal is to measure whether non‑Markov reverse conditioning and signature features improve sample quality or likelihood.


## 1) Problem setup and objectives

Let x_0 ∼ p_data be an image, and let t ∈ {1,…,T}. We choose a monotonically decreasing schedule {γ_t} with γ_0 = 1 and γ_T ≈ 0.

Forward (noising) process variants:

- Markov (baseline, DDPM):
  - q(x_t | x_{t−1}) = Normal(√α_t x_{t−1}, (1−α_t) I), forming a Markov chain where x_t only depends on x_{t-1}.
  - The marginal is q(x_t | x_0) = Normal(√γ_t x_0, (1−γ_t) I) with γ_t = ∏_{s=1}^t α_s.

- Non‑Markov (this work):
  - **Fixed, non-Markovian** inference process: q(x_t | x_{0:t−1}) = q(x_t | x_0) = Normal(√γ_t x_0, (1−γ_t) I), ∀ t ∈ [1, T].
  - Note: x_t depends on the entire history x_{0:t-1} through x_0, making it non-Markovian (not just dependent on x_{t-1}).
  - **Key property**: Each x_t is conditionally independent given x_0, enabling parallel sampling.
  - Implementation: x_t = √γ_t x_0 + √(1−γ_t) ε_t, where ε_t ∼ Normal(0, I) are independent across t.

Reverse (generative) models:

- Markov baseline: p_θ(x_{t−1} | x_t).
- Non‑Markov: p_θ(x_{t−1} | x_{t:T}), which uses the future suffix as context (a smoothing‑style reverse model).

Variational objective (non‑Markov ELBO, provided):

$$\mathcal{L}^{\text{ELBO}}_{\theta,\phi} = \mathbb{E}_{\mathbf{x}_{1:T} \sim q_\phi(\mathbf{x})} \Big[ \sum_{t=1}^T \log p_\theta(\mathbf{x}_{t-1}\,|\, \mathbf{x}_{t:T}) + \log p_\theta(\mathbf{x}_T) - \log q_\phi(\mathbf{x}_{1:T}\,|\,\mathbf{x}_0) \Big] $$

Notes:

- In the Markov baseline, the common “simple” loss is a weighted MSE between true noise ε and a network prediction ε_θ(x_t, t).
- In the non‑Markov variant, we allow ε_θ to condition on a representation of the suffix x_{t:T} via an encoder Enc(·): ε_θ(x_t, t, Enc(x_{t:T})).
- Boundary terms (prior on x_T, data likelihood p_θ(x_0 | x_1)) follow standard DDPM practice unless otherwise stated.


## 2) From Markov to non‑Markov training losses

- Markov baseline (DDPM‑style):
  - **Forward**: Sequential Markov chain q(x_t | x_{t-1}).
  - **Training**: Sample t ∼ Uniform({1,…,T}), ε ∼ Normal(0,I), form x_t = √γ_t x_0 + √(1−γ_t) ε.
  - **Loss**: Minimize w_t · ||ε − ε_θ(x_t, t)||² with schedule‑dependent weighting w_t.
  - **Reverse**: Model p_θ(x_{t-1} | x_t) uses only current state.

- Non‑Markov with suffix context:
  - **Forward**: Non-Markovian process q(x_t | x_{0:t-1}) = q(x_t | x_0), where each x_t is conditionally independent given x_0.
  - **Training**: For each minibatch, draw independent {ε_s}_{s=t}^{t+k} and construct the suffix {x_s}_{s=t}^{t+k} using:
    ```
    x_s = √γ_s x_0 + √(1−γ_s) ε_s,  for s ∈ {t, t+1, ..., t+k}
    ```
    This is parallelizable across timesteps s since they are independent given x_0.
  - **Encoding**: Encode the future suffix c_t = Enc(x_{t:t+k}, {t, t+1, ..., t+k}). Options: Transformer/Conv1D over time; or signature/log‑signature (Section 3).
  - **Loss**: Minimize w_t · ||ε_t − ε_θ(x_t, t, c_t)||², plus standard prior/likelihood terms.
  - **Reverse**: Model p_θ(x_{t-1} | x_{t:T}) uses suffix context for improved denoising.

**Key insight**: While the forward marginals q(x_t | x_0) look identical to DDPM, the underlying meaning is different:
- DDPM: x_t is the result of a Markov chain through x_1, ..., x_{t-1}
- Non-Markov: x_t is directly sampled from x_0, making {x_t} conditionally independent given x_0

This allows efficient parallel suffix construction during training while enabling non-Markov reverse conditioning.


## 3) Signature/log‑signature encoding of x_{t:T}

Path signatures provide hierarchical statistics of sequences via iterated integrals. For a time‑augmented path Z_s = [x_s, s/T], the (log‑)signature up to order m yields a fixed‑dimensional, order‑aware summary with provable universality properties.

- Practical recipe for images:
  - We summarize the temporal path across diffusion steps for each spatial location (or for a pooled feature map) and then average or project to a global context vector c_t.
  - Use a truncated log‑signature up to order m=2 or m=3 for stability and manageable dimension.
  - Libraries: Signatory (PyTorch) and iisignature. If GPU wheels are unavailable, fall back to a small Transformer over time.

- Encoder interface (conceptual):
  - c_t = SignatureEncoder(x_{t:T}, t; order=m, time_channel=True)
  - Feed c_t into the noise predictor or into a mean/variance head of p_θ(x_{t−1} | x_{t:T}).

Benefits to test empirically:

- Strong, permutation‑sensitive summaries of the suffix without heavy sequence models.
- Compact fixed‑size context even when T is large.


## 4) Datasets and metrics (simple, image)

**Supported Datasets:**

- **MNIST** (28×28, grayscale): ✅ Implemented - quickest sanity check for likelihood and sample quality.
- **CIFAR‑10** (32×32, RGB): ✅ Implemented - standard benchmark for FID/IS comparisons.

The pipeline automatically handles both datasets via the `dataset` field in config files. See `CIFAR10_GUIDE.md` for detailed CIFAR-10 usage instructions.

Evaluation metrics:

- FID (use 50k generated vs test set; library: torch‑fidelity or CleanFID).
- Inception Score (less stable; report alongside FID on CIFAR‑10).
- NLL estimates (bits/dim) if computing the full variational bound.


## 5) Repository layout (planned)

```
non_markov_signature_diffusion/
  README.md
  requirements.txt            # torch, torchvision, signatory or iisignature, torch-fidelity, einops, omegaconf, tqdm, wandb (optional)
  pyproject.toml              # (optional) if packaging as a module
  configs/
    markov_mnist.yaml
    nonmarkov_mnist_signature.yaml
    nonmarkov_cifar_signature.yaml
  src/
    nmsd/
      __init__.py
      diffusion/
        schedulers.py         # Markov and Non‑Markov forward samplers (γ_t handling)
        losses.py             # Markov DDPM loss, non‑Markov suffix‑conditional loss
        sampler.py            # ancestral / non‑Markov reverse sampling
      encoders/
        signature.py          # Signature/log‑signature wrappers
        transformer_context.py# Baseline temporal encoder
      models/
        unet.py               # U‑Net backbone
        heads.py              # noise / mean-variance heads
      data/
        datasets.py           # MNIST/CIFAR‑10 loaders, preprocessing
      training/
        train.py              # training loop, EMA, checkpointing
        ema.py
      evaluation/
        fid.py                # wrapper around torch‑fidelity / CleanFID
      utils/
        distributed.py, io.py, schedule.py, seed.py
  scripts/
    download_data.sh
    train_markov.sh
    train_nonmarkov_sig.sh
    sample.sh
    eval_fid.sh
  experiments/
    runs/                     # logs, tensorboard
    checkpoints/
```


## 6) Experiment plan

Phase A — Baseline (Markov):

1. Train DDPM‑style on MNIST with a small U‑Net. Verify convergence and sample quality.
2. Move to CIFAR‑10 at 32×32; reproduce standard FID ranges for a small model.

Phase B — Non‑Markov reverse without signatures:

3. Keep the same forward marginal q(x_t | x_0) but drop the Markov reverse assumption. Condition ε_θ on a lightweight temporal encoder over x_{t:T}.
4. Verify training stability and compare FID to the baseline.

Phase C — Signature context:

5. Replace temporal encoder with Signature/LogSignature on x_{t:T}. Tune order m ∈ {2,3}.
6. Ablations: encoder type (Transformer vs Signature), context dimensionality, schedule {γ_t}, T.
7. Report FID/IS and likelihood where practical.


## 7) Setup notes

- Python ≥ 3.10, PyTorch ≥ 2.2, torchvision.
- Optional: `signatory` (GPU build) or `iisignature`. If unavailable, use the Transformer context encoder.
- Metrics: `torch-fidelity` (or `clean-fid`).

Quick start:

```bash
pip install -r requirements.txt

# Baseline Markov on MNIST
python -m nmsd.training.train configs/markov_mnist.yaml

# Non‑Markov + Signature on MNIST
python -m nmsd.training.train_nonmarkov configs/nonmarkov_mnist_signature.yaml

# DART with Signature on MNIST
python -m nmsd.training.train_dart configs/dart_mnist_signature.yaml

# CIFAR-10 Examples
python -m nmsd.training.train configs/markov_cifar10.yaml
python -m nmsd.training.train_nonmarkov configs/nonmarkov_cifar10_signature.yaml
python -m nmsd.training.train_dart configs/dart_cifar10_signature.yaml
```

See `QUICKSTART.md` for basic usage and `CIFAR10_GUIDE.md` for CIFAR-10 specific instructions.


## 8) Implementation guidance (high‑level)

- Schedulers: implement utilities for γ_t and conversions to DDPM α_t when needed.
- Losses: share the same backbone but switch the context: empty for Markov; suffix encoder for non‑Markov.
- Sampling: in the non‑Markov case, predict parameters of p_θ(x_{t−1} | x_{t:T}) at each step using the suffix context computed on‑the‑fly.
- Efficiency: leverage parallel construction of {x_s}_{s=t}^T under the non‑Markov forward (independent given x_0).


## 9) References and tools

- DDPM and variants for the Markov baseline.
- Path Signatures and Log‑Signatures (Lyons et al.). Libraries: Signatory (PyTorch), iisignature.


— End —
