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
  - q(x_t | x_{t−1}) = Normal(√α_t x_{t−1}, (1−α_t) I), so that q(x_t | x_0) = Normal(√γ_t x_0, (1−γ_t) I) with γ_t = ∏_{s=1}^t α_s.

- Non‑Markov (parallelized, conditionally independent given x_0):
  - q(x_t | x_{0:t−1}) = q(x_t | x_0) = Normal(√γ_t x_0, (1−γ_t) I), for all t ∈ [1, T].
  - Each x_t can be sampled in parallel: x_t = √γ_t x_0 + √(1−γ_t) ε_t, ε_t ∼ Normal(0, I).

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
  - Sample t ∼ Uniform({1,…,T}), ε ∼ Normal(0,I), form x_t = √γ_t x_0 + √(1−γ_t) ε.
  - Minimize w_t · ||ε − ε_θ(x_t, t)||² with schedule‑dependent weighting w_t.

- Non‑Markov with suffix context:
  - For each minibatch, draw independent {ε_s} and construct {x_s}_{s=t}^T using x_s = √γ_s x_0 + √(1−γ_s) ε_s (parallelizable across s).
  - Encode the future suffix c_t = Enc(x_{t:T}, t). Options: Transformer/Conv1D over time; or signature/log‑signature (Section 3).
  - Minimize w_t · ||ε_t − ε_θ(x_t, t, c_t)||², plus standard prior/likelihood terms.

This keeps the computational profile close to DDPM while allowing non‑Markov reverse conditioning on the entire suffix.


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

Recommended progression:

- MNIST (28×28, grayscale): quickest sanity check for likelihood and sample quality.
- CIFAR‑10 (32×32, RGB): standard benchmark for FID/IS comparisons.

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

Quick start (to be implemented once code is added):

```
pip install -r requirements.txt

# Baseline Markov on MNIST
bash scripts/train_markov.sh configs/markov_mnist.yaml

# Non‑Markov + Signature on CIFAR‑10
bash scripts/train_nonmarkov_sig.sh configs/nonmarkov_cifar_signature.yaml

# Evaluate FID
bash scripts/eval_fid.sh --dataset cifar10 --ckpt experiments/checkpoints/best.pt
```


## 8) Implementation guidance (high‑level)

- Schedulers: implement utilities for γ_t and conversions to DDPM α_t when needed.
- Losses: share the same backbone but switch the context: empty for Markov; suffix encoder for non‑Markov.
- Sampling: in the non‑Markov case, predict parameters of p_θ(x_{t−1} | x_{t:T}) at each step using the suffix context computed on‑the‑fly.
- Efficiency: leverage parallel construction of {x_s}_{s=t}^T under the non‑Markov forward (independent given x_0).


## 9) References and tools

- DDPM and variants for the Markov baseline.
- Path Signatures and Log‑Signatures (Lyons et al.). Libraries: Signatory (PyTorch), iisignature.


— End —
