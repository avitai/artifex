#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
r"""
# Training a Flow Model on MNIST with RealNVP

## Overview

This tutorial demonstrates how to train a RealNVP normalizing flow model using
Artifex's configuration-based API. Instead of implementing flow transformations
from scratch, we use Artifex's `RealNVP` class with `RealNVPConfig` and
`CouplingNetworkConfig` for clean, production-ready training.

**Key Artifex Components Used:**
- `RealNVP` - RealNVP flow model implementation
- `RealNVPConfig` - Frozen dataclass configuration for RealNVP
- `CouplingNetworkConfig` - Coupling network configuration
- `DataRax` - Efficient GPU-accelerated data loading

## Training Best Practices Applied

Based on RealNVP and flow model research:
- Dequantization for discrete data (add uniform noise)
- 12 coupling layers with 4-layer MLPs (512 units)
- Learning rate warmup + cosine decay
- Gradient clipping for stability
- Maximum likelihood training (negative log-likelihood)
- JIT compilation for GPU performance

## Expected Results

- **Training time:** ~50 minutes (GPU), ~4-5 hours (CPU)
- **Final NLL:** ~-2500 to -2700 (more negative is better)
- **Generated samples:** Clear, recognizable digits

## Prerequisites

```bash
# Install Artifex
uv sync
```

---
"""

# %%
# IMPORTANT: Set memory env vars BEFORE importing TensorFlow or JAX
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Don't pre-allocate GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # JAX: don't pre-allocate
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # JAX: use 90% of GPU memory

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from datarax import from_source
from datarax.sources import TfdsDataSourceConfig, TFDSSource
from flax import nnx
from tqdm import tqdm

# Artifex imports
from artifex.generative_models.core.configuration.flow_config import (
    CouplingNetworkConfig,
    RealNVPConfig,
)
from artifex.generative_models.models.flow.real_nvp import RealNVP


print("=" * 70)
print("Artifex RealNVP Training - MNIST")
print("Using: RealNVP, RealNVPConfig, CouplingNetworkConfig, DataRax, nnx.jit")
print("=" * 70)

# %% [markdown]
r"""
## Step 1: Configuration

Training configuration based on RealNVP best practices.
"""

# %%
# Configuration (based on RealNVP best practices)
SEED = 42
NUM_EPOCHS = 100  # 100 epochs for good quality
BATCH_SIZE = 512  # Larger batch size for better GPU utilization
BASE_LR = 1e-3  # Scale LR with batch size
WARMUP_STEPS = 200  # Warmup steps

print("\nConfiguration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {BASE_LR} (with {WARMUP_STEPS} warmup steps)")

# %% [markdown]
r"""
## Step 2: Data Loading with DataRax

Flow models require continuous data. MNIST is discrete (0-255), so we apply
dequantization. We use DataRax for efficient GPU-accelerated data loading.
"""


# %%
print("\n📊 Loading MNIST data with DataRax...")

# Initialize data RNGs
data_key = jax.random.key(SEED + 1)
data_rngs = nnx.Rngs(default=data_key)

# Configure MNIST data source using DataRax
train_source_config = TfdsDataSourceConfig(
    name="mnist",
    split="train",
    shuffle=True,
    shuffle_buffer_size=10000,
)
train_source = TFDSSource(train_source_config, rngs=data_rngs)

print(f"  ✅ MNIST train dataset loaded: {len(train_source)} samples")

# Create training pipeline with batching and JIT compilation
train_pipeline = from_source(train_source, batch_size=BATCH_SIZE, jit_compile=True)

# Calculate number of batches per epoch
n_batches = len(train_source) // BATCH_SIZE
print(f"  ✅ Training pipeline created: {n_batches} batches per epoch")


def preprocess_batch(batch, key):
    """Preprocess MNIST batch for flow models with dequantization.

    Flow models require continuous data, so we:
    1. Normalize to [0, 1]
    2. Flatten to 784-dim vectors
    3. Apply dequantization (add uniform noise)
    4. Scale to [-1, 1]
    """
    # Extract images from batch
    images = batch["image"]

    # Normalize to [0, 1]
    images = images.astype(jnp.float32) / 255.0

    # Flatten to (batch_size, 784)
    images = images.reshape(images.shape[0], -1)

    # Dequantization: add uniform noise for continuous data
    noise = jax.random.uniform(key, images.shape) / 256.0
    images = images + noise

    # Scale to [-1, 1]
    images = (images - 0.5) / 0.5

    return images


# %% [markdown]
r"""
## Step 3: Create RealNVP Using Artifex's API

Use Artifex's `RealNVP` class with configuration objects.
"""

# %%
# Initialize RNGs
key = jax.random.key(SEED)
params_key, noise_key, sample_key, dropout_key = jax.random.split(key, 4)
rngs = nnx.Rngs(
    params=params_key,
    noise=noise_key,
    sample=sample_key,
    dropout=dropout_key,
)

# Coupling network config (4 hidden layers with 512 units each for better capacity)
coupling_config = CouplingNetworkConfig(
    name="coupling_mlp",
    hidden_dims=(512, 512, 512, 512),  # 4 hidden layers with more capacity
    activation="relu",
    network_type="mlp",
    scale_activation="tanh",
)

# RealNVP config (12 coupling layers for better expressiveness)
flow_config = RealNVPConfig(
    name="realnvp_mnist",
    coupling_network=coupling_config,
    input_dim=784,  # 28*28
    base_distribution="normal",
    num_coupling_layers=12,  # More layers for better expressiveness
    mask_type="checkerboard",
)

model = RealNVP(flow_config, rngs=rngs)
print("\n✅ RealNVP created:")
print(f"   Coupling layers: {flow_config.num_coupling_layers}")
print(f"   Hidden dims: {coupling_config.hidden_dims}")

# %% [markdown]
r"""
## Step 4: Create Optimizer with LR Schedule

Use learning rate warmup with cosine decay for stable training.
"""

# %%
# Calculate total training steps for learning rate schedule
total_steps = NUM_EPOCHS * n_batches
print(f"   Total training steps: {total_steps}")

# Learning rate schedule: warmup + cosine decay
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=BASE_LR,
    warmup_steps=WARMUP_STEPS,
    decay_steps=total_steps,
    end_value=BASE_LR * 0.01,
)

# Optimizer with gradient clipping and LR schedule
optimizer = nnx.Optimizer(
    model,
    optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule)),
    wrt=nnx.Param,
)
print(f"   Optimizer: Adam with warmup ({WARMUP_STEPS} steps) + cosine decay")

# %% [markdown]
r"""
## Step 5: Training Step

Define the training step for maximum likelihood training.
"""


# %%
def train_step(model, optimizer, batch):
    """Training step for RealNVP (maximum likelihood)."""

    def loss_fn(model):
        outputs = model(batch, training=True)
        log_prob = outputs["log_prob"]
        loss = -jnp.mean(log_prob)  # Negative log-likelihood
        return loss, {"nll": loss, "log_prob": jnp.mean(log_prob)}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return metrics


# JIT-compile training step for performance
jit_train_step = nnx.jit(train_step)
print("   Training step JIT-compiled")


# %% [markdown]
r"""
## Step 6: Training Loop

Train the model for multiple epochs using DataRax pipeline.
"""

# %%
history = {"step": [], "loss": [], "log_prob": [], "epoch": [], "lr": []}
train_key = jax.random.key(999)
global_step = 0

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print("-" * 60)

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    epoch_log_probs = []

    pbar = tqdm(train_pipeline, total=n_batches, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch in pbar:
        train_key, dequant_key = jax.random.split(train_key)

        # Preprocess batch with dequantization (fresh noise each batch)
        batch_processed = preprocess_batch(batch, dequant_key)

        # JIT-compiled training step
        metrics = jit_train_step(model, optimizer, batch_processed)

        # Get current learning rate
        current_lr = float(lr_schedule(global_step))

        epoch_losses.append(float(metrics["nll"]))
        epoch_log_probs.append(float(metrics["log_prob"]))
        history["step"].append(global_step)
        history["loss"].append(float(metrics["nll"]))
        history["log_prob"].append(float(metrics["log_prob"]))
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)

        global_step += 1

        pbar.set_postfix({"NLL": f"{metrics['nll']:.2f}", "lr": f"{current_lr:.2e}"})

    avg_loss = np.mean(epoch_losses)
    avg_log_prob = np.mean(epoch_log_probs)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: NLL={avg_loss:.2f}, Log-prob={avg_log_prob:.2f}")

print("-" * 60)
print("Training complete!")

# %% [markdown]
r"""
## Step 7: Generate Samples

Generate new digits from the trained model.
"""

# %%
print("\nGenerating samples...")
n_samples = 16

generated_samples = model.generate(n_samples=n_samples)

# Denormalize from [-1, 1] to [0, 1]
generated_samples = (generated_samples * 0.5) + 0.5
generated_samples = jnp.clip(generated_samples, 0, 1)

# Reshape to images
generated_images = generated_samples.reshape(n_samples, 28, 28)

print(f"✅ Generated {n_samples} samples")

# %% [markdown]
r"""
## Step 8: Visualize Results
"""

# %%
os.makedirs("examples_output", exist_ok=True)

# Samples
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    if i < len(generated_images):
        ax.imshow(np.array(generated_images[i]), cmap="gray")
    ax.axis("off")

plt.suptitle("RealNVP Generated MNIST Digits", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig("examples_output/flow_samples.png", dpi=150, bbox_inches="tight")
print("\nSaved: examples_output/flow_samples.png")
plt.close()


# Training curves with smoothing and LR plot
def smooth(values, window=100):
    """Apply moving average smoothing to a sequence of values."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

steps = np.array(history["step"])
loss = np.array(history["loss"])
log_prob = np.array(history["log_prob"])
lr = np.array(history["lr"])

window = 100

# NLL Loss
axes[0].plot(steps, loss, alpha=0.15, color="tab:blue", linewidth=0.5)
if len(loss) > window:
    smoothed_loss = smooth(loss, window)
    axes[0].plot(
        steps[window - 1 :],
        smoothed_loss,
        linewidth=2,
        label="NLL (smoothed)",
        color="tab:blue",
    )
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Negative Log-Likelihood")
axes[0].set_title(f"Training Loss ({NUM_EPOCHS} epochs)", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Log Probability
axes[1].plot(steps, log_prob, alpha=0.15, color="tab:green", linewidth=0.5)
if len(log_prob) > window:
    smoothed_lp = smooth(log_prob, window)
    axes[1].plot(
        steps[window - 1 :],
        smoothed_lp,
        linewidth=2,
        label="Log-prob (smoothed)",
        color="tab:green",
    )
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Log Probability")
axes[1].set_title("Average Log Probability", fontsize=14, fontweight="bold")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate
axes[2].plot(steps, lr, color="tab:orange", linewidth=1.5)
axes[2].set_xlabel("Step")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("examples_output/flow_training_curve.png", dpi=150, bbox_inches="tight")
print("Saved: examples_output/flow_training_curve.png")
plt.close()

print("\n✅ Done!")
