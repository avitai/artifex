#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %%
# IMPORTANT: Set memory env vars BEFORE importing TensorFlow or JAX
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Don't pre-allocate GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # JAX: don't pre-allocate
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # JAX: use 80% of GPU memory

# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %%
r"""Training a Diffusion Model on MNIST - Complete End-to-End Tutorial

## Overview

This tutorial provides a complete, production-ready example of training a DDPM
(Denoising Diffusion Probabilistic Model) on the MNIST dataset. You'll learn how
to train from scratch, evaluate quality, and generate realistic handwritten digits.

**Key Artifex Components Used:**
- `DDPMModel` - Artifex's DDPM implementation
- `DiffusionTrainer` - Training utilities with SOTA techniques
- Cosine noise schedule, Huber loss, warmup + cosine LR decay

## Training Best Practices Applied

Based on research (HuggingFace Annotated Diffusion, labml.ai DDPM):
- 1000 timesteps with **cosine noise schedule** (smoother than linear)
- Image padding to 32x32 for optimal UNet downsampling
- **Learning rate warmup** (1000 steps) + **cosine decay**
- **Huber loss** for stable training (more robust than MSE)
- Uniform timestep sampling for training stability

## Expected Results

- **Training time:** ~30 minutes (50 epochs, GPU)
- **Final loss:** ~0.027 (benchmark: 0.021 for quality digits)
- **Sample quality:** Clear, readable handwritten digits

## Prerequisites

```bash
# Install Artifex with CUDA support (recommended)
uv sync --extra cuda-dev

# Or CPU-only
uv sync
```

---
"""

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# DataRax imports for data loading
from datarax import from_source
from datarax.sources import TfdsDataSourceConfig, TFDSSource
from flax import nnx
from tqdm import tqdm

from artifex.generative_models.core.configuration.backbone_config import UNetBackboneConfig
from artifex.generative_models.core.configuration.diffusion_config import (
    DDPMConfig,
    NoiseScheduleConfig,
)
from artifex.generative_models.core.noise_schedule import create_noise_schedule
from artifex.generative_models.models.diffusion.ddpm import DDPMModel
from artifex.generative_models.training.trainers.diffusion_trainer import (
    DiffusionTrainer,
    DiffusionTrainingConfig,
)


print("=" * 70)
print("Artifex Diffusion Training - MNIST")
print("Using: DiffusionTrainer, DDPMModel, nnx.jit")
print("=" * 70)

# %% [markdown]
r"""
## 1. Configuration

Training configuration based on research best practices for diffusion models.
"""

# %%
# Configuration (tuned based on research best practices)
SEED = 42
NUM_EPOCHS = 50  # Research shows 40-100 epochs needed for quality results
BATCH_SIZE = 256  # Balance between memory and gradient stability
NUM_TIMESTEPS = 1000
IMAGE_SIZE = 32  # Pad MNIST to 32x32 (original DDPM used 32x32 images)

# Learning rate with warmup schedule
BASE_LR = 1e-4  # Conservative LR (labml.ai uses 2e-5 at batch 64)
WARMUP_STEPS = 500  # ~2 epochs of warmup (234 batches/epoch)

print("\nConfiguration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE} (MNIST padded)")
print(f"  Timesteps: {NUM_TIMESTEPS}")
print(f"  Learning rate: {BASE_LR} (with {WARMUP_STEPS} warmup steps)")

# %% [markdown]
r"""
## 2. Data Loading and Preprocessing

Use DataRax to load MNIST and create a batched data pipeline.
Pad to 32x32 for optimal UNet downsampling (32 -> 16 -> 8 -> 4).
"""


# %%
# Initialize RNG for data loading
data_rngs = nnx.Rngs(SEED)

# Configure MNIST data source using DataRax
train_source_config = TfdsDataSourceConfig(
    name="mnist",
    split="train",
    shuffle=True,  # Shuffle training data
    shuffle_buffer_size=10000,  # Buffer size for shuffling
)
train_source = TFDSSource(train_source_config, rngs=data_rngs)

print(f"\n📊 MNIST train dataset loaded: {len(train_source)} samples")

# Create training pipeline with batching and JIT compilation
train_pipeline = from_source(train_source, batch_size=BATCH_SIZE, jit_compile=True)

# Calculate number of batches per epoch
n_batches = len(train_source) // BATCH_SIZE
print(f"  ✅ Training pipeline created: {n_batches} batches per epoch")


# %%
def preprocess_batch(batch):
    """Preprocess MNIST batch for diffusion training.

    Args:
        batch: Dictionary with 'image' key (uint8, shape: [B, 28, 28, 1])

    Returns:
        Dictionary with normalized and padded images (float32, shape: [B, 32, 32, 1])
    """
    image = batch["image"]

    # Convert to float and normalize to [-1, 1]
    image = image.astype(jnp.float32)
    image = (image / 127.5) - 1.0

    # Pad 28x28 to 32x32 (2 pixels on each side)
    # This enables clean UNet downsampling: 32 -> 16 -> 8 -> 4
    image = jnp.pad(
        image,
        ((0, 0), (2, 2), (2, 2), (0, 0)),  # Batch, height, width, channels
        mode="constant",
        constant_values=-1.0,  # Background value after normalization
    )

    return {"image": image}


# Test the pipeline
print("  Testing pipeline...")
for raw_batch in train_pipeline:
    batch = preprocess_batch(raw_batch)
    print(f"  ✅ Batch shape: {batch['image'].shape}")
    print(
        f"  ✅ Value range: [{float(batch['image'].min()):.2f}, {float(batch['image'].max()):.2f}]"
    )
    break

# %% [markdown]
r"""
## 3. Model Creation

Configure the DDPM model with cosine noise schedule and Huber loss.
"""

# %%
# Initialize RNGs
key = jax.random.key(SEED)
params_key, noise_key, sample_key, dropout_key, timestep_key = jax.random.split(key, 5)
rngs = nnx.Rngs(
    params=params_key,
    noise=noise_key,
    sample=sample_key,
    dropout=dropout_key,
    timestep=timestep_key,
)

# UNet backbone (memory-efficient version)
backbone_config = UNetBackboneConfig(
    name="unet_backbone",
    hidden_dims=(64, 128, 256),  # 3 levels for 32x32 images
    activation="gelu",
    in_channels=1,
    out_channels=1,
    time_embedding_dim=128,
    attention_resolutions=(8,),  # Attention at 8x8 resolution only
    num_res_blocks=2,
    channel_mult=(1, 2, 4),
    dropout_rate=0.0,
)

# Noise schedule (cosine schedule for smoother training)
noise_schedule_config = NoiseScheduleConfig(
    name="cosine_schedule",
    schedule_type="cosine",  # Cosine: smoother noise progression, better gradients
    num_timesteps=NUM_TIMESTEPS,
    beta_start=1e-4,
    beta_end=0.02,
)

# DDPM config with Huber loss
ddpm_config = DDPMConfig(
    name="ddpm_mnist",
    backbone=backbone_config,
    noise_schedule=noise_schedule_config,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),  # 32x32 padded MNIST
    loss_type="huber",  # Huber loss for stable training (recommended)
    clip_denoised=True,
)

# Create model
model = DDPMModel(ddpm_config, rngs=rngs)

print("\n✅ DDPMModel created:")
print(f"   UNet: hidden_dims={backbone_config.hidden_dims}")
print(f"   Channel mults: {backbone_config.channel_mult}")
print(f"   Noise schedule: {noise_schedule_config.schedule_type}")
print(f"   Loss type: {ddpm_config.loss_type}")
print(f"   Timesteps: {NUM_TIMESTEPS}")
print(f"   JAX backend: {jax.default_backend()}")
print(f"   Devices: {jax.devices()}")

# %% [markdown]
r"""
## 4. Training Setup

Configure optimizer with warmup + cosine decay learning rate schedule.
"""

# %%
# Create noise schedule for trainer
noise_schedule = create_noise_schedule(noise_schedule_config)

# Calculate total training steps for learning rate schedule
total_steps = NUM_EPOCHS * n_batches
print(f"   Total training steps: {total_steps}")

# Learning rate schedule: warmup + cosine decay
# This prevents early training instability and allows gradual learning
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=BASE_LR,
    warmup_steps=WARMUP_STEPS,
    decay_steps=total_steps,
    end_value=BASE_LR * 0.01,  # End at 1% of peak
)

# Optimizer with warmup schedule and gradient clipping
optimizer = nnx.Optimizer(
    model,
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=1e-5),  # AdamW with weight decay
    ),
    wrt=nnx.Param,
)
print(f"   Optimizer: AdamW with warmup ({WARMUP_STEPS} steps) + cosine decay")

# %% [markdown]
r"""
## 5. Initialize Trainer

Use Artifex's DiffusionTrainer with uniform timestep sampling for stability.
"""

# %%
diffusion_config = DiffusionTrainingConfig(
    prediction_type="epsilon",  # Classic DDPM epsilon prediction
    timestep_sampling="uniform",  # Uniform sampling (more stable for training)
    loss_weighting="uniform",  # Start with uniform weighting for stability
    ema_decay=0.9999,  # EMA as per original DDPM paper
    ema_update_every=10,
)

trainer = DiffusionTrainer(noise_schedule, diffusion_config)

# JIT-compile training step
jit_train_step = nnx.jit(trainer.train_step)

print("\n✅ DiffusionTrainer initialized:")
print(f"   Prediction: {diffusion_config.prediction_type}")
print(f"   Timestep sampling: {diffusion_config.timestep_sampling}")
print(f"   Loss weighting: {diffusion_config.loss_weighting}")
print("   Training step JIT-compiled")

# %% [markdown]
r"""
## 6. Training Loop

Train the model with progress tracking and learning rate monitoring.
"""

# %%
history = {"step": [], "loss": [], "epoch": [], "lr": []}
train_key = jax.random.key(999)
global_step = 0

print(f"\nTraining for {NUM_EPOCHS} epochs ({total_steps} steps)...")
print("-" * 60)

for epoch in range(NUM_EPOCHS):
    epoch_losses = []

    # DataRax pipeline handles batching and shuffling
    pbar = tqdm(train_pipeline, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", total=n_batches)
    for raw_batch in pbar:
        train_key, step_key = jax.random.split(train_key)

        # Preprocess batch (normalize and pad)
        batch = preprocess_batch(raw_batch)

        # Training step (JIT-compiled)
        loss, metrics = jit_train_step(model, optimizer, batch, step_key)

        # Update EMA (outside JIT)
        if global_step % diffusion_config.ema_update_every == 0:
            trainer.update_ema(model)

        # Get current learning rate from schedule
        current_lr = float(lr_schedule(global_step))

        epoch_losses.append(float(loss))
        history["step"].append(global_step)
        history["loss"].append(float(loss))
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)

        global_step += 1

        # Show loss and LR in progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{current_lr:.2e}"})

    avg_loss = np.mean(epoch_losses)
    current_lr = float(lr_schedule(global_step - 1))
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: avg_loss = {avg_loss:.4f}, lr = {current_lr:.2e}")

print("-" * 60)
print("Training complete!")

# %% [markdown]
r"""
## 7. Generate Samples

Use DDIM for fast sampling with good quality.
"""

# %%
print("\nGenerating samples...")
n_samples = 16

# Use DDIM for faster sampling with more steps for better quality
samples = model.sample(
    n_samples_or_shape=n_samples,
    scheduler="ddim",
    steps=100,  # More steps for better quality
)

print(f"✅ Generated {n_samples} samples")

# %% [markdown]
r"""
## 8. Visualization

Save training curves and generated samples.
"""

# %%
os.makedirs("examples_output", exist_ok=True)


def visualize_samples(images, title="Samples", n_cols=4, save_path=None):
    """Visualize a grid of images."""
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, (ax, img) in enumerate(zip(axes, images)):
        img = (np.array(img) + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        ax.imshow(img.squeeze(), cmap="gray")
        ax.axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.close()
    return fig


# Save samples
visualize_samples(
    samples,
    title="DDPM Generated MNIST Digits",
    save_path="examples_output/diffusion_samples.png",
)

# %%
# Training curve with dual axis (loss + learning rate)
fig, ax1 = plt.subplots(figsize=(12, 5))

# Loss curve (left axis)
color_loss = "tab:blue"
ax1.plot(history["step"], history["loss"], alpha=0.3, linewidth=0.5, color=color_loss)
# Smooth curve
if len(history["loss"]) > 100:
    window = 100
    smoothed = np.convolve(history["loss"], np.ones(window) / window, mode="valid")
    ax1.plot(
        history["step"][window - 1 :],
        smoothed,
        linewidth=2,
        label="Loss (smoothed)",
        color=color_loss,
    )
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss", color=color_loss)
ax1.tick_params(axis="y", labelcolor=color_loss)
ax1.grid(True, alpha=0.3)

# Learning rate curve (right axis)
ax2 = ax1.twinx()
color_lr = "tab:orange"
ax2.plot(history["step"], history["lr"], linewidth=1.5, color=color_lr, label="Learning Rate")
ax2.set_ylabel("Learning Rate", color=color_lr)
ax2.tick_params(axis="y", labelcolor=color_lr)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax1.set_title(
    f"Diffusion Training ({NUM_EPOCHS} epochs, warmup + cosine decay)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
fig.savefig("examples_output/diffusion_training_curve.png", dpi=150, bbox_inches="tight")
print("Saved: examples_output/diffusion_training_curve.png")
plt.close()

# %% [markdown]
r"""
## 9. Summary

### Training Results

- **Final loss:** ~0.027 (close to benchmark 0.021)
- **Sample quality:** Clear, readable digits
- **Training time:** ~30 minutes on GPU

### Key Techniques Used

1. **Cosine noise schedule** - Smoother than linear, better gradients
2. **Huber loss** - More robust than MSE
3. **LR warmup + cosine decay** - Prevents early instability
4. **32x32 padding** - Optimal UNet downsampling
5. **Uniform timestep sampling** - Training stability

### Next Steps

- Try conditional generation (class labels)
- Experiment with v-prediction instead of epsilon
- Apply to other datasets (Fashion-MNIST, CIFAR-10)
"""

# %%
print("\n" + "=" * 70)
print("Training Summary")
print("=" * 70)
print(f"Final loss: {history['loss'][-1]:.4f}")
print(f"Total steps: {global_step}")
print("Samples saved: examples_output/diffusion_samples.png")
print("Training curve: examples_output/diffusion_training_curve.png")
print("=" * 70)
print("\nDone!")
