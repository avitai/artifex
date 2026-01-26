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
# Training a GAN on 2D Data with Artifex's GANTrainer

## Overview

This tutorial demonstrates how to train a GAN using Artifex's high-level training
API. Instead of implementing training loops from scratch, we use `GANTrainer`
which provides state-of-the-art techniques like WGAN-GP (Wasserstein GAN with
Gradient Penalty) for stable training.

**Key Artifex Components Used:**
- `Generator`, `Discriminator` - Configurable neural networks
- `GANTrainer` - Training framework with WGAN-GP support
- `GANTrainingConfig` - Configuration for loss type, gradient penalty, etc.

## Training Best Practices Applied

Based on the official WGAN-GP implementation:
- WGAN-GP loss for training stability (no mode collapse)
- Gradient penalty λ=0.1 for toy data (faster convergence)
- 5 critic (discriminator) iterations per generator iteration
- Adam optimizer with beta1=0.5, beta2=0.9

## Expected Results

- **Training time:** ~2-3 minutes (GPU/CPU)
- **Final Wasserstein distance:** Near 0 (distributions match)
- **Generated samples:** Points forming a circle matching real data

## Prerequisites

```bash
# Install Artifex
uv sync
```

---
"""

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from tqdm import tqdm

# Artifex imports
from artifex.generative_models.core.configuration.network_configs import (
    DiscriminatorConfig,
    GeneratorConfig,
)
from artifex.generative_models.models.gan import Discriminator, Generator
from artifex.generative_models.training.trainers.gan_trainer import (
    GANTrainer,
    GANTrainingConfig,
)


print("=" * 70)
print("Artifex GAN Training - 2D Circular Data")
print("Using: GANTrainer, Generator, Discriminator, nnx.jit")
print("=" * 70)

# %% [markdown]
r"""
## Step 1: Configuration

Training configuration based on the official WGAN-GP implementation.
"""

# %%
# Configuration (based on official WGAN-GP toy implementation)
SEED = 42
NUM_STEPS = 5000  # Training iterations
BATCH_SIZE = 256
LATENT_DIM = 2  # Match output dim for simpler mapping
N_CRITIC = 5  # Critic iterations per generator step
LR = 1e-4  # Adam learning rate
GP_WEIGHT = 0.1  # Gradient penalty weight (0.1 for toy data)
HIDDEN_DIM = 128  # Hidden layer dimension

print("\nConfiguration:")
print(f"  Steps: {NUM_STEPS}, Batch: {BATCH_SIZE}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Hidden dim: {HIDDEN_DIM}")
print(f"  N_critic: {N_CRITIC}")
print(f"  Learning rate: {LR}")
print(f"  Gradient penalty: {GP_WEIGHT}")

# %% [markdown]
r"""
## Step 2: Data Generation

We use a simple circular distribution for visualization. The goal is for the
generator to learn to produce points that form a circle.
"""


# %%
def generate_circle_data(key, batch_size):
    """Generate 2D points on a unit circle with noise."""
    theta_key, noise_key = jax.random.split(key)
    theta = jax.random.uniform(theta_key, (batch_size,)) * 2 * jnp.pi
    r = 1.0 + jax.random.normal(noise_key, (batch_size,)) * 0.05
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.stack([x, y], axis=-1)


# Test data generation
test_data = generate_circle_data(jax.random.key(0), 500)
print(f"\n📊 Test data shape: {test_data.shape}")
print(f"   Range: x=[{test_data[:, 0].min():.2f}, {test_data[:, 0].max():.2f}]")

# %% [markdown]
r"""
## Step 3: Create Models Using Artifex's API

Use Artifex's `Generator` and `Discriminator` classes with configuration objects.
"""

# %%
# Initialize RNGs
key = jax.random.key(SEED)
gen_key, disc_key, train_key = jax.random.split(key, 3)

# Generator configuration
gen_config = GeneratorConfig(
    name="circle_generator",
    hidden_dims=(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM),  # 3-layer MLP
    output_shape=(1, 2),  # 2D output
    latent_dim=LATENT_DIM,
    activation="relu",
    batch_norm=False,  # No batch norm for WGAN-GP
    dropout_rate=0.0,
)

# Discriminator (critic) configuration
disc_config = DiscriminatorConfig(
    name="circle_discriminator",
    input_shape=(1, 2),  # 2D input
    hidden_dims=(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM),
    activation="relu",
    batch_norm=False,
    dropout_rate=0.0,
)

# Create models
gen_rngs = nnx.Rngs(params=gen_key)
disc_rngs = nnx.Rngs(params=disc_key)

generator = Generator(config=gen_config, rngs=gen_rngs)
discriminator = Discriminator(config=disc_config, rngs=disc_rngs)

print("\n✅ Artifex models created:")
print(f"   Generator: {gen_config.hidden_dims}, latent_dim={LATENT_DIM}")
print(f"   Discriminator: {disc_config.hidden_dims}")

# %% [markdown]
r"""
## Step 4: Create Optimizers and GANTrainer

Use Artifex's `GANTrainer` with WGAN-GP configuration for stable training.
"""

# %%
# Optimizers (Adam with beta1=0.5, beta2=0.9 as per official WGAN-GP)
gen_optimizer = nnx.Optimizer(
    generator,
    optax.adam(LR, b1=0.5, b2=0.9),
    wrt=nnx.Param,
)
disc_optimizer = nnx.Optimizer(
    discriminator,
    optax.adam(LR, b1=0.5, b2=0.9),
    wrt=nnx.Param,
)

# GANTrainer configuration with WGAN-GP
gan_config = GANTrainingConfig(
    loss_type="wasserstein",  # WGAN loss
    n_critic=N_CRITIC,
    gp_weight=GP_WEIGHT,  # Gradient penalty
    gp_target=1.0,
    r1_weight=0.0,
    label_smoothing=0.0,
)

trainer = GANTrainer(config=gan_config)

# JIT-compile training steps for performance
jit_d_step = nnx.jit(trainer.discriminator_step)
jit_g_step = nnx.jit(trainer.generator_step)

print("\n✅ GANTrainer initialized:")
print(f"   Loss type: {gan_config.loss_type}")
print(f"   N_critic: {gan_config.n_critic}")
print(f"   GP weight: {gan_config.gp_weight}")
print("   Training steps JIT-compiled")

# %% [markdown]
r"""
## Step 5: Training Loop

The training loop uses Artifex's `GANTrainer` methods:
- `trainer.discriminator_step()` - Updates critic with gradient penalty
- `trainer.generator_step()` - Updates generator
"""

# %%
history = {"step": [], "d_loss": [], "g_loss": [], "w_dist": []}

print(f"\nTraining for {NUM_STEPS} steps...")
print("-" * 60)

pbar = tqdm(range(NUM_STEPS), desc="Training")
for step in pbar:
    train_key, *step_keys = jax.random.split(train_key, 2 + N_CRITIC * 2)

    # Train Discriminator (N_CRITIC steps)
    for i in range(N_CRITIC):
        d_data_key, d_z_key, d_gp_key = jax.random.split(step_keys[i], 3)

        real_data = generate_circle_data(d_data_key, BATCH_SIZE)
        z = jax.random.normal(d_z_key, (BATCH_SIZE, LATENT_DIM))

        # Use Artifex's JIT-compiled discriminator step
        d_loss, d_metrics = jit_d_step(
            generator, discriminator, disc_optimizer, real_data, z, d_gp_key
        )

    # Train Generator (1 step)
    z_gen_key = step_keys[-1]
    z = jax.random.normal(z_gen_key, (BATCH_SIZE, LATENT_DIM))

    # Use Artifex's JIT-compiled generator step
    g_loss, g_metrics = jit_g_step(generator, discriminator, gen_optimizer, z)

    # Record history
    w_dist = d_metrics.get("d_real", 0.0) - d_metrics.get("d_fake", 0.0)
    history["step"].append(step)
    history["d_loss"].append(float(d_loss))
    history["g_loss"].append(float(g_loss))
    history["w_dist"].append(float(w_dist))

    # Update progress bar
    pbar.set_postfix({"D": f"{d_loss:.3f}", "G": f"{g_loss:.3f}", "W": f"{w_dist:.3f}"})

print("-" * 60)
print("Training complete!")

# %% [markdown]
r"""
## Step 6: Generate Samples and Evaluate
"""

# %%
print("\nGenerating samples...")
n_samples = 1000

final_real = generate_circle_data(jax.random.key(5000), n_samples)
z_final = jax.random.normal(jax.random.key(6000), (n_samples, LATENT_DIM))
final_fake = generator(z_final)

# Statistics
real_radius = jnp.sqrt(jnp.sum(final_real**2, axis=1))
fake_radius = jnp.sqrt(jnp.sum(final_fake**2, axis=1))

print(f"\n📊 Evaluation ({n_samples} samples):")
print(f"   Real: mean_radius = {jnp.mean(real_radius):.4f} ± {jnp.std(real_radius):.4f}")
print(f"   Fake: mean_radius = {jnp.mean(fake_radius):.4f} ± {jnp.std(fake_radius):.4f}")

# %% [markdown]
r"""
## Step 7: Visualizations
"""

# %%
import os


os.makedirs("examples_output", exist_ok=True)

# Results visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(final_real[:, 0], final_real[:, 1], alpha=0.5, s=8, c="#2196F3")
axes[0].set_title("Real Data Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlim(-2, 2)
axes[0].set_ylim(-2, 2)
axes[0].set_aspect("equal")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(final_fake[:, 0], final_fake[:, 1], alpha=0.5, s=8, c="#FF9800")
axes[1].set_title("Generated Data Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlim(-2, 2)
axes[1].set_ylim(-2, 2)
axes[1].set_aspect("equal")
axes[1].grid(True, alpha=0.3)

axes[2].scatter(final_real[:, 0], final_real[:, 1], alpha=0.4, s=8, c="#2196F3", label="Real")
axes[2].scatter(final_fake[:, 0], final_fake[:, 1], alpha=0.4, s=8, c="#FF9800", label="Generated")
axes[2].set_title("Overlay Comparison", fontsize=14, fontweight="bold")
axes[2].set_xlim(-2, 2)
axes[2].set_ylim(-2, 2)
axes[2].set_aspect("equal")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("examples_output/simple_gan_results.png", dpi=150, bbox_inches="tight")
print("\nSaved: examples_output/simple_gan_results.png")
plt.close()

# Training curves with smoothing
import numpy as np


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


# Smoothing function
def smooth(values, window=100):
    """Apply moving average smoothing to a sequence of values."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


steps = np.array(history["step"])
d_loss = np.array(history["d_loss"])
g_loss = np.array(history["g_loss"])
w_dist = np.array(history["w_dist"])

# Plot raw data with low alpha
ax1.plot(steps, d_loss, alpha=0.15, color="tab:blue", linewidth=0.5)
ax1.plot(steps, g_loss, alpha=0.15, color="tab:orange", linewidth=0.5)

# Plot smoothed data
window = 100
if len(d_loss) > window:
    smoothed_d = smooth(d_loss, window)
    smoothed_g = smooth(g_loss, window)
    ax1.plot(steps[window - 1 :], smoothed_d, label="Critic Loss (smoothed)", linewidth=2)
    ax1.plot(steps[window - 1 :], smoothed_g, label="Generator Loss (smoothed)", linewidth=2)

ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.set_title("WGAN-GP Training Losses", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Wasserstein distance with smoothing
ax2.plot(steps, w_dist, alpha=0.15, color="green", linewidth=0.5)
if len(w_dist) > window:
    smoothed_w = smooth(w_dist, window)
    ax2.plot(steps[window - 1 :], smoothed_w, color="green", linewidth=2, label="W-dist")
ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Target (0)")
ax2.set_xlabel("Step")
ax2.set_ylabel("Wasserstein Distance Estimate")
ax2.set_title("Wasserstein Distance During Training", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("examples_output/simple_gan_training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: examples_output/simple_gan_training_curves.png")
plt.close()

print("\n✅ Done!")
