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
"""
# Quickstart: Train Your First VAE

This quickstart demonstrates how to train a Variational Autoencoder (VAE) on MNIST
using Artifex's high-performance training infrastructure.

**What you'll learn:**
- Load data with `TFDSEagerSource` (pure JAX, no TensorFlow during training)
- Configure a CNN-based VAE with `VAEConfig`
- Train using JIT-compiled training loops for maximum performance
- Generate and visualize samples

**Expected runtime:** ~30 seconds on GPU, ~2 minutes on CPU
"""

# %%
# Cell 1: Imports
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from datarax.sources import TFDSEagerSource
from datarax.sources.tfds_source import TFDSEagerConfig
from flax import nnx

from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    VAEConfig,
)
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.training import train_epoch_staged
from artifex.generative_models.training.trainers import VAETrainer, VAETrainingConfig


# %% [markdown]
"""
## Step 1: Load Data with TFDSEagerSource

`TFDSEagerSource` loads the entire dataset into JAX arrays at initialization.
This eliminates TensorFlow overhead during training - pure JAX from start to finish.
"""

# %%
# Load MNIST with TFDSEagerSource
print("Loading MNIST...")
tfds_config = TFDSEagerConfig(name="mnist", split="train", shuffle=True, seed=42)
mnist_source = TFDSEagerSource(tfds_config, rngs=nnx.Rngs(0))

# Get images as JAX array and normalize to [0, 1]
images = mnist_source.data["image"].astype(jnp.float32) / 255.0
num_samples = len(mnist_source)
print(f"Loaded {num_samples} images, shape: {images.shape}")

# %% [markdown]
"""
## Step 2: Configure the VAE Model

We use a CNN architecture for better image quality:
- **Encoder**: 3-layer CNN (32 -> 64 -> 128 channels) mapping images to 20-dim latent space
- **Decoder**: Symmetric CNN reconstructing images from latent codes
- **KL weight**: 1.0 (standard VAE)
"""

# %%
# Configure encoder
encoder = EncoderConfig(
    name="mnist_cnn_encoder",
    input_shape=(28, 28, 1),
    latent_dim=20,
    hidden_dims=(32, 64, 128),
    activation="relu",
    use_batch_norm=False,
)

# Configure decoder (symmetric to encoder)
decoder = DecoderConfig(
    name="mnist_cnn_decoder",
    latent_dim=20,
    output_shape=(28, 28, 1),
    hidden_dims=(32, 64, 128),
    activation="relu",
    batch_norm=False,
)

# Combine into VAE config
model_config = VAEConfig(
    name="mnist_cnn_vae",
    encoder=encoder,
    decoder=decoder,
    encoder_type="cnn",
    kl_weight=1.0,
)

print("Model configured:")
print(f"  Latent dimension: {encoder.latent_dim}")
print(f"  Encoder type: CNN with dims {encoder.hidden_dims}")

# %% [markdown]
"""
## Step 3: Create Model, Optimizer, and Trainer

- **Model**: VAE with CNN encoder/decoder
- **Optimizer**: Adam with learning rate 2e-3
- **Trainer**: VAETrainer with linear KL annealing (gradual warmup of KL term)

KL annealing helps training stability by letting the model learn good reconstructions
first before the KL regularization kicks in.
"""

# %%
# Create model and optimizer
model = VAE(model_config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(2e-3), wrt=nnx.Param)

# Create trainer with KL annealing
trainer = VAETrainer(
    VAETrainingConfig(
        kl_annealing="linear",
        kl_warmup_steps=2000,  # ~4 epochs of warmup
        beta=1.0,
    )
)

# Count parameters
state_leaves = jax.tree.leaves(nnx.state(model))
param_count = sum(p.size for p in state_leaves if hasattr(p, "size"))
print(f"Model created with ~{param_count / 1e3:.1f}K parameters")

# %% [markdown]
"""
## Step 4: Train with JIT-Compiled Training Loop

We use `train_epoch_staged` which:
1. Pre-stages data on GPU with `jax.device_put()`
2. Uses JIT-compiled training steps
3. Achieves 100-500x speedup over naive Python loops

The first epoch includes JIT compilation overhead; subsequent epochs are much faster.
"""

# %%
# Stage data on GPU for maximum performance
print()
print("Staging data on GPU...")
staged_data = jax.device_put(images)

# Training configuration
NUM_EPOCHS = 20
BATCH_SIZE = 128

# Warmup JIT compilation (don't count this in training time)
print("Warming up JIT compilation...")
warmup_rng = jax.random.key(999)
# Create loss function - step is passed dynamically inside train_epoch_staged
loss_fn = trainer.create_loss_fn(loss_type="bce")
_ = train_epoch_staged(
    model,
    optimizer,
    staged_data[:256],
    batch_size=128,
    rng=warmup_rng,
    loss_fn=loss_fn,
)
print("JIT warmup complete.")
print()

# Training loop
print(f"Training for {NUM_EPOCHS} epochs...")
print("-" * 50)

# IMPORTANT: Reuse the same loss_fn across epochs for JIT cache hits
# The warmup already created and cached the epoch runner for this loss_fn
step = 0
for epoch in range(NUM_EPOCHS):
    rng = jax.random.key(epoch)

    # Train one epoch (reuses JIT-compiled function from warmup)
    step, metrics = train_epoch_staged(
        model,
        optimizer,
        staged_data,
        batch_size=BATCH_SIZE,
        rng=rng,
        loss_fn=loss_fn,  # Reuse same loss_fn for JIT caching
        base_step=step,
    )

    print(f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Loss: {metrics['loss']:7.2f}")

print("-" * 50)
print("Training complete!")

# %% [markdown]
"""
## Step 5: Generate and Reconstruct Images

Now let's test the trained model:
- **Generation**: Sample from the prior p(z) = N(0, I) and decode
- **Reconstruction**: Encode test images to latent space, then decode back
"""

# %%
# Generate new samples
print()
print("Generating samples...")
samples = model.sample(n_samples=16)
print(f"Generated {samples.shape[0]} samples")

# Reconstruct test images
print("Testing reconstruction...")
test_images = jnp.array(images[:8])
reconstructed = model.reconstruct(test_images, deterministic=True)
print(f"Reconstructed {reconstructed.shape[0]} images")

# %% [markdown]
"""
## Step 6: Visualize Results

Let's visualize the generated samples and reconstructions to verify
the model learned meaningful representations.
"""

# %%
# Plot generated samples (4x4 grid)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
fig.suptitle("Generated Samples from VAE", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("vae_samples.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved samples to vae_samples.png")

# Plot reconstructions (original vs reconstructed)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.text(0.02, 0.75, "Original", fontsize=12, fontweight="bold", va="center")
fig.text(0.02, 0.25, "Reconstructed", fontsize=12, fontweight="bold", va="center")

for i in range(8):
    axes[0, i].imshow(test_images[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].axis("off")

fig.suptitle("VAE Reconstruction Quality", fontsize=14, y=1.02)
plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig("vae_reconstruction.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved reconstruction to vae_reconstruction.png")

print()
print("Success! You've trained your first VAE with Artifex!")

# %% [markdown]
"""
## What You Just Did

1. **Loaded data efficiently** with `TFDSEagerSource` - pure JAX, no TF overhead
2. **Configured a CNN VAE** using Artifex's modular config system
3. **Used VAETrainer** with KL annealing for stable training
4. **Trained with JIT-compiled loops** for maximum performance
5. **Generated new samples** from the learned latent space
6. **Reconstructed images** to verify encoder-decoder quality

## Next Steps

- **Core Concepts**: Learn about Artifex's architecture and design principles
- **VAE Guide**: Advanced techniques like beta-VAE, conditional VAE, VQ-VAE
- **Other Models**: Try Diffusion models, GANs, Flow models
- **Custom Data**: Load your own datasets with datarax
"""
