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
# # Quickstart: Train Your First VAE.
#
# This quickstart trains a Variational Autoencoder (VAE) on MNIST end-to-end
# and saves four visual artifacts to the current working directory:
#
# - `vae_loss_curve.png`        — training loss per epoch
# - `vae_reconstruction.png`    — original vs reconstructed digits
# - `vae_samples.png`           — random draws from the prior, decoded
# - `vae_latent_interpolation.png` — smooth morph between two real digits in latent space
#
# **Recipe** — small MLP VAE that gives sharp MNIST samples on CPU:
#
# - Two hidden layers of 512 → 256 units, **32-dim** latent, ReLU
# - Sigmoid decoder output → BCE reconstruction (sum over the 784 pixels,
#   mean over the batch — matches Kingma & Welling 2014 and the canonical
#   PyTorch reference)
# - Adam(lr = 1e-3), batch size 128, full 60K MNIST training set
# - 60 epochs with a one-epoch linear KL warmup
#   (Bowman et al., 2015, https://arxiv.org/abs/1511.06349)
#
# References:
# - Kingma & Welling (2014). *Auto-Encoding Variational Bayes.*
#   https://arxiv.org/abs/1312.6114
# - PyTorch official VAE example.
#   https://github.com/pytorch/examples/tree/main/vae
#
# **Expected runtime:** ~30 seconds on GPU, ~2 minutes on CPU.

# %%
# Cell 1: Imports
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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


print(f"JAX backend: {jax.default_backend()}")


# %% [markdown]
# ## Step 1: Load Data with TFDSEagerSource.
#
# `TFDSEagerSource` loads the entire dataset into JAX arrays at initialization.
# This eliminates TensorFlow overhead during training - pure JAX from start to finish.

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
# ## Step 2: Configure the VAE Model.
#
# Two hidden layers of 512 → 256 and a 32-dim latent — small enough to
# train in ~75 s on a single CPU core, large enough to give sharp,
# diverse MNIST samples. ReLU activations and a sigmoid decoder output
# match the canonical PyTorch reference.
#
# - **Encoder**: 784 → 512 → 256 → 32-dim latent (mean + log-var)
# - **Decoder**: 32 → 256 → 512 → 784, sigmoid output
# - **β = 1.0** — full ELBO with a short KL warmup

# %%
LATENT_DIM = 32
HIDDEN_DIMS = (512, 256)
ACTIVATION = "relu"

# Configure encoder (MLP, dense type)
encoder = EncoderConfig(
    name="mnist_mlp_encoder",
    input_shape=(28, 28, 1),
    latent_dim=LATENT_DIM,
    hidden_dims=HIDDEN_DIMS,
    activation=ACTIVATION,
    use_batch_norm=False,
)

# Configure decoder (symmetric MLP — MLPDecoder reverses hidden_dims)
decoder = DecoderConfig(
    name="mnist_mlp_decoder",
    latent_dim=LATENT_DIM,
    output_shape=(28, 28, 1),
    hidden_dims=HIDDEN_DIMS,
    activation=ACTIVATION,
    batch_norm=False,
)

# Combine into VAE config
model_config = VAEConfig(
    name="mnist_mlp_vae",
    encoder=encoder,
    decoder=decoder,
    encoder_type="dense",  # dense → MLPEncoder / MLPDecoder
    kl_weight=1.0,
)

print("Model configured:")
print(f"  Latent dimension: {LATENT_DIM}")
print(f"  Hidden dims:      {HIDDEN_DIMS}")

# %% [markdown]
# ## Step 3: Create Model, Optimizer, and Trainer.
#
# - **Model**: MLP VAE
# - **Optimizer**: Adam with learning rate 1e-3
# - **Trainer**: VAETrainer with β = 1.0 and a one-epoch KL warmup
#
# Linear KL annealing (Bowman et al., 2015) ramps the KL weight from 0
# to 1 during the first epoch only. This avoids posterior collapse early
# while keeping the loss curve visually monotonic (the warmup completes
# inside epoch 1, so it isn't visible at the per-epoch resolution).

# %%
# Create model and optimizer
model = VAE(model_config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# 469 steps ≈ 1 epoch at 60K/128 batches per epoch — warmup completes
# inside epoch 1, so the per-epoch loss curve stays monotone.
trainer = VAETrainer(
    VAETrainingConfig(
        kl_annealing="linear",
        kl_warmup_steps=469,
        beta=1.0,
    )
)

# Count parameters
state_leaves = jax.tree.leaves(nnx.state(model))
param_count = sum(p.size for p in state_leaves if hasattr(p, "size"))
print(f"Model created with ~{param_count / 1e3:.1f}K parameters")

# %% [markdown]
# ## Step 4: Train with a JIT-Compiled Training Loop.
#
# `train_epoch_staged` wraps the **entire epoch** in `@nnx.jit` and runs a
# `jax.lax.fori_loop` over batches inside the compiled program. The factory
# is cached on `loss_fn` identity, so reusing the same `loss_fn` across
# epochs avoids recompilation.
#
# The first epoch includes JIT compilation overhead; subsequent epochs are much faster.

# %%
# Stage data on GPU for maximum performance
print()
print("Staging data on device...")
staged_data = jax.device_put(images)

# Training configuration. 60 epochs is the sweet spot — sharp digits and
# full latent diversity, completes in ~75-90 s on a single CPU core.
NUM_EPOCHS = 60
BATCH_SIZE = 128

# Warmup JIT compilation (don't count this in training time)
print("Warming up JIT compilation...")
warmup_rng = jax.random.key(999)
loss_fn = trainer.create_loss_fn(loss_type="bce")
_ = train_epoch_staged(
    model,
    optimizer,
    staged_data[: BATCH_SIZE * 2],
    batch_size=BATCH_SIZE,
    rng=warmup_rng,
    loss_fn=loss_fn,
)
print("JIT warmup complete.")
print()

# Training loop — IMPORTANT: reuse the same loss_fn across epochs for JIT cache hits
print(f"Training for {NUM_EPOCHS} epochs...")
print("-" * 50)
step = 0
epoch_losses: list[float] = []
start = time.time()
for epoch in range(NUM_EPOCHS):
    rng = jax.random.key(epoch)
    step, metrics = train_epoch_staged(
        model,
        optimizer,
        staged_data,
        batch_size=BATCH_SIZE,
        rng=rng,
        loss_fn=loss_fn,
        base_step=step,
    )
    epoch_losses.append(float(metrics["loss"]))
    elapsed = time.time() - start
    print(
        f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | Loss: {metrics['loss']:7.2f} | "
        f"Elapsed: {elapsed:6.1f}s"
    )

print("-" * 50)
print(f"Training complete in {time.time() - start:.1f}s")

# %% [markdown]
# ## Step 5: Generate, Reconstruct, and Traverse the Latent Space.
#
# - **Generation**: sample $z \sim \mathcal{N}(0, I)$ and decode.
# - **Reconstruction**: encode test images and decode the posterior mean.
# - **Latent interpolation**: encode two real digits, linearly interpolate
#   their latent codes, and decode each step — direct evidence that the
#   learned latent space is smooth and semantically meaningful.

# %%
# Generate new samples
print()
print("Generating samples...")
samples = model.sample(n_samples=16)
print(f"Generated {samples.shape[0]} samples")

# Reconstruct test images
print("Computing reconstructions...")
test_images = jnp.array(images[:8])
reconstructed = model.reconstruct(test_images, deterministic=True)
print(f"Reconstructed {reconstructed.shape[0]} images")

# Build a latent interpolation between two real digits.
# 1. Pick two test digits with visibly different shapes.
# 2. Encode each, take the posterior means.
# 3. Linearly interpolate from z_a to z_b in N_STEPS, then decode each point.
print("Building latent interpolation...")
N_STEPS = 10
img_a, img_b = jnp.array(images[0:1]), jnp.array(images[7:8])
mean_a, _ = model.encoder(img_a)
mean_b, _ = model.encoder(img_b)
alphas = jnp.linspace(0.0, 1.0, N_STEPS).reshape(-1, 1)
z_interp = (1.0 - alphas) * mean_a + alphas * mean_b
interp = model.decoder(z_interp)

# %% [markdown]
# ## Step 6: Save Visualizations to PNG Files.
#
# Four PNGs are written to the current working directory.

# %%
# 1. Loss curve
fig, ax = plt.subplots(figsize=(8, 5))
epochs_axis = list(range(1, len(epoch_losses) + 1))
ax.plot(epochs_axis, epoch_losses, marker="o", linewidth=2, markersize=5, color="#1f77b4")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training loss (BCE + KL)", fontsize=12)
ax.set_title("VAE Training Loss on MNIST", fontsize=14)
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
plt.savefig("vae_loss_curve.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved vae_loss_curve.png")

# 2. Generated samples (4x4 grid)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(np.asarray(samples[i].squeeze()), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
fig.suptitle("Generated Samples from VAE", fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig("vae_samples.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved vae_samples.png")

# 3. Reconstruction comparison (originals on top, reconstructions on bottom)
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.text(0.02, 0.75, "Original", fontsize=12, fontweight="bold", va="center")
fig.text(0.02, 0.25, "Reconstructed", fontsize=12, fontweight="bold", va="center")
for i in range(8):
    axes[0, i].imshow(np.asarray(test_images[i].squeeze()), cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[1, i].imshow(np.asarray(reconstructed[i].squeeze()), cmap="gray", vmin=0, vmax=1)
    axes[1, i].axis("off")
fig.suptitle("VAE Reconstruction Quality", fontsize=14, y=1.02)
plt.tight_layout()
plt.subplots_adjust(left=0.08)
plt.savefig("vae_reconstruction.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved vae_reconstruction.png")

# 4. Latent interpolation strip: smooth morph from digit A to digit B.
interp_arr = np.asarray(interp).reshape(N_STEPS, 28, 28)
fig, axes = plt.subplots(1, N_STEPS, figsize=(N_STEPS * 1.4, 2.0))
for i in range(N_STEPS):
    axes[i].imshow(interp_arr[i], cmap="gray", vmin=0, vmax=1)
    axes[i].axis("off")
fig.suptitle(
    "Latent interpolation: linearly interpolate two real digits' encoded means",
    fontsize=12,
    y=1.05,
)
plt.tight_layout()
plt.savefig(
    "vae_latent_interpolation.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)
print("Saved vae_latent_interpolation.png")

print()
print("Success! You've trained your first VAE with Artifex!")

# %% [markdown]
# ## What You Just Did.
#
# 1. **Loaded MNIST efficiently** with `TFDSEagerSource` - pure JAX, no TF overhead
# 2. **Configured an MLP VAE** using Artifex's modular `VAEConfig`/`EncoderConfig`/`DecoderConfig`
# 3. **Trained with `VAETrainer`** and linear KL annealing over the first ~10 epochs
# 4. **Compiled the entire epoch with `@nnx.jit`** via `train_epoch_staged`
# 5. **Generated samples**, reconstructions, and a 2D latent-space traversal
# 6. **Saved four PNGs** for inclusion in docs, slides, or notebooks
#
# ## Next Steps
#
# - **Core Concepts**: Learn about Artifex's architecture and design principles
# - **VAE Guide**: Advanced techniques like beta-VAE, conditional VAE, VQ-VAE
# - **Other Models**: Try Diffusion models, GANs, Flow models
# - **Custom Data**: Load your own datasets with datarax
