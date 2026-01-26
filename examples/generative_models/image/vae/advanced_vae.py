#!/usr/bin/env python
r"""Advanced VAE Examples - Showcase Artifex's Advanced VAE Features

## Overview

This example demonstrates Artifex's advanced VAE implementations on real MNIST data:
1. **β-VAE**: Disentangled representations with β weighting and annealing
2. **β-VAE with Capacity Control**: Burgess et al. capacity-based training
3. **Conditional VAE**: Label-conditioned generation for controlled sampling
4. **VQ-VAE**: Discrete latent codes with vector quantization

All models use Artifex's production-ready implementations with proper training
on real MNIST data.

## Source Code Dependencies

**Validated:** 2026-01-13

This example uses Artifex's VAE implementations with frozen dataclass configs:
- `artifex.generative_models.models.vae.BetaVAE` - β-VAE implementation
- `artifex.generative_models.models.vae.BetaVAEWithCapacity` - Capacity control
- `artifex.generative_models.models.vae.ConditionalVAE` - Conditional VAE
- `artifex.generative_models.models.vae.VQVAE` - Vector-quantized VAE
- `artifex.generative_models.core.configuration.vae_config` - BetaVAEConfig, etc.
- `artifex.generative_models.core.configuration.network_configs` - EncoderConfig, DecoderConfig

**Validation Status:**
- ✅ All dependencies use Flax NNX best practices
- ✅ Proper RNG handling throughout
- ✅ Production-ready implementations from Artifex

## What You'll Learn

- [x] Using Artifex's BetaVAE with different β values
- [x] Implementing β annealing for gradual disentanglement
- [x] Capacity control for stable β-VAE training
- [x] Conditional generation with ConditionalVAE
- [x] Monitoring VQ-VAE codebook usage and perplexity
- [x] Training advanced VAEs on real MNIST data
- [x] Evaluating and visualizing latent representations

## Prerequisites

- Artifex installed (run `source activate.sh`)
- Understanding of standard VAEs (see basic vae-mnist tutorial)
- Familiarity with JAX and Flax NNX
- Knowledge of variational inference concepts

## Usage

```bash
source activate.sh
python examples/generative_models/image/vae/advanced_vae.py
```

## Expected Output

The example will:
1. Load real MNIST dataset (60,000 training images)
2. Train four advanced VAE variants (β-VAE, Capacity β-VAE, Conditional VAE, VQ-VAE)
3. Generate reconstructions and samples from each model
4. Save visualizations to `examples_output/advanced_vae/`
5. Display training metrics and convergence curves

## Key Concepts

### β-VAE

β-VAE adds weight β > 1 to KL divergence term for disentanglement:
$$\mathcal{L}_{\beta} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \beta \cdot \text{KL}(q(z|x) \| p(z))$$

Higher β encourages independence between latent dimensions.

### β-VAE with Capacity Control

Capacity control (Burgess et al.) gradually increases KL capacity:
$$\mathcal{L}_{C} = \mathbb{E}_{q(z|x)}[\log p(x|z)] + \gamma \cdot |KL - C|$$

Where C increases from 0 to max_capacity during training.

### Conditional VAE

Conditional VAE adds label information to encoder and decoder, enabling
controlled generation of specific classes.

### VQ-VAE

VQ-VAE uses discrete latent codes from a learnable codebook, enabling
better compression and sharper reconstructions.

## Estimated Runtime

- **CPU**: ~20-30 minutes total (all 4 variants)
- **GPU**: ~5-8 minutes total (if available)

## Author

Artifex Team

## Last Updated

2026-01-13
"""

# %% [markdown]
"""
# Advanced VAE Examples

This notebook demonstrates Artifex's advanced VAE implementations on real MNIST data.

## Learning Objectives

By the end of this example, you will understand:
1. How to use Artifex's BetaVAE with different β values
2. Implementing β annealing and capacity control
3. Conditional generation with ConditionalVAE
4. Monitoring VQ-VAE codebook usage
5. Training and evaluating advanced VAE variants
"""

# %%
# Cell 1: Import Dependencies
"""
Import Artifex's VAE implementations and utilities for training advanced variants.
"""

from itertools import islice
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
    ConditionalVAEConfig,
    VQVAEConfig,
)
from artifex.generative_models.models.vae import BetaVAE, ConditionalVAE, VQVAE
from artifex.generative_models.models.vae.beta_vae import BetaVAEWithCapacity


# Create output directory
OUTPUT_DIR = Path("examples_output/advanced_vae")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"JAX devices: {jax.devices()}")
print(f"Output directory: {OUTPUT_DIR}")


# %% [markdown]
"""
## Data Loading

Load real MNIST dataset using TensorFlow Datasets. This ensures we train on
actual handwritten digits with proper training/test splits.

**Dataset Properties:**
- Training: 60,000 images
- Test: 10,000 images
- Image size: 28×28×1 (grayscale)
- Labels: 0-9 (digit classes)
- Values: [0, 1] (normalized)
"""


# %%
# Cell 2: Real MNIST Data Loading with Grain
def load_real_mnist(batch_size=128):
    """Load real MNIST dataset using Grain framework (JAX best practice).

    Uses Hugging Face datasets for initial loading, then Grain for efficient
    batching and iteration. This follows JAX/Grain best practices for small datasets.

    Args:
        batch_size: Batch size for training

    Returns:
        Tuple of (train_loader, test_loader) as Grain DataLoaders
    """
    import grain.python as grain
    from datasets import load_dataset

    print("\nLoading MNIST dataset...")

    # Load MNIST from Hugging Face (avoids TensorFlow dependency)
    ds = load_dataset("mnist")

    # Convert to lists of indices (Grain will fetch data via __getitem__)
    # We'll create a custom RandomAccessDataSource to handle complex data types
    class MNISTDataSource(grain.RandomAccessDataSource):
        """Custom data source for MNIST that handles numpy/JAX arrays."""

        def __init__(self, hf_dataset):
            self.dataset = hf_dataset
            # Pre-convert all images to JAX arrays for efficiency
            self.images = []
            self.labels = []
            for item in hf_dataset:
                # Convert PIL image to JAX array
                image = jnp.array(item["image"], dtype=jnp.float32)
                image = image / 255.0  # Normalize to [0, 1]
                image = image[..., jnp.newaxis]  # Add channel: (28, 28, 1)
                self.images.append(image)
                self.labels.append(item["label"])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return {"image": self.images[index], "label": self.labels[index]}

    train_source = MNISTDataSource(ds["train"])
    test_source = MNISTDataSource(ds["test"])

    print(f"✓ MNIST loaded: {len(train_source)} training images, {len(test_source)} test images")

    # Create samplers
    train_sampler = grain.IndexSampler(
        num_records=len(train_source),
        shuffle=True,
        seed=42,
        num_epochs=None,  # Infinite epochs (we'll manually control batches per epoch)
    )

    test_sampler = grain.IndexSampler(
        num_records=len(test_source),
        shuffle=False,
        seed=42,
        num_epochs=1,  # Single pass for evaluation
    )

    # Create DataLoaders with batching
    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=0,  # Single-process for simplicity
    )

    test_loader = grain.DataLoader(
        data_source=test_source,
        sampler=test_sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=False)],
        worker_count=0,
    )

    print(f"✓ Created Grain DataLoaders (batch_size={batch_size})")

    return train_loader, test_loader


# %% [markdown]
"""
## Example 1: β-VAE with β Annealing

Demonstrate Artifex's BetaVAE with gradual β annealing from 1.0 to 4.0.
This helps avoid posterior collapse while achieving disentanglement.
"""


# %%
# Cell 3: β-VAE Setup
def create_beta_vae(beta=4.0, warmup_steps=1000):
    """Create a β-VAE model using Artifex's config-based API.

    Args:
        beta: Weight for KL divergence (β > 1 for disentanglement)
        warmup_steps: Steps to anneal β from 0 to beta

    Returns:
        BetaVAE model
    """
    # Create encoder config
    encoder_config = EncoderConfig(
        name="beta_vae_encoder",
        hidden_dims=(512, 256),  # Tuple for frozen dataclass
        latent_dim=10,
        activation="relu",
        input_shape=(28, 28, 1),
    )

    # Create decoder config
    decoder_config = DecoderConfig(
        name="beta_vae_decoder",
        hidden_dims=(256, 512),  # Tuple for frozen dataclass
        latent_dim=10,
        output_shape=(28, 28, 1),
        activation="relu",
    )

    # Create β-VAE config with annealing
    config = BetaVAEConfig(
        name="beta_vae_mnist",
        encoder=encoder_config,
        decoder=decoder_config,
        encoder_type="dense",
        beta_default=beta,
        beta_warmup_steps=warmup_steps,
        reconstruction_loss_type="mse",
    )

    # Create model from config
    model = BetaVAE(config=config, rngs=nnx.Rngs(0))

    return model


# %%
# Cell 4: Common Training Infrastructure
@nnx.jit(donate_argnums=(1,))  # Donate optimizer for memory efficiency
def train_step(model, optimizer, images, step, labels=None):
    """Single training step (JIT-compiled for speed).

    Following JAX best practices: step is traced (not static) to avoid recompilation.
    JIT compilation provides 3-50x speedup on compute-intensive operations.

    Args:
        model: VAE model (any variant) with internal RNGs
        optimizer: NNX optimizer (donated for memory efficiency)
        images: Batch of images
        step: Current training step (traced, not static - used for β/capacity annealing)
        labels: Optional labels for conditional VAE

    Returns:
        Dictionary of losses from model.loss_fn()
    """

    def loss_fn(model):
        # Handle conditional vs non-conditional models
        # Models use their internal RNGs, no need to pass rngs parameter
        if labels is not None:
            outputs = model(images, y=labels)
        else:
            outputs = model(images)

        # Compute losses (step parameter needed for β/capacity annealing)
        losses = model.loss_fn(x=images, outputs=outputs, step=step)
        return losses["loss"], losses

    # Compute gradients
    (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Update parameters (NNX 0.11.0+ API)
    optimizer.update(model, grads)

    return losses


# %%
# Cell 5: β-VAE Training
def train_beta_vae(model, train_ds, num_epochs=5, learning_rate=1e-3, batches_per_epoch=468):
    """Train β-VAE with monitoring of β value and loss components.

    Uses shared JIT-compiled train step for performance (JAX best practice).

    Args:
        model: BetaVAE model
        train_ds: Training dataset (infinite iterator)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batches_per_epoch: Number of batches per epoch (default 468 for MNIST with batch_size=128)

    Returns:
        Dictionary with training history
    """
    print()
    print("=" * 70)
    print(f"Training β-VAE (β={model.beta_default}, warmup={model.beta_warmup_steps} steps)")
    print("=" * 70)

    # Optimizer (NNX 0.11.0+ API requires wrt parameter)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    history = {"loss": [], "recon_loss": [], "kl_loss": [], "beta": []}
    step_count = 0

    # Set model to training mode (enables dropout, batch norm training, etc.)
    model.train()

    print("Compiling train step with JIT (first iteration will be slower)...")

    for epoch in range(num_epochs):
        epoch_metrics = {"loss": [], "recon_loss": [], "kl_loss": [], "beta": []}

        # Limit batches per epoch from infinite iterator
        epoch_ds = islice(train_ds, batches_per_epoch)

        # Progress bar for batches (leave=True to show progress for each epoch)
        pbar = tqdm(
            epoch_ds,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=batches_per_epoch,
            leave=True,
            unit=" batch",
        )
        for batch in pbar:
            images = jnp.array(batch["image"])  # Convert to JAX array

            # Execute JIT-compiled train step (model uses internal RNGs)
            losses = train_step(model, optimizer, images, step_count)

            # Track metrics
            epoch_metrics["loss"].append(float(losses["loss"]))
            epoch_metrics["recon_loss"].append(float(losses["reconstruction_loss"]))
            epoch_metrics["kl_loss"].append(float(losses["kl_loss"]))
            epoch_metrics["beta"].append(float(losses["beta"]))

            # Update progress bar with current metrics
            pbar.set_postfix(
                {
                    "loss": f"{losses['loss']:.4f}",
                    "recon": f"{losses['reconstruction_loss']:.4f}",
                    "kl": f"{losses['kl_loss']:.4f}",
                    "β": f"{losses['beta']:.3f}",
                }
            )

            step_count += 1

        # Average metrics
        avg_loss = np.mean(epoch_metrics["loss"])
        avg_recon = np.mean(epoch_metrics["recon_loss"])
        avg_kl = np.mean(epoch_metrics["kl_loss"])
        avg_beta = np.mean(epoch_metrics["beta"])

        history["loss"].append(avg_loss)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["beta"].append(avg_beta)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, "
            f"KL: {avg_kl:.4f}, β: {avg_beta:.3f}"
        )

    print("✓ β-VAE training complete")
    return history


# %% [markdown]
"""
## Example 2: β-VAE with Capacity Control

Demonstrate Artifex's BetaVAEWithCapacity which uses capacity control
(Burgess et al.) for more stable training.
"""


# %%
# Cell 5: Capacity β-VAE Setup
def create_capacity_beta_vae(capacity_max=25.0, capacity_steps=5000):
    """Create a β-VAE with capacity control using config-based API.

    Args:
        capacity_max: Maximum KL capacity in nats
        capacity_steps: Steps to reach max capacity

    Returns:
        BetaVAEWithCapacity model
    """
    # Create encoder config
    encoder_config = EncoderConfig(
        name="capacity_vae_encoder",
        hidden_dims=(512, 256),  # Tuple for frozen dataclass
        latent_dim=10,
        activation="relu",
        input_shape=(28, 28, 1),
    )

    # Create decoder config
    decoder_config = DecoderConfig(
        name="capacity_vae_decoder",
        hidden_dims=(256, 512),  # Tuple for frozen dataclass
        latent_dim=10,
        output_shape=(28, 28, 1),
        activation="relu",
    )

    # Create β-VAE with capacity control config
    config = BetaVAEWithCapacityConfig(
        name="capacity_beta_vae_mnist",
        encoder=encoder_config,
        decoder=decoder_config,
        encoder_type="dense",
        beta_default=1.0,  # β fixed at 1.0 when using capacity control
        beta_warmup_steps=0,
        reconstruction_loss_type="mse",
        use_capacity_control=True,
        capacity_max=capacity_max,
        capacity_num_iter=capacity_steps,
        gamma=1000.0,
    )

    # Create model from config
    model = BetaVAEWithCapacity(config=config, rngs=nnx.Rngs(10))

    return model


# %%
# Cell 6: Capacity β-VAE Training
def train_capacity_beta_vae(
    model, train_ds, num_epochs=5, learning_rate=1e-3, batches_per_epoch=468
):
    """Train β-VAE with capacity control monitoring.

    Uses shared JIT-compiled train step for performance.

    Args:
        model: BetaVAEWithCapacity model
        train_ds: Training dataset (infinite iterator)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batches_per_epoch: Number of batches per epoch (default 468 for MNIST with batch_size=128)

    Returns:
        Dictionary with training history
    """
    print()
    print("=" * 70)
    print(f"Training β-VAE with Capacity Control (C_max={model.capacity_max} nats)")
    print("=" * 70)

    # Optimizer (NNX 0.11.0+ API requires wrt parameter)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    history = {"loss": [], "recon_loss": [], "kl_loss": [], "capacity": [], "capacity_loss": []}
    step_count = 0

    # Set model to training mode (enables dropout, batch norm training, etc.)
    model.train()

    print("Compiling train step with JIT (first iteration will be slower)...")

    for epoch in range(num_epochs):
        epoch_metrics = {
            "loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "capacity": [],
            "capacity_loss": [],
        }

        # Limit batches per epoch from infinite iterator
        epoch_ds = islice(train_ds, batches_per_epoch)

        # Progress bar for batches (leave=True to show progress for each epoch)
        pbar = tqdm(
            epoch_ds,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=batches_per_epoch,
            leave=True,
            unit=" batch",
        )
        for batch in pbar:
            images = jnp.array(batch["image"])  # Convert to JAX array

            # Execute JIT-compiled train step (model uses internal RNGs)
            losses = train_step(model, optimizer, images, step_count)

            # Track metrics
            epoch_metrics["loss"].append(float(losses["loss"]))
            epoch_metrics["recon_loss"].append(float(losses["reconstruction_loss"]))
            epoch_metrics["kl_loss"].append(float(losses["kl_loss"]))
            epoch_metrics["capacity"].append(float(losses["current_capacity"]))
            epoch_metrics["capacity_loss"].append(float(losses["capacity_loss"]))

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{losses['loss']:.4f}",
                    "recon": f"{losses['reconstruction_loss']:.4f}",
                    "C": f"{losses['current_capacity']:.2f}",
                }
            )

            step_count += 1

        # Average metrics
        avg_loss = np.mean(epoch_metrics["loss"])
        avg_recon = np.mean(epoch_metrics["recon_loss"])
        avg_kl = np.mean(epoch_metrics["kl_loss"])
        avg_cap = np.mean(epoch_metrics["capacity"])
        avg_cap_loss = np.mean(epoch_metrics["capacity_loss"])

        history["loss"].append(avg_loss)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["capacity"].append(avg_cap)
        history["capacity_loss"].append(avg_cap_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, "
            f"KL: {avg_kl:.4f}, C: {avg_cap:.2f}"
        )

    print("✓ Capacity β-VAE training complete")
    return history


# %% [markdown]
"""
## Example 3: Conditional VAE

Demonstrate Artifex's ConditionalVAE for label-conditioned generation,
enabling controlled sampling of specific digit classes.
"""


# %%
# Cell 7: Conditional VAE Setup
def create_conditional_vae():
    """Create a Conditional VAE model using Artifex's config-based API.

    The ConditionalVAE automatically handles the encoder input adjustment
    and decoder latent adjustment for conditioning by wrapping base encoders
    and decoders with ConditionalEncoder and ConditionalDecoder.

    Returns:
        ConditionalVAE model
    """
    # MNIST image dimensions - base encoder receives flattened images + condition
    # ConditionalVAE handles the input size adjustment internally

    # Create encoder config (base encoder dimensions before conditioning adjustment)
    encoder_config = EncoderConfig(
        name="cvae_encoder",
        hidden_dims=(512, 256),  # Tuple for frozen dataclass
        latent_dim=20,
        activation="relu",
        input_shape=(28, 28, 1),  # Original input shape (conditioning added internally)
    )

    # Create decoder config (base decoder dimensions)
    decoder_config = DecoderConfig(
        name="cvae_decoder",
        hidden_dims=(256, 512),  # Tuple for frozen dataclass
        latent_dim=20,  # Base latent dim (conditioning added internally)
        output_shape=(28, 28, 1),
        activation="relu",
    )

    # Create Conditional VAE config
    config = ConditionalVAEConfig(
        name="conditional_vae_mnist",
        encoder=encoder_config,
        decoder=decoder_config,
        encoder_type="dense",
        num_classes=10,  # 10 digit classes for MNIST
        condition_dim=10,  # Embedding dimension for class labels
        condition_type="concat",
    )

    # Create model from config (internally wraps encoder/decoder with conditional layers)
    model = ConditionalVAE(config=config, rngs=nnx.Rngs(20))

    return model


# %%
# Cell 8: Conditional VAE Training
def train_conditional_vae(model, train_ds, num_epochs=5, learning_rate=1e-3, batches_per_epoch=468):
    """Train Conditional VAE with label conditioning.

    Uses shared JIT-compiled train step for performance.

    Args:
        model: ConditionalVAE model
        train_ds: Training dataset (infinite iterator)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batches_per_epoch: Number of batches per epoch (default 468 for MNIST with batch_size=128)

    Returns:
        Dictionary with training history
    """
    print()
    print("=" * 70)
    print(f"Training Conditional VAE (condition_dim={model.condition_dim})")
    print("=" * 70)

    # Optimizer (NNX 0.11.0+ API requires wrt parameter)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    history = {"loss": [], "recon_loss": [], "kl_loss": []}
    step_count = 0

    # Set model to training mode (enables dropout, batch norm training, etc.)
    model.train()

    print("Compiling train step with JIT (first iteration will be slower)...")

    for epoch in range(num_epochs):
        epoch_metrics = {"loss": [], "recon_loss": [], "kl_loss": []}

        # Limit batches per epoch from infinite iterator
        epoch_ds = islice(train_ds, batches_per_epoch)

        # Progress bar for batches (leave=True to show progress for each epoch)
        pbar = tqdm(
            epoch_ds,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=batches_per_epoch,
            leave=True,
            unit=" batch",
        )
        for batch in pbar:
            images = jnp.array(batch["image"])  # Convert to JAX array
            labels = jnp.array(batch["label"])  # Convert to JAX array

            # Execute JIT-compiled train step with labels (model uses internal RNGs)
            losses = train_step(model, optimizer, images, step_count, labels=labels)

            # Track metrics
            epoch_metrics["loss"].append(float(losses["loss"]))
            epoch_metrics["recon_loss"].append(float(losses["reconstruction_loss"]))
            epoch_metrics["kl_loss"].append(float(losses["kl_loss"]))

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{losses['loss']:.4f}",
                    "recon": f"{losses['reconstruction_loss']:.4f}",
                    "kl": f"{losses['kl_loss']:.4f}",
                }
            )

            step_count += 1

        # Average metrics
        avg_loss = np.mean(epoch_metrics["loss"])
        avg_recon = np.mean(epoch_metrics["recon_loss"])
        avg_kl = np.mean(epoch_metrics["kl_loss"])

        history["loss"].append(avg_loss)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}"
        )

    print("✓ Conditional VAE training complete")
    return history


# %% [markdown]
"""
## Example 4: VQ-VAE

Demonstrate Artifex's VQVAE with codebook monitoring and perplexity tracking
to ensure healthy codebook usage.
"""


# %%
# Cell 9: VQ-VAE Setup
def create_vqvae(num_embeddings=512, embedding_dim=64):
    """Create a VQ-VAE model using Artifex's config-based API.

    Args:
        num_embeddings: Size of codebook
        embedding_dim: Dimension of each embedding

    Returns:
        VQVAE model
    """
    # Create encoder config
    encoder_config = EncoderConfig(
        name="vqvae_encoder",
        hidden_dims=(512, 256),  # Tuple for frozen dataclass
        latent_dim=embedding_dim,
        activation="relu",
        input_shape=(28, 28, 1),
    )

    # Create decoder config
    decoder_config = DecoderConfig(
        name="vqvae_decoder",
        hidden_dims=(256, 512),  # Tuple for frozen dataclass
        latent_dim=embedding_dim,
        output_shape=(28, 28, 1),
        activation="relu",
    )

    # Create VQ-VAE config
    config = VQVAEConfig(
        name="vqvae_mnist",
        encoder=encoder_config,
        decoder=decoder_config,
        encoder_type="dense",
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
    )

    # Create model from config
    model = VQVAE(config=config, rngs=nnx.Rngs(30))

    return model


# %%
# Cell 10: VQ-VAE Training
def train_vqvae(model, train_ds, num_epochs=5, learning_rate=1e-3, batches_per_epoch=468):
    """Train VQ-VAE with codebook monitoring.

    Uses shared JIT-compiled train step for performance.

    Args:
        model: VQVAE model
        train_ds: Training dataset (infinite iterator)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batches_per_epoch: Number of batches per epoch (default 468 for MNIST with batch_size=128)

    Returns:
        Dictionary with training history
    """
    print()
    print("=" * 70)
    print(f"Training VQ-VAE (codebook={model.num_embeddings}, dim={model.embedding_dim})")
    print("=" * 70)

    # Optimizer (NNX 0.11.0+ API requires wrt parameter)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    history = {"loss": [], "recon_loss": [], "vq_loss": [], "perplexity": []}
    step_count = 0

    # Set model to training mode (enables dropout, batch norm training, etc.)
    model.train()

    print("Compiling train step with JIT (first iteration will be slower)...")

    for epoch in range(num_epochs):
        epoch_metrics = {"loss": [], "recon_loss": [], "vq_loss": [], "perplexity": []}

        # Limit batches per epoch from infinite iterator
        epoch_ds = islice(train_ds, batches_per_epoch)

        # Progress bar for batches (leave=True to show progress for each epoch)
        pbar = tqdm(
            epoch_ds,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            total=batches_per_epoch,
            leave=True,
            unit=" batch",
        )
        for batch in pbar:
            images = jnp.array(batch["image"])  # Convert to JAX array

            # Execute JIT-compiled train step (model uses internal RNGs)
            losses = train_step(model, optimizer, images, step_count)

            # Track metrics
            epoch_metrics["loss"].append(float(losses["loss"]))
            epoch_metrics["recon_loss"].append(float(losses["reconstruction_loss"]))
            epoch_metrics["vq_loss"].append(float(losses["vq_loss"]))
            epoch_metrics["perplexity"].append(float(losses.get("perplexity", 0.0)))

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{losses['loss']:.4f}",
                    "recon": f"{losses['reconstruction_loss']:.4f}",
                    "vq": f"{losses['vq_loss']:.4f}",
                }
            )

            step_count += 1

        # Average metrics
        avg_loss = np.mean(epoch_metrics["loss"])
        avg_recon = np.mean(epoch_metrics["recon_loss"])
        avg_vq = np.mean(epoch_metrics["vq_loss"])
        avg_perp = np.mean(epoch_metrics["perplexity"])

        history["loss"].append(avg_loss)
        history["recon_loss"].append(avg_recon)
        history["vq_loss"].append(avg_vq)
        history["perplexity"].append(avg_perp)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, "
            f"VQ: {avg_vq:.4f}, Perplexity: {avg_perp:.1f}"
        )

    print("✓ VQ-VAE training complete")
    return history


# %% [markdown]
"""
## Visualization Functions

Create visualizations for each VAE variant showing reconstructions and
generated samples.
"""


# %%
# Cell 11: Visualization Functions
def visualize_vae_results(model, test_ds, variant_name, conditional=False):
    """Visualize VAE reconstructions and generated samples.

    Args:
        model: Trained VAE model
        test_ds: Test dataset
        variant_name: Name of the variant (for plot title)
        conditional: Whether this is a conditional VAE

    Returns:
        matplotlib figure
    """
    # Set model to evaluation mode (disables dropout, uses running stats for batch norm)
    model.eval()

    # Get test batch from Grain DataLoader
    test_batch = next(iter(test_ds))
    test_images = test_batch["image"][:10]  # Already numpy arrays from Grain

    # Reconstructions (model uses internal RNGs in eval mode)
    if conditional:
        test_labels = test_batch["label"][:10]  # Already numpy arrays from Grain
        outputs = model(test_images, y=test_labels)
    else:
        outputs = model(test_images)

    reconstructions = outputs.get("reconstructed", outputs.get("reconstruction"))

    # Plot
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
    for i in range(10):
        # Original
        axes[0, i].imshow(test_images[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)

        # Reconstructed
        recon_img = reconstructions[i]
        if recon_img.shape != test_images[i].shape:
            recon_img = recon_img.reshape(test_images[i].shape)
        axes[1, i].imshow(recon_img[:, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=10)

    plt.suptitle(f"{variant_name} Results", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    filename = variant_name.lower().replace(" ", "_").replace("-", "_") + "_results.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to {output_path}")

    return fig


def plot_training_curves(history, variant_name):
    """Plot training curves for a VAE variant.

    Args:
        history: Training history dictionary
        variant_name: Name of the variant (for plot title)

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total loss
    axes[0].plot(history["loss"], label="Total Loss", linewidth=2, color="tab:blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Component losses
    axes[1].plot(history["recon_loss"], label="Reconstruction", linewidth=2, color="tab:orange")
    if "kl_loss" in history:
        axes[1].plot(history["kl_loss"], label="KL Divergence", linewidth=2, color="tab:green")
    if "vq_loss" in history:
        axes[1].plot(history["vq_loss"], label="VQ Loss", linewidth=2, color="tab:red")
    if "capacity_loss" in history:
        axes[1].plot(
            history["capacity_loss"], label="Capacity Loss", linewidth=2, color="tab:purple"
        )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss Component")
    axes[1].set_title("Loss Components")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{variant_name} Training Curves", fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    filename = variant_name.lower().replace(" ", "_").replace("-", "_") + "_training.png"
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved training curves to {output_path}")

    return fig


# %% [markdown]
"""
## Main Execution

Train and evaluate all four advanced VAE variants using Artifex's
production-ready implementations.
"""


# %%
# Cell 12: Main Execution
def main():
    """Main execution: demonstrate all advanced VAE variants."""
    print()
    print("=" * 80)
    print("ADVANCED VAE EXAMPLES - Artifex's Advanced Features on MNIST")
    print("=" * 80)

    # Load data
    train_ds, test_ds = load_real_mnist(batch_size=128)

    # ========== β-VAE ==========
    print()
    print("=" * 80)
    print("1. β-VAE: Disentangled Representations with β Annealing")
    print("=" * 80)

    beta_vae = create_beta_vae(beta=4.0, warmup_steps=1000)
    beta_history = train_beta_vae(beta_vae, train_ds, num_epochs=5)
    plot_training_curves(beta_history, "β-VAE")
    visualize_vae_results(beta_vae, test_ds, "β-VAE")

    # ========== Capacity β-VAE ==========
    print()
    print("=" * 80)
    print("2. β-VAE with Capacity Control: Burgess et al. Method")
    print("=" * 80)

    capacity_vae = create_capacity_beta_vae(capacity_max=25.0, capacity_steps=5000)
    capacity_history = train_capacity_beta_vae(capacity_vae, train_ds, num_epochs=5)
    plot_training_curves(capacity_history, "Capacity β-VAE")
    visualize_vae_results(capacity_vae, test_ds, "Capacity β-VAE")

    # ========== Conditional VAE ==========
    print()
    print("=" * 80)
    print("3. Conditional VAE: Label-Conditioned Generation")
    print("=" * 80)

    cvae = create_conditional_vae()
    cvae_history = train_conditional_vae(cvae, train_ds, num_epochs=5)
    plot_training_curves(cvae_history, "Conditional VAE")
    visualize_vae_results(cvae, test_ds, "Conditional VAE", conditional=True)

    # ========== VQ-VAE ==========
    print()
    print("=" * 80)
    print("4. VQ-VAE: Discrete Latent Codes with Codebook Monitoring")
    print("=" * 80)

    vqvae = create_vqvae(num_embeddings=512, embedding_dim=64)
    vqvae_history = train_vqvae(vqvae, train_ds, num_epochs=5)
    plot_training_curves(vqvae_history, "VQ-VAE")
    visualize_vae_results(vqvae, test_ds, "VQ-VAE")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Trained 4 advanced VAE variants using Artifex's implementations")
    print(f"✓ Generated visualizations in {OUTPUT_DIR}")
    print("✓ All models trained on real MNIST data (60,000 images)")
    print()
    print("Variants demonstrated:")
    print("  1. β-VAE with β annealing (β=1.0→4.0)")
    print("  2. β-VAE with capacity control (C_max=25 nats)")
    print("  3. Conditional VAE (10 classes)")
    print("  4. VQ-VAE (codebook_size=512)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
