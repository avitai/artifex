#!/usr/bin/env python
r"""Advanced GAN Examples - Showcase Artifex's Advanced GAN Features

## Overview

This example demonstrates Artifex's advanced GAN implementations on real MNIST data:
1. **Conditional GAN**: Class-conditional generation for controlled digit synthesis
2. **WGAN-GP**: Wasserstein GAN with gradient penalty for stable training
3. **DCGAN**: Deep Convolutional GAN with proven architecture
4. **LSGAN**: Least Squares GAN for improved stability

All models use Artifex's production-ready implementations with proper training
on real MNIST data.

## Source Code Dependencies

**Validated:** 2025-10-25

This example uses Artifex's GAN implementations:
- `artifex.generative_models.models.gan.ConditionalGAN` - Conditional GAN
- `artifex.generative_models.models.gan.WGAN` - Wasserstein GAN with GP
- `artifex.generative_models.models.gan.DCGAN` - Deep Convolutional GAN
- `artifex.generative_models.models.gan.LSGAN` - Least Squares GAN

**Validation Status:**
- ✅ All dependencies use Flax NNX best practices
- ✅ Proper RNG handling throughout
- ✅ Production-ready implementations from Artifex

## What You'll Learn

- [x] Using Artifex's ConditionalGAN for label-conditioned generation
- [x] Implementing WGAN-GP for stable adversarial training
- [x] Training DCGAN with convolutional architecture
- [x] Using LSGAN for improved training stability
- [x] Training advanced GANs on real MNIST data
- [x] Evaluating and visualizing generated samples

## Prerequisites

- Artifex installed (run `source activate.sh`)
- Understanding of standard GANs (see basic gan-mnist tutorial)
- Familiarity with JAX and Flax NNX
- Knowledge of adversarial training concepts

## Usage

```bash
source activate.sh
python examples/generative_models/image/gan/advanced_gan.py
```

## Expected Output

The example will:
1. Load real MNIST dataset (60,000 training images)
2. Train four advanced GAN variants (Conditional, WGAN-GP, DCGAN, LSGAN)
3. Generate samples from each model
4. Save visualizations to `examples_output/advanced_gan/`
5. Display training metrics and convergence curves

## Key Concepts

### Conditional GAN

Conditional GAN extends standard GAN by conditioning on labels:
$$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x,y}[\\log D(x|y)] +$$
$$\\mathbb{E}_{z,y}[\\log(1 - D(G(z|y)|y))]$$

Where y is the class label.

### WGAN-GP

Wasserstein GAN with gradient penalty uses Wasserstein distance:
$$\\mathcal{L}_D = \\mathbb{E}[D(G(z))] - \\mathbb{E}[D(x)] +$$
$$\\lambda \\mathbb{E}[(\\|\\nabla_{\\hat{x}} D(\\hat{x})\\|_2 - 1)^2]$$

Provides more stable training than standard GAN.

### DCGAN

Deep Convolutional GAN uses architectural guidelines:
- Use strided convolutions instead of pooling
- Use batch normalization in both G and D
- Remove fully connected hidden layers
- Use ReLU in G (except output: tanh)
- Use LeakyReLU in D

### LSGAN

Least Squares GAN uses least squares loss instead of cross-entropy:
$$\\min_D \\mathbb{E}[(D(x) - 1)^2] + \\mathbb{E}[D(G(z))^2]$$
$$\\min_G \\mathbb{E}[(D(G(z)) - 1)^2]$$

Provides more stable gradients.

## Estimated Runtime

- **CPU**: ~20-30 minutes total (all 4 variants)
- **GPU**: ~5-8 minutes total (if available)

## Author

Artifex Team

## Last Updated

2025-10-25
"""

# %% [markdown]
"""
# Advanced GAN Examples

This notebook demonstrates Artifex's advanced GAN implementations on real MNIST data.

## Learning Objectives

By the end of this example, you will understand:
1. How to use Artifex's ConditionalGAN for label-controlled generation
2. Using WGAN-GP for stable adversarial training
3. Training DCGAN with convolutional architecture
4. Using LSGAN for improved stability
5. Training and evaluating advanced GAN variants
"""

# %%
# Cell 1: Import Dependencies
"""
Import Artifex's GAN implementations and utilities for training advanced variants.
"""

from itertools import islice
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from tqdm import tqdm

from artifex.generative_models.core.configuration.network_configs import (
    ConditionalDiscriminatorConfig,
    ConditionalGeneratorConfig,
    ConditionalParams,
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.core.losses.adversarial import (
    least_squares_discriminator_loss,
    least_squares_generator_loss,
    vanilla_discriminator_loss,
    vanilla_generator_loss,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
)
from artifex.generative_models.core.losses.regularization import gradient_penalty
from artifex.generative_models.models.gan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    DCGANDiscriminator,
    DCGANGenerator,
    LSGANDiscriminator,
    LSGANGenerator,
    WGANDiscriminator,
    WGANGenerator,
)


# Create output directory
OUTPUT_DIR = Path("examples_output/advanced_gan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"JAX devices: {jax.devices()}")
print(f"Output directory: {OUTPUT_DIR}")


# %% [markdown]
"""
## Data Loading

Load real MNIST dataset using Hugging Face datasets. This ensures we train on
actual handwritten digits with proper training/test splits.

**Dataset Properties:**
- Training: 60,000 images
- Test: 10,000 images
- Image size: 28×28×1 (grayscale)
- Labels: 0-9 (digit classes)
- Values: [-1, 1] (normalized for GAN training)
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
                # Reshape to (H, W, C) format expected by Artifex GANs
                image = image[..., jnp.newaxis]  # Add channel: (28, 28, 1)
                # Scale to [-1, 1] for GAN training (standard practice)
                image = (image - 0.5) * 2.0
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
        num_epochs=None,  # Infinite epochs
    )

    test_sampler = grain.IndexSampler(
        num_records=len(test_source),
        shuffle=False,
        seed=42,
        num_epochs=1,
    )

    # Create DataLoaders with batching
    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=0,
    )

    test_loader = grain.DataLoader(
        data_source=test_source,
        sampler=test_sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=False)],
        worker_count=0,
    )

    print(f"✓ Created Grain DataLoaders (batch_size={batch_size})")

    return train_loader, test_loader


# Load data once for all examples
train_loader, test_loader = load_real_mnist(batch_size=64)


# %% [markdown]
"""
## Training Infrastructure

Common training utilities used by all GAN variants.
"""


# %%
# Cell 3: Common Training Infrastructure
def train_gan_model(
    generator,
    discriminator,
    train_loader,
    model_name: str,
    loss_type: str = "vanilla",
    num_epochs: int = 10,
    latent_dim: int = 100,
    learning_rate: float = 2e-4,
    n_critic: int = 1,
    lambda_gp: float = 10.0,
):
    """Train a GAN model using Artifex's loss functions.

    This function demonstrates how to use Artifex's modular loss functions
    with different GAN architectures. Training steps are JIT-compiled for
    performance (following JAX best practices).

    Args:
        generator: Generator model (Artifex's Generator class)
        discriminator: Discriminator model (Artifex's Discriminator class)
        train_loader: Training data loader
        model_name: Name for logging
        loss_type: Loss type ('vanilla', 'wgan', 'lsgan')
        num_epochs: Number of training epochs
        latent_dim: Latent dimension
        learning_rate: Learning rate
        n_critic: Number of discriminator updates per generator update
        lambda_gp: Gradient penalty coefficient (for WGAN-GP)

    Returns:
        Tuple of (generator, discriminator, metrics_dict)
    """
    print()
    print("=" * 70)
    print(f"Training {model_name}")
    print(f"Loss type: {loss_type}")
    print("=" * 70)

    # Create optimizers using Artifex's pattern
    optimizer_g = nnx.Optimizer(generator, optax.adam(learning_rate, b1=0.5), wrt=nnx.Param)
    optimizer_d = nnx.Optimizer(discriminator, optax.adam(learning_rate, b1=0.5), wrt=nnx.Param)

    # Determine if models are conditional
    is_conditional = isinstance(generator, ConditionalGenerator)

    # Define JIT-compiled training steps for better performance
    @nnx.jit
    def train_discriminator_step(disc, opt_d, real_imgs, fake_imgs, labels_oh=None, seed=0):
        """JIT-compiled discriminator training step."""

        def d_loss_fn(disc):
            if is_conditional:
                real_scores = disc(real_imgs, labels_oh)
                fake_scores = disc(fake_imgs, labels_oh)
            else:
                real_scores = disc(real_imgs)
                fake_scores = disc(fake_imgs)

            # Use Artifex's loss functions
            if loss_type == "vanilla":
                # Vanilla loss expects probabilities, so apply sigmoid to logits
                return vanilla_discriminator_loss(
                    nnx.sigmoid(real_scores), nnx.sigmoid(fake_scores)
                )
            elif loss_type == "wgan":
                # WGAN loss + gradient penalty (works with raw scores)
                w_loss = wasserstein_discriminator_loss(real_scores, fake_scores)
                # Compute gradient penalty using Artifex's function
                disc_fn = (lambda x: disc(x, labels_oh)) if is_conditional else (lambda x: disc(x))
                gp = gradient_penalty(
                    real_imgs,
                    fake_imgs,
                    disc_fn,
                    lambda_gp=lambda_gp,
                    key=jax.random.key(seed),
                )
                return w_loss + gp
            elif loss_type == "lsgan":
                # LSGAN works with raw scores
                return least_squares_discriminator_loss(real_scores, fake_scores)

        # Update discriminator
        d_loss, grads = nnx.value_and_grad(d_loss_fn)(disc)
        opt_d.update(disc, grads)
        return d_loss

    @nnx.jit
    def train_generator_step(gen, disc, opt_g, z, labels_oh=None):
        """JIT-compiled generator training step."""

        def g_loss_fn(gen):
            if is_conditional:
                fake = gen(z, labels_oh)
                fake_scores = disc(fake, labels_oh)
            else:
                fake = gen(z)
                fake_scores = disc(fake)

            # Use Artifex's loss functions
            if loss_type == "vanilla":
                return vanilla_generator_loss(nnx.sigmoid(fake_scores))
            elif loss_type == "wgan":
                return wasserstein_generator_loss(fake_scores)
            elif loss_type == "lsgan":
                return least_squares_generator_loss(fake_scores)

        # Update generator
        g_loss, grads = nnx.value_and_grad(g_loss_fn)(gen)
        opt_g.update(gen, grads)
        return g_loss

    # Training metrics
    g_losses = []
    d_losses = []

    batches_per_epoch = 468  # 60,000 / 64 ≈ 937, but we'll use 468 for faster training

    print("Compiling JIT-compiled training steps (first iteration will be slower)...")

    # Training loop
    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batch_count = 0

        # Set models to training mode
        generator.train()
        discriminator.train()

        # Progress bar for this epoch
        pbar = tqdm(
            islice(train_loader, batches_per_epoch),
            total=batches_per_epoch,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
        )

        critic_steps = 0

        for batch in pbar:
            real_images = batch["image"]
            labels = batch.get("label", None)
            batch_size = real_images.shape[0]

            # Convert images from (H, W, C) to (C, H, W) for Artifex GANs
            real_images = jnp.transpose(real_images, (0, 3, 1, 2))

            # Generate fake images
            seed = epoch * 1000 + batch_count
            z = jax.random.normal(jax.random.key(seed), (batch_size, latent_dim))

            # Handle conditional vs unconditional generation
            labels_onehot = None
            if is_conditional:
                # Convert labels to one-hot
                labels_onehot = jax.nn.one_hot(labels, 10)
                fake_images = generator(z, labels_onehot)
            else:
                fake_images = generator(z)

            # Crop WGAN-GP output from 32x32 to 28x28 if needed
            if loss_type == "wgan" and fake_images.shape[-1] == 32:
                # Center crop: 32x32 -> 28x28 (remove 2 pixels from each side)
                fake_images = fake_images[:, :, 2:30, 2:30]

            # Train discriminator (JIT-compiled)
            d_loss = train_discriminator_step(
                discriminator, optimizer_d, real_images, fake_images, labels_onehot, seed
            )
            epoch_d_loss += float(d_loss)

            critic_steps += 1

            # Train generator every n_critic steps (JIT-compiled)
            if critic_steps >= n_critic:
                critic_steps = 0

                # Generate new latent vectors for generator update
                z = jax.random.normal(jax.random.key(seed + 1), (batch_size, latent_dim))

                g_loss = train_generator_step(
                    generator, discriminator, optimizer_g, z, labels_onehot
                )
                epoch_g_loss += float(g_loss)

            batch_count += 1
            pbar.set_postfix(
                {
                    "g_loss": f"{epoch_g_loss / max(1, batch_count // n_critic):.4f}",
                    "d_loss": f"{epoch_d_loss / batch_count:.4f}",
                }
            )

        # Average losses for epoch
        avg_g_loss = epoch_g_loss / max(1, batch_count // n_critic)
        avg_d_loss = epoch_d_loss / batch_count
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}"
        )

        # Save sample images every 5 epochs to track quality improvement
        if (epoch + 1) % 5 == 0 or epoch == 0:
            generator.eval()  # Set to eval mode for sampling

            # Generate samples for visualization
            sample_rngs = nnx.Rngs(999 + epoch)
            if is_conditional:
                # Generate one sample per class
                z_sample = jax.random.normal(sample_rngs.params(), (10, latent_dim))
                labels_sample = jax.nn.one_hot(jnp.arange(10), 10)
                samples = generator(z_sample, labels_sample)
            else:
                z_sample = jax.random.normal(sample_rngs.params(), (10, latent_dim))
                samples = generator(z_sample)

            # Crop WGAN-GP output from 32x32 to 28x28 if needed
            if loss_type == "wgan" and samples.shape[-1] == 32:
                samples = samples[:, :, 2:30, 2:30]

            # Convert from (C, H, W) to (H, W, C) and denormalize
            samples = jnp.transpose(samples, (0, 2, 3, 1))
            samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]

            # Save epoch samples
            fig, axes = plt.subplots(1, 10, figsize=(15, 2))
            for i in range(10):
                axes[i].imshow(samples[i, :, :, 0], cmap="gray")
                axes[i].axis("off")
                if is_conditional:
                    axes[i].set_title(f"{i}", fontsize=10)

            plt.tight_layout()
            epoch_dir = OUTPUT_DIR / model_name.lower().replace(" ", "_").replace("-", "_")
            epoch_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(epoch_dir / f"epoch_{epoch + 1:02d}.png", dpi=100, bbox_inches="tight")
            plt.close()

            generator.train()  # Set back to training mode

    # Set to eval mode
    generator.eval()
    discriminator.eval()

    print(f"\n✓ {model_name} training complete")
    epoch_dir_name = model_name.lower().replace(" ", "_").replace("-", "_")
    print(f"Epoch samples saved to: {OUTPUT_DIR / epoch_dir_name}")

    return generator, discriminator, {"g_losses": g_losses, "d_losses": d_losses}


# %% [markdown]
"""
## Example 1: Conditional GAN

Conditional GAN allows generation of specific digit classes by conditioning
both generator and discriminator on label information.

**Artifex Components:**
- `ConditionalGenerator`: Generator with label embedding
- `ConditionalDiscriminator`: Discriminator with label conditioning
- `ConditionalGAN`: Combined model with conditional loss

**Key Features:**
- Class-controlled generation
- Label consistency in generated samples
- Improved training stability through conditioning

**Training Notes:**
- Uses learning rate of 0.0003 (Keras 2024 best practice)
- Trains for 20 epochs (recommended for good convergence)
- LeakyReLU with slope 0.2 (standard GAN practice)
- Batch size 64 (proven stable for MNIST)
- **Loss behavior**: Losses stabilize around G~0.7, D~1.3 after epoch 2
  - This is normal! GANs reach Nash equilibrium, not minimize loss
  - Fluctuating losses indicate healthy adversarial balance
  - Quality improves even when losses stay constant
- **Progress tracking**: Samples saved every 5 epochs + epoch 1
  - See `examples_output/advanced_gan/conditional_gan/` for progression
  - Visual quality improves even when losses plateau
- Future improvement: Add label smoothing (0.9/0.1) for enhanced stability
"""


# %%
# Cell 4: Create and Train Conditional GAN
def create_conditional_gan():
    """Create Conditional GAN using Artifex's Generator and Discriminator."""
    rngs = nnx.Rngs(42)

    # MNIST is (H, W, C) = (28, 28, 1), but Artifex GANs expect (C, H, W)
    output_shape = (1, 28, 28)

    # Create conditional params (shared by generator and discriminator)
    cond_params = ConditionalParams(
        num_classes=10,
        embedding_dim=100,  # Standard embedding dimension
    )

    # Generator config with conditional params
    gen_config = ConditionalGeneratorConfig(
        name="cond_generator",
        output_shape=output_shape,
        latent_dim=100,
        hidden_dims=(256, 128),  # Smaller for MNIST
        batch_norm=True,
        activation="relu",
        conditional=cond_params,
    )

    generator = ConditionalGenerator(
        config=gen_config,
        rngs=rngs,
    )

    # Discriminator config with conditional params
    disc_config = ConditionalDiscriminatorConfig(
        name="cond_discriminator",
        input_shape=output_shape,
        hidden_dims=(128, 256),
        batch_norm=False,  # No batch norm in discriminator
        activation="leaky_relu",
        conditional=cond_params,
    )

    discriminator = ConditionalDiscriminator(
        config=disc_config,
        rngs=nnx.Rngs(43),
    )

    return generator, discriminator


# Create and train using Artifex's loss functions
# Following best practices from Keras CGAN tutorial (2024):
# - Learning rate: 0.0003 (slightly higher than standard 0.0002)
# - Batch size: 64 (already set in data loader)
# - Epochs: 20 (increased from 10 for better convergence)
# - Latent dim: 100 (standard, could try 128)
cond_generator, cond_discriminator = create_conditional_gan()
cond_generator, cond_discriminator, cond_metrics = train_gan_model(
    cond_generator,
    cond_discriminator,
    train_loader,
    "Conditional GAN",
    loss_type="vanilla",
    num_epochs=20,  # Increased from 10 (Keras recommendation)
    learning_rate=3e-4,  # 0.0003 (Keras best practice)
)


# %%
# Cell 5: Visualize Conditional GAN Results
def visualize_conditional_samples(generator, filename="conditional_gan_samples.png"):
    """Generate and visualize samples for each digit class."""
    rngs = nnx.Rngs(123)

    # Generate samples for each class
    num_classes = 10
    samples_per_class = 8

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(12, 15))

    for class_id in range(num_classes):
        z = jax.random.normal(rngs.params(), (samples_per_class, 100))
        labels_onehot = jax.nn.one_hot(jnp.ones(samples_per_class, dtype=jnp.int32) * class_id, 10)

        # Generate using Artifex's ConditionalGenerator
        generated = generator(z, labels_onehot)

        # Convert from (C, H, W) to (H, W, C)
        generated = jnp.transpose(generated, (0, 2, 3, 1))

        for i in range(samples_per_class):
            ax = axes[class_id, i]
            img = generated[i, :, :, 0]
            img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Class {class_id}", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved conditional samples to {OUTPUT_DIR / filename}")


visualize_conditional_samples(cond_generator)


# %% [markdown]
"""
## Example 2: WGAN-GP (Wasserstein GAN with Gradient Penalty)

WGAN-GP uses Wasserstein distance instead of standard GAN loss, providing
more stable training and meaningful loss curves.

**Artifex Components:**
- `WGANGenerator`: Generator for WGAN
- `WGANDiscriminator`: Critic (no sigmoid output)
- `WGAN`: Combined model with Wasserstein loss
- Artifex's `gradient_penalty`: Enforces Lipschitz constraint

**Key Features:**
- Wasserstein distance (Earth Mover's Distance)
- Gradient penalty for Lipschitz constraint (λ=10)
- No sigmoid in discriminator (critic outputs raw scores)
- More stable training dynamics
- Critic trained 5x per generator update
- Meaningful loss metric (Wasserstein distance)

**MNIST-Specific Settings (Research-Based):**
- Learning rate: 1e-4 (optimal from original paper)
- Training epochs: 30+ (WGANs need more epochs than vanilla GANs)
- Batch size: 64 (can use up to 512 for better stability)
- No BatchNorm in discriminator (incompatible with gradient penalty)
- Generator outputs 32x32, center-cropped to 28x28 for MNIST

Note: WGAN-GP typically takes longer to converge than vanilla GANs, but
provides more stable training with less mode collapse. The Wasserstein
loss correlates well with sample quality.
"""


# %%
# Cell 6: Create and Train WGAN-GP
def create_wgan_gp():
    """Create WGAN-GP using Artifex's components.

    Note: WGANGenerator produces 32x32 output (4->8->16->32 upsampling path).
    We crop to 28x28 to match MNIST dimensions.
    """
    rngs = nnx.Rngs(45)

    # WGANGenerator outputs 32x32, we'll crop to 28x28 for MNIST
    generator_output_shape = (1, 32, 32)
    mnist_shape = (1, 28, 28)

    # WGAN architecture based on original paper and Keras implementation
    # Generator: Initial 4x4, then upsample with decreasing channels
    # With hidden_dims=(A, B, C): initial 4x4@A, then A->B (4->8), B->C (8->16), C->out (16->32)
    # Original paper uses DIM=64: (1024, 512, 256) for generator
    gen_config = ConvGeneratorConfig(
        name="wgan_generator",
        output_shape=generator_output_shape,
        latent_dim=100,  # Standard latent dimension (original paper uses 128, but 100 is common)
        hidden_dims=(512, 256, 128),  # Larger capacity than before (originally 1024, 512, 256)
        batch_norm=True,  # Generator uses BatchNorm
        activation="relu",
    )

    generator = WGANGenerator(
        config=gen_config,
        rngs=rngs,
    )

    # Discriminator: No BatchNorm (incompatible with gradient penalty)
    # Progressive downsampling with increasing channels
    # Original paper uses: (64, 128, 256) or Keras uses (64, 128, 256, 512)
    disc_config = ConvDiscriminatorConfig(
        name="wgan_discriminator",
        input_shape=mnist_shape,
        hidden_dims=(64, 128, 256),  # Standard WGAN-GP progression
        batch_norm=False,  # No BatchNorm (incompatible with gradient penalty)
        use_instance_norm=True,  # WGAN-GP uses instance norm instead of batch norm
        activation="leaky_relu",
    )

    discriminator = WGANDiscriminator(
        config=disc_config,
        rngs=nnx.Rngs(46),
    )

    return generator, discriminator


# Create and train using Artifex's WGAN loss + gradient penalty
# Based on research from Keras docs, original WGAN-GP paper, and empirical studies (2024)
wgan_generator, wgan_discriminator = create_wgan_gp()
wgan_generator, wgan_discriminator, wgan_metrics = train_gan_model(
    wgan_generator,
    wgan_discriminator,
    train_loader,
    "WGAN-GP",
    loss_type="wgan",
    num_epochs=30,  # WGANs need more epochs (research shows 20-50 minimum)
    learning_rate=1e-4,  # Optimal for WGAN-GP on MNIST (original paper setting)
    n_critic=5,  # Train critic 5x per generator update (standard WGAN-GP)
    lambda_gp=10.0,  # Gradient penalty coefficient (consistent across all sources)
)


# %%
# Cell 7: Visualize WGAN-GP Results
def visualize_gan_samples(generator, filename, num_samples=64, crop_to_28=False):
    """Generate and visualize GAN samples.

    Args:
        generator: The generator model
        filename: Output filename for the visualization
        num_samples: Number of samples to generate
        crop_to_28: If True, crop 32x32 outputs to 28x28 (for WGAN-GP)
    """
    rngs = nnx.Rngs(456)

    z = jax.random.normal(rngs.params(), (num_samples, 100))
    generated = generator(z)

    # Crop WGAN-GP output from 32x32 to 28x28 if needed
    if crop_to_28 and generated.shape[-1] == 32:
        generated = generated[:, :, 2:30, 2:30]

    # Convert from (C, H, W) to (H, W, C)
    generated = jnp.transpose(generated, (0, 2, 3, 1))

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for i in range(8):
        for j in range(8):
            idx = i * 8 + j
            ax = axes[i, j]
            img = generated[idx, :, :, 0]
            img = (img + 1) / 2  # Denormalize
            ax.imshow(img, cmap="gray")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved samples to {OUTPUT_DIR / filename}")


visualize_gan_samples(wgan_generator, "wgan_gp_samples.png", crop_to_28=True)


# %% [markdown]
"""
## Example 3: DCGAN (Deep Convolutional GAN)

DCGAN introduced architectural guidelines that became standard for GAN training:
- Replace pooling with strided convolutions
- Use batch normalization in generator
- Remove fully connected layers
- Use ReLU in generator, LeakyReLU in discriminator

**Artifex Components:**
- `DCGANGenerator`: Generator with transposed convolutions
- `DCGANDiscriminator`: Discriminator with strided convolutions
- `DCGAN`: Combined model following DCGAN guidelines

**Key Features:**
- Proven convolutional architecture
- Stable training with batch normalization
- Good image quality
- Well-established best practices
"""


# %%
# Cell 8: Create and Train DCGAN
def create_dcgan():
    """Create DCGAN using Artifex's components."""
    rngs = nnx.Rngs(48)

    output_shape = (1, 28, 28)

    gen_config = ConvGeneratorConfig(
        name="dcgan_generator",
        output_shape=output_shape,
        latent_dim=100,
        hidden_dims=(256, 128),
        batch_norm=True,
        activation="relu",
    )

    generator = DCGANGenerator(
        config=gen_config,
        rngs=rngs,
    )

    disc_config = ConvDiscriminatorConfig(
        name="dcgan_discriminator",
        input_shape=output_shape,
        hidden_dims=(128, 256),
        batch_norm=False,
        activation="leaky_relu",
    )

    discriminator = DCGANDiscriminator(
        config=disc_config,
        rngs=nnx.Rngs(49),
    )

    return generator, discriminator


# Create and train using Artifex's vanilla GAN loss
dcgan_generator, dcgan_discriminator = create_dcgan()
dcgan_generator, dcgan_discriminator, dcgan_metrics = train_gan_model(
    dcgan_generator,
    dcgan_discriminator,
    train_loader,
    "DCGAN",
    loss_type="vanilla",
    num_epochs=10,
    learning_rate=2e-4,
)


# %%
# Cell 9: Visualize DCGAN Results
visualize_gan_samples(dcgan_generator, "dcgan_samples.png")


# %% [markdown]
"""
## Example 4: LSGAN (Least Squares GAN)

LSGAN replaces the cross-entropy loss with a least squares loss, providing
more stable gradients during training.

**Artifex Components:**
- `LSGANGenerator`: Generator optimized for LSGAN
- `LSGANDiscriminator`: Discriminator with least squares loss
- `LSGAN`: Combined model with LS loss

**Key Features:**
- Least squares loss instead of cross-entropy
- More stable gradients
- Better convergence properties
- Reduced vanishing gradients
"""


# %%
# Cell 10: Create and Train LSGAN
def create_lsgan():
    """Create LSGAN using Artifex's components."""
    rngs = nnx.Rngs(51)

    output_shape = (1, 28, 28)

    gen_config = ConvGeneratorConfig(
        name="lsgan_generator",
        output_shape=output_shape,
        latent_dim=100,
        hidden_dims=(256, 128),
        batch_norm=True,
        activation="relu",
    )

    generator = LSGANGenerator(
        config=gen_config,
        rngs=rngs,
    )

    disc_config = ConvDiscriminatorConfig(
        name="lsgan_discriminator",
        input_shape=output_shape,
        hidden_dims=(128, 256),
        batch_norm=False,
        activation="leaky_relu",
    )

    discriminator = LSGANDiscriminator(
        config=disc_config,
        rngs=nnx.Rngs(52),
    )

    return generator, discriminator


# Create and train using Artifex's least squares GAN loss
lsgan_generator, lsgan_discriminator = create_lsgan()
lsgan_generator, lsgan_discriminator, lsgan_metrics = train_gan_model(
    lsgan_generator,
    lsgan_discriminator,
    train_loader,
    "LSGAN",
    loss_type="lsgan",
    num_epochs=10,
    learning_rate=2e-4,
)


# %%
# Cell 11: Visualize LSGAN Results
visualize_gan_samples(lsgan_generator, "lsgan_samples.png")


# %% [markdown]
"""
## Summary and Comparison

Let's compare the training dynamics and results of all four GAN variants.
"""


# %%
# Cell 12: Compare All Models
def compare_all_models():
    """Compare training curves and generate comparison grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Conditional GAN losses
    ax = axes[0, 0]
    ax.plot(cond_metrics["g_losses"], label="Generator", linewidth=2)
    ax.plot(cond_metrics["d_losses"], label="Discriminator", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Conditional GAN Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: WGAN-GP losses
    ax = axes[0, 1]
    ax.plot(wgan_metrics["g_losses"], label="Generator", linewidth=2)
    ax.plot(wgan_metrics["d_losses"], label="Critic", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("WGAN-GP Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: DCGAN losses
    ax = axes[1, 0]
    ax.plot(dcgan_metrics["g_losses"], label="Generator", linewidth=2)
    ax.plot(dcgan_metrics["d_losses"], label="Discriminator", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DCGAN Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: LSGAN losses
    ax = axes[1, 1]
    ax.plot(lsgan_metrics["g_losses"], label="Generator", linewidth=2)
    ax.plot(lsgan_metrics["d_losses"], label="Discriminator", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("LSGAN Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved training comparison to {OUTPUT_DIR}/training_comparison.png")

    # Generate sample comparison grid
    rngs = nnx.Rngs(999)
    num_samples = 8

    fig, axes = plt.subplots(4, num_samples, figsize=(16, 8))

    # Conditional GAN - one sample per class
    for i in range(num_samples):
        z = jax.random.normal(rngs.params(), (1, 100))
        label_onehot = jax.nn.one_hot(jnp.array([i]), 10)
        img = cond_generator(z, label_onehot)
        img = jnp.transpose(img, (0, 2, 3, 1))[0, :, :, 0]
        img = (img + 1) / 2
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Conditional", fontsize=10)
        axes[0, i].set_title(f"{i}", fontsize=9)

    # WGAN-GP
    z = jax.random.normal(rngs.params(), (num_samples, 100))
    wgan_samples = wgan_generator(z)
    wgan_samples = jnp.transpose(wgan_samples, (0, 2, 3, 1))
    for i in range(num_samples):
        img = wgan_samples[i, :, :, 0]
        img = (img + 1) / 2
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("WGAN-GP", fontsize=10)

    # DCGAN
    z = jax.random.normal(rngs.params(), (num_samples, 100))
    dcgan_samples = dcgan_generator(z)
    dcgan_samples = jnp.transpose(dcgan_samples, (0, 2, 3, 1))
    for i in range(num_samples):
        img = dcgan_samples[i, :, :, 0]
        img = (img + 1) / 2
        axes[2, i].imshow(img, cmap="gray")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("DCGAN", fontsize=10)

    # LSGAN
    z = jax.random.normal(rngs.params(), (num_samples, 100))
    lsgan_samples = lsgan_generator(z)
    lsgan_samples = jnp.transpose(lsgan_samples, (0, 2, 3, 1))
    for i in range(num_samples):
        img = lsgan_samples[i, :, :, 0]
        img = (img + 1) / 2
        axes[3, i].imshow(img, cmap="gray")
        axes[3, i].axis("off")
        if i == 0:
            axes[3, i].set_ylabel("LSGAN", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved model comparison to {OUTPUT_DIR}/model_comparison.png")


compare_all_models()


# %% [markdown]
"""
## Key Takeaways

### Model Comparison

1. **Conditional GAN**
   - Best for: Controlled generation with class labels
   - Stability: Moderate (depends on label embedding quality)
   - Use Artifex's: `ConditionalGenerator`, `ConditionalDiscriminator`
   - Loss: `vanilla_discriminator_loss`, `vanilla_generator_loss`

2. **WGAN-GP**
   - Best for: Stable training with meaningful metrics
   - Stability: Excellent (Wasserstein distance provides stable gradients)
   - Use Artifex's: `WGANGenerator`, `WGANDiscriminator`
   - Loss: `wasserstein_discriminator_loss`, `wasserstein_generator_loss`, `gradient_penalty`

3. **DCGAN**
   - Best for: Proven convolutional architecture
   - Stability: Good (batch normalization helps)
   - Use Artifex's: `DCGANGenerator`, `DCGANDiscriminator`
   - Loss: `vanilla_discriminator_loss`, `vanilla_generator_loss`

4. **LSGAN**
   - Best for: Reduced vanishing gradients
   - Stability: Good (least squares loss provides stable gradients)
   - Use Artifex's: `LSGANGenerator`, `LSGANDiscriminator`
   - Loss: `least_squares_discriminator_loss`, `least_squares_generator_loss`

### Artifex Features Demonstrated

- **Modular Architecture**: Using Generator and Discriminator classes independently
- **Loss Functions**: Leveraging Artifex's adversarial loss library
  - `vanilla_*_loss` for standard GAN training
  - `wasserstein_*_loss` for WGAN training
  - `least_squares_*_loss` for LSGAN training
  - `gradient_penalty` for WGAN-GP regularization
- **Configurable Components**: Setting hidden dimensions, batch normalization, dropout
- **Training Patterns**: Implementing alternating updates with n_critic
- **Multi-variant Comparison**: Evaluating different GAN architectures side-by-side

### Training Tips

- **Conditional GAN**: Use proper label embedding dimensions
- **WGAN-GP**: Train critic more frequently (n_critic=5), use lower learning rate
- **DCGAN**: Follow architectural guidelines (batch norm, strided convs)
- **LSGAN**: Standard GAN training applies, but with better stability

### Next Steps

1. Try different architectures (hidden dimensions)
2. Experiment with hyperparameters (learning rate, batch size)
3. Apply to higher-resolution datasets
4. Explore other Artifex GAN variants (StyleGAN, CycleGAN)
5. Add evaluation metrics (FID, IS)
"""

# %%
print()
print("=" * 70)
print("Advanced GAN Examples Complete!")
print("=" * 70)
print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - conditional_gan_samples.png")
print("  - wgan_gp_samples.png")
print("  - dcgan_samples.png")
print("  - lsgan_samples.png")
print("  - training_comparison.png")
print("  - model_comparison.png")
print("\nExperiment with Artifex's other GAN variants!")
print("See: artifex.generative_models.models.gan")
