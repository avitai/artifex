# GAN Trainer

**Module:** `artifex.generative_models.training.trainers.gan_trainer`

The GAN Trainer provides specialized training utilities for Generative Adversarial Networks, including multiple loss variants, gradient penalty regularization, and R1 regularization for stable training.

## Overview

GAN training requires careful balancing between generator and discriminator. The GAN Trainer handles this through:

- **Multiple Loss Types**: Vanilla, Wasserstein, Hinge, and Least Squares GAN
- **Gradient Penalty**: WGAN-GP regularization for stable Wasserstein training
- **R1 Regularization**: Gradient penalty on real data for improved stability
- **Label Smoothing**: One-sided smoothing to prevent overconfidence

## Quick Start

```python
from artifex.generative_models.training.trainers import (
    GANTrainer,
    GANTrainingConfig,
)
from flax import nnx
import optax
import jax

# Create models and optimizers
generator = create_generator(rngs=nnx.Rngs(0))
discriminator = create_discriminator(rngs=nnx.Rngs(1))

g_optimizer = nnx.Optimizer(generator, optax.adam(1e-4, b1=0.5), wrt=nnx.Param)
d_optimizer = nnx.Optimizer(discriminator, optax.adam(1e-4, b1=0.5), wrt=nnx.Param)

# Configure GAN training
config = GANTrainingConfig(
    loss_type="wasserstein",
    n_critic=5,
    gp_weight=10.0,
)

trainer = GANTrainer(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
    config=config,
)

# Training loop
key = jax.random.key(0)
latent_dim = 128

for step, batch in enumerate(train_loader):
    key, d_key, g_key, z_key = jax.random.split(key, 4)
    real = batch["image"]
    z = jax.random.normal(z_key, (real.shape[0], latent_dim))

    # Train discriminator
    d_loss, d_metrics = trainer.discriminator_step(real, z, d_key)

    # Train generator every n_critic steps
    if step % config.n_critic == 0:
        z = jax.random.normal(z_key, (real.shape[0], latent_dim))
        g_loss, g_metrics = trainer.generator_step(z)
```

## Configuration

::: artifex.generative_models.training.trainers.gan_trainer.GANTrainingConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_type` | `str` | `"vanilla"` | Loss variant: `"vanilla"`, `"wasserstein"`, `"hinge"`, `"lsgan"` |
| `n_critic` | `int` | `1` | Discriminator updates per generator update |
| `gp_weight` | `float` | `10.0` | Gradient penalty weight (WGAN-GP) |
| `gp_target` | `float` | `1.0` | Target gradient norm for GP |
| `r1_weight` | `float` | `0.0` | R1 regularization weight |
| `label_smoothing` | `float` | `0.0` | One-sided label smoothing for real labels |

## Loss Types

### Vanilla GAN (Non-Saturating)

Standard GAN with non-saturating generator loss for numerical stability:

```python
config = GANTrainingConfig(loss_type="vanilla")
# Uses log(1 - sigmoid) form with softplus for stability
```

### Wasserstein GAN

Earth Mover's distance with gradient penalty:

```python
config = GANTrainingConfig(
    loss_type="wasserstein",
    gp_weight=10.0,  # Required for WGAN-GP
    n_critic=5,      # More D updates per G update
)
```

### Hinge Loss

Hinge loss used in BigGAN and StyleGAN2:

```python
config = GANTrainingConfig(loss_type="hinge")
# D loss: relu(1 - D(real)) + relu(1 + D(fake))
# G loss: -D(fake)
```

### Least Squares GAN

Mean squared error between predictions and targets:

```python
config = GANTrainingConfig(loss_type="lsgan")
# More stable gradients than vanilla GAN
```

## Regularization Techniques

### Gradient Penalty (WGAN-GP)

Enforces 1-Lipschitz constraint via gradient penalty on interpolated samples:

```python
config = GANTrainingConfig(
    loss_type="wasserstein",
    gp_weight=10.0,
    gp_target=1.0,  # Target gradient norm
)
```

The gradient penalty is computed as:

$$\lambda \mathbb{E}_{\hat{x}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]$$

where $\hat{x}$ is interpolated between real and fake samples.

### R1 Regularization

Gradient penalty on real data only, used in StyleGAN:

```python
config = GANTrainingConfig(
    loss_type="hinge",
    r1_weight=10.0,  # R1 penalty weight
)
```

R1 penalty is computed as:

$$\frac{\gamma}{2} \mathbb{E}_{x \sim p_{data}}[||\nabla_x D(x)||_2^2]$$

### Label Smoothing

One-sided label smoothing to prevent discriminator overconfidence:

```python
config = GANTrainingConfig(
    loss_type="vanilla",
    label_smoothing=0.1,  # Real labels: 0.9 instead of 1.0
)
```

## API Reference

::: artifex.generative_models.training.trainers.gan_trainer.GANTrainer
    options:
      show_root_heading: true
      members_order: source

## Training Patterns

### Standard GAN Training

```python
for step, batch in enumerate(train_loader):
    key, subkey = jax.random.split(key)
    real = batch["image"]
    z = jax.random.normal(subkey, (batch_size, latent_dim))

    # Train discriminator
    d_loss, d_metrics = trainer.discriminator_step(real, z, subkey)

    # Train generator (every step for vanilla/hinge/lsgan)
    g_loss, g_metrics = trainer.generator_step(z)
```

### WGAN Training (Multiple D Updates)

```python
for step, batch in enumerate(train_loader):
    key, d_key, g_key = jax.random.split(key, 3)
    real = batch["image"]
    z = jax.random.normal(d_key, (batch_size, latent_dim))

    # Multiple discriminator updates
    for _ in range(config.n_critic):
        d_loss, d_metrics = trainer.discriminator_step(real, z, d_key)

    # Single generator update
    z = jax.random.normal(g_key, (batch_size, latent_dim))
    g_loss, g_metrics = trainer.generator_step(z)
```

### Progressive Training

For high-resolution generation, progressively grow resolution:

```python
resolutions = [4, 8, 16, 32, 64, 128]

for resolution in resolutions:
    # Update model for this resolution
    generator.grow_layer()
    discriminator.grow_layer()

    # Train at this resolution
    for step in range(steps_per_resolution):
        # ... training step ...
```

## Model Requirements

### Generator Interface

```python
class Generator(nnx.Module):
    def __call__(self, z: jax.Array) -> jax.Array:
        """Generate images from latent vectors.

        Args:
            z: Latent vectors, shape (batch, latent_dim).

        Returns:
            Generated images, shape (batch, H, W, C).
        """
        ...
```

### Discriminator Interface

```python
class Discriminator(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        """Classify real/fake images.

        Args:
            x: Images, shape (batch, H, W, C).

        Returns:
            Logits (unbounded scores), shape (batch,) or (batch, 1).
        """
        ...
```

## Training Metrics

### Discriminator Metrics

| Metric | Description |
|--------|-------------|
| `d_loss` | Base discriminator loss |
| `d_loss_total` | Total loss including regularization |
| `d_real` | Mean discriminator output on real samples |
| `d_fake` | Mean discriminator output on fake samples |
| `gp_loss` | Gradient penalty loss (if enabled) |
| `r1_loss` | R1 regularization loss (if enabled) |

### Generator Metrics

| Metric | Description |
|--------|-------------|
| `g_loss` | Generator loss |
| `d_fake_g` | Mean discriminator output on generated samples |

## Loss Functions

The GAN Trainer uses loss functions from `artifex.generative_models.core.losses.adversarial`:

```python
from artifex.generative_models.core.losses import (
    # Vanilla GAN (non-saturating)
    ns_vanilla_generator_loss,
    ns_vanilla_discriminator_loss,
    # Wasserstein
    wasserstein_generator_loss,
    wasserstein_discriminator_loss,
    # Hinge
    hinge_generator_loss,
    hinge_discriminator_loss,
    # Least Squares
    least_squares_generator_loss,
    least_squares_discriminator_loss,
)
```

## References

- [WGAN-GP: Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [Hinge Loss for GANs (SAGAN)](https://arxiv.org/abs/1802.05957)
- [R1 Regularization (StyleGAN)](https://arxiv.org/abs/1801.04406)
- [LSGAN: Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
