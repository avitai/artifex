# VAE Trainer

**Module:** `artifex.generative_models.training.trainers.vae_trainer`

The VAE Trainer provides specialized training utilities for Variational Autoencoders, including KL divergence annealing schedules, beta-VAE weighting for disentanglement, and free bits constraints to prevent posterior collapse.

## Overview

Training VAEs requires balancing reconstruction quality against latent space regularization. The VAE Trainer handles this balance through:

- **KL Annealing**: Gradual increase of KL weight to prevent posterior collapse
- **Beta-VAE Weighting**: Control disentanglement vs reconstruction trade-off
- **Free Bits Constraint**: Minimum KL per dimension to ensure information flow

## Quick Start

```python
from artifex.generative_models.training.trainers import (
    VAETrainer,
    VAETrainingConfig,
)
from flax import nnx
import optax

# Create model and optimizer
model = create_vae_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Configure VAE-specific training
config = VAETrainingConfig(
    kl_annealing="cyclical",
    kl_warmup_steps=5000,
    beta=4.0,
    free_bits=0.5,
)

trainer = VAETrainer(model, optimizer, config)

# Training loop
for step, batch in enumerate(train_loader):
    loss, metrics = trainer.train_step(batch, step=step)
    if step % 100 == 0:
        print(f"Step {step}: loss={metrics['loss']:.4f}, "
              f"recon={metrics['recon_loss']:.4f}, kl={metrics['kl_loss']:.4f}")
```

## Configuration

::: artifex.generative_models.training.trainers.vae_trainer.VAETrainingConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kl_annealing` | `str` | `"linear"` | KL weight schedule: `"none"`, `"linear"`, `"sigmoid"`, `"cyclical"` |
| `kl_warmup_steps` | `int` | `10000` | Steps to reach full KL weight |
| `beta` | `float` | `1.0` | Final KL weight (beta-VAE parameter) |
| `free_bits` | `float` | `0.0` | Minimum KL per latent dimension |
| `cyclical_period` | `int` | `10000` | Period for cyclical annealing |

## KL Annealing Schedules

### None (Constant)

No annealing - use full beta weight from the start:

```python
config = VAETrainingConfig(kl_annealing="none", beta=1.0)
# KL weight = 1.0 at all steps
```

### Linear Warmup

Linearly increase KL weight from 0 to beta:

```python
config = VAETrainingConfig(
    kl_annealing="linear",
    kl_warmup_steps=10000,
    beta=1.0,
)
# KL weight = beta * min(1.0, step / warmup_steps)
```

### Sigmoid Warmup

S-shaped warmup curve centered at half the warmup steps:

```python
config = VAETrainingConfig(
    kl_annealing="sigmoid",
    kl_warmup_steps=10000,
    beta=1.0,
)
```

### Cyclical Annealing

Periodically reset KL weight to encourage information flow:

```python
config = VAETrainingConfig(
    kl_annealing="cyclical",
    cyclical_period=5000,
    beta=4.0,
)
# KL weight cycles: 0 -> beta -> 0 -> beta -> ...
```

Cyclical annealing helps prevent posterior collapse by periodically "reopening" information pathways.

## Beta-VAE Training

Higher beta values encourage disentangled representations at the cost of reconstruction quality:

```python
# Standard VAE (beta=1)
standard_config = VAETrainingConfig(beta=1.0)

# Beta-VAE for disentanglement (beta=4)
disentangled_config = VAETrainingConfig(beta=4.0)

# Strong regularization (beta=10)
strong_reg_config = VAETrainingConfig(beta=10.0)
```

## Free Bits Constraint

Prevent posterior collapse by ensuring minimum KL per latent dimension:

```python
config = VAETrainingConfig(
    free_bits=0.5,  # Minimum 0.5 nats per dimension
    beta=1.0,
)
```

The free bits constraint ensures each latent dimension carries at least the specified amount of information.

## API Reference

::: artifex.generative_models.training.trainers.vae_trainer.VAETrainer
    options:
      show_root_heading: true
      members_order: source

## Integration with Base Trainer

The VAE Trainer provides a `create_loss_fn()` method for seamless integration with the base Trainer's callbacks, checkpointing, and logging infrastructure:

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.trainers import VAETrainer, VAETrainingConfig
from artifex.generative_models.training.callbacks import (
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
)

# Create VAE-specific trainer
vae_config = VAETrainingConfig(kl_annealing="cyclical", beta=4.0)
vae_trainer = VAETrainer(model, optimizer, vae_config)

# Create loss function for a specific training step
# Note: step is required for KL annealing
def make_loss_fn(step: int):
    return vae_trainer.create_loss_fn(step=step)

# Use with base Trainer for callbacks
callbacks = [
    EarlyStopping(EarlyStoppingConfig(monitor="val_loss", patience=10)),
    ModelCheckpoint(CheckpointConfig(dirpath="checkpoints", monitor="val_loss")),
]
```

## Model Requirements

The VAE Trainer expects models with the following interface:

```python
class VAEModel(nnx.Module):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass returning (reconstruction, mean, logvar).

        Args:
            x: Input data, shape (batch, ...).

        Returns:
            Tuple of:
                - recon_x: Reconstructed data, shape (batch, ...)
                - mean: Latent mean, shape (batch, latent_dim)
                - logvar: Latent log-variance, shape (batch, latent_dim)
        """
        ...
```

## Reconstruction Loss Types

The trainer supports MSE and BCE reconstruction losses:

```python
# Mean Squared Error (default, for continuous data)
loss, metrics = trainer.train_step(batch, step=100, loss_type="mse")

# Binary Cross-Entropy (for images normalized to [0, 1])
loss, metrics = trainer.train_step(batch, step=100, loss_type="bce")
```

## Training Metrics

The trainer returns detailed metrics for monitoring:

| Metric | Description |
|--------|-------------|
| `loss` | Total ELBO loss |
| `recon_loss` | Reconstruction loss |
| `kl_loss` | KL divergence (unweighted) |
| `kl_weight` | Current KL weight from annealing |

## References

- [Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
- [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](https://arxiv.org/abs/1903.10145)
- [Free Bits for VAEs](https://arxiv.org/abs/1606.04934)
