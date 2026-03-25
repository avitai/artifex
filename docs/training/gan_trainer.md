# GAN Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.trainers.gan_trainer`

**Source:** `src/artifex/generative_models/training/trainers/gan_trainer.py`

`GANTrainer` is a stateless step helper. It owns GAN-specific loss logic, but
the caller owns generator/discriminator state, optimizers, and update cadence.

## Quick Start

```python
from flax import nnx
import jax
import optax

from artifex.generative_models.training.trainers import (
    GANTrainer,
    GANTrainingConfig,
)

generator = create_generator(rngs=nnx.Rngs(0))
discriminator = create_discriminator(rngs=nnx.Rngs(1))
g_optimizer = nnx.Optimizer(generator, optax.adam(1e-4, b1=0.5), wrt=nnx.Param)
d_optimizer = nnx.Optimizer(discriminator, optax.adam(1e-4, b1=0.5), wrt=nnx.Param)

config = GANTrainingConfig(loss_type="wasserstein", gp_weight=10.0)
trainer = GANTrainer(config)

key = jax.random.key(0)
key, d_key, g_key, z_key = jax.random.split(key, 4)
real = batch["image"]
z = jax.random.normal(z_key, (real.shape[0], latent_dim))

d_loss, d_metrics = trainer.discriminator_step(
    generator,
    discriminator,
    d_optimizer,
    real,
    z,
    d_key,
)
g_loss, g_metrics = trainer.generator_step(
    generator,
    discriminator,
    g_optimizer,
    z,
)
```

## Configuration

::: artifex.generative_models.training.trainers.gan_trainer.GANTrainingConfig
    options:
      show_root_heading: true
      members_order: source

### Runtime-Active Fields

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_type` | `"vanilla"` | GAN loss family |
| `gp_weight` | `10.0` | Gradient-penalty weight |
| `gp_target` | `1.0` | Target gradient norm for GP |
| `r1_weight` | `0.0` | R1 regularization weight |
| `label_smoothing` | `0.0` | One-sided smoothing for real labels |

## Update Cadence

Artifex does not hide GAN scheduling inside `GANTrainingConfig`. If you want
multiple discriminator steps per generator step, make that explicit in the loop:

```python
for _ in range(5):
    d_loss, d_metrics = trainer.discriminator_step(
        generator,
        discriminator,
        d_optimizer,
        real,
        z,
        d_key,
    )

g_loss, g_metrics = trainer.generator_step(
    generator,
    discriminator,
    g_optimizer,
    z,
)
```

## Supported Loss Families

- `vanilla`
- `wasserstein`
- `hinge`
- `lsgan`

Use `gp_weight` for Wasserstein-style gradient penalties and `r1_weight` when
you want R1 regularization on real data.

## Related Documentation

- [Training Systems](index.md)
- [GAN Models](../user-guide/models/gan-guide.md)
