# Diffusion Trainer

**Module:** `artifex.generative_models.training.trainers.diffusion_trainer`

The Diffusion Trainer provides state-of-the-art training utilities for diffusion models, including multiple prediction types, advanced timestep sampling strategies, loss weighting schemes, and EMA model updates.

## Overview

Modern diffusion model training benefits from several advanced techniques. The Diffusion Trainer provides:

- **Prediction Types**: Epsilon, v-prediction, and x-prediction
- **Timestep Sampling**: Uniform, logit-normal, and mode-seeking strategies
- **Loss Weighting**: Uniform, SNR, min-SNR, and EDM-style weighting
- **EMA Updates**: Exponential moving average for stable inference

## Quick Start

```python
from artifex.generative_models.training.trainers import (
    DiffusionTrainer,
    DiffusionTrainingConfig,
)
from artifex.generative_models.core.noise_schedules import CosineNoiseSchedule
from flax import nnx
import optax
import jax

# Create model and optimizer
model = create_diffusion_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
noise_schedule = CosineNoiseSchedule(num_timesteps=1000)

# Configure diffusion training with SOTA techniques
config = DiffusionTrainingConfig(
    prediction_type="v_prediction",
    timestep_sampling="logit_normal",
    loss_weighting="min_snr",
    snr_gamma=5.0,
)

trainer = DiffusionTrainer(model, optimizer, noise_schedule, config)

# Training loop
key = jax.random.key(0)

for step, batch in enumerate(train_loader):
    key, subkey = jax.random.split(key)
    loss, metrics = trainer.train_step(batch, subkey)

    if step % 100 == 0:
        print(f"Step {step}: loss={metrics['loss']:.4f}")
```

## Configuration

::: artifex.generative_models.training.trainers.diffusion_trainer.DiffusionTrainingConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_type` | `str` | `"epsilon"` | Model prediction: `"epsilon"`, `"v_prediction"`, `"x_start"` |
| `timestep_sampling` | `str` | `"uniform"` | Timestep distribution: `"uniform"`, `"logit_normal"`, `"mode"` |
| `loss_weighting` | `str` | `"uniform"` | Loss weighting: `"uniform"`, `"snr"`, `"min_snr"`, `"edm"` |
| `snr_gamma` | `float` | `5.0` | Gamma for min-SNR weighting |
| `logit_normal_loc` | `float` | `-0.5` | Logit-normal location parameter |
| `logit_normal_scale` | `float` | `1.0` | Logit-normal scale parameter |
| `ema_decay` | `float` | `0.9999` | EMA decay rate |
| `ema_update_every` | `int` | `10` | Steps between EMA updates |

## Prediction Types

### Epsilon Prediction (DDPM)

The classic approach - model predicts the noise added:

```python
config = DiffusionTrainingConfig(prediction_type="epsilon")
# Target: noise that was added to x_0
```

### V-Prediction

Model predicts v = sqrt(alpha) *noise - sqrt(1-alpha)* x_0:

```python
config = DiffusionTrainingConfig(prediction_type="v_prediction")
# Provides more consistent gradients across timesteps
# Used in Stable Diffusion 3 and Imagen Video
```

V-prediction often leads to faster convergence and better sample quality.

### X-Start Prediction

Model directly predicts the clean data:

```python
config = DiffusionTrainingConfig(prediction_type="x_start")
# Target: original clean data x_0
```

## Timestep Sampling Strategies

### Uniform Sampling

Standard uniform sampling over all timesteps:

```python
config = DiffusionTrainingConfig(timestep_sampling="uniform")
# Equal probability for all timesteps
```

### Logit-Normal Sampling

Favors middle timesteps where learning is most effective:

```python
config = DiffusionTrainingConfig(
    timestep_sampling="logit_normal",
    logit_normal_loc=-0.5,
    logit_normal_scale=1.0,
)
# Used in Stable Diffusion 3 for improved convergence
```

### Mode-Seeking Sampling

Favors high-noise timesteps for improved generation quality:

```python
config = DiffusionTrainingConfig(timestep_sampling="mode")
# Quadratic bias toward larger timesteps
```

## Loss Weighting Schemes

### Uniform Weighting

No weighting - all timesteps contribute equally:

```python
config = DiffusionTrainingConfig(loss_weighting="uniform")
```

### SNR Weighting

Weight by signal-to-noise ratio:

```python
config = DiffusionTrainingConfig(loss_weighting="snr")
# weight = alpha / (1 - alpha)
```

### Min-SNR-Gamma Weighting

Clips high SNR weights for 3.4x faster convergence:

```python
config = DiffusionTrainingConfig(
    loss_weighting="min_snr",
    snr_gamma=5.0,
)
# weight = min(SNR, gamma) / SNR
# Down-weights low-noise timesteps where SNR is high
```

Min-SNR-gamma is the recommended weighting scheme for most use cases.

### EDM Weighting

EDM-style weighting based on sigma:

```python
config = DiffusionTrainingConfig(loss_weighting="edm")
# weight = 1 / (sigma^2 + 1)
```

## EMA (Exponential Moving Average)

Maintain an EMA of model parameters for stable inference:

```python
config = DiffusionTrainingConfig(
    ema_decay=0.9999,
    ema_update_every=10,
)

# After training, get EMA parameters
ema_params = trainer.get_ema_params()

# Apply EMA params to model for inference
from flax import nnx
nnx.update(model, ema_params)
```

## API Reference

::: artifex.generative_models.training.trainers.diffusion_trainer.DiffusionTrainer
    options:
      show_root_heading: true
      members_order: source

## Noise Schedule Protocol

The trainer works with any noise schedule implementing the `NoiseScheduleProtocol`:

```python
from typing import Protocol
import jax

class NoiseScheduleProtocol(Protocol):
    """Protocol for noise schedules used by diffusion trainers."""

    num_timesteps: int
    alphas_cumprod: jax.Array

    def add_noise(
        self,
        x: jax.Array,
        noise: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Add noise to data at given timesteps."""
        ...
```

Artifex provides several noise schedule implementations:

```python
from artifex.generative_models.core.noise_schedules import (
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SquaredCosineNoiseSchedule,
)

# Linear schedule (DDPM default)
schedule = LinearNoiseSchedule(num_timesteps=1000)

# Cosine schedule (improved for images)
schedule = CosineNoiseSchedule(num_timesteps=1000)
```

## Integration with Base Trainer

Use `create_loss_fn()` for integration with callbacks and checkpointing:

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.trainers import (
    DiffusionTrainer,
    DiffusionTrainingConfig,
)
from artifex.generative_models.training.callbacks import (
    ModelCheckpoint,
    CheckpointConfig,
)

# Create diffusion trainer
diff_config = DiffusionTrainingConfig(
    prediction_type="v_prediction",
    loss_weighting="min_snr",
)
diff_trainer = DiffusionTrainer(model, optimizer, noise_schedule, diff_config)

# Get loss function for base Trainer
loss_fn = diff_trainer.create_loss_fn()

# Use with base Trainer for callbacks
callbacks = [
    ModelCheckpoint(CheckpointConfig(dirpath="checkpoints", monitor="loss")),
]
```

## Model Requirements

The Diffusion Trainer expects models with the following interface:

```python
class DiffusionModel(nnx.Module):
    def __call__(
        self,
        x_noisy: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Predict noise/v/x_0 from noisy input.

        Args:
            x_noisy: Noisy data, shape (batch, ...).
            t: Integer timesteps, shape (batch,).

        Returns:
            Prediction matching prediction_type, shape (batch, ...).
        """
        ...
```

## Training Metrics

| Metric | Description |
|--------|-------------|
| `loss` | Weighted loss (affected by loss_weighting) |
| `loss_unweighted` | Raw MSE loss without weighting |

## Recommended Configurations

### High-Quality Image Generation

```python
config = DiffusionTrainingConfig(
    prediction_type="v_prediction",
    timestep_sampling="logit_normal",
    loss_weighting="min_snr",
    snr_gamma=5.0,
    ema_decay=0.9999,
)
```

### Fast Training

```python
config = DiffusionTrainingConfig(
    prediction_type="epsilon",
    timestep_sampling="uniform",
    loss_weighting="min_snr",
    snr_gamma=5.0,
)
```

### Large Models (EDM-style)

```python
config = DiffusionTrainingConfig(
    prediction_type="epsilon",
    timestep_sampling="logit_normal",
    loss_weighting="edm",
    ema_decay=0.9999,
    ema_update_every=1,
)
```

## References

- [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [V-Prediction: Progressive Distillation](https://arxiv.org/abs/2202.00512)
- [Min-SNR Weighting](https://arxiv.org/abs/2303.09556)
- [EDM: Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
