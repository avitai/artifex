# Flow Trainer

**Module:** `artifex.generative_models.training.trainers.flow_trainer`

The Flow Trainer provides specialized training utilities for flow matching models, including Conditional Flow Matching (CFM), Optimal Transport CFM (OT-CFM), and various time sampling strategies.

## Overview

Flow matching enables simulation-free training of continuous normalizing flows. The Flow Trainer provides:

- **Flow Types**: Standard CFM, OT-CFM, and Rectified Flow
- **Time Sampling**: Uniform, logit-normal, and U-shaped strategies
- **Linear Interpolation**: Straight paths from noise to data
- **Minimal Noise**: Configurable sigma_min for path endpoints

## Quick Start

```python
from artifex.generative_models.training.trainers import (
    FlowTrainer,
    FlowTrainingConfig,
)
from flax import nnx
import optax
import jax

# Create model and optimizer
model = create_flow_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Configure flow matching training
config = FlowTrainingConfig(
    flow_type="cfm",
    time_sampling="logit_normal",
    sigma_min=0.001,
)

trainer = FlowTrainer(model, optimizer, config)

# Training loop
key = jax.random.key(0)

for step, batch in enumerate(train_loader):
    key, subkey = jax.random.split(key)
    loss, metrics = trainer.train_step(batch, subkey)

    if step % 100 == 0:
        print(f"Step {step}: loss={metrics['loss']:.4f}")
```

## Configuration

::: artifex.generative_models.training.trainers.flow_trainer.FlowTrainingConfig
    options:
      show_root_heading: true
      members_order: source

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flow_type` | `str` | `"cfm"` | Flow type: `"cfm"`, `"ot_cfm"`, `"rectified_flow"` |
| `time_sampling` | `str` | `"uniform"` | Time distribution: `"uniform"`, `"logit_normal"`, `"u_shaped"` |
| `sigma_min` | `float` | `0.001` | Minimum noise level for paths |
| `use_ot` | `bool` | `False` | Enable optimal transport coupling |
| `ot_regularization` | `float` | `0.01` | Sinkhorn regularization for OT |
| `logit_normal_loc` | `float` | `0.0` | Logit-normal location parameter |
| `logit_normal_scale` | `float` | `1.0` | Logit-normal scale parameter |

## Flow Types

### Conditional Flow Matching (CFM)

Standard CFM with linear interpolation paths:

```python
config = FlowTrainingConfig(flow_type="cfm")
# Learns velocity field: v(x_t, t) = x_1 - x_0
```

The interpolation path is defined as:

$$x_t = (1 - t) x_0 + t x_1$$

where $x_0$ is noise and $x_1$ is data.

### Optimal Transport CFM (OT-CFM)

CFM with optimal transport coupling for straighter paths:

```python
config = FlowTrainingConfig(
    flow_type="ot_cfm",
    use_ot=True,
    ot_regularization=0.01,
)
# Uses minibatch OT to pair noise and data samples
```

### Rectified Flow

Straighten paths through reflow iterations:

```python
config = FlowTrainingConfig(flow_type="rectified_flow")
# Single reflow iteration typically sufficient
```

## Time Sampling Strategies

### Uniform Sampling

Standard uniform sampling in [0, 1]:

```python
config = FlowTrainingConfig(time_sampling="uniform")
# Equal probability across all time values
```

### Logit-Normal Sampling

Favors middle time values for improved convergence:

```python
config = FlowTrainingConfig(
    time_sampling="logit_normal",
    logit_normal_loc=0.0,
    logit_normal_scale=1.0,
)
```

### U-Shaped Sampling

Favors endpoints (t=0 and t=1), useful for rectified flows:

```python
config = FlowTrainingConfig(time_sampling="u_shaped")
# More samples near 0 and 1 where endpoint behavior is critical
```

U-shaped sampling is computed as:

$$t = \sin^2(\pi u / 2)$$

where $u \sim \text{Uniform}(0, 1)$.

## API Reference

::: artifex.generative_models.training.trainers.flow_trainer.FlowTrainer
    options:
      show_root_heading: true
      members_order: source

## Flow Matching Theory

Flow matching learns a velocity field $v_\theta(x_t, t)$ that transports samples from noise distribution to data distribution.

### Training Objective

The CFM loss is:

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \|v_\theta(x_t, t) - u_t\|^2$$

where:

- $x_0 \sim \mathcal{N}(0, I)$ (source noise)
- $x_1 \sim p_{\text{data}}$ (target data)
- $x_t = (1-t) x_0 + t x_1$ (interpolated point)
- $u_t = x_1 - x_0$ (target velocity)

### Sampling

Generate samples by solving the ODE from t=0 to t=1:

$$\frac{dx}{dt} = v_\theta(x, t)$$

## Integration with Base Trainer

Use `create_loss_fn()` for integration with callbacks and checkpointing:

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.trainers import FlowTrainer, FlowTrainingConfig
from artifex.generative_models.training.callbacks import (
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
)

# Create flow trainer
flow_config = FlowTrainingConfig(
    flow_type="cfm",
    time_sampling="logit_normal",
)
flow_trainer = FlowTrainer(model, optimizer, flow_config)

# Get loss function for base Trainer
loss_fn = flow_trainer.create_loss_fn()

# Use with callbacks
callbacks = [
    EarlyStopping(EarlyStoppingConfig(monitor="loss", patience=10)),
    ModelCheckpoint(CheckpointConfig(dirpath="checkpoints", monitor="loss")),
]
```

## Model Requirements

The Flow Trainer expects models with the following interface:

```python
class FlowModel(nnx.Module):
    def __call__(
        self,
        x_t: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Predict velocity at (x_t, t).

        Args:
            x_t: Points along flow path, shape (batch, ...).
            t: Time values in [0, 1], shape (batch,).

        Returns:
            Predicted velocity field, shape (batch, ...).
        """
        ...
```

## Training Metrics

| Metric | Description |
|--------|-------------|
| `loss` | MSE between predicted and target velocity |

## Recommended Configurations

### Standard CFM Training

```python
config = FlowTrainingConfig(
    flow_type="cfm",
    time_sampling="uniform",
    sigma_min=0.001,
)
```

### High-Quality Generation

```python
config = FlowTrainingConfig(
    flow_type="cfm",
    time_sampling="logit_normal",
    logit_normal_loc=0.0,
    logit_normal_scale=1.0,
)
```

### Rectified Flow

```python
config = FlowTrainingConfig(
    flow_type="rectified_flow",
    time_sampling="u_shaped",
)
```

## Sampling from Trained Models

After training, generate samples using ODE integration:

```python
from jax.experimental.ode import odeint
import jax.numpy as jnp

def sample(model, shape, key, num_steps=100):
    """Generate samples from trained flow model."""
    # Start from noise
    x_0 = jax.random.normal(key, shape)

    # Define ODE function
    def velocity_fn(x, t):
        t_batch = jnp.full((x.shape[0],), t)
        return model(x, t_batch)

    # Integrate from t=0 to t=1
    ts = jnp.linspace(0, 1, num_steps)
    trajectory = odeint(velocity_fn, x_0, ts)

    # Return final sample at t=1
    return trajectory[-1]

# Generate samples
samples = sample(model, (batch_size, *data_shape), key)
```

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [OT-CFM: Improving and Simplifying Flow Matching](https://arxiv.org/abs/2302.00482)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
