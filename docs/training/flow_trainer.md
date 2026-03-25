# Flow Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.trainers.flow_trainer`

**Source:** `src/artifex/generative_models/training/trainers/flow_trainer.py`

`FlowTrainer` implements the flow-matching runtime that Artifex actually ships:
linear Gaussian-noise-to-data interpolation plus configurable time sampling.

## Quick Start

```python
from flax import nnx
import jax
import optax

from artifex.generative_models.training.trainers import (
    FlowTrainer,
    FlowTrainingConfig,
)

model = create_flow_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
trainer = FlowTrainer(
    FlowTrainingConfig(
        time_sampling="logit_normal",
        logit_normal_loc=0.0,
        logit_normal_scale=1.0,
    )
)

key = jax.random.key(0)
loss, metrics = trainer.train_step(model, optimizer, batch, key)
```

## Configuration

::: artifex.generative_models.training.trainers.flow_trainer.FlowTrainingConfig
    options:
      show_root_heading: true
      members_order: source

### Runtime-Active Fields

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_sampling` | `"uniform"` | Time distribution used for interpolation samples |
| `logit_normal_loc` | `0.0` | Mean of the latent normal before the logistic transform |
| `logit_normal_scale` | `1.0` | Scale of the latent normal before the logistic transform |

## Time Sampling Strategies

### Uniform

```python
FlowTrainingConfig(time_sampling="uniform")
```

### Logit-Normal

```python
FlowTrainingConfig(
    time_sampling="logit_normal",
    logit_normal_loc=0.0,
    logit_normal_scale=1.0,
)
```

### U-Shaped

```python
FlowTrainingConfig(time_sampling="u_shaped")
```

## Objective

The trainer uses the linear interpolation path

$$x_t = (1 - t)x_0 + tx_1$$

with target velocity

$$u_t = x_1 - x_0$$

and minimizes mean-squared error between the model prediction and `u_t`.

## Shared Trainer Integration

`FlowTrainer` can also provide a step-aware objective for the shared `Trainer`:

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.callbacks import CallbackList

flow_trainer = FlowTrainer(FlowTrainingConfig(time_sampling="logit_normal"))

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=flow_trainer.create_loss_fn(),
    callbacks=CallbackList([]),
)
```

## Model Contract

The model is expected to implement:

```python
class FlowModel(nnx.Module):
    def __call__(self, x_t, t):
        ...
```

where `x_t` matches the sample shape and `t` is a `(batch,)` tensor of sampled
times.

## Related Documentation

- [Training Systems](index.md)
- [Shared Trainer](../api/training/trainer.md)
