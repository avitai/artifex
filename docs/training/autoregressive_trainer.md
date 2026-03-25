# Autoregressive Trainer

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.trainers.autoregressive_trainer`

**Source:** `src/artifex/generative_models/training/trainers/autoregressive_trainer.py`

`AutoregressiveTrainer` is the retained sequence-training owner for teacher forcing, scheduled sampling, label smoothing, and causal or padding masks. The caller still owns the model, optimizer, and outer loop.

## Quick Start

```python
from flax import nnx
import jax
import optax

from artifex.generative_models.training.trainers import (
    AutoregressiveTrainer,
    AutoregressiveTrainingConfig,
)

model = TransformerModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
trainer = AutoregressiveTrainer(
    AutoregressiveTrainingConfig(
        use_teacher_forcing=True,
        scheduled_sampling="linear",
        label_smoothing=0.1,
        pad_token_id=0,
    )
)

key = jax.random.key(0)
loss, metrics = trainer.train_step(model, optimizer, batch, step=10, key=key)
```

## JIT-Friendly Step Boundary

The trainer keeps model state and optimizer state explicit in `train_step(...)`, so the step can be wrapped by `nnx.jit` in the caller when that is appropriate.

```python
jit_step = nnx.jit(trainer.train_step)
loss, metrics = jit_step(model, optimizer, batch, step=10, key=key)
```

## Mask Helpers

The module also exports three mask helpers:

- `create_causal_mask(seq_length)`
- `create_padding_mask(tokens, pad_token_id)`
- `create_combined_mask(tokens, pad_token_id=None)`
