# Gradient Accumulation

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.gradient_accumulation`

**Source:** `src/artifex/generative_models/training/gradient_accumulation.py`

`artifex.generative_models.training.gradient_accumulation` is the current owner for two shared utilities:

- `GradientAccumulator` for larger effective batch sizes across microbatches
- `DynamicLossScaler` for mixed-precision-friendly loss scaling and overflow handling

## GradientAccumulator

```python
from artifex.generative_models.training import (
    GradientAccumulator,
    GradientAccumulatorConfig,
)

accumulator = GradientAccumulator(
    GradientAccumulatorConfig(accumulation_steps=4, normalize_gradients=True)
)

for step, grads in enumerate(microbatch_gradients):
    accumulator.accumulate(grads)
    if accumulator.should_update(step):
        final_grads = accumulator.get_gradients()
        optimizer.update(final_grads)
```

## DynamicLossScaler

```python
from artifex.generative_models.training import DynamicLossScaler, DynamicLossScalerConfig

scaler = DynamicLossScaler(
    DynamicLossScalerConfig(initial_scale=2**15, growth_interval=2000)
)

scaled_loss = scaler.scale_loss(loss)
grads = compute_gradients(scaled_loss)
grads = scaler.unscale_gradients(grads)
overflow = scaler.check_overflow(grads)
scaler.update_scale(overflow)
```

Mixed precision currently lives through `DynamicLossScaler` on this shared utility module, not through a separate `artifex.generative_models.training.mixed_precision` module.
