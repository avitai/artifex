# Advanced Training Features

This guide covers two techniques for training under memory and precision
limits: accumulating gradients across microbatches for a larger effective batch,
and dynamic loss scaling for float16 or bfloat16 training. Artifex ships neither
as its own class. Both come from the libraries every artifex trainer already
builds on, and the trainers accept them through the optimizer and the loss
function they are given.

## Overview

- **Gradient accumulation**: `optax.MultiSteps` wraps any optax optimizer so it
  accumulates `k` microbatch gradients and applies one averaged update on the
  `k`-th call. Wrapped in `nnx.Optimizer`, it drops into every artifex trainer
  unchanged.
- **Dynamic loss scaling**: `flax.training.dynamic_scale.DynamicScale` scales the
  loss before differentiation, returns unscaled gradients with a finiteness flag,
  and adapts the scale after overflows and after runs of finite steps.

## Gradient Accumulation

### Why Use Gradient Accumulation?

When training large models or using high-resolution inputs, GPU memory often
limits the batch size you can use. Gradient accumulation solves this by:

1. Running multiple forward/backward passes with smaller batches
2. Accumulating the gradients from each pass
3. Applying a single optimizer update with the accumulated gradients

Effective batch size = `micro_batch_size * accumulation_steps`.

### Basic Usage

```python
import optax
from flax import nnx

accumulation_steps = 4
optimizer = nnx.Optimizer(
    model,
    optax.MultiSteps(optax.adam(1e-3), every_k_schedule=accumulation_steps),
    wrt=nnx.Param,
)
```

`MultiSteps` keeps the running sum inside the optimizer state. The first `k - 1`
calls to `optimizer.update` apply a zero update; the `k`-th applies the inner
optimizer to the mean of the `k` gradients. The repository test
`test_gradient_accumulation.py` pins that `k` microbatches through `MultiSteps`
equal one Adam step on the averaged gradient.

`every_k_schedule` also accepts a function of the step, so the accumulation
window can change over training; `use_grad_mean=False` sums instead of averaging.

### Training Loop Integration

Because accumulation lives in the optimizer, the training loop is the plain one:

```python
import optax
from flax import nnx


def train_with_accumulation(model, train_loader, num_epochs, accumulation_steps=4, learning_rate=1e-3):
    """Training with gradient accumulation."""
    optimizer = nnx.Optimizer(
        model,
        optax.MultiSteps(optax.adam(learning_rate), every_k_schedule=accumulation_steps),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            outputs = model(batch["images"], training=True)
            loss_dict = model.loss_fn(batch, outputs)
            return loss_dict["total_loss"], loss_dict

        (loss, loss_dict), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return loss

    for _epoch in range(num_epochs):
        for step, batch in enumerate(train_loader()):
            loss = train_step(model, optimizer, batch)
            if (step + 1) % accumulation_steps == 0 and (step + 1) % (100 * accumulation_steps) == 0:
                print(f"Update {(step + 1) // accumulation_steps}: Loss = {loss:.4f}")

    return model
```

Every artifex trainer takes an `nnx.Optimizer`, so the same wrapper gives the
REINFORCE, PPO, GRPO and DPO trainers accumulation without any change to their
`train_step`.

## Dynamic Loss Scaling

### Why Use Dynamic Loss Scaling?

Mixed-precision training with float16 or bfloat16 provides significant speedups
but introduces numerical challenges:

- **Underflow**: small gradients become zero in lower precision
- **Overflow**: large values exceed the representable range

Dynamic loss scaling multiplies the loss before differentiation so small
gradients survive, divides the gradients back afterwards, skips the update when
a gradient is not finite, and adapts the scale from what it observes.

### Basic Usage

```python
from flax.training.dynamic_scale import DynamicScale

dynamic_scale = DynamicScale(
    growth_factor=2.0,      # multiplier after growth_interval finite steps
    backoff_factor=0.5,     # multiplier after a non-finite step
    growth_interval=2000,   # finite steps between growth attempts
    scale=65536.0,          # starting loss scale (2**16)
)
```

`DynamicScale` is an immutable Flax struct: every step returns a new instance
carrying the updated `scale` and `fin_steps`.

### Training Loop Integration

`DynamicScale.value_and_grad` differentiates a function of a parameter pytree,
so split the module into its graph definition and parameters, differentiate the
merged model, and apply the update only when the gradients are finite:

```python
import jax
import optax
from flax import nnx
from flax.training.dynamic_scale import DynamicScale


def train_with_mixed_precision(model, train_loader, num_epochs, learning_rate=1e-3):
    """Training with dynamic loss scaling for mixed precision."""
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    dynamic_scale = DynamicScale()
    graphdef, params = nnx.split(model, nnx.Param)

    def train_step(dynamic_scale, params, batch):
        def loss_fn(params):
            outputs = nnx.merge(graphdef, params)(batch["images"], training=True)
            loss_dict = model.loss_fn(batch, outputs)
            return loss_dict["total_loss"], loss_dict

        dynamic_scale, is_finite, (loss, _), grads = dynamic_scale.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        if bool(is_finite):
            optimizer.update(model, grads)
        return dynamic_scale, nnx.state(model, nnx.Param), loss, bool(is_finite)

    for epoch in range(num_epochs):
        skipped = 0
        for step, batch in enumerate(train_loader()):
            dynamic_scale, params, loss, is_finite = train_step(dynamic_scale, params, batch)
            skipped += not is_finite
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss:.4f}, Scale = {float(dynamic_scale.scale):.0f}")
        print(f"Epoch {epoch}: skipped {skipped} non-finite steps")

    return model
```

The gradients `value_and_grad` returns are already divided by the scale, so the
optimizer sees true gradient magnitudes; `test_gradient_accumulation.py` checks
them against `jax.grad` of the unscaled loss. Inside `jax.jit`, replace the
Python `if` with `jax.lax.cond` on `is_finite`, or use
`optax.apply_if_finite` around the optimizer.

## Combining Both Features

Accumulation lives in the optimizer and scaling in the loss, so they compose
without extra code: build the optimizer with `MultiSteps` and take gradients
through `DynamicScale`. A non-finite microbatch is skipped and does not enter
the running sum; the window closes on the next finite `k`-th call.

## Best Practices

### Gradient Accumulation

1. **Choose accumulation steps from the target batch size**:

   ```python
   target_batch_size = 256
   micro_batch_size = 32  # What fits in memory
   accumulation_steps = target_batch_size // micro_batch_size  # = 8
   ```

2. **Keep the default averaging** (`use_grad_mean=True`) so gradient magnitudes
   do not depend on the window size.

3. **Adjust the learning rate** when changing the effective batch size. The
   linear scaling rule suggests scaling the learning rate with batch size.

### Dynamic Loss Scaling

1. **Start with the default scale** (`2**16`); it works for most models.

2. **Monitor skipped steps.** Frequent non-finite steps indicate a learning rate
   that is too high, a numerically unstable model, or an initial scale that is
   too high.

3. **Use bfloat16 when possible.** It has the dynamic range of float32, so
   overflow is rarer than with float16.

4. **Consider gradient clipping** as a complementary technique:

   ```python
   optimizer = optax.chain(
       optax.clip_by_global_norm(1.0),
       optax.adam(learning_rate),
   )
   ```

## API Reference

- [`optax.MultiSteps`](https://optax.readthedocs.io/en/latest/api/optimizer_wrappers.html#optax.MultiSteps)
- [`flax.training.dynamic_scale.DynamicScale`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#dynamic-scale)
- [Trainer API Reference](../../api/training/trainer.md)

## Related Documentation

- [Training Guide](training-guide.md) - Core training patterns and callbacks
- [Logging & Experiment Tracking](logging.md) - W&B, TensorBoard, and progress bar integration
- [Performance Profiling](profiling.md) - JAX trace profiling and memory tracking
- [RL Training](rl-training.md) - Reinforcement learning for model fine-tuning (REINFORCE, PPO, GRPO, DPO)
- [Distributed Training](../advanced/distributed.md) - Multi-device training with gradient accumulation
- [Configuration System](configuration.md) - Training configuration options
