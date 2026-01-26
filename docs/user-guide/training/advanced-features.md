# Advanced Training Features

This guide covers advanced training utilities in Artifex for optimizing training performance and handling challenging scenarios like large batch training and mixed-precision optimization.

## Overview

Artifex provides two key utilities for advanced training:

- **GradientAccumulator**: Enables training with larger effective batch sizes by accumulating gradients across multiple forward/backward passes
- **DynamicLossScaler**: Handles numerical stability for mixed-precision training (float16/bfloat16) through automatic loss scaling

## Gradient Accumulation

### Why Use Gradient Accumulation?

When training large models or using high-resolution inputs, GPU memory often limits the batch size you can use. Gradient accumulation solves this by:

1. Running multiple forward/backward passes with smaller batches
2. Accumulating the gradients from each pass
3. Applying a single optimizer update with the accumulated gradients

This simulates training with a larger effective batch size without requiring more memory.

**Effective batch size = micro_batch_size × accumulation_steps**

### Basic Usage

```python
from artifex.generative_models.training import (
    GradientAccumulator,
    GradientAccumulatorConfig,
)

# Configure accumulation
config = GradientAccumulatorConfig(
    accumulation_steps=4,      # Accumulate over 4 micro-batches
    normalize_gradients=True,  # Average gradients (recommended)
)

# Create accumulator
accumulator = GradientAccumulator(config)
```

### GradientAccumulatorConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accumulation_steps` | `int` | `1` | Number of steps to accumulate before update |
| `normalize_gradients` | `bool` | `True` | Whether to average gradients by accumulation_steps |

### Training Loop Integration

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from artifex.generative_models.training import (
    GradientAccumulator,
    GradientAccumulatorConfig,
)

def train_with_accumulation(
    model: nnx.Module,
    train_loader,
    num_epochs: int,
    micro_batch_size: int = 32,
    accumulation_steps: int = 4,
    learning_rate: float = 1e-3,
):
    """Training with gradient accumulation."""
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nnx.state(model))

    # Setup accumulator
    accumulator = GradientAccumulator(
        GradientAccumulatorConfig(
            accumulation_steps=accumulation_steps,
            normalize_gradients=True,
        )
    )

    @nnx.jit
    def compute_gradients(model, batch):
        """Compute gradients for a single micro-batch."""
        def loss_fn(model):
            outputs = model(batch["images"], training=True)
            return outputs["loss"], outputs

        (loss, outputs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        return grads, loss, outputs

    @nnx.jit
    def apply_gradients(model, opt_state, grads):
        """Apply accumulated gradients to model."""
        updates, new_opt_state = optimizer.update(grads, opt_state)
        nnx.update(model, nnx.apply_updates(nnx.state(model), updates))
        return new_opt_state

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        for batch in train_loader(batch_size=micro_batch_size):
            # Compute and accumulate gradients
            grads, loss, outputs = compute_gradients(model, batch)
            accumulator.accumulate(grads)
            global_step += 1

            # Apply update when accumulation is complete
            if accumulator.should_update(global_step):
                accumulated_grads = accumulator.get_gradients()
                opt_state = apply_gradients(model, opt_state, accumulated_grads)
                accumulator.reset()

                effective_step = global_step // accumulation_steps
                if effective_step % 100 == 0:
                    print(f"Step {effective_step}: Loss = {loss:.4f}")

    return model
```

### Key Methods

#### `accumulate(grads)`

Add gradients from a micro-batch to the accumulator.

```python
grads, loss = compute_gradients(model, batch)
accumulator.accumulate(grads)
```

#### `should_update(step)`

Check if enough gradients have been accumulated. Returns `True` when `step % accumulation_steps == 0`.

```python
if accumulator.should_update(global_step):
    # Time to apply optimizer update
    ...
```

#### `get_gradients()`

Retrieve the accumulated (and optionally normalized) gradients.

```python
accumulated_grads = accumulator.get_gradients()
# If normalize_gradients=True, returns grads / accumulation_steps
```

#### `reset()`

Clear the accumulator after applying an update.

```python
accumulator.reset()
```

## Dynamic Loss Scaling

### Why Use Dynamic Loss Scaling?

Mixed-precision training with float16 or bfloat16 provides significant speedups but introduces numerical challenges:

- **Underflow**: Small gradients become zero in lower precision
- **Overflow**: Large values exceed the representable range

Dynamic loss scaling addresses these issues by:

1. **Scaling up** the loss before backward pass (prevents underflow)
2. **Unscaling** gradients before optimizer update
3. **Adjusting scale dynamically** based on gradient overflow detection

### Basic Usage

```python
from artifex.generative_models.training import (
    DynamicLossScaler,
    DynamicLossScalerConfig,
)

# Configure loss scaler
config = DynamicLossScalerConfig(
    initial_scale=2**15,      # Starting loss scale
    growth_factor=2.0,        # Scale growth multiplier
    backoff_factor=0.5,       # Scale reduction on overflow
    growth_interval=2000,     # Steps before attempting scale increase
    min_scale=1.0,            # Minimum allowed scale
    max_scale=2**24,          # Maximum allowed scale
)

# Create scaler
scaler = DynamicLossScaler(config)
```

### DynamicLossScalerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_scale` | `float` | `2**15` | Starting loss scale value |
| `growth_factor` | `float` | `2.0` | Multiplier when increasing scale |
| `backoff_factor` | `float` | `0.5` | Multiplier when reducing scale (on overflow) |
| `growth_interval` | `int` | `2000` | Steps without overflow before increasing scale |
| `min_scale` | `float` | `1.0` | Minimum allowed scale value |
| `max_scale` | `float` | `2**24` | Maximum allowed scale value |

### Training Loop Integration

```python
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from artifex.generative_models.training import (
    DynamicLossScaler,
    DynamicLossScalerConfig,
)

def train_with_mixed_precision(
    model: nnx.Module,
    train_loader,
    num_epochs: int,
    learning_rate: float = 1e-3,
):
    """Training with dynamic loss scaling for mixed precision."""
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nnx.state(model))

    # Setup loss scaler
    scaler = DynamicLossScaler(DynamicLossScalerConfig())

    def train_step(model, opt_state, batch):
        """Single training step with loss scaling."""
        def loss_fn(model):
            # Forward pass (in lower precision if model uses fp16/bf16)
            outputs = model(batch["images"], training=True)
            return outputs["loss"], outputs

        # Compute loss and gradients
        (loss, outputs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Scale loss for numerical stability
        scaled_loss = scaler.scale_loss(loss)

        # Unscale gradients before optimizer update
        unscaled_grads = scaler.unscale_gradients(grads)

        # Check for overflow (NaN or Inf in gradients)
        overflow = scaler.check_overflow(unscaled_grads)

        # Update scale based on overflow status
        scaler.update_scale(overflow)

        if not overflow:
            # Apply gradients only if no overflow
            updates, opt_state = optimizer.update(unscaled_grads, opt_state)
            nnx.update(model, nnx.apply_updates(nnx.state(model), updates))

        return opt_state, loss, overflow

    # Training loop
    for epoch in range(num_epochs):
        total_overflow_steps = 0
        for step, batch in enumerate(train_loader()):
            opt_state, loss, overflow = train_step(model, opt_state, batch)

            if overflow:
                total_overflow_steps += 1

            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss:.4f}, Scale = {scaler.scale:.0f}")

        print(f"Epoch {epoch}: Overflow steps = {total_overflow_steps}")

    return model
```

### Key Methods

#### `scale_loss(loss)`

Multiply the loss by the current scale factor before backward pass.

```python
scaled_loss = scaler.scale_loss(loss)
# Now compute gradients with respect to scaled_loss
```

#### `unscale_gradients(grads)`

Divide gradients by the scale factor to get the true gradient values.

```python
unscaled_grads = scaler.unscale_gradients(grads)
```

#### `check_overflow(grads)`

Check if any gradient contains NaN or Inf values.

```python
overflow = scaler.check_overflow(unscaled_grads)
if overflow:
    # Skip this update, reduce scale
    pass
```

#### `update_scale(overflow_detected)`

Adjust the scale based on whether overflow was detected.

```python
scaler.update_scale(overflow)
# If overflow: scale *= backoff_factor
# If no overflow for growth_interval steps: scale *= growth_factor
```

## Combining Both Features

For optimal training of large models, combine gradient accumulation with dynamic loss scaling:

```python
from artifex.generative_models.training import (
    GradientAccumulator,
    GradientAccumulatorConfig,
    DynamicLossScaler,
    DynamicLossScalerConfig,
)

def train_with_accumulation_and_scaling(
    model: nnx.Module,
    train_loader,
    num_epochs: int,
    accumulation_steps: int = 4,
    learning_rate: float = 1e-3,
):
    """Combined gradient accumulation and dynamic loss scaling."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nnx.state(model))

    # Initialize utilities
    accumulator = GradientAccumulator(
        GradientAccumulatorConfig(
            accumulation_steps=accumulation_steps,
            normalize_gradients=True,
        )
    )
    scaler = DynamicLossScaler(DynamicLossScalerConfig())

    def compute_scaled_gradients(model, batch):
        """Compute gradients with loss scaling."""
        def loss_fn(model):
            outputs = model(batch["images"], training=True)
            # Scale loss before backward pass
            scaled_loss = scaler.scale_loss(outputs["loss"])
            return scaled_loss, outputs

        (_, outputs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Unscale gradients immediately
        unscaled_grads = scaler.unscale_gradients(grads)
        return unscaled_grads, outputs["loss"]

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        for batch in train_loader():
            # Compute and accumulate gradients
            grads, loss = compute_scaled_gradients(model, batch)
            accumulator.accumulate(grads)
            global_step += 1

            # Apply update when accumulation is complete
            if accumulator.should_update(global_step):
                accumulated_grads = accumulator.get_gradients()

                # Check for overflow in accumulated gradients
                overflow = scaler.check_overflow(accumulated_grads)
                scaler.update_scale(overflow)

                if not overflow:
                    updates, opt_state = optimizer.update(
                        accumulated_grads, opt_state
                    )
                    nnx.update(
                        model,
                        nnx.apply_updates(nnx.state(model), updates)
                    )

                accumulator.reset()

    return model
```

## Best Practices

### Gradient Accumulation

1. **Choose accumulation steps based on target batch size**:

   ```python
   target_batch_size = 256
   micro_batch_size = 32  # What fits in memory
   accumulation_steps = target_batch_size // micro_batch_size  # = 8
   ```

2. **Always normalize gradients** unless you have a specific reason not to. This ensures consistent gradient magnitudes regardless of accumulation steps.

3. **Adjust learning rate** when changing effective batch size. The linear scaling rule suggests scaling learning rate proportionally with batch size.

### Dynamic Loss Scaling

1. **Start with moderate initial scale** (default `2**15` works well for most cases).

2. **Monitor overflow frequency**. Frequent overflows indicate:
   - Learning rate may be too high
   - Model may have numerical instability
   - Initial scale may be too high

3. **Use with bfloat16 when possible**. bfloat16 has the same dynamic range as float32, reducing overflow issues compared to float16.

4. **Consider gradient clipping** as a complementary technique:

   ```python
   optimizer = optax.chain(
       optax.clip_by_global_norm(1.0),
       optax.adam(learning_rate),
   )
   ```

## Integration with Model-Specific Trainers

The advanced features integrate seamlessly with Artifex's model-specific trainers:

```python
from artifex.generative_models.training.trainers import VAETrainer
from artifex.generative_models.training import (
    GradientAccumulator,
    GradientAccumulatorConfig,
)

# Create trainer
trainer = VAETrainer(model, optimizer, config)

# Use accumulator in custom training loop
accumulator = GradientAccumulator(
    GradientAccumulatorConfig(accumulation_steps=4)
)

for batch in dataloader:
    grads, metrics = trainer.compute_gradients(batch)
    accumulator.accumulate(grads)

    if accumulator.should_update(step):
        trainer.apply_gradients(accumulator.get_gradients())
        accumulator.reset()
```

## API Reference

For complete API documentation, see the [Trainer API Reference](../../api/training/trainer.md).

The gradient accumulation and dynamic loss scaling utilities are exported from the main training module:

```python
from artifex.generative_models.training import (
    GradientAccumulator,
    GradientAccumulatorConfig,
    DynamicLossScaler,
    DynamicLossScalerConfig,
)
```

## Related Documentation

- [Training Guide](training-guide.md) - Core training patterns and callbacks
- [Logging & Experiment Tracking](logging.md) - W&B, TensorBoard, and progress bar integration
- [Performance Profiling](profiling.md) - JAX trace profiling and memory tracking
- [RL Training](rl-training.md) - Reinforcement learning for model fine-tuning (REINFORCE, PPO, GRPO, DPO)
- [Distributed Training](../advanced/distributed.md) - Multi-device training with gradient accumulation
- [Configuration System](configuration.md) - Training configuration options
