# Distributed Metrics

**Module:** `artifex.generative_models.training.distributed.metrics`

**Source:** `src/artifex/generative_models/training/distributed/metrics.py`

## Overview

The `DistributedMetrics` class provides unified utilities for collecting and aggregating metrics across multiple devices in distributed training settings. It supports both static method usage (stateless) and NNX module instantiation (stateful).

## Classes

### DistributedMetrics

Unified distributed metrics collection utilities for Artifex, implemented as an NNX Module.

```python
class DistributedMetrics(nnx.Module):
    """Unified distributed metrics collection utilities for Artifex.

    This class provides methods for aggregating metrics across multiple devices
    in a distributed training setting, including mean, sum, and custom reduction
    operations.

    Supports both static method usage (stateless) and instance method
    usage (stateful).
    """
```

#### Constructor

```python
def __init__(self) -> None:
    """Initialize DistributedMetrics module."""
```

#### Methods

##### all_gather

Gather metrics from all devices.

```python
def all_gather(
    self,
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Gather metrics from all devices.

    Args:
        metrics: The metrics to gather.
        axis_name: The name of the axis to gather across.

    Returns:
        A dictionary of gathered metrics.
    """
```

##### reduce_mean

Compute the mean of metrics across devices.

```python
def reduce_mean(
    self,
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Compute the mean of metrics across devices.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of mean metrics.
    """
```

##### reduce_sum

Compute the sum of metrics across devices.

```python
def reduce_sum(
    self,
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Compute the sum of metrics across devices.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of summed metrics.
    """
```

##### reduce_max

Compute the maximum of metrics across devices.

```python
def reduce_max(
    self,
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Compute the maximum of metrics across devices.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of maximum metrics.
    """
```

##### reduce_min

Compute the minimum of metrics across devices.

```python
def reduce_min(
    self,
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Compute the minimum of metrics across devices.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of minimum metrics.
    """
```

##### reduce_custom

Apply custom reduction operations to metrics.

```python
def reduce_custom(
    self,
    metrics: dict[str, Any],
    reduce_fn: dict[str, str | None] | None = None,
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Apply custom reduction operations to metrics.

    Args:
        metrics: The metrics to reduce.
        reduce_fn: A dictionary mapping metric names to reduction operations.
            Each operation should be one of {"mean", "sum", "max", "min"}.
            If None, defaults to "mean" for all metrics.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of reduced metrics.
    """
```

##### collect_from_devices

Collect metrics from all devices (outside pmap context).

```python
def collect_from_devices(
    self,
    metrics: dict[str, Any],
) -> dict[str, list[Any] | Any]:
    """Collect metrics from all devices.

    This function should be called outside of a pmapped function to collect
    metrics from all devices.

    Args:
        metrics: The metrics from all devices, with the first dimension
            corresponding to the device axis.

    Returns:
        A dictionary of metrics, with each value being a list of the values
        from each device.
    """
```

#### Static Methods

All instance methods have static equivalents with `_static` suffix:

- `all_gather_static(metrics, axis_name="batch")`
- `reduce_mean_static(metrics, axis_name="batch")`
- `reduce_sum_static(metrics, axis_name="batch")`
- `reduce_max_static(metrics, axis_name="batch")`
- `reduce_min_static(metrics, axis_name="batch")`
- `reduce_custom_static(metrics, reduce_fn=None, axis_name="batch")`
- `collect_from_devices_static(metrics)`

## Usage Examples

### Basic Metrics Reduction

```python
from artifex.generative_models.training.distributed import DistributedMetrics
import jax
import jax.numpy as jnp

# Create instance
dm = DistributedMetrics()

# Inside a pmap context, reduce metrics across devices
@jax.pmap
def train_step(params, batch):
    # ... compute loss and gradients ...
    loss = compute_loss(params, batch)

    # Reduce metrics across devices
    metrics = {"loss": loss, "accuracy": acc}
    metrics = dm.reduce_mean(metrics)

    return params, metrics
```

### Static Method Usage

```python
from artifex.generative_models.training.distributed import DistributedMetrics
import jax

# Static methods don't require instantiation
@jax.pmap
def train_step(params, batch):
    loss = compute_loss(params, batch)

    # Use static method
    metrics = {"loss": loss}
    metrics = DistributedMetrics.reduce_mean_static(metrics)

    return params, metrics
```

### Custom Reduction Operations

```python
from artifex.generative_models.training.distributed import DistributedMetrics

dm = DistributedMetrics()

# Define custom reduction per metric
@jax.pmap
def train_step(params, batch):
    loss = compute_loss(params, batch)
    batch_size = batch["data"].shape[0]
    max_grad_norm = compute_grad_norm(grads)

    metrics = {
        "loss": loss,
        "total_samples": batch_size,
        "max_grad_norm": max_grad_norm,
    }

    # Custom reductions: mean for loss, sum for samples, max for grad norm
    reduce_ops = {
        "loss": "mean",
        "total_samples": "sum",
        "max_grad_norm": "max",
    }
    metrics = dm.reduce_custom(metrics, reduce_fn=reduce_ops)

    return params, metrics
```

### Gathering Metrics from All Devices

```python
from artifex.generative_models.training.distributed import DistributedMetrics

dm = DistributedMetrics()

@jax.pmap
def evaluate_step(params, batch):
    predictions = model(params, batch["data"])

    # Gather predictions from all devices (not reduce)
    results = {"predictions": predictions}
    results = dm.all_gather(results)

    return results
```

### Collecting Per-Device Metrics

```python
from artifex.generative_models.training.distributed import DistributedMetrics

dm = DistributedMetrics()

# After pmap returns metrics with device dimension
metrics_per_device = train_step(params, batch)
# Shape of metrics_per_device["loss"]: (num_devices,)

# Collect into list of per-device values
collected = dm.collect_from_devices(metrics_per_device)
# collected["loss"] is now a list of values, one per device

# Useful for debugging or logging per-device metrics
for i, loss in enumerate(collected["loss"]):
    print(f"Device {i} loss: {loss}")
```

### Integration with Training Loop

```python
from artifex.generative_models.training.distributed import (
    DeviceMeshManager,
    DataParallel,
    DistributedMetrics,
)
import jax
import jax.numpy as jnp

# Setup distributed training
manager = DeviceMeshManager()
mesh = manager.create_data_parallel_mesh()
dp = DataParallel()
dm = DistributedMetrics()

# Create shardings
data_sharding = dp.create_data_parallel_sharding(mesh)

@jax.jit
def train_step(params, optimizer_state, batch):
    def loss_fn(p):
        logits = model.apply(p, batch["data"])
        loss = jnp.mean((logits - batch["targets"]) ** 2)
        return loss, {"loss": loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Aggregate gradients and metrics
    grads = dp.all_reduce_gradients(grads, reduce_type="mean")

    # Update parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, metrics

# Training loop
for batch in dataloader:
    # Shard batch
    sharded_batch = dp.shard_batch(batch, data_sharding)

    # Training step
    params, optimizer_state, metrics = train_step(
        params, optimizer_state, sharded_batch
    )

    print(f"Loss: {metrics['loss']}")
```

### Complete Multi-GPU Example

```python
from artifex.generative_models.training.distributed import (
    DeviceMeshManager,
    DataParallel,
    DistributedMetrics,
)
from flax import nnx
import jax
import jax.numpy as jnp
import optax

# Model definition
class SimpleModel(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.dense1 = nnx.Linear(784, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)

# Setup
manager = DeviceMeshManager()
mesh = manager.create_data_parallel_mesh()
dp = DataParallel()
dm = DistributedMetrics()

# Create model and optimizer (wrt=nnx.Param required in NNX 0.11.0+)
model = SimpleModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# Training step with metrics aggregation
def train_step(model, optimizer, batch):
    def loss_fn(m):
        logits = m(batch["images"])
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits, batch["labels"]
            )
        )
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch["labels"])
        return loss, {"loss": loss, "accuracy": accuracy}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)  # NNX 0.11.0+ API

    return metrics

# Training loop
data_sharding = dp.create_data_parallel_sharding(mesh)

for batch in dataloader:
    sharded_batch = dp.shard_batch(batch, data_sharding)
    metrics = train_step(model, optimizer, sharded_batch)
    print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
```

## Module Statistics

- **Classes:** 1 (`DistributedMetrics`)
- **Instance Methods:** 7
- **Static Methods:** 7
