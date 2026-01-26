# Data Parallel Training

**Module:** `artifex.generative_models.training.distributed.data_parallel`

**Source:** `src/artifex/generative_models/training/distributed/data_parallel.py`

## Overview

The `DataParallel` class provides utilities for data-parallel training across multiple devices, including batch sharding, model state distribution, and gradient aggregation.

## Classes

### DataParallel

Data parallel training utilities for Artifex, implemented as an NNX Module.

```python
class DataParallel(nnx.Module):
    """Data parallel training utilities for Artifex.

    This class provides methods for creating data-parallel shardings,
    distributing batches across devices, and aggregating gradients.

    Supports both static method usage (stateless) and instance method
    usage (stateful).
    """
```

#### Constructor

```python
def __init__(self) -> None:
    """Initialize DataParallel module."""
```

#### Methods

##### create_data_parallel_sharding

Create a NamedSharding for data parallelism.

```python
def create_data_parallel_sharding(
    self,
    mesh: Mesh,
    data_axis: str = "data",
) -> NamedSharding:
    """Create a NamedSharding for data parallelism.

    Args:
        mesh: The device mesh to use for sharding.
        data_axis: Name of the data parallel axis in the mesh.

    Returns:
        A NamedSharding that distributes the first dimension across
        the data axis.
    """
```

##### shard_batch

Shard a batch of data across devices.

```python
def shard_batch(
    self,
    batch: Any,
    sharding: NamedSharding,
) -> Any:
    """Shard a batch of data across devices.

    Args:
        batch: PyTree of data to shard (dict, array, etc.).
        sharding: The sharding specification to apply.

    Returns:
        The sharded batch distributed across devices.
    """
```

##### shard_model_state

Shard model state across devices.

```python
def shard_model_state(
    self,
    state: Any,
    mesh: Mesh,
    param_sharding: Literal["replicate", "shard"] = "replicate",
) -> Any:
    """Shard model state across devices.

    Args:
        state: The model state (parameters, optimizer state, etc.).
        mesh: The device mesh to use.
        param_sharding: How to shard parameters:
            - "replicate": Copy parameters to all devices (default)
            - "shard": Shard parameters across devices

    Returns:
        The sharded model state.
    """
```

##### all_reduce_gradients

Aggregate gradients across devices.

```python
def all_reduce_gradients(
    self,
    gradients: Any,
    reduce_type: Literal["mean", "sum"] = "mean",
    axis_name: str = "batch",
) -> Any:
    """Aggregate gradients across devices.

    This should be called inside a pmap/shard_map context.

    Args:
        gradients: PyTree of gradients to aggregate.
        reduce_type: Type of reduction ("mean" or "sum").
        axis_name: The axis name for the parallel reduction.

    Returns:
        The aggregated gradients.

    Raises:
        ValueError: If reduce_type is not "mean" or "sum".
    """
```

##### replicate_params

Replicate parameters across all devices.

```python
def replicate_params(
    self,
    params: Any,
    mesh: Mesh,
) -> Any:
    """Replicate parameters across all devices.

    Args:
        params: PyTree of parameters to replicate.
        mesh: The device mesh to use.

    Returns:
        The replicated parameters.
    """
```

#### Static Methods

All instance methods have static equivalents with `_static` suffix:

- `create_data_parallel_sharding_static(mesh, data_axis="data")`
- `shard_batch_static(batch, sharding)`
- `all_reduce_gradients_static(gradients, reduce_type="mean", axis_name="batch")`

## Usage Examples

### Basic Data Parallel Training

```python
from artifex.generative_models.training.distributed import (
    DeviceMeshManager,
    DataParallel,
)
import jax.numpy as jnp

# Create mesh and data parallel utilities
manager = DeviceMeshManager()
mesh = manager.create_data_parallel_mesh()
dp = DataParallel()

# Create sharding for batch data
sharding = dp.create_data_parallel_sharding(mesh)

# Shard batch data
batch = {"inputs": jnp.ones((32, 784)), "targets": jnp.zeros((32,))}
sharded_batch = dp.shard_batch(batch, sharding)
```

### Sharding Model State

```python
# Replicate model parameters across all devices (default for data parallelism)
sharded_state = dp.shard_model_state(model_state, mesh, param_sharding="replicate")

# Or shard parameters across devices (for model parallelism)
sharded_state = dp.shard_model_state(model_state, mesh, param_sharding="shard")
```

### Gradient Aggregation

```python
import jax

# Inside a pmap context, aggregate gradients
@jax.pmap
def train_step(params, batch):
    def loss_fn(p):
        return compute_loss(p, batch)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    # Average gradients across devices
    grads = DataParallel.all_reduce_gradients_static(grads, reduce_type="mean")

    return grads, loss
```

### Using Static Methods

```python
# Static methods don't require instantiation
from artifex.generative_models.training.distributed import DataParallel

mesh = manager.create_device_mesh({"data": 4})
sharding = DataParallel.create_data_parallel_sharding_static(mesh)
sharded_batch = DataParallel.shard_batch_static(batch, sharding)
```

### Complete Training Example

```python
from artifex.generative_models.training.distributed import (
    DeviceMeshManager,
    DataParallel,
)
from flax import nnx
import jax
import jax.numpy as jnp

# Setup
manager = DeviceMeshManager()
mesh = manager.create_data_parallel_mesh()
dp = DataParallel()

# Create model and optimizer (wrt=nnx.Param required in NNX 0.11.0+)
model = MyModel(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

# Create shardings
data_sharding = dp.create_data_parallel_sharding(mesh)

# Replicate model state
model_state = nnx.state(model)
sharded_model_state = dp.shard_model_state(model_state, mesh)

# Training loop
for batch in dataloader:
    # Shard batch
    sharded_batch = dp.shard_batch(batch, data_sharding)

    # Compute gradients and update
    # (In practice, wrap this in @jax.jit for performance)
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Update model (NNX 0.11.0+ API)
    optimizer.update(model, grads)
```

## Module Statistics

- **Classes:** 1 (`DataParallel`)
- **Instance Methods:** 5
- **Static Methods:** 3
