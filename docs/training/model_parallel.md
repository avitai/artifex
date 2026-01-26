# Model Parallel

**Module:** `artifex.generative_models.training.distributed.model_parallel`

**Source:** `src/artifex/generative_models/training/distributed/model_parallel.py`

## Overview

Model parallelism (tensor parallelism) splits model layers across devices, useful when models don't fit in single-device memory. This module is planned for future implementation.

## Current Status

This module is part of the planned distributed training infrastructure and is scheduled for implementation in a future release. For current model parallelism needs, you can use JAX's native sharding APIs directly.

## Planned Features

The model parallel module will provide:

- **FSDP-style parameter sharding** - Full parameter sharding across devices
- **Gradient-only sharding (ZeRO-2)** - Shard only optimizer states and gradients
- **Automatic resharding** - Manage parameter gathering and scattering
- **Memory-efficient training** - Enable training of models larger than single-device memory

## Using JAX Native Sharding (Current Approach)

Until the model parallel module is implemented, use JAX's native sharding APIs:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx

# Create 2D device mesh: (data_parallel, model_parallel)
devices = jax.devices()
mesh = Mesh(
    devices.reshape(2, 2),  # 2 data parallel, 2 model parallel
    axis_names=("data", "model")
)

# Define sharding for model parameters
# Shard weights along model axis, replicate bias
weight_sharding = NamedSharding(mesh, P(None, "model"))  # (in_features, out_features)
bias_sharding = NamedSharding(mesh, P("model"))  # (out_features,)

# Create model with sharded parameters
class ShardedLinear(nnx.Module):
    """Linear layer with sharded weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        super().__init__()

        # Create weight with sharding
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(),
                (in_features, out_features)
            )
        )

        # Apply sharding
        self.weight = jax.device_put(
            self.weight,
            NamedSharding(mesh, P(None, "model"))
        )

        # Create bias with sharding
        self.bias = nnx.Param(
            jnp.zeros(out_features)
        )
        self.bias = jax.device_put(
            self.bias,
            NamedSharding(mesh, P("model"))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Computation automatically distributed
        return x @ self.weight + self.bias
```

## Related Documentation

For current distributed training capabilities, see:

- [Device Mesh Management](mesh.md) - Creating and managing device meshes
- [Data Parallel Training](data_parallel.md) - Data parallelism utilities
- [Device Placement](device_placement.md) - Explicit device placement
- [Distributed Metrics](distributed_metrics.md) - Aggregating metrics across devices

## Module Statistics

- **Classes:** 0 (Planned for future release)
- **Functions:** 0 (Planned for future release)
- **Imports:** 0
