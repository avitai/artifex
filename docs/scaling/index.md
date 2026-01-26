# Scaling & Distributed Training

Comprehensive tools for scaling generative model training across multiple devices and accelerators.

## Overview

Artifex provides robust infrastructure for scaling model training from single-GPU experiments to multi-node distributed setups. The scaling module offers:

<div class="grid cards" markdown>

- :material-grid:{ .lg .middle } **Device Mesh Management**

    ---

    Create and optimize device meshes for different workloads

    [:octicons-arrow-right-24: Mesh Utilities](#device-mesh-management)

- :material-layers-triple:{ .lg .middle } **Sharding Strategies**

    ---

    Data, tensor, FSDP, and pipeline parallelism

    [:octicons-arrow-right-24: Sharding Strategies](#sharding-strategies)

- :material-tune:{ .lg .middle } **Multi-Dimensional Parallelism**

    ---

    Combine strategies for optimal performance

    [:octicons-arrow-right-24: Multi-Dimensional](#multi-dimensional-parallelism)

- :material-cog:{ .lg .middle } **Configuration**

    ---

    Flexible configuration for complex setups

    [:octicons-arrow-right-24: Configuration](#configuration)

</div>

---

## Quick Start

### Basic Data Parallel Training

```python
import jax
from artifex.generative_models.scaling import mesh_utils, sharding

# Get available devices
devices = jax.devices()
print(f"Available devices: {len(devices)}")

# Create sharding config for data parallelism
config = sharding.ShardingConfig.from_device_count(len(devices))

# Create parallelism config with mesh topology
parallel_config = sharding.ParallelismConfig.from_sharding_config(config)

# Create device mesh
mesh_manager = mesh_utils.DeviceMeshManager(
    mesh_shape=parallel_config.mesh_shape,
    axis_names=parallel_config.mesh_axis_names,
)
mesh = mesh_manager.create_mesh_from_config(parallel_config)

print(f"Mesh shape: {parallel_config.mesh_shape}")
print(f"Axis names: {parallel_config.mesh_axis_names}")
```

### Tensor Parallel Setup

```python
from artifex.generative_models.scaling.sharding import (
    ShardingConfig,
    ParallelismConfig,
    TensorParallelStrategy,
)

# Configure tensor parallelism for large models
config = ShardingConfig(
    data_parallel_size=2,
    tensor_parallel_size=4,  # 8 GPUs total
)

# Create tensor parallel strategy
tensor_strategy = TensorParallelStrategy(
    axis_name="model",
    mesh_axis=1,
    shard_dimension="out_features",
)

# Get partition specs for attention layers
qkv_spec = tensor_strategy.get_attention_qkv_spec()
output_spec = tensor_strategy.get_attention_output_spec()
```

---

## Device Mesh Management

The `DeviceMeshManager` provides utilities for creating and optimizing device meshes.

### Creating a Device Mesh

```python
from artifex.generative_models.scaling.mesh_utils import (
    DeviceMeshManager,
    create_device_mesh,
)

# Simple mesh creation
mesh = create_device_mesh(
    mesh_shape=(4, 2),        # 4 data parallel x 2 tensor parallel
    axis_names=("data", "model"),
)

# Using DeviceMeshManager for more control
manager = DeviceMeshManager(
    mesh_shape=(4, 2),
    axis_names=("data", "model"),
)

# Get optimal mesh shape for device count
optimal_shape = manager.get_optimal_mesh_shape(
    device_count=8,
    dimensions=2,
)
print(f"Optimal shape for 8 devices: {optimal_shape}")
```

### Optimizing for Transformers

```python
# Optimize mesh for transformer workloads
optimal_shape = manager.optimize_for_transformer(
    device_count=8,
    model_size="7B",
    sequence_length=2048,
)
print(f"Optimal shape for 7B model: {optimal_shape}")

# For larger models, more tensor parallelism
large_shape = manager.optimize_for_transformer(
    device_count=32,
    model_size="70B",
    sequence_length=4096,
)
print(f"Optimal shape for 70B model: {large_shape}")
```

### Validation

```python
# Validate mesh configuration before use
is_valid = manager.validate_mesh_config(
    mesh_shape=(4, 2),
    device_count=8,
)
print(f"Configuration valid: {is_valid}")
```

---

## Sharding Strategies

Artifex provides multiple sharding strategies for different parallelism approaches.

### Data Parallel Strategy

Shards data across devices while replicating model parameters.

```python
from artifex.generative_models.scaling.sharding import DataParallelStrategy

strategy = DataParallelStrategy(axis_name="data", mesh_axis=0)

# Get partition spec for a batch of data
# Shape: (batch, sequence, hidden)
spec = strategy.get_partition_spec(("batch", "sequence", "hidden"))
# Result: PartitionSpec("data", None, None)

# Apply sharding to data
sharded_data = strategy.apply_sharding(batch_data, mesh)
```

### FSDP Strategy

Fully Sharded Data Parallel for memory-efficient training.

```python
from artifex.generative_models.scaling.sharding import FSDPStrategy

strategy = FSDPStrategy(
    axis_name="fsdp",
    mesh_axis=0,
    min_weight_size=1024,  # Only shard weights >= 1024 in first dim
)

# Check if a weight should be sharded
should_shard = strategy.should_shard_weight(large_weight_matrix)

# Apply FSDP sharding
sharded_weights = strategy.apply_sharding(weights, mesh)
```

### Tensor Parallel Strategy

Shards model computation across devices.

```python
from artifex.generative_models.scaling.sharding import TensorParallelStrategy

strategy = TensorParallelStrategy(
    axis_name="model",
    mesh_axis=1,
    shard_dimension="out_features",
)

# Get specs for attention layers
qkv_spec = strategy.get_attention_qkv_spec()      # Shard output
output_spec = strategy.get_attention_output_spec() # Shard input

# Get specs for linear layers
linear_spec = strategy.get_linear_weight_spec()
```

### Pipeline Parallel Strategy

Distributes model layers across devices.

```python
from artifex.generative_models.scaling.sharding import PipelineParallelStrategy

strategy = PipelineParallelStrategy(
    axis_name="pipeline",
    mesh_axis=2,
    num_stages=4,
)

# Assign 24 transformer layers to 4 pipeline stages
layer_assignments = strategy.assign_layers_to_stages(num_layers=24)
# Result: [6, 6, 6, 6] - 6 layers per stage

# Get communication patterns
forward_pattern = strategy.get_forward_communication_pattern()
backward_pattern = strategy.get_backward_communication_pattern()
```

---

## Multi-Dimensional Parallelism

Combine multiple strategies for optimal large-scale training.

```python
from artifex.generative_models.scaling.sharding import (
    MultiDimensionalStrategy,
    DataParallelStrategy,
    TensorParallelStrategy,
    FSDPStrategy,
    ParallelismConfig,
    ShardingConfig,
)

# Create individual strategies
data_strategy = DataParallelStrategy(axis_name="data", mesh_axis=0)
tensor_strategy = TensorParallelStrategy(
    axis_name="model",
    mesh_axis=1,
    shard_dimension="out_features",
)
fsdp_strategy = FSDPStrategy(axis_name="data", mesh_axis=0)

# Combine into multi-dimensional strategy
config = ShardingConfig(
    data_parallel_size=4,
    tensor_parallel_size=2,
    fsdp_enabled=True,
)
parallel_config = ParallelismConfig.from_sharding_config(config)

multi_strategy = MultiDimensionalStrategy(
    strategies={
        "data": data_strategy,
        "tensor": tensor_strategy,
        "fsdp": fsdp_strategy,
    },
    config=parallel_config,
)

# Get combined partition spec for a tensor
combined_spec = multi_strategy.get_combined_partition_spec(
    tensor_name="attention.query",
    tensor_shape=("batch", "sequence", "hidden"),
)
```

---

## Configuration

### ShardingConfig

Defines parallelism dimensions.

```python
from artifex.generative_models.scaling.sharding import ShardingConfig

# Manual configuration
config = ShardingConfig(
    data_parallel_size=4,
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    fsdp_enabled=True,
    fsdp_min_weight_size=1024,
)

# Auto-configure from device count
auto_config = ShardingConfig.from_device_count(device_count=8)

# Get total device requirement
total_devices = config.get_total_device_count()  # 4 * 2 * 1 = 8
```

### ParallelismConfig

Complete parallelism configuration with mesh topology.

```python
from artifex.generative_models.scaling.sharding import ParallelismConfig

# Create from sharding config
parallel_config = ParallelismConfig.from_sharding_config(config)

# Access mesh configuration
print(f"Mesh shape: {parallel_config.mesh_shape}")
print(f"Axis names: {parallel_config.mesh_axis_names}")

# Validate configuration
is_valid = parallel_config.is_valid()
```

---

## Best Practices

### Choosing Parallelism Strategy

| Model Size | Devices | Recommended Strategy |
|------------|---------|---------------------|
| < 1B params | 1-8 | Data Parallel |
| 1B - 10B | 8-32 | Data + Tensor Parallel |
| 10B - 100B | 32-128 | Data + Tensor + FSDP |
| > 100B | 128+ | All strategies + Pipeline |

### Memory Optimization

```python
# For memory-constrained setups, enable FSDP
config = ShardingConfig(
    data_parallel_size=4,
    fsdp_enabled=True,
    fsdp_min_weight_size=512,  # Shard smaller weights
)

# For very large models, combine with tensor parallel
config = ShardingConfig(
    data_parallel_size=2,
    tensor_parallel_size=4,
    fsdp_enabled=True,
)
```

### Performance Tips

1. **Balance dimensions**: Avoid extreme ratios in mesh shape
2. **Match workload**: Use transformer-optimized shapes for transformers
3. **Validate configs**: Always validate before creating meshes
4. **Monitor memory**: Enable FSDP for memory-constrained scenarios

---

## API Reference

### Mesh Utilities

::: artifex.generative_models.scaling.mesh_utils
    options:
      show_root_heading: true
      members:
        - DeviceMeshManager
        - create_device_mesh
        - get_optimal_mesh_shape

### Sharding

::: artifex.generative_models.scaling.sharding
    options:
      show_root_heading: true
      members:
        - ShardingConfig
        - ParallelismConfig
        - ShardingStrategy
        - DataParallelStrategy
        - FSDPStrategy
        - TensorParallelStrategy
        - PipelineParallelStrategy
        - MultiDimensionalStrategy

---

## Related Documentation

- [Distributed Training Guide](../user-guide/advanced/distributed.md) - User guide for distributed training
- [Model Parallelism](../user-guide/advanced/parallelism.md) - Model parallelism techniques
- [Training Guide](../user-guide/training/training-guide.md) - Core training concepts
- [Device Management](../api/core/device-manager.md) - Device manager API
