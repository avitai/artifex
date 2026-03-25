# Sharding

**Module:** `generative_models.scaling.sharding`

**Source:** `generative_models/scaling/sharding.py`

## Overview

Sharding strategies and parallelism configuration for scalable training.

This module exposes the retained sharding strategies and configuration
objects used by `artifex.generative_models.scaling`. It focuses on explicit
strategy composition and mesh metadata instead of claiming a generic
parameter-name to `PartitionSpec` inference layer.

## Classes

### DataParallelStrategy

```python
class DataParallelStrategy
```

### FSDPStrategy

```python
class FSDPStrategy
```

### MultiDimensionalStrategy

```python
class MultiDimensionalStrategy
```

### ParallelismConfig

```python
class ParallelismConfig
```

### PipelineParallelStrategy

```python
class PipelineParallelStrategy
```

### ShardingConfig

```python
class ShardingConfig
```

### ShardingStrategy

```python
class ShardingStrategy
```

### TensorParallelStrategy

```python
class TensorParallelStrategy
```

## Functions

### **init**

```python
def __init__()
```

### **init**

```python
def __init__()
```

### **init**

```python
def __init__()
```

### **init**

```python
def __init__()
```

### **init**

```python
def __init__()
```

### apply_sharding

```python
def apply_sharding()
```

### apply_sharding

```python
def apply_sharding()
```

### apply_sharding

```python
def apply_sharding()
```

### apply_sharding

```python
def apply_sharding()
```

### apply_sharding

```python
def apply_sharding()
```

### assign_layers_to_stages

```python
def assign_layers_to_stages()
```

### from_device_count

```python
def from_device_count()
```

### from_sharding_config

```python
def from_sharding_config()
```

### get_attention_output_spec

```python
def get_attention_output_spec()
```

### get_attention_qkv_spec

```python
def get_attention_qkv_spec()
```

### get_backward_communication_pattern

```python
def get_backward_communication_pattern()
```

### get_combined_partition_spec

```python
def get_combined_partition_spec()
```

### get_forward_communication_pattern

```python
def get_forward_communication_pattern()
```

### get_gradient_partition_spec

```python
def get_gradient_partition_spec()
```

### get_linear_weight_spec

```python
def get_linear_weight_spec()
```

### get_partition_spec

```python
def get_partition_spec()
```

### get_partition_spec

```python
def get_partition_spec()
```

### get_partition_spec

```python
def get_partition_spec()
```

### get_partition_spec

```python
def get_partition_spec()
```

### get_partition_spec

```python
def get_partition_spec()
```

### get_sharding_constraints

```python
def get_sharding_constraints()
```

### get_total_device_count

```python
def get_total_device_count()
```

### is_valid

```python
def is_valid()
```

### resolve_sharding_conflicts

```python
def resolve_sharding_conflicts()
```

### should_shard_weight

```python
def should_shard_weight()
```

## Module Statistics

- **Classes:** 8
- **Functions:** 31
- **Imports:** 8
