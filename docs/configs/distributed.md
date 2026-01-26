# Distributed

**Module:** `configs.schema.distributed`

**Source:** `configs/schema/distributed.py`

## Overview

Enhanced distributed training configuration schema with improved validation.

## Classes

### DistributedBackend

```python
class DistributedBackend
```

### DistributedConfig

```python
class DistributedConfig
```

## Functions

### get_data_parallel_size

```python
def get_data_parallel_size()
```

### get_mesh_config

```python
def get_mesh_config()
```

### is_local_main_process

```python
def is_local_main_process()
```

### is_main_process

```python
def is_main_process()
```

### validate_distributed_consistency

```python
def validate_distributed_consistency()
```

### validate_port

```python
def validate_port()
```

### validate_positive_int

```python
def validate_positive_int()
```

### validate_rank_non_negative

```python
def validate_rank_non_negative()
```

## Module Statistics

- **Classes:** 2
- **Functions:** 8
- **Imports:** 4
