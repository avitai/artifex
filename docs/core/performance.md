# Performance

**Module:** `generative_models.core.performance`

**Source:** `generative_models/core/performance.py`

## Overview

Performance infrastructure for roofline analysis and optimization.

This module provides core performance analysis capabilities including:

- Hardware detection and specification
- Roofline model analysis for performance estimation
- FLOP counting and arithmetic intensity calculation
- JAX function profiling and benchmarking

All implementations follow JAX/Flax NNX best practices and avoid numpy
usage within any performance-critical code paths.

## Classes

### HardwareDetector

```python
class HardwareDetector
```

### HardwareSpecs

```python
class HardwareSpecs
```

### PerformanceEstimator

```python
class PerformanceEstimator
```

### RooflineMetrics

```python
class RooflineMetrics
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

### analyze_roofline

```python
def analyze_roofline()
```

### benchmark_operation

```python
def benchmark_operation()
```

### calculate_arithmetic_intensity

```python
def calculate_arithmetic_intensity()
```

### detect_hardware

```python
def detect_hardware()
```

### estimate_flops_attention

```python
def estimate_flops_attention()
```

### estimate_flops_linear

```python
def estimate_flops_linear()
```

### estimate_memory_usage

```python
def estimate_memory_usage()
```

### estimate_transformer_layer_performance

```python
def estimate_transformer_layer_performance()
```

### get_batch_size_recommendation

```python
def get_batch_size_recommendation()
```

### get_critical_batch_size

```python
def get_critical_batch_size()
```

### get_optimal_batch_size

```python
def get_optimal_batch_size()
```

### is_batch_size_optimal

```python
def is_batch_size_optimal()
```

### profile_jax_function

```python
def profile_jax_function()
```

## Module Statistics

- **Classes:** 4
- **Functions:** 15
- **Imports:** 4
