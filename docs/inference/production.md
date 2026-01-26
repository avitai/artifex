# Production

**Module:** `generative_models.inference.optimization.production`

**Source:** `generative_models/inference/optimization/production.py`

## Overview

> **Note**: This module is experimental and under active development as part of ongoing work towards production-ready inference capabilities.

Inference optimization infrastructure for scaled models.

This module provides optimization infrastructure including:

- Automatic optimization pipeline selection
- Inference optimization strategies
- Model adapter classes for different architectures
- Monitoring and debugging tools

All implementations follow JAX/Flax NNX best practices and prioritize
performance through hardware-aware optimization.

## Classes

### CompiledModel

```python
class CompiledModel
```

### MonitoringMetrics

```python
class MonitoringMetrics
```

### OptimizationResult

```python
class OptimizationResult
```

### OptimizationTarget

```python
class OptimizationTarget
```

### ProductionMonitor

```python
class ProductionMonitor
```

### ProductionOptimizer

```python
class ProductionOptimizer
```

### ProductionPipeline

```python
class ProductionPipeline
```

## Functions

### **call**

```python
def __call__()
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

### compiled_forward

```python
def compiled_forward()
```

### create_production_optimizer

```python
def create_production_optimizer()
```

### create_production_pipeline

```python
def create_production_pipeline()
```

### create_production_pipeline

```python
def create_production_pipeline()
```

### get_metrics

```python
def get_metrics()
```

### get_monitoring_metrics

```python
def get_monitoring_metrics()
```

### optimize_for_production

```python
def optimize_for_production()
```

### predict

```python
def predict()
```

### predict_batch

```python
def predict_batch()
```

### record_request

```python
def record_request()
```

### reset

```python
def reset()
```

### reset_monitoring

```python
def reset_monitoring()
```

## Module Statistics

- **Classes:** 7
- **Functions:** 17
- **Imports:** 7
