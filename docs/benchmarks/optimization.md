# Optimization

**Module:** `benchmarks.performance.optimization`

**Source:** `benchmarks/performance/optimization.py`

## Overview

Optimization benchmark for generative models.

This module provides benchmarks for evaluating the training performance and
optimization strategies for generative models, measuring convergence rates,
loss curves, and training efficiency.

## Classes

### OptimizationBenchmark

```python
class OptimizationBenchmark
```

### OptimizationMetricsConfig

```python
class OptimizationMetricsConfig
```

### OptimizerComparisonBenchmark

```python
class OptimizerComparisonBenchmark
```

### TrainerProtocol

```python
class TrainerProtocol
```

### TrainingConvergenceBenchmark

```python
class TrainingConvergenceBenchmark
```

### TrainingCurvePoint

```python
@dataclass(frozen=True, kw_only=True)
class TrainingCurvePoint:
    iteration: int
    metrics: dict[str, float]
    timestamp: float | None = None
```

A point on a training curve, and the record it takes in
`BenchmarkResult.metadata["training_curve"]`: `to_record()` writes it and
`TrainingCurvePoint.from_record(record)` reads it back, so the benchmark and the optimization
plots share one format. Values are read with calibrax's `read_metadata`, so a metric recorded as
an array scalar reads as the number it holds, and a value of the wrong kind is refused with its
path (`training_curve.metrics.loss`).

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

### evaluate

```python
def evaluate()
```

### init

```python
def init()
```

### run

```python
def run()
```

### run

```python
def run()
```

### train_step

```python
def train_step()
```

### training_curve_from_metadata

```python
def training_curve_from_metadata(
    metadata: Mapping[str, MetadataValue],
) -> list[TrainingCurvePoint]
```

The training curve an optimization benchmark wrote into its result's metadata. Refuses metadata
with no curve, an empty curve, or one that is not a list.

## Module Statistics

- **Classes:** 6
- **Functions:** 9
- **Imports:** 10
