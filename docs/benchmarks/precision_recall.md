# Precision Recall

**Module:** `benchmarks.metrics.precision_recall`

**Source:** `benchmarks/metrics/precision_recall.py`

## Overview

Precision-recall metrics for evaluating generative models.

This module implements precision and recall metrics for generative models as
described in "Improved Precision and Recall Metric for Assessing Generative
Models" (Kynkäänniemi et al., 2019).

The implementation uses clustering to identify modes in the data distribution
and computes precision and recall based on cluster coverage.

## Classes

### KMeansModule

```python
class KMeansModule
```

### PrecisionRecallBenchmark

```python
class PrecisionRecallBenchmark
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

### compute_cluster_based_metrics

```python
def compute_cluster_based_metrics()
```

### compute_distance_based_metrics

```python
def compute_distance_based_metrics()
```

### compute_precision_recall

```python
def compute_precision_recall()
```

### fit

```python
def fit()
```

### is_well_separated_clusters

```python
def is_well_separated_clusters()
```

### run

```python
def run()
```

## Module Statistics

- **Classes:** 2
- **Functions:** 8
- **Imports:** 5
