# Optimization Plots

**Module:** `benchmarks.visualization.optimization_plots`

**Source:** `benchmarks/visualization/optimization_plots.py`

## Overview

Visualization tools for optimization benchmark results.

The plots read their result's metadata through
`benchmarks.performance.optimization.training_curve_from_metadata` and calibrax's
`read_metadata`, so they validate a curve exactly as the benchmark wrote it. A figure is saved
at `save_path` when one is given.

## Functions

### plot_convergence_speed

```python
def plot_convergence_speed()
```

### plot_optimizer_comparison

```python
def plot_optimizer_comparison()
```

### plot_training_curve

```python
def plot_training_curve()
```

## Module Statistics

- **Classes:** 0
- **Functions:** 3
- **Imports:** 8
