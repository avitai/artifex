# Multi Beta Vae Suite

**Module:** `benchmarks.suites.multi_beta_vae_suite`

**Source:** `benchmarks/suites/multi_beta_vae_suite.py`

## Overview

Multi-β VAE controllable generation benchmark suite.

This module provides a comprehensive benchmark suite for evaluating
multi-β VAE controllable generation models, targeting the Week 9-12 objectives:

- MIG Score >0.3 (Mutual Information Gap for disentanglement)
- FID Score <50 on CelebA (Fréchet Inception Distance)
- Reconstruction Quality: LPIPS <0.2, SSIM >0.8
- Training Efficiency: <8h per epoch on CelebA subset

## Classes

### MultiBetaVAEBenchmark

```python
class MultiBetaVAEBenchmark
```

### MultiBetaVAEBenchmarkSuite

```python
class MultiBetaVAEBenchmarkSuite
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

### run

```python
def run()
```

### run_all

```python
def run_all()
```

## Module Statistics

- **Classes:** 2
- **Functions:** 4
- **Imports:** 10
