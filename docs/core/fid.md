# Fid

**Module:** `generative_models.core.evaluation.metrics.image.fid`

**Source:** `generative_models/core/evaluation/metrics/image/fid.py`

## Overview

Fréchet Inception Distance (FID) implementation using JAX and NNX.

FID is a metric that calculates the distance between feature vectors
of real and generated images. It uses the Inception-v3 network to extract features
and then computes the Fréchet distance between distributions.

## Classes

### FrechetInceptionDistance

```python
class FrechetInceptionDistance
```

## Functions

### **init**

```python
def __init__()
```

### calculate_frechet_distance

```python
def calculate_frechet_distance()
```

### compute

```python
def compute()
```

### feature_extractor

```python
def feature_extractor()
```

## Module Statistics

- **Classes:** 1
- **Functions:** 4
- **Imports:** 6
