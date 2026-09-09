# Image

**Module:** `benchmarks.metrics.image`

**Source:** `benchmarks/metrics/image.py`

## Overview

Image metrics for generative models.

This module provides metrics for evaluating the quality of generated images,
including FID, IS (Inception Score), LPIPS, and SSIM metrics for image generation.

## Classes

### FIDMetric

```python
class FIDMetric
```

### ISMetric

```python
class ISMetric
```

### LPIPSMetric

```python
class LPIPSMetric
```

### MockInceptionModel

```python
class MockInceptionModel
```

Demo-only stand-in for an Inception backbone: one key for the life of the model, so
the same images map to the same features and logits.

### SSIMMetric

```python
class SSIMMetric
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

### compute

```python
def compute()
```

### compute

```python
def compute()
```

### compute

```python
def compute()
```

### compute

```python
def compute()
```

### create_fid_metric

```python
def create_fid_metric()
```

### create_is_metric

```python
def create_is_metric()
```

### fid_from_backbone

```python
def fid_from_backbone(real, generated, *, feature_extractor) -> float
```

### inception_score_from_backbone

```python
def inception_score_from_backbone(generated, *, classifier, splits=10) -> float
```

Both register in `calibrax.metrics.MetricRegistry` as `frozen_backbone` metrics;
`ISMetric.compute` reports `inception_score` and `inception_score_std` over the
splits and raises when `splits` exceeds the sample count.

### create_lpips_metric

```python
def create_lpips_metric()
```

### create_ssim_metric

```python
def create_ssim_metric()
```

### extract_features

```python
def extract_features()
```

### validate_inputs

```python
def validate_inputs()
```

### validate_inputs

```python
def validate_inputs()
```

### validate_inputs

```python
def validate_inputs()
```

### validate_inputs

```python
def validate_inputs()
```

## Module Statistics

- **Classes:** 5
- **Functions:** 18
- **Imports:** 10
