# Regularization

**Module:** `generative_models.core.losses.regularization`

**Source:** `generative_models/core/losses/regularization.py`

## Overview

Regularization losses module.

This module provides regularization terms that can be added to other losses
to improve model stability, generalization, and prevent overfitting.
All functions are JAX-compatible and work with NNX modules.

## Classes

### DropoutRegularization

```python
class DropoutRegularization
```

### SpectralNormRegularization

```python
class SpectralNormRegularization
```

## Functions

### **call**

```python
def __call__()
```

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

### disc_interpolated

```python
def disc_interpolated()
```

### exclude_bias_predicate

```python
def exclude_bias_predicate()
```

### exclude_norm_predicate

```python
def exclude_norm_predicate()
```

### gradient_penalty

```python
def gradient_penalty()
```

### l1_regularization

```python
def l1_regularization()
```

### l2_regularization

```python
def l2_regularization()
```

### only_conv_predicate

```python
def only_conv_predicate()
```

### orthogonal_regularization

```python
def orthogonal_regularization()
```

### spectral_norm_regularization

```python
def spectral_norm_regularization()
```

### total_variation_loss

```python
def total_variation_loss()
```

## Module Statistics

- **Classes:** 2
- **Functions:** 14
- **Imports:** 5
