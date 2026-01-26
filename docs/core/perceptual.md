# Perceptual

**Module:** `generative_models.core.losses.perceptual`

**Source:** `generative_models/core/losses/perceptual.py`

## Overview

Perceptual losses module.

This module provides loss functions that compare features extracted from
neural networks, rather than direct pixel-wise comparisons. These losses
are especially useful for image generation tasks.

## Classes

### PerceptualLoss

```python
class PerceptualLoss
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

### compute_contextual_loss_single

```python
def compute_contextual_loss_single()
```

### contextual_loss

```python
def contextual_loss()
```

### create_vgg_perceptual_loss

```python
def create_vgg_perceptual_loss()
```

### feature_reconstruction_loss

```python
def feature_reconstruction_loss()
```

### style_loss

```python
def style_loss()
```

## Module Statistics

- **Classes:** 1
- **Functions:** 7
- **Imports:** 6
