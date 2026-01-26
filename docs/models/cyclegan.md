# Cyclegan

**Module:** `generative_models.models.gan.cyclegan`

**Source:** `generative_models/models/gan/cyclegan.py`

## Overview

CycleGAN implementation for unpaired image-to-image translation.

Based on the paper "Unpaired Image-to-Image Translation using Cycle-Consistent
Adversarial Networks" by Zhu et al. (2017).

CycleGAN learns mappings between two image domains X and Y without requiring
paired training examples. It uses cycle consistency loss to enforce that
translated images can be mapped back to the original domain.

## Classes

### CycleGAN

```python
class CycleGAN
```

### CycleGANDiscriminator

```python
class CycleGANDiscriminator
```

### CycleGANGenerator

```python
class CycleGANGenerator
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

### **init**

```python
def __init__()
```

### compute_cycle_loss

```python
def compute_cycle_loss()
```

### compute_identity_loss

```python
def compute_identity_loss()
```

### generate

```python
def generate()
```

### loss_fn

```python
def loss_fn()
```

## Module Statistics

- **Classes:** 3
- **Functions:** 9
- **Imports:** 9
