# Flash Attention

**Module:** `generative_models.core.layers.flash_attention`

**Source:** `generative_models/core/layers/flash_attention.py`

## Overview

Flash Attention implementation for Flax NNX with kvax optimizations.

This module provides a Flash Attention implementation designed to serve
as a drop-in replacement for Flax NNX's MultiHeadAttention with performance improvements and additional features.

Based on:

- Flash Attention paper: <https://arxiv.org/abs/2205.14135>
- Flash Attention 2: <https://arxiv.org/abs/2307.08691>
- kvax implementation: <https://github.com/nebius/kvax>

## Classes

### AttentionBackend

```python
class AttentionBackend
```

### AttentionMask

```python
class AttentionMask
```

### FlashAttentionConfig

```python
class FlashAttentionConfig
```

### FlashMultiHeadAttention

```python
class FlashMultiHeadAttention
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

### create_attention_mask

```python
def create_attention_mask()
```

### flash_attention_forward_kernel

```python
def flash_attention_forward_kernel()
```

### flash_attention_triton

```python
def flash_attention_triton()
```

### init_cache

```python
def init_cache()
```

### make_causal_mask

```python
def make_causal_mask()
```

### make_segment_mask

```python
def make_segment_mask()
```

## Module Statistics

- **Classes:** 4
- **Functions:** 9
- **Imports:** 20
