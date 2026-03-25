# Flash Attention

**Module:** `generative_models.core.layers.flash_attention`

**Source:** `generative_models/core/layers/flash_attention.py`

## Overview

Flash-attention-style helpers for Flax NNX.

This page documents the retained single JAX fallback implementation. It does not
publish backend switches, Triton-specific runtime guarantees, or broader
performance claims beyond the code that actually ships in this repository.

Based on:

- Flash Attention paper: <https://arxiv.org/abs/2205.14135>
- Flash Attention 2: <https://arxiv.org/abs/2307.08691>

## Classes

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

### create_attention_mask

```python
def create_attention_mask()
```

### flash_attention

```python
def flash_attention()
```

### init_cache

```python
def init_cache()
```

## Module Statistics

- **Classes:** 3
- **Functions:** 5
