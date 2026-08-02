# Transformers

**Module:** `generative_models.core.layers.transformers`

**Source:** `generative_models/core/layers/transformers.py`

## Overview

Transformer encoder and decoder implementations supporting several kinds of
positional encoding, built on the Flax NNX API.

## Dropout and RNG Streams

Passing `dropout_rate > 0` builds a dropout layer, and a plain `nnx.Rngs(seed)`
is enough to drive it: `nnx.Rngs` resolves an unknown stream name to its
`default` stream, so no separately named `dropout` stream is required.

```python
block = TransformerEncoderBlock(hidden_dim=32, num_heads=2, dropout_rate=0.1, rngs=nnx.Rngs(0))
```

An `nnx.Rngs` built only from named streams, such as `nnx.Rngs(params=key)`, has
neither a `dropout` nor a `default` stream and so cannot supply dropout
randomness. Constructing a dropout-bearing block from one raises rather than
quietly disabling the dropout you asked for. Add a stream it can use:

```python
rngs = nnx.Rngs(params=params_key, dropout=dropout_key)
```

## Autoregressive Decoding

`decode=True` uses the cached key/value path, which must be allocated once,
before the first decode step, for the full length the sequence will reach. This
follows the `nnx.MultiHeadAttention` contract:

```python
decoder.init_cache((batch_size, max_length, hidden_dim))

for _ in range(max_length):
    output = decoder(next_token, encoder_output, deterministic=True, decode=True)
```

Calling `init_cache(...)` again discards everything decoded so far, so do not
call it inside the decoding loop. Running a `decode=True` pass without an
allocated cache raises from `nnx.MultiHeadAttention`.

`TransformerDecoderBlock` exposes the same `init_cache(...)` for use on its own.

## Classes

### FeedForwardNetwork

```python
class FeedForwardNetwork
```

### TransformerDecoder

```python
class TransformerDecoder
```

### TransformerDecoderBlock

```python
class TransformerDecoderBlock
```

### TransformerEncoder

```python
class TransformerEncoder
```

### TransformerEncoderBlock

```python
class TransformerEncoderBlock
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

### **call**

```python
def __call__()
```

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

### create_transformer

```python
def create_transformer()
```

## Module Statistics

- **Classes:** 5
- **Functions:** 12
- **Imports:** 6
