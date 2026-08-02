# Flash Attention

**Module:** `generative_models.core.layers.flash_attention`

**Source:** `generative_models/core/layers/flash_attention.py`

## Overview

`FlashMultiHeadAttention` is a drop-in replacement for `flax.nnx.MultiHeadAttention`.
It subclasses it rather than reimplementing it, so the projections, decode cache,
sown attention weights and dropout are the reference implementations. The only
things this module adds are the choice of attention kernel and a `causal` flag.

Two kernels are reachable, both through the public JAX API:

| Backend | Used when | Notes |
| --- | --- | --- |
| `cudnn` | Inputs are eligible and nothing forces the portable path | NVIDIA's fused kernel |
| `xla`   | Otherwise | Delegates to `flax.nnx.nn.attention.dot_product_attention` |

Selection happens in artifex because `jax.nn.dot_product_attention` does not
choose for itself: passing `implementation=None` always takes the XLA path.

Based on:

- Flash Attention paper: <https://arxiv.org/abs/2205.14135>
- Flash Attention 2: <https://arxiv.org/abs/2307.08691>

## Eligibility

The fused kernel is used only when every one of these holds:

- The computation dtype is `bfloat16` or `float16`. **`float32` is rejected by
  cuDNN**, and `param_dtype` defaults to `float32`, so the default configuration
  runs on the portable path. Set `dtype=jnp.bfloat16` to reach the fused kernel.
- `head_dim` is a multiple of 8, and at most 128 below sm90 or 256 from sm90 up.
- A CUDA device is present.

Two capabilities always force the portable path:

- **`sow_weights=True`** — a fused kernel never materialises the attention
  matrix, so there is nothing to sow.
- **Live dropout** — the fused kernel takes its dropout seed as a compile-time
  constant and serialises it into the custom-call backend config. A jitted
  training step would reuse a single dropout mask for every step, so fusing live
  dropout would be silently wrong rather than merely fast. Inference and training
  with `dropout_rate=0.0` are unaffected.

Call `resolve_backend()` to see which kernel a configuration will reach:

```python
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.layers import FlashMultiHeadAttention

attention = FlashMultiHeadAttention(
    num_heads=8,
    in_features=512,
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    decode=False,
    rngs=nnx.Rngs(0),
)
attention.resolve_backend()  # AttentionBackend.CUDNN on an eligible GPU
```

## Padding

Padding is expressed with sequence lengths rather than segment identifiers,
because cuDNN accelerates the former natively:

```python
from artifex.generative_models.core.layers import flash_dot_product_attention

output = flash_dot_product_attention(
    query, key, value,
    query_seq_lengths=lengths,
    key_value_seq_lengths=lengths,
)
```

Packing several documents into one sequence is not supported. The public JAX API
exposes no packed layout, and the private one requires Hopper.

## Classes

### FlashMultiHeadAttention

```python
class FlashMultiHeadAttention(nnx.MultiHeadAttention)
```

Accepts every `nnx.MultiHeadAttention` argument, plus:

- `causal: bool = False` — apply causal masking without building an explicit
  mask. The fused kernel skips the masked region rather than computing and
  discarding it.

### AttentionBackend

```python
class AttentionBackend(StrEnum)
```

Members: `CUDNN`, `XLA`.

## Functions

### flash_dot_product_attention

```python
def flash_dot_product_attention(query, key, value, bias=None, mask=None, *, ...)
```

Satisfies the `attention_fn` contract of `nnx.MultiHeadAttention`, so it can be
passed to any nnx attention module.

### is_cudnn_eligible

```python
def is_cudnn_eligible(*, dtype, head_dim, device_kind, compute_capability) -> bool
```

Pure predicate over the cuDNN constraints, so the rules stay testable without a GPU.

### select_attention_backend

```python
def select_attention_backend(*, dtype, head_dim, device_kind, compute_capability,
                             deterministic, dropout_rate, sow_weights) -> AttentionBackend
```

Applies the eligibility rules together with the two forced-fallback conditions.
