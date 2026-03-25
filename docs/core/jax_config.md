# Jax Config

**Module:** `generative_models.core.jax_config`

**Source:** `generative_models/core/jax_config.py`

## Overview

JAX configuration optimization for maximum performance.

Current contract:

- `configure_jax()` configures cache, precision, and memory settings
- default calls do not force `XLA_FLAGS`
- deterministic mode adds `--xla_gpu_deterministic_ops=true` explicitly

## Functions

### configure_jax

```python
def configure_jax()
```

## Module Statistics

- **Classes:** 0
- **Functions:** 1
- **Imports:** 5
