# Sampling API Reference

API reference for sampling methods and utilities in Artifex.

## Overview

This module provides sampling utilities for generating outputs from trained generative models.

## Core Sampling Functions

### Basic Sampling

```python
from artifex.generative_models.core.sampling import sample_from_model

samples = sample_from_model(
    model=model,
    num_samples=64,
    rng=jax.random.key(0),
)
```

### Temperature Scaling

```python
from artifex.generative_models.core.sampling import temperature_sample

samples = temperature_sample(
    model=model,
    num_samples=64,
    temperature=0.8,
    rng=jax.random.key(0),
)
```

## Model-Specific Sampling

For detailed sampling methods for each model type, see:

- [VAE Sampling](../user-guide/inference/sampling.md#vae-sampling-methods)
- [GAN Sampling](../user-guide/inference/sampling.md#gan-sampling-methods)
- [Diffusion Sampling](../user-guide/inference/sampling.md#diffusion-sampling-methods)
- [Flow Sampling](../user-guide/inference/sampling.md#flow-sampling-methods)

## Related Documentation

- [Sampling User Guide](../user-guide/inference/sampling.md) - Comprehensive sampling guide
- [Inference Optimization](../user-guide/inference/optimization.md) - Performance optimization
- [Core API](core/base.md) - Core module reference
