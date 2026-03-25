# Distribution Base

**Module:** `generative_models.core.distributions.base`

**Source:** `generative_models/core/distributions/base.py`

## Overview

Concrete shared base class for Artifex probability distributions.

This module owns the reusable runtime behavior for:

- sampling with `nnx.Rngs`
- log-probability evaluation
- entropy and KL caching
- numerical-stability helpers
- batching and chunked vectorization helpers

## Classes

### Distribution

```python
class Distribution
```

Shared concrete distribution foundation for Artifex distribution modules.

## Key Methods

### sample

```python
def sample()
```

Draw samples using the distribution's RNG-aware runtime.

### log_prob

```python
def log_prob()
```

Compute log probabilities with parameter validation and finite-value checks.

### entropy

```python
def entropy()
```

Compute entropy with optional cache reuse in non-JIT contexts.

### kl_divergence

```python
def kl_divergence()
```

Compute KL divergence against another Artifex distribution with optional cache reuse.
