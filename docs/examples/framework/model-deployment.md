# Model Deployment

Guide to deploying trained Artifex models for inference in production environments.

## Overview

This guide covers strategies for deploying generative models for production inference.

<div class="grid cards" markdown>

- :material-export:{ .lg .middle } **Model Export**

    ---

    Export trained models for deployment

    [:octicons-arrow-right-24: Model Export](#model-export)

- :material-rocket-launch:{ .lg .middle } **Inference Optimization**

    ---

    Optimize models for fast inference

    [:octicons-arrow-right-24: Inference Optimization](#inference-optimization)

- :material-server:{ .lg .middle } **Serving Patterns**

    ---

    Common deployment architectures

    [:octicons-arrow-right-24: Serving Patterns](#serving-patterns)

</div>

---

## Model Export

Save and load trained models for deployment.

```python
from artifex.generative_models.core import save_model, load_model

# Save trained model
save_model(model, "checkpoints/vae_model")

# Load for inference
model = load_model("checkpoints/vae_model")
```

---

## Inference Optimization

Optimize models for production inference speed.

```python
import jax

# JIT compile for faster inference
@jax.jit
def generate(model, rng_key, num_samples):
    return model.sample(num_samples=num_samples, rng=rng_key)

# Warm up JIT compilation
_ = generate(model, jax.random.key(0), 1)

# Fast inference
samples = generate(model, jax.random.key(42), 64)
```

---

## Serving Patterns

Common patterns for serving generative models.

### Batch Processing

```python
def batch_generate(model, batch_size=64, total_samples=1000):
    """Generate samples in batches for efficiency."""
    samples = []
    for i in range(0, total_samples, batch_size):
        batch = model.sample(
            num_samples=min(batch_size, total_samples - i),
            rng=jax.random.key(i),
        )
        samples.append(batch)
    return jnp.concatenate(samples, axis=0)
```

### API Endpoint

```python
from fastapi import FastAPI
import jax.numpy as jnp

app = FastAPI()

@app.post("/generate")
async def generate_samples(num_samples: int = 16):
    samples = model.sample(num_samples=num_samples, rng=jax.random.key(0))
    return {"samples": samples.tolist()}
```

---

## Related Documentation

- [Inference Overview](../../user-guide/inference/overview.md) - Inference fundamentals
- [Optimization Guide](../../user-guide/inference/optimization.md) - Performance optimization
- [Sampling Methods](../../user-guide/inference/sampling.md) - Sampling techniques
