# Production

**Status:** `Supported runtime inference surface`
**Module:** `artifex.generative_models.inference.optimization.production`
**Source:** `src/artifex/generative_models/inference/optimization/production.py`

## Overview

This page documents the retained experimental production inference helpers.

The current runtime owns one real shared optimization step plus request-level
monitoring:

- `jit_compilation` through `ProductionOptimizer.optimize_for_production(...)`
- request/latency monitoring through `ProductionMonitor` and
  `ProductionPipeline`

Quantization, pruning, caching, and dynamic batching remain internal
placeholders and are not reported as applied optimization techniques.

## Retained Classes

- `OptimizationTarget`
- `OptimizationResult`
- `MonitoringMetrics`
- `ProductionOptimizer`
- `ProductionPipeline`
- `ProductionMonitor`

## Current Semantics

- `optimize_for_production(...)` currently reports only `jit_compilation` in
  `OptimizationResult.optimization_techniques`.
- `ProductionPipeline.predict(...)` and `predict_batch(...)` record request
  count, latency, throughput, and error rate.
- `memory_usage_gb` and `cache_hit_rate` are unavailable in
  `MonitoringMetrics` and remain `None` until live instrumentation exists.

## Example

```python
from flax import nnx
import jax.numpy as jnp

from artifex.generative_models.inference.optimization.production import (
    OptimizationTarget,
    ProductionOptimizer,
)

optimizer = ProductionOptimizer()
target = OptimizationTarget(latency_ms=50.0)
sample_inputs = (jnp.ones((8, 64)),)

result = optimizer.optimize_for_production(model, target, sample_inputs)
assert result.optimization_techniques == ["jit_compilation"]

pipeline = optimizer.create_production_pipeline(model, result)
outputs = pipeline.predict(sample_inputs[0])
metrics = pipeline.get_monitoring_metrics()
```
