# Diffusion

**Module:** `artifex.generative_models.core.sampling.diffusion`

**Source:** `src/artifex/generative_models/core/sampling/diffusion.py`

## Overview

`DiffusionSampler` retains two supported responsibilities:

- stateful DDPM-style stepping through `init(...)` and `step(...)`
- a wrapper-only `sample(...)` entrypoint that delegates to `model.sample(...)`

The public `sample(...)` method does not implement a standalone generic
direct-sampling path. If you want `DiffusionSampler.sample(...)`, initialize the
sampler with a model that already owns a real `sample(...)` implementation.

## Supported Sampling Contract

- `DiffusionSampler.sample(...)` is wrapper-only.
- It delegates to `model.sample(...)` when the sampler was created with a
  compatible model owner.
- It forwards `scheduler`, optional `steps`, and optional `rngs` to that model.
- Without a model-owned sampling implementation, the method raises
  `NotImplementedError` instead of pretending a generic fallback exists.

## Stepper Contract

`init(...)` and `step(...)` still provide the retained low-level diffusion
stepping utility for explicit state dictionaries containing `x`, `key`, and `t`.
Use this surface when you want to own the outer loop yourself.
