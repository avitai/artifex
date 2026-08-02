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

`step(...)` reports `x0_prediction`, `mean`, and `variance` for the reverse
transition. It does not derive those quantities itself: the sampler builds a
`NoiseSchedule` from its `beta_schedule`, `beta_start`, `beta_end`, and
`num_timesteps` arguments and delegates to `predict_start_from_noise(...)` and
`q_posterior_mean_variance(...)`. The schedule algebra therefore exists in one
place, and `sampler.noise_schedule` is available if you need the underlying
arrays. The familiar `betas`, `alphas`, `alphas_cumprod`, `posterior_variance`
and related attributes remain on the sampler.
