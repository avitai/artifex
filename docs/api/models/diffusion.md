# Diffusion Models API Reference

This page documents the live top-level diffusion package surface.

## Public Imports

```python
from artifex.generative_models.models.diffusion import (
    ClassifierFreeGuidance,
    ClassifierGuidance,
    ConditionalDiffusionMixin,
    DDPMModel,
    DiTModel,
    DiffusionModel,
    GuidedDiffusionModel,
    LDMModel,
    ScoreDiffusionModel,
    apply_guidance,
    cosine_guidance_schedule,
    linear_guidance_schedule,
)
```

## Shared Runtime Contract

The live base diffusion surface uses stored module RNG state and NNX train/eval
mode management.

- `DiffusionModel.__call__(x, timesteps, *, conditioning=None, **kwargs)`
- `DiffusionModel.q_sample(x_start, t, noise=None)`
- `model.train()`
- `model.eval()`

## DDPMModel

`DDPMModel` is the canonical denoising diffusion implementation currently
exported from the top-level package.

- `DDPMModel.sample(n_samples_or_shape, scheduler="ddpm", steps=None)`
- `DDPMConfig.input_shape` uses the public HWC convention and must match the
  configured backbone channel count

## Other Live Exports

- `ScoreDiffusionModel`
- `LDMModel`
- `DiTModel`
- `ClassifierFreeGuidance`
- `ClassifierGuidance`
- `GuidedDiffusionModel`
- `ConditionalDiffusionMixin`
- `apply_guidance(...)`
- `linear_guidance_schedule(...)`
- `cosine_guidance_schedule(...)`

## Minimal Example

```python
from flax import nnx
from artifex.generative_models.core.configuration import DDPMConfig
from artifex.generative_models.models.diffusion import DDPMModel

rngs = nnx.Rngs(0)
model = DDPMModel(config=DDPMConfig(), rngs=rngs)
model.train()
losses = model.loss_fn({'x': x})
model.eval()
samples = model.sample(4)
```

`loss_fn` runs the forward diffusion and the denoising pass itself. A prediction
alone does not identify the timestep and noise it was produced from, so there is
nothing meaningful to score in a finished output dictionary; the objective has to
draw the pair and evaluate the model on that exact corruption. The reconstruction
loss it applies comes from `DDPMConfig.loss_type` (`mse`, `l1`, or `huber`).

Call the model directly when you want its outputs:

```python
outputs = model(x, timesteps)
```
