# Sampling Methods

Sampling in Artifex stays family-owned. The runtime does not ship a second
generic sampler framework on top of the model families, so the supported entry
points are the methods owned by each retained model package.

## Live Sampling Surface

| Family | Live owners | Retained sampling entry points |
| --- | --- | --- |
| VAE | `artifex.generative_models.models.vae` | `sample(...)`, `reconstruct(...)`, `decode(...)` |
| GAN | `artifex.generative_models.models.gan` | `generate(...)` |
| Diffusion | `artifex.generative_models.models.diffusion` | `generate(...)`, `DDPMModel.sample(..., scheduler="ddim")` |
| Flow | `artifex.generative_models.models.flow` | `sample(...)`, `generate(...)`, `log_prob(...)` |

There is no exported `FlowModel` alias and no standalone `DDIMSampler`,
`DPMSolver`, or `StyleGANSampler` helper in the current runtime.

## Imports

```python
import jax
from flax import nnx

from artifex.generative_models.models.diffusion import DDPMModel
from artifex.generative_models.models.flow import NormalizingFlow
from artifex.generative_models.models.gan import GAN
from artifex.generative_models.models.vae import VAE
```

## VAE Sampling And Reconstruction

```python
def sample_vae(vae: VAE, batch: jax.Array) -> tuple[jax.Array, jax.Array]:
    samples = vae.sample(8, temperature=0.8)
    reconstructions = vae.reconstruct(batch, deterministic=True)
    return samples, reconstructions
```

## GAN Generation

```python
def sample_gan(gan: GAN) -> jax.Array:
    return gan.generate(8)
```

## Diffusion Generation

Use the retained family owner directly. `generate(...)` keeps the default DDPM
path, while `DDPMModel.sample(..., scheduler="ddim")` is the live fast-path
entry point when you want fewer denoising steps.

```python
def sample_ddpm(ddpm: DDPMModel) -> tuple[jax.Array, jax.Array]:
    full_schedule = ddpm.generate(4)
    fast_schedule = ddpm.sample(4, scheduler="ddim", steps=50)
    return full_schedule, fast_schedule
```

## Flow Sampling And Scoring

Flows keep both generation and density evaluation on the model itself.

```python
def sample_flow(flow: NormalizingFlow, batch: jax.Array) -> tuple[jax.Array, jax.Array]:
    samples = flow.sample(8, rngs=nnx.Rngs(3))
    log_probs = flow.log_prob(batch, rngs=nnx.Rngs(4))
    return samples, log_probs
```

## Practical Rule

If a sampling example needs a new wrapper class, exported helper namespace, or
cross-family orchestration object, check the runtime owner first. Keep the docs
on the family method that already exists unless a real shared inference owner is
added to the codebase.
