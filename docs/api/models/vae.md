# VAE API Reference

This page documents the live top-level VAE package surface.

## Public Imports

```python
from flax import nnx
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae import BetaVAE, ConditionalVAE, VAE, VQVAE
```

## Base VAE Contract

The base VAE now uses the typed config surface rather than raw constructor parts.

- `VAE(config: VAEConfig, *, rngs: nnx.Rngs, precision=None)`
- `encode(x)`
- `decode(z)`
- `reparameterize(mean, log_var)`
- `__call__(x)`
- `loss_fn(batch, model_outputs, *, beta=None, reconstruction_loss_fn=None)`
- `sample(n_samples=1, *, temperature=1.0)`

The module uses stored RNG state on the instance, so these base methods no longer
teach call-time RNG arguments. The config owns encoder and decoder structure;
do not construct the base VAE by passing raw encoder, decoder, and latent-dim
objects directly.

## Live Top-Level Exports

- `VAE`
- `BetaVAE`
- `ConditionalVAE`
- `VQVAE`

## Minimal Example

```python
from flax import nnx
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae import VAE

rngs = nnx.Rngs(0)
config = VAEConfig()
model = VAE(config=config, rngs=rngs)
outputs = model(x)
losses = model.loss_fn({'x': x}, outputs)
```
