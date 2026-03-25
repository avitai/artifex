# Inference Overview

Artifex does not currently ship one shared inference framework across every
model family. Loading and generation stay family-owned, while the only separate
inference package surface today is the experimental production optimizer in
`artifex.generative_models.inference.optimization.production`.

## What Exists Today

- Family-owned loading: build the concrete model from its typed config, then
  restore weights with `setup_checkpoint_manager(...)` and `load_checkpoint(...)`.
- Family-owned generation: call the retained model-native methods such as
  `generate(...)`, `sample(...)`, `encode(...)`, `decode(...)`, or
  `log_prob(...)`.
- Experimental production optimization: `ProductionOptimizer` measures and
  wraps compiled inference calls; only `jit_compilation` currently changes
  runtime behavior.

## Load A Concrete Family Model From Checkpoint

The loading pattern is: build the same typed model template you trained, then
restore Orbax checkpoint state into that template.

```python
from flax import nnx

from artifex.generative_models.core.checkpointing import (
    load_checkpoint,
    setup_checkpoint_manager,
)
from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    VAEConfig,
)
from artifex.generative_models.models.vae import VAE


def build_vae_template() -> VAE:
    latent_dim = 32
    vae_config = VAEConfig(
        name="mnist-vae",
        encoder=EncoderConfig(
            input_shape=(28, 28, 1),
            latent_dim=latent_dim,
            hidden_dims=(256, 128),
            activation="relu",
        ),
        decoder=DecoderConfig(
            latent_dim=latent_dim,
            output_shape=(28, 28, 1),
            hidden_dims=(128, 256),
            activation="relu",
            output_activation="sigmoid",
        ),
        encoder_type="dense",
    )
    return VAE(vae_config, rngs=nnx.Rngs(0))


def load_vae_from_checkpoint(checkpoint_dir: str) -> tuple[VAE, int]:
    checkpoint_manager, _ = setup_checkpoint_manager(checkpoint_dir)
    model_template = build_vae_template()
    restored_model, step = load_checkpoint(checkpoint_manager, model_template)

    if restored_model is None or step is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    return restored_model, step
```

The same pattern applies to other retained families: instantiate the real model
owner with its typed config, then restore checkpoint state into that template.

## Use Family-Owned Inference Entry Points

After loading, keep inference on the model family that owns the surface.
Batched inference is just ordinary batched JAX arrays passed to the retained
methods.

```python
import jax.numpy as jnp

vae, step = load_vae_from_checkpoint("./checkpoints/vae")

samples = vae.sample(8, temperature=0.8)
reconstructions = vae.reconstruct(jnp.zeros((8, 28, 28, 1)), deterministic=True)
```

## Streaming Boundary

Artifex does not currently ship a shared streaming helper or generic async
inference server. If you need request streaming, WebSocket delivery, thread
pools, or application-specific backpressure, build that orchestration in your
application layer around the family-owned methods above.

## Experimental Production Optimization

If you need the retained shared inference package, use the experimental
production optimizer directly and treat the result as a thin compiled pipeline
wrapper, not as a full serving framework.

```python
import jax.numpy as jnp

from artifex.generative_models.inference.optimization.production import (
    OptimizationTarget,
    ProductionOptimizer,
)

optimizer = ProductionOptimizer()
target = OptimizationTarget(latency_ms=50.0)

result = optimizer.optimize_for_production(
    model=vae,
    optimization_target=target,
    sample_inputs=(jnp.zeros((1, 28, 28, 1)),),
)

assert result.optimization_techniques == ["jit_compilation"]
```

See [optimization.md](optimization.md) for the narrower experimental optimizer
surface that exists today.
