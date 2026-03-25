# Centralized Factory System

`artifex.generative_models.factory` is the canonical model-creation surface for
Artifex. It accepts typed dataclass configs, chooses the correct builder from
the config type, and optionally applies modality adaptation and extensions.

## Public Surface

- `create_model`
- `create_model_with_extensions`
- `ModelFactory`

The factory is the only supported public runtime entry point for model
creation. The legacy zoo preset path has been retired and now raises a
migration error instead of loading named presets.

## Basic Usage

```python
from flax import nnx

from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    VAEConfig,
)
from artifex.generative_models.factory import create_model

encoder = EncoderConfig(
    name="vae_encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(256, 128),
)
decoder = DecoderConfig(
    name="vae_decoder",
    output_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(128, 256),
)
config = VAEConfig(
    name="mnist_vae",
    encoder=encoder,
    decoder=decoder,
    kl_weight=1.0,
)

model = create_model(config, rngs=nnx.Rngs(0))
```

## Supported Config Families

The factory infers the builder from the config type.

- `VAEConfig`, `BetaVAEConfig`, `ConditionalVAEConfig`, `VQVAEConfig`
- `DCGANConfig`, `WGANConfig`, `LSGANConfig`, `ConditionalGANConfig`, `CycleGANConfig`
- `DiffusionConfig`, `DDPMConfig`, `ScoreDiffusionConfig`
- `EBMConfig`, `DeepEBMConfig`
- `FlowConfig`
- `AutoregressiveConfig`, `TransformerConfig`, `PixelCNNConfig`, `WaveNetConfig`
- `GeometricConfig`, `PointCloudConfig`, `MeshConfig`, `VoxelConfig`, `GraphConfig`

Base `GANConfig` remains an abstract shared config parent and is rejected by
`create_model(...)`. Use a concrete GAN config subclass instead.

## Extensions

Use `create_model_with_extensions()` when the runtime should materialize both
the model and its extension instances together.

```python
from flax import nnx

from artifex.generative_models.factory import create_model_with_extensions

model, extensions = create_model_with_extensions(
    config,
    extensions_config=extension_configs,
    rngs=nnx.Rngs(0),
)
```

The extension configs should already be typed and validated before they reach
the factory boundary.

## Migrating From Zoo Presets

If you still have code that imports `artifex.generative_models.zoo`, replace it
with an explicit typed config constructor and pass that config to
`create_model()`.

Typical migration shape:

1. Materialize a typed config such as `VAEConfig`, `DDPMConfig`, or `PointCloudConfig`.
2. Keep any project-specific preset naming or YAML lookup in your own code.
3. Call `create_model(config, rngs=...)` with that typed config.

## Builder Architecture

`ModelFactory` owns the internal builder registry and registers one builder per
model family in `factory/builders/`.

- `vae.py`
- `gan.py`
- `diffusion.py`
- `flow.py`
- `ebm.py`
- `autoregressive.py`
- `geometric.py`

Each builder is responsible for translating a typed config into the
corresponding model implementation.

## Adding A New Model Family

1. Add a new builder under `factory/builders/`.
2. Register it in `ModelFactory._register_default_builders()`.
3. Extend config-type dispatch in `ModelFactory._extract_model_type()`.
4. Add focused tests for:
   - builder registration
   - config-type dispatch
   - model creation

The builder-registry internals live in `factory/registry.py` and are reviewed as
implementation details rather than part of the normal top-level package API.

## Contract

- the public factory surface is `artifex.generative_models.factory`
- callers pass typed dataclass configs, not raw dictionaries
- the config type determines the model family
- modality adaptation is optional and explicit
- extension creation is explicit and typed
- named zoo presets are not part of the supported runtime contract
