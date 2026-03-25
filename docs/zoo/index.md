# Model Zoo Migration

The legacy Artifex model zoo has been removed as a live runtime surface.

`artifex.generative_models.zoo` remains only as a compatibility boundary that
raises a migration error. It does not provide a working preset registry,
configuration listing API, or a live zoo model-construction API anymore.

## Why It Was Removed

The old zoo taught a generic preset and `model_class` story that no longer
matched the supported factory contract. The canonical runtime path is now:

1. materialize a family-specific typed config
2. pass that typed config to `artifex.generative_models.factory.create_model`

That keeps model creation aligned with the actual builders the runtime supports.

## Supported Replacement

Use a concrete config family such as `VAEConfig`, `DDPMConfig`, `WGANConfig`,
or `PointCloudConfig`.

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
    latent_dim=32,
    output_shape=(28, 28, 1),
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

## Migrating Existing Presets

If your project used named zoo presets, move that lookup into your own project
code and make it return typed configs.

Recommended pattern:

- keep preset names in your application layer
- map each preset name to a typed config constructor or YAML loader
- call `create_model(config, rngs=...)` after materializing that typed config

## YAML And Template Guidance

If you still want reusable presets, keep them as project-local typed config
documents and load them with the matching config class. For example:

- `VAEConfig.from_dict(...)`
- `DDPMConfig.from_dict(...)`
- `PointCloudConfig.from_dict(...)`

Do not rebuild a generic universal preset registry around `model_class`
strings.

## Summary

- the old zoo runtime has been removed
- the supported creation path is typed config plus factory
- project-specific presets should stay outside `artifex.generative_models.zoo`

- ❌ Modify zoo configs directly (use overrides instead)
- ❌ Use generic names that might conflict

## Related Documentation

- [Model Factory](../factory/index.md) - Low-level model creation
- [Configuration System](../user-guide/training/configuration.md) - Configuration details
- [Model Gallery](../models/index.md) - All available models
