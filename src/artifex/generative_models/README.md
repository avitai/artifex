# artifex.generative_models

Artifex generative models are built around three explicit contracts:

- typed frozen dataclass configs
- factory-based model creation
- optional modality and extension composition

The canonical public entry points are:

- `artifex.generative_models.core.configuration` for runtime config families
- `artifex.configs` for retained config assets, templates, and loaders
- `artifex.generative_models.factory` for model creation
- `artifex.generative_models.extensions` for runtime extension instances
- `artifex.generative_models.modalities` for modality adaptation

## Package Layout

```text
generative_models/
├── core/          # shared protocols, configuration, sampling, evaluation
├── factory/       # typed model creation
├── models/        # model family implementations
├── modalities/    # modality adaptation
├── extensions/    # domain-specific runtime extensions
├── training/      # trainers and optimizer/scheduler factories
├── inference/     # inference helpers
├── scaling/       # sharding and scaling utilities
└── zoo/           # retired preset compatibility boundary (migration error only)
```

## Creating A Model

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

The factory accepts typed config objects, not raw dictionaries and not
`model_class`-driven catch-all configs.

The legacy `artifex.generative_models.zoo` preset path is intentionally not a
supported runtime entry point. Importing it yields a migration error rather
than a working preset registry.

## Adding Extensions

```python
from flax import nnx

from artifex.generative_models.core.configuration import (
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
)
from artifex.generative_models.extensions.protein import create_protein_extensions

bundle = ProteinExtensionsConfig(
    name="protein_extensions",
    bond_length=ProteinExtensionConfig(
        name="bond_length",
        weight=1.0,
        bond_length_weight=1.0,
    ),
    bond_angle=ProteinExtensionConfig(
        name="bond_angle",
        weight=0.5,
        bond_angle_weight=0.5,
    ),
)

extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```

## Applying A Modality

```python
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.factory import create_model

config = PointCloudConfig(
    name="protein_point_cloud",
    network=PointCloudNetworkConfig(
        name="protein_point_cloud_network",
        hidden_dims=(256, 256, 256),
        embed_dim=256,
        num_heads=8,
        num_layers=6,
    ),
    num_points=1024,
)

model = create_model(config, modality="protein", rngs=nnx.Rngs(0))
```

## Related Package Docs

- [factory/README.md](factory/README.md)
- [extensions/README.md](extensions/README.md)
- [modalities/README.md](modalities/README.md)
- [core/README.md](core/README.md)
- [models/energy/README.md](models/energy/README.md)
