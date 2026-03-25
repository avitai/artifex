# Modality Framework

The modality layer adapts typed model configs and shared model families to
domain-specific data without copying model implementations into every domain
package.

## What A Modality Owns

- domain-specific preprocessing metadata
- model adapters for supported model families
- modality-specific runtime extensions
- evaluation and representation helpers where needed

## Registry Surface

```python
from artifex.generative_models.modalities import (
    get_modality,
    list_modalities,
)

available = list_modalities()
protein_modality = get_modality("protein", rngs=rngs)
```

## Factory Integration

Pass the typed model config to the factory and specify the modality explicitly:

```python
from flax import nnx

from artifex.configs import PointCloudConfig, PointCloudNetworkConfig
from artifex.generative_models.factory import create_model

config = PointCloudConfig(
    name="protein_point_cloud",
    network=PointCloudNetworkConfig(
        name="point_cloud_network",
        hidden_dims=(256, 256, 256),
        embed_dim=256,
        num_heads=8,
        num_layers=6,
    ),
    num_points=1024,
)

model = create_model(config, modality="protein", rngs=nnx.Rngs(0))
```

For `modality="protein"`, the shared factory still returns the generic model family selected by the typed config. It does not swap in `ProteinPointCloudModel` or `ProteinGraphModel`; retained protein-specific runtime behavior lives on the typed protein extension bundle and adapter utilities.

## Modality Configs

Use `ModalityConfig` when the modality itself needs structured metadata:

```python
from artifex.configs import ModalityConfig
from artifex.generative_models.modalities.text import TextModality
from flax import nnx

config = ModalityConfig(
    name="text_modality",
    modality_name="text",
    metadata={
        "text_params": {
            "vocab_size": 10000,
            "max_length": 512,
            "pad_token_id": 0,
            "unk_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
        }
    },
)

text_modality = TextModality(config=config, rngs=nnx.Rngs(0))
```

## Design Rules

- model configs stay typed and model-family specific
- modality choice is explicit at the factory boundary
- domain behavior belongs in modality and extension packages, not model base
  classes
- new modalities should follow the existing registry + adapter pattern
