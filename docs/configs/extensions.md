# Extension Configs

Artifex configures extension composition through typed frozen dataclasses. The
canonical protein surface is `ProteinExtensionsConfig`, which bundles the
optional protein-aware extensions that can be attached to a model.

## Public Imports

```python
from artifex.configs import (
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
    get_protein_extensions_config,
)
```

## Shipped Protein Bundle

The retained default protein extension bundle lives at
`src/artifex/configs/defaults/extensions/protein.yaml` and loads as a real typed
config object:

```python
from artifex.configs import get_protein_extensions_config

bundle = get_protein_extensions_config("protein")
```

## Programmatic Composition

```python
from artifex.configs import (
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
)

bundle = ProteinExtensionsConfig(
    name="protein_extensions",
    bond_length=ProteinExtensionConfig(
        name="bond_length",
        weight=1.0,
        bond_length_weight=1.0,
        ideal_bond_lengths={"N-CA": 1.45, "CA-C": 1.52, "C-N": 1.33},
    ),
    bond_angle=ProteinExtensionConfig(
        name="bond_angle",
        weight=0.5,
        bond_angle_weight=0.5,
        ideal_bond_angles={"CA-C-N": 2.025, "C-N-CA": 2.11, "N-CA-C": 1.94},
    ),
    mixin=ProteinMixinConfig(
        name="protein_mixin",
        embedding_dim=16,
        num_aa_types=21,
    ),
)
```

## Runtime Materialization

```python
from flax import nnx

from artifex.generative_models.extensions.protein import create_protein_extensions

extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```

## Current Config Types

- `ExtensionConfig`
- `ConstraintExtensionConfig`
- `ProteinExtensionConfig`
- `ProteinExtensionsConfig`
- `ProteinMixinConfig`
- `ArchitectureExtensionConfig`
- `SamplingExtensionConfig`
- `LossExtensionConfig`
- `EvaluationExtensionConfig`
