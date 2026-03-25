# Extensions

Extensions are how Artifex adds domain-specific behavior to otherwise
domain-agnostic models. The protein extension surface remains the design
reference for typed composition, but the supported shared extension surface is
broader than protein alone.

## Design Principles

- Base models stay domain-agnostic.
- Domain knowledge is attached explicitly as extensions.
- Composition is typed and validated through config objects.
- Registries expose discovery and factory hooks without hard-coding modality
  knowledge into every model package.
- The top-level `artifex.generative_models.extensions` barrel stays curated so
  import-time costs remain low while shipped family subpackages remain public.

## Current Extension Scope

Today Artifex ships one curated protein bundle flow plus broader registry-backed
extension families:

- `protein`: canonical typed bundle flow and the reference implementation
- `chemical`: molecular validation and descriptor helpers
- `vision`: image augmentation helpers
- `audio_processing`: spectral and temporal analysis helpers
- `nlp`: tokenization and embedding helpers

The top-level `artifex.generative_models.extensions` barrel stays curated: it
exports the shared registry/base types plus protein convenience helpers. The
other shipped families are supported through their own subpackages and through
`ExtensionsRegistry`.

## Protein Extensions

The protein surface currently includes:

- `ProteinMixinExtension`
- `BondLengthExtension`
- `BondAngleExtension`
- `ProteinBackboneConstraint`
- `ProteinDihedralConstraint`

## Canonical Config Surface

Protein extensions compose through the typed `ProteinExtensionsConfig` bundle.

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
    ),
    bond_angle=ProteinExtensionConfig(
        name="bond_angle",
        weight=0.5,
        bond_angle_weight=0.5,
    ),
    mixin=ProteinMixinConfig(
        name="protein_mixin",
        embedding_dim=16,
        num_aa_types=21,
    ),
)
```

## Runtime Composition

```python
from flax import nnx

from artifex.generative_models.extensions.protein import create_protein_extensions

extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```

Those materialized extensions can then be passed into a model:

```python
model = PointCloudModel(model_config, extensions=extensions, rngs=nnx.Rngs(0))
```

## Registry

The registry is the discovery surface for extension classes and capabilities:

```python
from artifex.generative_models.extensions import get_extensions_registry

registry = get_extensions_registry()
registry.list_all_extensions()
registry.get_extensions_for_modality("protein")
registry.get_extensions_for_modality("molecular")
registry.get_extensions_for_modality("image")
registry.get_extensions_for_modality("audio")
registry.get_extensions_for_modality("text")
```

`ExtensionsRegistry` currently auto-registers the canonical protein extension
classes plus the shipped `chemical`, `vision`, `audio_processing`, and `nlp`
family entries so modality and factory layers can discover them without reaching
into implementation-specific packages.

## Recommended Usage Pattern

1. Use the top-level barrel for registry/base types and protein convenience
   helpers.
2. Load or construct a `ProteinExtensionsConfig` when you need the curated
   protein bundle flow.
3. Use family subpackages such as `extensions.chemical`, `extensions.vision`,
   `extensions.audio_processing`, and `extensions.nlp` when you need those
   shipped helpers directly.
4. Use `get_extensions_registry()` when you need the authoritative discovery
   surface across all shipped extension families.

## Related Documentation

- [Config Surface](../configs/extensions.md)
- [Protein Extensions Guide](../guides/protein-extensions.md)
- [Registry](registry.md)
- [Protein API](../api/extensions/protein.md)
