# Extension Mechanism for Generative Models

This package hosts the canonical Artifex extension surface. Extensions let the
repo add domain-specific behavior to shared generative-model code without
hard-wiring modality logic into every model implementation.

## Package Structure

```text
extensions/
├── __init__.py
├── base/
│   └── extensions.py
├── protein/
│   ├── __init__.py
│   ├── backbone.py
│   ├── constraints.py
│   ├── mixin.py
│   └── utils.py
├── chemical/
│   ├── __init__.py
│   ├── constraints.py
│   └── features.py
├── vision/
│   ├── __init__.py
│   └── augmentation.py
├── audio_processing/
│   ├── __init__.py
│   ├── spectral.py
│   └── temporal.py
├── nlp/
│   ├── __init__.py
│   ├── embeddings.py
│   └── tokenization.py
└── registry.py
```

## Current Shared Surface

The protein extensions are the reference implementation for the Artifex
extension architecture. They exercise the intended system end to end:

- typed config bundle composition
- explicit runtime materialization
- registry-based discovery
- modality integration without polluting base model packages

That curated protein flow is not the whole supported surface, though. The live
registry also ships broader registry-backed family subpackages for:

- `chemical`
- `vision`
- `audio_processing`
- `nlp`

The top-level `artifex.generative_models.extensions` package remains a curated convenience barrel: it lazily exports the shared registry/base types plus the protein helpers that anchor the canonical typed-bundle workflow. The broader registry-backed family subpackages stay importable through their own package paths and through `get_extensions_registry()`.

## Curated Protein Surface

The currently exported protein extension classes are:

- `BondLengthExtension`
- `BondAngleExtension`
- `ProteinBackboneConstraint`
- `ProteinDihedralConstraint`
- `ProteinMixinExtension`

## Canonical Composition Contract

Use `ProteinExtensionsConfig` as the single supported bundle surface for the
curated protein flow:

```python
from flax import nnx

from artifex.configs import (
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
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
    mixin=ProteinMixinConfig(
        name="protein_mixin",
        embedding_dim=16,
        num_aa_types=21,
    ),
)

extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```

## Registry

Use `get_extensions_registry()` to inspect registered extension classes and
their capabilities:

```python
from artifex.generative_models.extensions import get_extensions_registry

registry = get_extensions_registry()
registry.get_extensions_for_modality("protein")
registry.get_extensions_for_modality("molecular")
registry.get_extensions_for_modality("image")
registry.get_extensions_for_modality("audio")
registry.get_extensions_for_modality("text")
```

The registry is the authoritative discovery surface for the shipped shared
extension families, even though the top-level barrel keeps curated convenience
exports.

## Design Rules

- Base model packages should not provide shadow extension implementations.
- Extension composition should stay typed and explicit.
- Domain-specific defaults should live in config assets, not in ad hoc dict
  shims.
- New extension families should copy the protein pattern rather than invent a
  parallel surface.
