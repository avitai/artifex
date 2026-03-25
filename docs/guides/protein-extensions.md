# Protein Extensions Guide

Protein extensions are the current reference implementation of the Artifex
extension architecture. They let protein-aware behavior live outside the base
model implementations while still integrating cleanly with training, inference,
and modality code.

## What Protein Extensions Provide

- bond-length monitoring
- bond-angle monitoring
- backbone constraints
- dihedral constraints
- amino-acid feature injection

## Canonical Composition Flow

1. Construct or load a `ProteinExtensionsConfig`.
2. Materialize runtime extensions with `create_protein_extensions()`.
3. Attach the resulting extension collection to a model.

```python
from flax import nnx

from artifex.configs import get_protein_extensions_config
from artifex.generative_models.extensions.protein import create_protein_extensions

bundle = get_protein_extensions_config("protein")
extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```

## Why the Bundle Matters

Artifex intentionally uses a typed bundle instead of a free-form mapping:

- the config shape is discoverable
- nested extension configs are validated
- YAML and programmatic construction share the same surface
- modality code and examples do not need translation helpers

## When to Instantiate Extensions Directly

Direct construction is still appropriate when you need to test or study one
extension in isolation:

```python
from flax import nnx

from artifex.configs import ProteinExtensionConfig
from artifex.generative_models.extensions.protein import BondLengthExtension

extension = BondLengthExtension(
    ProteinExtensionConfig(
        name="bond_length",
        weight=1.0,
        bond_length_weight=1.0,
    ),
    rngs=nnx.Rngs(0),
)
```

## Design Direction

Protein extensions are the template for future extension families:

- one canonical package
- one canonical config bundle
- one registry surface
- no duplicate compatibility implementations inside model packages
