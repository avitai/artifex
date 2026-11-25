# Protein Extensions Example

This example demonstrates the canonical typed bundle workflow for protein-aware
extensions in Artifex.

## Files

- **Python Script**: [`examples/generative_models/protein/protein_extensions_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_example.py)
- **Jupyter Notebook**: [`examples/generative_models/protein/protein_extensions_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_example.ipynb)

## Quick Start

```bash
pip install avitai-artifex
python examples/generative_models/protein/protein_extensions_example.py
```

## What It Shows

- composition with `ProteinExtensionsConfig`
- runtime materialization with `create_protein_extensions()`
- attachment of extensions to `PointCloudModel`
- direct instantiation of individual protein extensions when needed
- explicit protein-shaped `atom_positions` tensors for coordinate-aware helpers

## Core Pattern

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
    bond_length=ProteinExtensionConfig(name="bond_length", bond_length_weight=1.0),
    bond_angle=ProteinExtensionConfig(name="bond_angle", bond_angle_weight=0.5),
    mixin=ProteinMixinConfig(name="protein_mixin", embedding_dim=16),
)
extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```

Coordinate-aware helpers in that bundle consume explicit protein-shaped
`atom_positions` tensors, not flattened point-cloud payloads.
