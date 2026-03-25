# Protein Extensions with Configuration System

This example demonstrates the configuration-driven protein extension workflow in
Artifex.

## Files

- **Python Script**: [`examples/generative_models/protein/protein_extensions_with_config.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_with_config.py)
- **Jupyter Notebook**: [`examples/generative_models/protein/protein_extensions_with_config.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/protein/protein_extensions_with_config.ipynb)

## Quick Start

```bash
pip install artifex
python examples/generative_models/protein/protein_extensions_with_config.py
```

## What It Shows

- loading the shipped protein bundle with `get_protein_extensions_config()`
- customizing a `ProteinExtensionsConfig` in code
- materializing runtime extensions from the typed bundle
- attaching the bundle-backed extensions to a model
- explicit protein-shaped `atom_positions` tensors for coordinate-aware helpers

## Core Pattern

```python
from flax import nnx

from artifex.configs import get_protein_extensions_config
from artifex.generative_models.extensions.protein import create_protein_extensions

bundle = get_protein_extensions_config("protein")
extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```

Coordinate-aware helpers in that bundle consume explicit protein-shaped
`atom_positions` tensors, not flattened point-cloud payloads.
