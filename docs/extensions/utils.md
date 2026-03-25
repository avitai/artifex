# Utils

**Module:** `generative_models.extensions.protein.utils`

## Overview

`create_protein_extensions()` is the canonical materialization helper for the
protein extension bundle. It expects a typed `ProteinExtensionsConfig` and
returns the live extension collection used by models and modality helpers.

## Example

```python
from flax import nnx

from artifex.configs import get_protein_extensions_config
from artifex.generative_models.extensions.protein import create_protein_extensions

bundle = get_protein_extensions_config("protein")
extensions = create_protein_extensions(bundle, rngs=nnx.Rngs(0))
```
