# SE3 Molecular Flow

**Module:** `generative_models.models.flow.se3_molecular`

**Source:** `generative_models/models/flow/se3_molecular.py`

## Overview

`SE3MolecularFlow` is a simplified molecular conformation flow baseline built
from geometry-conditioned helper layers and affine coordinate transforms.

The current runtime is useful as a molecular flow baseline, but it does not
provide verified global rotation or translation guarantees.

## Public Surface

### `SE3MolecularFlow`

Module-local molecular flow model for conformation scoring and sampling.

Retained behavior:

- `log_prob(coordinates, atom_types, atom_mask)` scores molecular coordinates
  under the current affine-coupling flow
- `sample(atom_types, atom_mask, num_samples, *, rngs)` draws molecular
  coordinate samples for the provided atom template
- `generate(n_samples=1, *, rngs, atom_types=..., atom_mask=...)` wraps the
  sampling path required by the shared generative-model interface
- `loss_fn(batch, model_outputs, **kwargs)` returns the canonical flow loss
  dictionary with `total_loss`, `nll_loss`, `log_prob`, and `avg_log_prob`

Helper classes in this module remain implementation details for the retained
molecular flow runtime.

## Example

```python
from flax import nnx
import jax.numpy as jnp

from artifex.generative_models.models.flow.se3_molecular import SE3MolecularFlow

model = SE3MolecularFlow(
    hidden_dim=64,
    num_layers=3,
    num_coupling_layers=4,
    max_atoms=29,
    atom_types=5,
    rngs=nnx.Rngs(0),
)

atom_types = jnp.ones((2, 29), dtype=jnp.int32)
atom_mask = jnp.ones((2, 29), dtype=jnp.bool_)
samples = model.generate(2, rngs=nnx.Rngs(1), atom_types=atom_types, atom_mask=atom_mask)
```
