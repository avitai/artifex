# Glow

**Module:** `generative_models.models.flow.glow`

**Source:** `generative_models/models/flow/glow.py`

## Overview

`Glow` is the retained single-scale image normalizing flow baseline.

The current runtime applies a stack of Glow blocks at the fixed channel width
from `image_shape[-1]`. It does not implement the squeeze/split multi-scale
stages from the original Glow paper.

## Public Surface

### `GlowConfig`

Typed configuration for the retained Glow runtime.

Retained fields include:

- `coupling_network`: the `CouplingNetworkConfig` used by each affine coupling
  layer
- `image_shape`: the image tensor shape consumed by the flow
- `blocks_per_scale`: the number of sequential Glow blocks in the retained
  single-scale stack
- shared `FlowConfig` fields such as `input_dim`, `latent_dim`, and the base
  distribution settings

### `Glow`

Single-scale image flow baseline.

Retained behavior:

- `forward(images, *, rngs)` and `inverse(latents, *, rngs)` transform tensors
  with the configured `image_shape`
- `__call__(images, *, rngs)` returns the shared flow output dictionary with
  `z`, `logdet`, and `log_prob`
- `generate(n_samples, *, rngs)` samples Gaussian latents with `image_shape`
  and maps them through the inverse flow
- `sample(...)` remains the shared alias for `generate(...)`

### `GlowBlock`, `ActNormLayer`, `InvertibleConv1x1`, `AffineCouplingLayer`

These helper layers remain implementation details of the retained Glow block
stack. Channel mismatches fail explicitly instead of silently no-oping.

## Example

```python
from flax import nnx

from artifex.generative_models.core.configuration import (
    CouplingNetworkConfig,
    GlowConfig,
)
from artifex.generative_models.models.flow import Glow

coupling_network = CouplingNetworkConfig(
    name="glow_coupling",
    hidden_dims=(512, 512),
    activation="relu",
    network_type="mlp",
)
config = GlowConfig(
    name="glow",
    coupling_network=coupling_network,
    input_dim=32 * 32 * 3,
    image_shape=(32, 32, 3),
    blocks_per_scale=6,
)

model = Glow(config, rngs=nnx.Rngs(0))
samples = model.generate(n_samples=4, rngs=nnx.Rngs(1))
```
