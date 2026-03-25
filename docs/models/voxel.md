# Voxel

**Module:** `generative_models.models.geometric.voxel`

**Source:** `generative_models/models/geometric/voxel.py`

## Overview

`VoxelModel` is the retained learned voxel decoder for geometric generation.
The public surface is intentionally small:

- `VoxelConfig` defines the typed voxel-model configuration
- `VoxelModel.__call__(latent, *, deterministic=False)` decodes latent vectors
  into voxel occupancy grids
- `VoxelModel.sample(...)` and `generate(...)` draw Gaussian latents and decode
  them through the learned 3D deconvolution stack

## Public Surface

### `VoxelConfig`

Typed configuration for the voxel model runtime.

Retained fields include:

- `network`: a `VoxelNetworkConfig`
- `voxel_size`: the side length of the cubic voxel grid
- `voxel_dim`: the number of channels per voxel cell
- `use_sparse`: whether sparse downstream handling is requested
- `loss_type`: one of the supported voxel loss contracts
- `focal_gamma`: the focal-loss parameter used when `loss_type="focal"`

### `VoxelModel`

Learned voxel decoder baseline.

Retained behavior:

- `__call__(latent, *, deterministic=False)` decodes latent vectors to voxel
  occupancy grids and returns `(voxels, auxiliary)`
- `sample(n_samples, *, rngs, threshold=None)` samples Gaussian latents and
  decodes them through the model parameters
- `generate(...)` remains the shared generative-model alias for `sample(...)`
- `get_loss_fn(...)` returns the canonical voxel loss dictionary with
  `total_loss` plus the configured voxel loss term

## Example

```python
from flax import nnx

from artifex.generative_models.core.configuration import VoxelConfig, VoxelNetworkConfig
from artifex.generative_models.models.geometric import VoxelModel

network = VoxelNetworkConfig(
    name="voxel_network",
    hidden_dims=(64, 64),
    activation="gelu",
    base_channels=64,
    num_layers=4,
)
config = VoxelConfig(
    name="voxel_model",
    network=network,
    voxel_size=16,
    voxel_dim=1,
    loss_type="focal",
    focal_gamma=2.0,
)
model = VoxelModel(config, rngs=nnx.Rngs(0))
samples = model.generate(n_samples=2, rngs=nnx.Rngs(1), threshold=0.5)
```
