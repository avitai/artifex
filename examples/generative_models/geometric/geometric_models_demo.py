# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   language_info:
#     name: python
# ---

# %% [markdown]
"""# Geometric Models Demo: Point Clouds, Meshes, and Voxels.

This example demonstrates how to configure and instantiate different types of
geometric models in the Artifex framework:

1. **Point Cloud Models**: Unordered sets of 3D points
2. **Mesh Models**: Connected vertex structures with topology
3. **Voxel Models**: Regular 3D grids decoded from learned latents

## Learning Objectives

- ✅ Understand the three main geometric representations
- ✅ Configure models using family-specific typed configs
- ✅ Use the unified factory pattern for model creation
- ✅ Understand model-specific parameters for each geometry type

## Prerequisites

- Basic understanding of 3D geometric representations
- Familiarity with JAX and Flax NNX
- Understanding of model configuration patterns

## Source Code Dependencies

This example uses Artifex's geometric model implementations:
- `artifex.generative_models.models.geometric.PointCloudModel`
- `artifex.generative_models.models.geometric.MeshModel`
- `artifex.generative_models.models.geometric.VoxelModel`

All models follow the unified `GeometricModel` protocol and use proper NNX patterns.

## Estimated Runtime

~5-10 seconds on CPU, ~2-3 seconds on GPU

## Usage

```bash
source ./activate.sh
uv run python examples/generative_models/geometric/geometric_models_demo.py
```
"""

# %%
import logging

import jax
from flax import nnx

from artifex.generative_models.core.configuration import (
    MeshConfig,
    MeshNetworkConfig,
    PointCloudConfig,
    PointCloudNetworkConfig,
    VoxelConfig,
    VoxelNetworkConfig,
)
from artifex.generative_models.factory import create_model


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def echo(message: object = "") -> None:
    """Emit example output through the standard example logger."""
    LOGGER.info("%s", message)


# %% [markdown]
"""## 1. Setup Random Number Generation.

We initialize the random number generator for reproducible model creation.
"""


# %%
def create_demo_rngs(seed: int = 42) -> nnx.Rngs:
    """Create reproducible RNGs for the demo."""
    rng = jax.random.PRNGKey(seed)
    return nnx.Rngs(params=rng)


# %% [markdown]
"""## 2. Point Cloud Model.

Point clouds are unordered sets of 3D points, commonly used for:

- LiDAR data processing
- 3D object detection
- Molecular structure modeling

Key parameters:

- `num_points`: Number of points in the cloud
- `embed_dim`: Embedding dimension for features
- `num_layers`: Depth of the network
- `dropout_rate`: Regularization strength for the transformer stack
- runtime training uses the model's family-local reconstruction objective
"""


# %%
def run_point_cloud_demo(rngs: nnx.Rngs) -> None:
    """Create a point-cloud model and generate one sample."""
    echo("Creating point cloud model...")

    pc_network_config = PointCloudNetworkConfig(
        name="point_cloud_network",
        hidden_dims=(128, 128, 128, 128),
        activation="gelu",
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout_rate=0.1,
    )

    point_cloud_config = PointCloudConfig(
        name="demo_point_cloud",
        network=pc_network_config,
        num_points=512,
        dropout_rate=0.1,
    )
    point_cloud_model = create_model(point_cloud_config, rngs=rngs)
    echo(f"Created model: {type(point_cloud_model).__name__}")

    sample = point_cloud_model.sample(1, rngs=rngs)
    echo(f"Sample shape: {sample.shape}")


# %% [markdown]
"""## 3. Mesh Model.

Meshes are connected vertex structures with explicit topology, ideal for:

- 3D graphics and rendering
- Shape analysis and generation
- Surface reconstruction

Key parameters:

- `num_vertices`: Number of mesh vertices
- `edge_features_dim`: Size of learned edge features in the mesh network
- retained topology is derived from the sphere template plus the chosen vertex budget
- runtime training uses vertex reconstruction; richer mesh losses stay as standalone primitives
"""


# %%
def run_mesh_demo(rngs: nnx.Rngs) -> None:
    """Create a mesh model."""
    echo("Creating mesh model...")

    mesh_network_config = MeshNetworkConfig(
        name="mesh_network",
        hidden_dims=(256, 128, 64),
        activation="gelu",
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        edge_features_dim=64,
    )

    mesh_config = MeshConfig(
        name="demo_mesh",
        network=mesh_network_config,
        num_vertices=512,
        dropout_rate=0.1,
    )
    mesh_model = create_model(mesh_config, rngs=rngs)
    echo(f"Created model: {type(mesh_model).__name__}")


# %% [markdown]
"""## 4. Voxel Model.

Voxels are regular 3D grids, which makes them useful for:

- medical imaging
- volumetric shape generation
- 3D scene understanding

This demo highlights the retained typed voxel configuration, learned latent
decoding, and focal-loss settings that are common in sparse voxel problems.
"""


# %%
def run_voxel_demo(rngs: nnx.Rngs) -> None:
    """Create a voxel model."""
    echo("Creating voxel model...")

    voxel_network_config = VoxelNetworkConfig(
        name="voxel_network",
        hidden_dims=(64, 64, 64, 64),
        activation="gelu",
        base_channels=64,
        num_layers=4,
        kernel_size=3,
        use_residual=True,
    )

    voxel_config = VoxelConfig(
        name="demo_voxel",
        network=voxel_network_config,
        voxel_size=16,
        voxel_dim=1,
        loss_type="focal",
        focal_gamma=2.0,
        dropout_rate=0.1,
    )
    voxel_model = create_model(voxel_config, rngs=rngs)
    echo(f"Created model: {type(voxel_model).__name__}")

    sample = voxel_model.generate(n_samples=1, rngs=rngs)
    echo(f"Sample shape: {sample.shape}")


# %% [markdown]
"""## Summary and Key Takeaways.

This example demonstrates three foundational geometric representations:

1. **Point Clouds**: Flexible, unordered, permutation-invariant structures
2. **Meshes**: Topological surface representations
3. **Voxels**: Regular grids that work naturally with CNN-style architectures

### Next Steps

- Explore `simple_point_cloud_example.py` for a deeper point-cloud walkthrough
- See `geometric_losses_demo.py` for specialized geometric losses
- Check `protein_point_cloud_example.py` for domain-specific applications
"""


# %%
def main() -> None:
    """Demonstrate configuration and creation of geometric models."""
    rngs = create_demo_rngs()

    run_point_cloud_demo(rngs)
    echo()
    run_mesh_demo(rngs)
    echo()
    run_voxel_demo(rngs)
    echo()
    echo("Demo completed successfully!")


if __name__ == "__main__":
    main()
