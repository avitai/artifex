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
"""
# Geometric Models Demo: Point Clouds, Meshes, and Voxels

This example demonstrates how to configure and instantiate different types of
geometric models in the Artifex framework:

1. **Point Cloud Models**: Unordered sets of 3D points
2. **Mesh Models**: Connected vertex structures with topology
3. **Voxel Models**: Regular 3D grids with optional conditioning

## Learning Objectives

- ✅ Understand the three main geometric representations
- ✅ Configure models using ModelConfig
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
source activate.sh
python examples/generative_models/geometric/geometric_models_demo.py
```
"""

# %%
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


# %% [markdown]
"""
## 1. Setup Random Number Generation

We initialize the random number generator for reproducible model creation.
"""


# %%
def main():
    """Demonstrate configuration and creation of geometric models."""
    # Set up random number generator using Flax NNX patterns
    rng = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(params=rng)

    # %% [markdown]
    #     #     #     ## 2. Point Cloud Model
    #     #     #
    #     #     #     Point clouds are unordered sets of 3D points, commonly used for:
    #     #     #     - LiDAR data processing
    #     #     #     - 3D object detection
    #     #     #     - Molecular structure modeling (proteins, molecules)
    #     #     #
    #     #     #     Key parameters:
    #     #     #     - `num_points`: Number of points in the cloud
    #     #     #     - `embed_dim`: Embedding dimension for features
    #     #     #     - `num_layers`: Depth of the network
    #     #     #     - `loss_type`: Distance metric (chamfer, earth mover's distance)
    #     #     #

    # %%
    print("Creating point cloud model...")
    # Create network configuration for point cloud
    pc_network_config = PointCloudNetworkConfig(
        name="point_cloud_network",
        hidden_dims=(128, 128, 128, 128),  # Tuple for frozen dataclass
        activation="gelu",
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout_rate=0.1,
    )

    point_cloud_config = PointCloudConfig(
        name="demo_point_cloud",
        network=pc_network_config,
        num_points=512,  # Smaller point cloud for demo
        dropout_rate=0.1,
    )
    point_cloud_model = create_model(point_cloud_config, rngs=rngs)
    print(f"Created model: {type(point_cloud_model).__name__}")

    # Generate a sample point cloud
    sample = point_cloud_model.sample(1, rngs=rngs)
    print(f"Sample shape: {sample.shape}")

    # %% [markdown]
    #     #     #     ## 3. Mesh Model
    #     #     #
    #     #     #     Meshes are connected vertex structures with explicit topology, ideal for:
    #     #     #     - 3D graphics and rendering
    #     #     #     - Shape analysis and generation
    #     #     #     - Surface reconstruction
    #     #     #
    #     #     #     Key parameters:
    #     #     #     - `num_vertices`: Number of mesh vertices
    #     #     #     - `template_type`: Initial mesh template (sphere, cube, etc.)
    #     #     #     - Loss weights balance different geometric properties:
    #     #     #         - `vertex_loss_weight`: Vertex position accuracy
    #     #     #         - `normal_loss_weight`: Surface normal consistency
    #     #     #         - `edge_loss_weight`: Edge length regularization
    #     #     #

    # %%
    print("\nCreating mesh model...")
    # Create network configuration for mesh
    mesh_network_config = MeshNetworkConfig(
        name="mesh_network",
        hidden_dims=(256, 128, 64),  # Tuple for frozen dataclass
        activation="gelu",
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        edge_features_dim=64,
    )

    mesh_config = MeshConfig(
        name="demo_mesh",
        network=mesh_network_config,
        num_vertices=512,  # Smaller mesh for demo
        num_faces=1024,
        dropout_rate=0.1,
    )
    mesh_model = create_model(mesh_config, rngs=rngs)
    print(f"Created model: {type(mesh_model).__name__}")

    # %% [markdown]
    #     #     #     ## 4. Voxel Model with Conditioning
    #     #     #
    #     #     #     Voxels are regular 3D grids, perfect for:
    #     #     #     - Medical imaging (CT, MRI scans)
    #     #     #     - 3D scene understanding
    #     #     #     - Volumetric shape generation
    #     #     #
    #     #     #     Key features demonstrated:
    #     #     #     - **Conditioning**: Enable class-conditional generation
    #     #     #     - **Focal Loss**: Handle class imbalance in sparse voxel data
    #     #     #     - **Multi-scale architecture**: Progressive refinement through channels
    #     #     #
    #     #     #     The focal loss with γ=2.0 focuses on hard-to-classify voxels.
    #     #     #

    # %%
    print("\nCreating voxel model with conditioning...")
    # Create network configuration for voxel
    voxel_network_config = VoxelNetworkConfig(
        name="voxel_network",
        hidden_dims=(64, 64, 64, 64),  # Tuple for frozen dataclass
        activation="gelu",
        base_channels=64,
        num_layers=4,
        kernel_size=3,
        use_residual=True,
    )

    voxel_config = VoxelConfig(
        name="demo_voxel",
        network=voxel_network_config,
        voxel_size=16,  # Low resolution for demo
        voxel_dim=1,
        loss_type="focal",
        focal_gamma=2.0,
        dropout_rate=0.1,
    )
    voxel_model = create_model(voxel_config, rngs=rngs)
    print(f"Created model: {type(voxel_model).__name__}")

    # %% [markdown]
    #     #     #     ## Summary and Key Takeaways
    #     #     #
    #     #     #     This example demonstrated three fundamental geometric representations:
    #     #     #
    #     #     #     1. **Point Clouds** - Flexible, unordered, permutation-invariant
    #     #     #     2. **Meshes** - Topological structure, surface-oriented
    #     #     #     3. **Voxels** - Regular grids, easy to process with CNNs
    #     #     #
    #     #     #     ### When to Use Each Representation
    #     #     #
    #     #     #     - **Point Clouds**: Raw sensor data, molecular structures, irregular shapes
    #     #     #     - **Meshes**: Graphics, animation, smooth surfaces
    #     #     #     - **Voxels**: Medical imaging, volumetric data, regular grids
    #     #     #
    #     #     #     ### Next Steps
    #     #     #
    #     #     #     - Explore `simple_point_cloud_example.py` for detailed point cloud modeling
    #     #     #     - See `geometric_losses_demo.py` for specialized geometric losses
    #     #     #     - Check `protein_point_cloud_example.py` for domain-specific applications
    #     #     #

    # %%
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
