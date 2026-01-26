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
# # Geometric Model Loss Functions Demo
#
# This example demonstrates different loss functions used for geometric models
# including point clouds, meshes, and voxel grids. Understanding these losses is
# crucial for training effective 3D generative models.
#
# ## Learning Objectives
#
# - Understand loss functions for point cloud models (Chamfer, Earth Mover's Distance)
# - Learn about mesh-specific losses (vertex, normal, edge)
# - Explore voxel loss functions (BCE, Focal, Dice)
# - See how to configure loss weights for different geometric representations
#
# ## Prerequisites
#
# - Basic understanding of 3D representations (point clouds, meshes, voxels)
# - Familiarity with loss functions in machine learning
# - Knowledge of JAX and Flax NNX
#
# ## Overview
#
# Geometric models require specialized loss functions that account for the structure
# of 3D data:
#
# - **Point Clouds**: Unordered sets of points -> need permutation-invariant losses
# - **Meshes**: Connected vertices and faces -> need topology-aware losses
# - **Voxels**: Regular 3D grids -> can use standard image-like losses
#
# This example shows how to configure and use these different loss types.

# %%
"""Demonstration of geometric model loss functions with configuration."""

import jax
import jax.numpy as jnp
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
# ## 1. Point Cloud Loss Functions
#
# Point clouds are unordered sets of 3D points. The main challenge is that
# permutations of the same shape should have zero loss, requiring specialized
# metrics like Chamfer Distance and Earth Mover's Distance.
#
# ### Chamfer Distance
#
# Measures the average distance from each point in one set to its nearest
# neighbor in the other set.
#
# ### Earth Mover's Distance (EMD)
#
# Finds the optimal transport plan between two point sets, more accurate but
# computationally expensive.


# %%
# Initialize RNG for point cloud demo
print("===== Point Cloud Loss Functions Demo =====")
rng_pc = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng_pc)

# %%
# Create a sample point cloud for testing
true_points = jnp.array(
    [[[i * 0.1, j * 0.1, k * 0.1] for i in range(5) for j in range(5) for k in range(5)]]
)  # Shape: [1, 125, 3]

# %%
# Test with Chamfer distance loss
# Create network configuration for point cloud
pc_network_config = PointCloudNetworkConfig(
    name="chamfer_network",
    hidden_dims=(64, 64),
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    dropout_rate=0.1,
)

chamfer_config = PointCloudConfig(
    name="chamfer_point_cloud",
    network=pc_network_config,
    num_points=125,
    dropout_rate=0.1,
)
chamfer_model = create_model(chamfer_config, rngs=rngs)
chamfer_loss_fn = chamfer_model.get_loss_fn()

# Generate a sample prediction
pred_points = chamfer_model.generate(n_samples=1)

# Compute the loss (loss function expects batch dict and outputs)
batch = {"target": true_points}
chamfer_loss = chamfer_loss_fn(batch, pred_points)
print(f"Chamfer distance loss: {chamfer_loss}")

# %%
# Test with Earth Mover's distance loss (if implemented)
# Create network configuration for point cloud
em_network_config = PointCloudNetworkConfig(
    name="earth_mover_network",
    hidden_dims=(64, 64),
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    dropout_rate=0.1,
)

earth_mover_config = PointCloudConfig(
    name="earth_mover_point_cloud",
    network=em_network_config,
    num_points=125,
    dropout_rate=0.1,
)
earth_mover_model = create_model(earth_mover_config, rngs=rngs)
earth_mover_loss_fn = earth_mover_model.get_loss_fn()

# Compute the loss
pred_points = earth_mover_model.generate(n_samples=1)
earth_mover_batch = {"target": true_points}
earth_mover_loss = earth_mover_loss_fn(earth_mover_batch, pred_points)
print(f"Earth Mover distance loss: {earth_mover_loss}")


# %% [markdown]
# ## 2. Mesh Loss Functions
#
# Meshes have vertices connected by edges and faces. Loss functions can focus
# on different aspects:
#
# - **Vertex Loss**: Measures vertex position accuracy
# - **Normal Loss**: Ensures surface normals are correct (smooth surfaces)
# - **Edge Loss**: Maintains edge lengths and connectivity
#
# The total loss is a weighted combination of these components.


# %%
# Initialize RNG for mesh demo
print("\n===== Mesh Loss Functions Demo =====")
rng_mesh = jax.random.PRNGKey(43)
rngs = nnx.Rngs(params=rng_mesh)

# %%
# Default weights - create network configuration for mesh
default_mesh_network = MeshNetworkConfig(
    name="default_mesh_network",
    hidden_dims=(128, 64),
    activation="gelu",
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    edge_features_dim=32,
)

default_config = MeshConfig(
    name="default_weights_mesh",
    network=default_mesh_network,
    num_vertices=100,
    num_faces=196,
    dropout_rate=0.1,
)
default_model = create_model(default_config, rngs=rngs)

# %%
# Custom weights emphasizing normal consistency
normal_mesh_network = MeshNetworkConfig(
    name="normal_mesh_network",
    hidden_dims=(128, 64),
    activation="gelu",
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    edge_features_dim=32,
)

normal_config = MeshConfig(
    name="normal_weights_mesh",
    network=normal_mesh_network,
    num_vertices=100,
    num_faces=196,
    dropout_rate=0.1,
)
normal_model = create_model(normal_config, rngs=rngs)

# %%
# Access configuration parameters
print(f"Default model num_vertices: {default_model.config.num_vertices}")
print(f"Default model num_faces: {default_model.config.num_faces}")
print(f"Default model network embed_dim: {default_model.config.network.embed_dim}")

print(f"Normal-focused model num_vertices: {normal_model.config.num_vertices}")
print(f"Normal-focused model num_faces: {normal_model.config.num_faces}")
print(f"Normal-focused model network embed_dim: {normal_model.config.network.embed_dim}")


# %% [markdown]
# ## 3. Voxel Loss Functions
#
# Voxel grids are regular 3D arrays similar to images. We can use standard
# image losses, but some are better for 3D shapes:
#
# ### Binary Cross-Entropy (BCE)
#
# Standard loss for binary occupancy.
#
# ### Focal Loss
#
# Reduces loss for well-classified voxels, focuses on hard examples.
#
# ### Dice Loss
#
# Measures overlap between predicted and true shapes.


# %%
# Initialize RNG for voxel demo
print("\n===== Voxel Loss Functions Demo =====")
rng_voxel = jax.random.PRNGKey(44)
rngs = nnx.Rngs(params=rng_voxel)

# %%
# Create a small voxel grid for testing
resolution = 8
true_voxels = jnp.zeros((1, resolution, resolution, resolution, 1))
# Set a simple shape (middle cube)
true_voxels = true_voxels.at[:, 2:6, 2:6, 2:6, :].set(1.0)

# %%
# BCE loss configuration - create network configuration for voxel
bce_voxel_network = VoxelNetworkConfig(
    name="bce_voxel_network",
    hidden_dims=(64, 32, 16),
    activation="gelu",
    base_channels=64,
    num_layers=4,
    kernel_size=3,
    use_residual=True,
)

bce_config = VoxelConfig(
    name="bce_voxel",
    network=bce_voxel_network,
    voxel_size=resolution,
    voxel_dim=1,
    loss_type="bce",
    dropout_rate=0.1,
)
bce_model = create_model(bce_config, rngs=rngs)
bce_loss_fn = bce_model.get_loss_fn()

# Generate a sample prediction
pred_voxels = bce_model.generate(n_samples=1)

# Compute the loss (voxel loss functions expect direct arrays)
bce_loss = bce_loss_fn(pred_voxels, true_voxels)
print(f"Binary cross-entropy loss: {bce_loss}")

# %%
# Focal loss configuration
focal_voxel_network = VoxelNetworkConfig(
    name="focal_voxel_network",
    hidden_dims=(64, 32, 16),
    activation="gelu",
    base_channels=64,
    num_layers=4,
    kernel_size=3,
    use_residual=True,
)

focal_config = VoxelConfig(
    name="focal_voxel",
    network=focal_voxel_network,
    voxel_size=resolution,
    voxel_dim=1,
    loss_type="focal",
    focal_gamma=2.0,
    dropout_rate=0.1,
)
focal_model = create_model(focal_config, rngs=rngs)
focal_loss_fn = focal_model.get_loss_fn()

# Compute the loss
pred_voxels = focal_model.generate(n_samples=1)
# Compute the loss (voxel loss functions expect direct arrays)
focal_loss = focal_loss_fn(pred_voxels, true_voxels)
print(f"Focal loss (gamma=2.0): {focal_loss}")

# %%
# Dice loss configuration (good for segmentation tasks)
dice_voxel_network = VoxelNetworkConfig(
    name="dice_voxel_network",
    hidden_dims=(64, 32, 16),
    activation="gelu",
    base_channels=64,
    num_layers=4,
    kernel_size=3,
    use_residual=True,
)

dice_config = VoxelConfig(
    name="dice_voxel",
    network=dice_voxel_network,
    voxel_size=resolution,
    voxel_dim=1,
    loss_type="dice",
    dropout_rate=0.1,
)
dice_model = create_model(dice_config, rngs=rngs)
dice_loss_fn = dice_model.get_loss_fn()

# Compute the loss
pred_voxels = dice_model.generate(n_samples=1)
dice_loss = dice_loss_fn(pred_voxels, true_voxels)
print(f"Dice loss: {dice_loss}")

# %%
# MSE loss configuration (good for regression tasks)
mse_voxel_network = VoxelNetworkConfig(
    name="mse_voxel_network",
    hidden_dims=(64, 32, 16),
    activation="gelu",
    base_channels=64,
    num_layers=4,
    kernel_size=3,
    use_residual=True,
)

mse_config = VoxelConfig(
    name="mse_voxel",
    network=mse_voxel_network,
    voxel_size=resolution,
    voxel_dim=1,
    loss_type="mse",
    dropout_rate=0.1,
)
mse_model = create_model(mse_config, rngs=rngs)
mse_loss_fn = mse_model.get_loss_fn()

# Compute the loss
pred_voxels = mse_model.generate(n_samples=1)
mse_loss = mse_loss_fn(pred_voxels, true_voxels)
print(f"MSE loss: {mse_loss}")

print("\nSupported voxel loss types: 'bce', 'dice', 'focal', 'mse'")


# %% [markdown]
# ## Summary
#
# This example demonstrated loss functions for three geometric representations:
#
# ### Point Clouds
#
# - **Chamfer Distance**: Fast, permutation-invariant, good for most uses
# - **Earth Mover's Distance**: More accurate but slower
#
# ### Meshes
#
# - **Vertex Loss**: Position accuracy
# - **Normal Loss**: Surface smoothness
# - **Edge Loss**: Topology preservation
# - Can adjust weights based on application requirements
#
# ### Voxels
#
# - **BCE**: Standard for balanced datasets
# - **Focal Loss**: Better for sparse objects (imbalanced data)
# - **Dice Loss**: Optimizes overlap directly
#
# ## Key Takeaways
#
# 1. **Choose loss based on representation**: Different 3D formats need different losses
# 2. **Point clouds need permutation invariance**: Use Chamfer or EMD
# 3. **Mesh losses are composite**: Balance vertex, normal, and edge terms
# 4. **Voxel losses handle sparsity**: Use Focal or Dice for sparse 3D shapes
# 5. **Loss weights are hyperparameters**: Tune them for your application
#
# ## Experiments to Try
#
# 1. Compare Chamfer vs EMD loss on the same point cloud
# 2. Adjust mesh loss weights for different surface types (smooth vs sharp edges)
# 3. Compare BCE, Focal, and Dice on sparse vs dense voxel grids
# 4. Try different focal_gamma values (0.5, 1.0, 2.0, 5.0)
# 5. Visualize the effect of different losses on generated shapes
#
# ## Next Steps
#
# - See `geometric_models_demo.py` for creating geometric models
# - See `simple_point_cloud_example.py` for point cloud generation
# - See `geometric_benchmark_demo.py` for evaluation metrics


# %%
print("\nLoss function demos completed!")
