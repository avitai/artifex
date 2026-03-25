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
# - Understand point-cloud loss primitives (Chamfer, Earth Mover's Distance, Hausdorff)
# - Learn how mesh losses are composed from explicit geometric primitives
# - Explore voxel loss functions (BCE, Focal, Dice)
# - See which losses are model runtime defaults versus standalone helpers
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

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import VoxelConfig, VoxelNetworkConfig
from artifex.generative_models.core.losses.geometric import (
    chamfer_distance,
    earth_mover_distance,
    get_mesh_loss,
    hausdorff_distance,
)
from artifex.generative_models.factory import create_model


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def echo(message: object = "") -> None:
    """Log example progress without raw print statements."""
    LOGGER.info("%s", message)


# %% [markdown]
# ## 1. Point Cloud Loss Functions
#
# Point clouds are unordered sets of 3D points. The main challenge is that
# permutations of the same shape should have zero loss, so the retained surface
# is a set of standalone geometric primitives rather than a PointCloudConfig
# loss-selection knob.
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
echo("===== Point Cloud Loss Functions Demo =====")

# %%
# Create a sample point cloud for testing
true_points = jnp.array(
    [[[i * 0.1, j * 0.1, k * 0.1] for i in range(5) for j in range(5) for k in range(5)]]
)  # Shape: [1, 125, 3]

# %%
# Create a slightly shifted prediction
pred_points = true_points + 0.05

# Compute retained standalone point-cloud losses directly
chamfer_loss = chamfer_distance(pred_points, true_points)
echo(f"Chamfer distance loss: {float(chamfer_loss):.4f}")

# %%
earth_mover_loss = earth_mover_distance(pred_points, true_points)
echo(f"Earth Mover distance loss: {float(earth_mover_loss):.4f}")

hausdorff_loss = hausdorff_distance(pred_points, true_points)
echo(f"Hausdorff distance loss: {float(hausdorff_loss):.4f}")


# %% [markdown]
# ## 2. Mesh Loss Functions
#
# Meshes have vertices connected by edges and faces. The retained runtime model
# reconstructs vertices generically; the richer mesh objective remains an
# explicit standalone primitive assembled with get_mesh_loss(...).
#
# - **Vertex Loss**: Measures vertex position accuracy
# - **Normal Loss**: Ensures surface normals are correct (smooth surfaces)
# - **Edge Loss**: Maintains edge lengths and connectivity
#
# The total loss is a weighted combination of these components.


# %%
echo("\n===== Mesh Loss Functions Demo =====")

# %%
# Create simple predicted and target meshes
pred_vertices = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
pred_faces = jnp.array([[0, 1, 2]])
pred_normals = jnp.array([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])

target_vertices = pred_vertices + 0.1
target_faces = pred_faces
target_normals = pred_normals

pred_mesh = (pred_vertices, pred_faces, pred_normals)
target_mesh = (target_vertices, target_faces, target_normals)

default_mesh_loss = get_mesh_loss()(pred_mesh, target_mesh)
smooth_mesh_loss = get_mesh_loss(vertex_weight=0.5, normal_weight=1.0, edge_weight=0.1)(
    pred_mesh, target_mesh
)

echo(f"Default mesh loss: {float(default_mesh_loss):.4f}")
echo(f"Smooth-surface weighted mesh loss: {float(smooth_mesh_loss):.4f}")


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
echo("\n===== Voxel Loss Functions Demo =====")
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
echo(f"Binary cross-entropy loss: {bce_loss}")

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
echo(f"Focal loss (gamma=2.0): {focal_loss}")

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
echo(f"Dice loss: {dice_loss}")

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
echo(f"MSE loss: {mse_loss}")

echo("\nSupported voxel loss types: 'bce', 'dice', 'focal', 'mse'")


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
# 3. **Mesh losses are standalone primitives**: Compose vertex, normal, and edge terms
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
echo("\nLoss function demos completed!")
