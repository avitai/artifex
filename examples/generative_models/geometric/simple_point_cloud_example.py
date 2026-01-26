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
# Simple Point Cloud Example

This example demonstrates point cloud generation and visualization using Artifex's
PointCloudModel with JAX and Flax NNX.

## Learning Objectives

- ✅ Understand point cloud representation and processing
- ✅ Configure and instantiate PointCloudModel
- ✅ Generate 3D point clouds with transformer-based architecture
- ✅ Visualize point clouds in 3D
- ✅ Control sampling diversity with temperature

## Prerequisites

- Basic understanding of 3D coordinates and point clouds
- Familiarity with JAX and Flax NNX
- Knowledge of attention mechanisms (helpful but not required)

## What Are Point Clouds?

Point clouds are sets of 3D points representing objects or scenes:
- **Unordered**: Point order doesn't matter (permutation-invariant)
- **Sparse**: Efficient for representing 3D shapes
- **Flexible**: Can represent complex, irregular geometries

Common sources:
- LiDAR sensors in autonomous vehicles
- 3D scanners and depth cameras
- Molecular structures (proteins, molecules)
- 3D reconstruction from images

## Estimated Runtime

~10-15 seconds on GPU, ~20-30 seconds on CPU

## Usage

```bash
source activate.sh
python examples/generative_models/geometric/simple_point_cloud_example.py
```
"""

# %%
import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 for 3D plotting

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.models.geometric import PointCloudModel


# %% [markdown]
"""
## 1. Visualization Function

First, we define a helper function to visualize 3D point clouds using matplotlib.
"""


# %%
def plot_point_cloud(points, filename=None):
    """Plot a 3D point cloud.

    Args:
        points: Point cloud with shape [N, 3]
        filename: Optional filename to save the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Normalize points for coloring
    norm = np.sqrt(np.sum(points**2, axis=1))
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)

    # Plot the point cloud
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=norm, cmap="viridis", s=20, alpha=0.7
    )

    # Add a color bar
    plt.colorbar(scatter)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Generated 3D Point Cloud")

    # Save if filename provided
    if filename:
        import os

        output_dir = "examples_output"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)

    plt.tight_layout()
    return fig


# %% [markdown]
"""
## 2. Model Configuration and Creation

We configure a PointCloudModel using Artifex's unified configuration system.

### Key Architecture Components:

**Transformer-Based Processing:**
- Self-attention layers for learning point relationships
- Permutation-invariant (order doesn't matter)
- Can model long-range dependencies

**Parameters:**
- `num_points`: 512 - Number of 3D points to generate
- `embed_dim`: 128 - Feature dimension for attention
- `num_layers`: 3 - Depth of transformer
- `num_heads`: 4 - Multi-head attention heads
"""


# %%
def main():
    """Run the point cloud generation example."""
    # Set random seed for reproducibility
    seed = 42
    key = jax.random.key(seed)
    rngs = nnx.Rngs(params=key)

    print("Creating point cloud model...")
    # Create a point cloud model with frozen dataclass configuration
    network_config = PointCloudNetworkConfig(
        name="point_cloud_network",
        hidden_dims=(128, 128, 128),  # Tuple for frozen dataclass
        activation="gelu",
        embed_dim=128,  # Size of hidden layers
        num_heads=4,  # Number of attention heads
        num_layers=3,  # Number of transformer layers
        dropout_rate=0.1,
    )

    config = PointCloudConfig(
        name="point_cloud_generator",
        network=network_config,
        num_points=512,  # Number of points to generate
        dropout_rate=0.1,
    )
    model = PointCloudModel(
        config=config,
        rngs=rngs,
    )

    print("Generating point clouds...")
    # Generate point clouds
    point_clouds = model.generate(
        rngs=rngs,
        n_samples=2,  # Generate two point clouds
        temperature=0.8,  # Temperature for sampling (higher = more diverse)
    )

    # Convert to numpy for visualization
    point_clouds_np = np.array(point_clouds)

    # Visualize the results
    print("Visualizing point clouds...")
    for i in range(len(point_clouds_np)):
        points = point_clouds_np[i]
        plot_point_cloud(points, f"point_cloud_{i + 1}.png")
        plt.show()

    print("Example completed! Point clouds saved as PNG files.")


# %% [markdown]
# ## 3. Generate Point Clouds
#
# The `generate()` method creates new point clouds from the learned distribution.
#
# **Temperature parameter** controls diversity:
# - Lower (0.5-0.7): More consistent, focused samples
# - Medium (0.8-1.0): Balanced diversity and quality
# - Higher (1.0+): More diverse but potentially noisy

# %%
# Cell content moved to main() function above
pass

# %% [markdown]
# ## 4. Visualize Results
#
# We visualize each generated point cloud in 3D, with colors representing
# distance from the origin (useful for understanding spatial distribution).
#
# The plots are saved to `examples_output/` directory.

# %%
# Cell content moved to main() function above
pass

# %% [markdown]
# ## Summary and Key Takeaways
#
# This example demonstrated:
#
# 1. **Point Cloud Representation**: Unordered sets of 3D coordinates
# 2. **Transformer Architecture**: Self-attention for point relationships
# 3. **Generation Process**: Using learned distributions with temperature control
# 4. **3D Visualization**: Plotting and saving point clouds
#
# ### Understanding the Architecture
#
# **PointCloudModel** uses a transformer-based architecture:
# - **Positional encoding**: None needed (permutation-invariant)
# - **Self-attention**: Each point attends to all others
# - **Multi-head**: Different attention patterns learned simultaneously
# - **Layer normalization**: Stable training across layers
#
# ### Practical Applications
#
# - **3D object generation**: Furniture, vehicles, buildings
# - **Shape completion**: Fill in missing parts from partial scans
# - **Upsampling**: Increase point density for higher resolution
# - **Molecular modeling**: Protein structures, drug design
#
# ### Next Steps
#
# - Try different `num_points` values (256, 1024, 2048)
# - Experiment with `temperature` for diversity control
# - Explore `protein_point_cloud_example.py` for domain-specific modeling
# - See `geometric_losses_demo.py` for specialized loss functions

# %%
if __name__ == "__main__":
    main()
