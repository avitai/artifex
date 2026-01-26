# Geometric Model Loss Functions Demo

![Level](https://img.shields.io/badge/Level-Intermediate-orange)
![Runtime](https://img.shields.io/badge/Runtime-~15s-green)
![Format](https://img.shields.io/badge/Format-Script%20%2B%20Notebook-blue)

Comprehensive demonstration of loss functions for geometric models including point clouds, meshes, and voxel grids.

## Files

- **Python Script**: [`geometric_losses_demo.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_losses_demo.py)
- **Jupyter Notebook**: [`geometric_losses_demo.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_losses_demo.ipynb)

## Quick Start

```bash
# Run the Python script
python examples/generative_models/geometric/geometric_losses_demo.py

# Or use Jupyter notebook
jupyter notebook examples/generative_models/geometric/geometric_losses_demo.ipynb
```

## Overview

This example provides a comprehensive tour of loss functions used for different 3D geometric representations. Understanding these losses is crucial for training effective generative models for 3D data.

### Learning Objectives

- [ ] Understand permutation-invariant losses for point clouds
- [ ] Learn composite loss functions for meshes
- [ ] Explore specialized losses for voxel grids
- [ ] See how to configure and balance different loss components
- [ ] Compare different loss types for the same representation

### Prerequisites

- Basic understanding of 3D representations (point clouds, meshes, voxels)
- Familiarity with loss functions in machine learning
- Knowledge of JAX and Artifex's configuration system

## Background: 3D Representations and Their Losses

### Why Geometric Losses Matter

Standard image losses (MSE, L1) don't work well for 3D data because:

1. **Point clouds are unordered**: Permuting points shouldn't change the loss
2. **Meshes have topology**: Need to preserve connectivity and smoothness
3. **Voxels are sparse**: Most voxels are empty, causing class imbalance

### The Three Representations

#### Point Clouds

Unordered sets of 3D points: ${(x_i, y_i, z_i)}_{i=1}^N$

- Compact representation
- Permutation invariance required
- Use Chamfer Distance or Earth Mover's Distance

#### Meshes

Vertices connected by edges and faces

- Explicit topology
- Need vertex, normal, and edge losses
- Balancing multiple objectives

#### Voxels

Regular 3D grids with occupancy values

- Like 3D images
- Sparse (mostly empty)
- Use BCE, Focal, or Dice loss

## Code Walkthrough

### 1. Point Cloud Losses

#### Chamfer Distance

The Chamfer Distance is the workhorse of point cloud generation. It measures how well two point sets match by finding nearest neighbors:

$$
\mathcal{L}_{\text{CD}}(X, Y) = \frac{1}{|X|}\sum_{x \in X} \min_{y \in Y} \|x - y\|^2 + \frac{1}{|Y|}\sum_{y \in Y} \min_{x \in X} \|x - y\|^2
$$

**Key properties:**

- Permutation invariant
- Fast to compute (O(N² ) with optimizations)
- Good for most applications

```python
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)

network_config = PointCloudNetworkConfig(
    name="chamfer_network",
    hidden_dims=(64,),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    dropout_rate=0.1,
)

chamfer_config = PointCloudConfig(
    name="chamfer_point_cloud",
    network=network_config,
    num_points=125,
    loss_type="chamfer",  # Chamfer distance loss
    dropout_rate=0.1,
)
```

#### Earth Mover's Distance (EMD)

EMD finds the optimal transport plan between point sets. More accurate but slower:

$$
\mathcal{L}_{\text{EMD}}(X, Y) = \min_{\phi: X \to Y} \sum_{x \in X} \|x - \phi(x)\|
$$

Where $\phi$ is a bijection between X and Y.

**When to use:**

- Quality is more important than speed
- Small point clouds (<1000 points)
- Fine geometric details matter

```python
# Same network config, different loss type
earth_mover_config = PointCloudConfig(
    name="earth_mover_point_cloud",
    network=network_config,
    num_points=125,
    loss_type="earth_mover",  # EMD loss
    dropout_rate=0.1,
)
```

### 2. Mesh Losses

Meshes require balancing multiple geometric properties:

$$
\mathcal{L}_{\text{mesh}} = w_v \mathcal{L}_{\text{vertex}} + w_n \mathcal{L}_{\text{normal}} + w_e \mathcal{L}_{\text{edge}}
$$

**Vertex Loss**: L2 distance between vertex positions

$$
\mathcal{L}_{\text{vertex}} = \|\mathbf{V}_{\text{pred}} - \mathbf{V}_{\text{true}}\|^2
$$

**Normal Loss**: Ensures surface smoothness

$$
\mathcal{L}_{\text{normal}} = 1 - \frac{1}{|F|}\sum_{f \in F} \mathbf{n}_{\text{pred}}^f \cdot \mathbf{n}_{\text{true}}^f
$$

**Edge Loss**: Preserves edge lengths

$$
\mathcal{L}_{\text{edge}} = \sum_{(i,j) \in E} (|\mathbf{v}_i - \mathbf{v}_j|_{\text{pred}} - |\mathbf{v}_i - \mathbf{v}_j|_{\text{true}})^2
$$

#### Configuring Weights

```python
from artifex.generative_models.core.configuration import (
    MeshConfig,
    MeshNetworkConfig,
)

mesh_network = MeshNetworkConfig(
    name="mesh_network",
    hidden_dims=(128, 64),  # Tuple for frozen dataclass
    activation="gelu",
)

# Smooth surfaces (e.g., CAD models)
normal_config = MeshConfig(
    name="smooth_mesh",
    network=mesh_network,
    num_vertices=512,
    vertex_loss_weight=0.5,   # Reduce vertex constraint
    normal_loss_weight=1.0,   # Emphasize smoothness
    edge_loss_weight=0.1,     # Light edge preservation
)

# Sharp edges (e.g., furniture)
edge_config = MeshConfig(
    name="sharp_mesh",
    network=mesh_network,
    num_vertices=512,
    vertex_loss_weight=0.5,
    normal_loss_weight=0.1,   # Less smoothing
    edge_loss_weight=1.0,     # Strong edge preservation
)
```

### 3. Voxel Losses

Voxel grids can use image-like losses, but some are better for 3D:

#### Binary Cross-Entropy (BCE)

Standard loss for binary voxels:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

**Best for:**

- Balanced datasets (50% occupied voxels)
- Dense 3D shapes

#### Focal Loss

Down-weights easy examples, focuses on hard ones:

$$
\mathcal{L}_{\text{focal}} = -\frac{1}{N}\sum_{i=1}^N (1-p_t)^\gamma \log(p_t)
$$

Where $p_t = \hat{y}_i$ if $y_i=1$, else $1-\hat{y}_i$.

**Best for:**

- Imbalanced data (sparse objects)
- $\gamma=2.0$ is typical
- Higher $\gamma$ → more focus on hard examples

#### Dice Loss

Directly optimizes overlap (similar to IoU):

$$
\mathcal{L}_{\text{dice}} = 1 - \frac{2\sum_i y_i \hat{y}_i + \epsilon}{\sum_i y_i + \sum_i \hat{y}_i + \epsilon}
$$

**Best for:**

- Segmentation-like tasks
- Maximizing shape overlap
- Handles class imbalance well

#### Comparison

```python
from artifex.generative_models.core.configuration import (
    VoxelConfig,
    VoxelNetworkConfig,
)

voxel_network = VoxelNetworkConfig(
    name="voxel_network",
    hidden_dims=(64, 32),  # Required base config
    activation="relu",
    base_channels=32,  # Base number of 3D CNN channels
    num_layers=4,       # Number of 3D convolutional layers
)

# Dense shapes → BCE
bce_config = VoxelConfig(
    name="dense_voxel",
    network=voxel_network,
    resolution=16,
    loss_type="bce",
)

# Sparse shapes → Focal
focal_config = VoxelConfig(
    name="sparse_voxel",
    network=voxel_network,
    resolution=16,
    loss_type="focal",
    focal_gamma=2.0,  # Adjust based on sparsity
)

# Overlap optimization → Dice
dice_config = VoxelConfig(
    name="overlap_voxel",
    network=voxel_network,
    resolution=16,
    loss_type="dice",
)
```

## Expected Output

```
===== Point Cloud Loss Functions Demo =====
Chamfer distance loss: {'total_loss': 2.92, 'mse_loss': 2.92}
Earth Mover distance loss: {'total_loss': 3.61, 'mse_loss': 3.61}

===== Mesh Loss Functions Demo =====
Default model vertex weight: 1.0
Default model normal weight: 1.0
Default model edge weight: 1.0
Normal-focused model vertex weight: 0.5
Normal-focused model normal weight: 1.0
Normal-focused model edge weight: 0.1

===== Voxel Loss Functions Demo =====
Binary cross-entropy loss: {'total_loss': 0.68, ...}
Focal loss (gamma=2.0): {'total_loss': 0.15, ...}
Dice loss: {'total_loss': 0.42, ...}

Loss function demos completed!
```

## Key Concepts

### Permutation Invariance

Point cloud losses must be invariant to point ordering:

```python
# These should have the same loss
points_A = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
points_B = [[2, 2, 2], [0, 0, 0], [1, 1, 1]]  # Same points, different order

loss(points_A, target) == loss(points_B, target)  # Must be true
```

### Loss Component Balancing

For composite losses (meshes), balance is key:

- Start with equal weights (1.0, 1.0, 1.0)
- Identify the most important property (smoothness vs sharp edges)
- Increase weight for that component
- Reduce others proportionally
- Validate on test shapes

### Class Imbalance in Voxels

Voxel grids are typically 90-99% empty:

```python
# Sparse object (5% occupied)
occupancy_ratio = 0.05

# BCE: Treats all voxels equally → biased toward empty
# Focal (γ=2): Down-weights easy empties → balanced
# Dice: Focuses on overlap → invariant to sparsity
```

## Experiments to Try

1. **Compare Chamfer vs EMD**

   Generate the same point cloud with both losses and compare quality/speed

2. **Mesh Weight Tuning**

   Try different weight combinations for different mesh types (organic vs geometric)

3. **Voxel Sparsity Study**

   Compare BCE, Focal, Dice on grids with 1%, 10%, 50% occupancy

4. **Focal Gamma Sweep**

   Test $\gamma \in [0.5, 1.0, 2.0, 5.0]$ on sparse voxels

5. **Visualization**

   Plot generated shapes with different losses to see visual differences

## Next Steps

Explore related examples to deepen your understanding:

<div class="grid cards" markdown>

- :material-cube-outline:{ .lg .middle } **Geometric Models Overview**

    ---

    Learn about the three geometric representations and when to use each.

    [:octicons-arrow-right-24: geometric_models_demo.py](geometric-models-demo.md)

- :material-chart-scatter-plot:{ .lg .middle } **Point Cloud Generation**

    ---

    Generate and visualize 3D point clouds with transformers.

    [:octicons-arrow-right-24: simple_point_cloud_example.py](simple-point-cloud-example.md)

- :material-chart-box:{ .lg .middle } **Geometric Benchmarks**

    ---

    Evaluate geometric models with specialized metrics.

    [:octicons-arrow-right-24: geometric_benchmark_demo.py](geometric-benchmark-demo.md)

- :material-molecule:{ .lg .middle } **Protein Modeling**

    ---

    Apply geometric models to protein structure prediction.

    [:octicons-arrow-right-24: ../protein/protein-point-cloud-example.md](../protein/protein-point-cloud-example.md)

</div>

## Troubleshooting

### High Chamfer Distance

**Problem**: Chamfer loss is unexpectedly high.

**Solutions**:

1. Check point cloud normalization (scale to [-1, 1])
2. Verify number of points matches between pred and target
3. Ensure points are in same coordinate system

### Mesh Loss Imbalance

**Problem**: One loss component dominates others.

**Solutions**:

1. Normalize each loss to [0, 1] range before weighting
2. Use relative weights (sum to 1.0)
3. Monitor individual losses during training

### Voxel Loss Not Decreasing

**Problem**: Loss plateaus early in training.

**Solutions**:

1. Switch from BCE to Focal for sparse grids
2. Adjust focal gamma (try 2.0 → 3.0)
3. Check for label imbalance (>95% empty → use Dice)

### Out of Memory

**Problem**: Voxel models run out of GPU memory.

**Solutions**:

1. Reduce voxel resolution (32³ → 16³)
2. Reduce batch size
3. Use gradient checkpointing
4. Consider point cloud representation instead

## Additional Resources

- [Loss Functions API](../../api/core/losses.md)
- [Point Cloud Models](../../models/point_cloud.md)

## Citation

If you use these loss functions in your research, please cite:

```bibtex
@software{artifex2025,
  title={Artifex: Modular Generative Modeling Library},
  author={Artifex Contributors},
  year={2025},
  url={https://github.com/avitai/artifex}
}
```

## References

1. Chamfer Distance: Fan et al., "A Point Set Generation Network for 3D Object Reconstruction from a Single Image", CVPR 2017
2. Earth Mover's Distance: Rubner et al., "The Earth Mover's Distance as a Metric for Image Retrieval", IJCV 2000
3. Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
4. Dice Loss: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016
