# Simple Point Cloud Example

<div class="example-badges">
  <span class="badge badge-beginner">Beginner</span>
  <span class="badge badge-runtime-fast">⚡ 10-15 seconds</span>
  <span class="badge badge-format-dual">📓 Dual Format</span>
</div>

Learn how to generate and visualize 3D point clouds using Artifex's PointCloudModel with transformer-based architecture.

## Files

- **Python Script**: [`simple_point_cloud_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/simple_point_cloud_example.py)
- **Jupyter Notebook**: [`simple_point_cloud_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/simple_point_cloud_example.ipynb)

## Quick Start

```bash
# Clone and setup
cd artifex
source activate.sh

# Run Python script
python examples/generative_models/geometric/simple_point_cloud_example.py

# Or use Jupyter notebook
jupyter notebook examples/generative_models/geometric/simple_point_cloud_example.ipynb
```

## Overview

This tutorial teaches you how to work with point clouds—the fundamental representation for 3D data in machine learning. Point clouds are unordered sets of 3D coordinates that represent objects, scenes, or molecular structures.

### Learning Objectives

- [x] Understand point cloud representation and properties
- [x] Configure PointCloudModel with transformer architecture
- [x] Generate 3D point clouds from learned distributions
- [x] Visualize point clouds in 3D space
- [x] Control generation diversity with temperature

### Prerequisites

- Basic understanding of 3D coordinates (x, y, z)
- Familiarity with JAX and Flax NNX
- Basic knowledge of attention mechanisms (helpful but not required)

## What Are Point Clouds?

Point clouds are collections of 3D points that represent the shape or structure of objects:

```
Point Cloud = {(x₁, y₁, z₁), (x₂, y₂, z₂), ..., (xₙ, yₙ, zₙ)}
```

### Key Properties

1. **Unordered (Permutation-Invariant)**: The order of points doesn't matter
   - `{A, B, C}` is the same as `{C, A, B}`
   - This is why transformers work well (they're permutation-invariant)

2. **Sparse**: Represents surfaces without filling volumes
   - A sphere needs only surface points, not interior
   - More efficient than voxels for many tasks

3. **Flexible**: Can represent arbitrary shapes
   - No fixed topology required
   - Handles complex, irregular geometries

### Common Sources

| Source | Example | Resolution |
|--------|---------|------------|
| **LiDAR** | Autonomous vehicles | 10K-1M points |
| **3D Scanners** | Industrial inspection | 100K-10M points |
| **Depth Cameras** | Robotics, AR/VR | 10K-100K points |
| **Molecular** | Protein structures | 100-10K atoms |
| **Photogrammetry** | 3D reconstruction | 100K-10M points |

## Transformer Architecture for Point Clouds

### Why Transformers?

Traditional CNNs require regular grids. Point clouds are irregular, so we need architectures that:

1. **Handle variable-size inputs**: Different objects have different numbers of points
2. **Are permutation-invariant**: Point order shouldn't matter
3. **Model long-range relationships**: Distant points may be related

**Transformers satisfy all three requirements!**

### Architecture Components

```
Input Points (N, 3)
      ↓
Point Embedding → (N, 128)
      ↓
┌─────────────────┐
│ Transformer     │ ← Self-Attention Layers (×3)
│  Layer 1        │    Each point attends to all others
├─────────────────┤
│ Transformer     │ ← Multi-Head (×4)
│  Layer 2        │    Different attention patterns
├─────────────────┤
│ Transformer     │ ← Layer Normalization
│  Layer 3        │    Stable training
└─────────────────┘
      ↓
Output Points (N, 3)
```

### Key Parameters

- **`num_points`**: 512 - Number of 3D points
- **`embed_dim`**: 128 - Feature dimension for attention
- **`num_layers`**: 3 - Transformer depth
- **`num_heads`**: 4 - Multi-head attention heads

## Code Walkthrough

### Step 1: Setup and Imports

```python
import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.models.geometric import PointCloudModel
```

We use:

- **JAX**: Fast numerical computing with GPU support
- **Flax NNX**: Modern neural network framework
- **Matplotlib**: 3D visualization

### Step 2: Configure the Model

```python
# Create network config for the point cloud transformer
network_config = PointCloudNetworkConfig(
    name="point_cloud_network",
    hidden_dims=(128, 128, 128),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=128,      # Feature dimension
    num_heads=4,        # Attention heads
    num_layers=3,       # Transformer depth
    dropout_rate=0.1,   # Regularization
)

# Create point cloud config with nested network config
config = PointCloudConfig(
    name="point_cloud_generator",
    network=network_config,
    num_points=512,     # Number of points
    dropout_rate=0.1,
)
```

**Configuration breakdown:**

- `input_dim` & `output_dim`: Point cloud shape (points × dimensions)
- `hidden_dims`: Internal processing dimensions
- `parameters`: Model-specific settings

### Step 3: Create the Model

```python
rngs = nnx.Rngs(params=jax.random.key(42))
model = PointCloudModel(config=config, rngs=rngs)
```

The model initializes its transformer layers with random weights.

### Step 4: Generate Point Clouds

```python
point_clouds = model.generate(
    rngs=rngs,
    n_samples=2,        # Generate 2 point clouds
    temperature=0.8,    # Control diversity
)
```

**Temperature effects:**

- **0.5-0.7**: Focused, consistent samples
- **0.8-1.0**: Balanced diversity ← Recommended
- **1.0+**: High diversity, may be noisy

### Step 5: Visualize in 3D

```python
def plot_point_cloud(points, filename=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color by distance from origin
    norm = np.sqrt(np.sum(points**2, axis=1))
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=norm, cmap="viridis", s=20, alpha=0.7)
    plt.colorbar(scatter)
    return fig
```

The visualization:

- Projects 3D points onto 2D screen
- Colors points by distance from origin
- Saves plots to `examples_output/` directory

## Expected Output

```
Creating point cloud model...
Generating point clouds...
Visualizing point clouds...
Example completed! Point clouds saved as PNG files.
```

**Generated files:**

- `examples_output/point_cloud_1.png` - First generated point cloud
- `examples_output/point_cloud_2.png` - Second generated point cloud

Each visualization shows a 3D scatter plot with:

- X, Y, Z axes labeled
- Color gradient indicating spatial distribution
- 512 points representing the generated shape

## Experiments to Try

### 1. Vary Number of Points

```python
# Sparse point cloud (faster, less detail)
"num_points": 256

# Dense point cloud (slower, more detail)
"num_points": 1024
```

**Tradeoff**: More points = better shape representation but slower generation

### 2. Adjust Temperature

```python
# Low temperature (focused samples)
temperature=0.6

# High temperature (diverse samples)
temperature=1.2
```

**Try it**: Generate 5 samples at different temperatures and compare diversity

### 3. Modify Architecture Depth

```python
# Shallow network (faster, simpler patterns)
"num_layers": 2

# Deep network (slower, complex patterns)
"num_layers": 6
```

**Note**: Deeper networks may need more training data

### 4. Multi-Head Attention

```python
# Fewer heads (simpler attention)
"num_heads": 2

# More heads (richer attention patterns)
"num_heads": 8
```

**Heads must divide embed_dim evenly**: e.g., 128 ÷ 4 = 32 ✓

## Understanding Point Cloud Applications

### 1. Autonomous Vehicles

LiDAR sensors generate point clouds for:

- Obstacle detection
- Lane tracking
- 3D scene understanding

**Typical setup**: 64-128 laser beams → 100K+ points per second

### 2. Robotics

Point clouds enable:

- Object grasping (find grip points)
- Navigation (3D mapping)
- Human-robot interaction (gesture recognition)

### 3. Molecular Modeling

Proteins as point clouds:

- Each atom is a 3D point
- Backbone atoms: N, C-alpha, C, O
- Sidechains: variable number of atoms

See [`protein_point_cloud_example.py`](../protein/protein-point-cloud-example.md) for details

### 4. 3D Content Creation

Generate 3D models for:

- Video games (procedural generation)
- Movies (digital assets)
- Virtual reality (environments)

## Next Steps

<div class="grid cards" markdown>

- :material-cube-outline: **Geometric Models Overview**

    ---

    Quick reference for point clouds, meshes, and voxels

    [:octicons-arrow-right-24: geometric_models_demo.py](geometric-models-demo.md)

- :material-chart-line: **Geometric Losses**

    ---

    Learn specialized loss functions for point clouds

    [:octicons-arrow-right-24: geometric_losses_demo.py](geometric-losses-demo.md)

- :material-atom: **Protein Point Clouds**

    ---

    Apply point clouds to protein structure modeling

    [:octicons-arrow-right-24: protein_point_cloud_example.py](../protein/protein-point-cloud-example.md)

- :material-flask: **Geometric Benchmarks**

    ---

    Evaluate on standard geometric datasets

    [:octicons-arrow-right-24: geometric_benchmark_demo.py](geometric-benchmark-demo.md)

</div>

## Troubleshooting

### Points look random/unstructured

**Cause**: Model is untrained or poorly trained

**Solutions**:

1. Train on a dataset first (this example uses pretrained weights)
2. Increase `num_layers` for more capacity
3. Adjust `temperature` for different sampling behavior

### "FigureCanvasAgg is non-interactive"

**Cause**: matplotlib trying to show plots in non-GUI environment

**Solution**: This is just a warning, plots are still saved. To suppress:

```python
import matplotlib
matplotlib.use('Agg')  # Add before importing pyplot
```

### Out of memory errors

**Solutions**:

```python
# Reduce number of points
"num_points": 256  # Instead of 512

# Reduce batch size in generation
n_samples=1  # Instead of 2

# Use CPU instead of GPU
JAX_PLATFORMS=cpu python simple_point_cloud_example.py
```

### Generated point clouds are identical

**Cause**: Not providing fresh random keys

**Solution**: Split RNG keys for each generation:

```python
for i in range(n_samples):
    key, subkey = jax.random.split(key)
    rngs_new = nnx.Rngs(params=subkey)
    point_cloud = model.generate(rngs=rngs_new, n_samples=1)
```

## Additional Resources

- [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593) - Original point cloud deep learning paper
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Point Cloud Models](../../models/point_cloud.md) - Model documentation
- [JAX Point Cloud Processing](https://github.com/google/jax/discussions/9806) - Community discussions
