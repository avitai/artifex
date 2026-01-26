# Geometric Models Demo

<div class="example-badges">
  <span class="badge badge-beginner">Beginner</span>
  <span class="badge badge-runtime-fast">⚡ 5-10 seconds</span>
  <span class="badge badge-format-dual">📓 Dual Format</span>
</div>

A quick reference guide demonstrating how to configure and instantiate three types of geometric models in Artifex: point clouds, meshes, and voxels.

## Files

- **Python Script**: [`geometric_models_demo.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_models_demo.py)
- **Jupyter Notebook**: [`geometric_models_demo.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_models_demo.ipynb)

## Quick Start

```bash
# Clone and setup
cd artifex
source activate.sh

# Run Python script
python examples/generative_models/geometric/geometric_models_demo.py

# Or use Jupyter notebook
jupyter notebook examples/generative_models/geometric/geometric_models_demo.ipynb
```

## Overview

This example provides a concise demonstration of Artifex's three main geometric representations:

### Learning Objectives

- [x] Understand point cloud, mesh, and voxel representations
- [x] Configure models using frozen dataclass configs (`PointCloudConfig`, `MeshConfig`, `VoxelConfig`)
- [x] Use the unified factory pattern with `create_model()`
- [x] Understand model-specific parameters for each geometry type

### Prerequisites

- Basic understanding of 3D geometric representations
- Familiarity with JAX and Flax NNX
- Artifex installed and activated

## Geometric Representations Explained

### 1. Point Clouds

**Unordered sets of 3D points** - flexible, permutation-invariant

```python
from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)

# Network config for point cloud transformer
network_config = PointCloudNetworkConfig(
    name="point_cloud_network",
    hidden_dims=(128, 128),  # Tuple for frozen dataclass
    activation="gelu",
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    dropout_rate=0.1,
)

# Point cloud config with nested network
point_cloud_config = PointCloudConfig(
    name="demo_point_cloud",
    network=network_config,
    num_points=512,
    loss_type="chamfer",
    dropout_rate=0.1,
)
point_cloud_model = create_model(point_cloud_config, rngs=rngs)
```

**Use cases:**

- LiDAR data processing
- 3D object detection
- Molecular structure modeling (proteins, molecules)

**Key parameters:**

- `num_points`: Number of points in the cloud
- `embed_dim`: Embedding dimension for features
- `num_layers`: Network depth
- `loss_type`: Distance metric (chamfer, earth mover's distance)

### 2. Mesh Models

**Connected vertex structures with topology** - surface-oriented

```python
from artifex.generative_models.core.configuration import (
    MeshConfig,
    MeshNetworkConfig,
)

# Network config for mesh processing
mesh_network_config = MeshNetworkConfig(
    name="mesh_network",
    hidden_dims=(256, 128, 64),  # Tuple for frozen dataclass
    activation="gelu",
)

# Mesh config with nested network and loss weights
mesh_config = MeshConfig(
    name="demo_mesh",
    network=mesh_network_config,
    num_vertices=512,
    template_type="sphere",
    vertex_loss_weight=1.0,
    normal_loss_weight=0.2,
    edge_loss_weight=0.1,
)
mesh_model = create_model(mesh_config, rngs=rngs)
```

**Use cases:**

- 3D graphics and rendering
- Shape analysis and generation
- Surface reconstruction

**Key parameters:**

- `num_vertices`: Number of mesh vertices
- `template_type`: Initial mesh template (sphere, cube, etc.)
- Loss weights balance geometric properties:
  - `vertex_loss_weight`: Vertex position accuracy
  - `normal_loss_weight`: Surface normal consistency
  - `edge_loss_weight`: Edge length regularization

### 3. Voxel Models

**Regular 3D grids** - easy to process with CNNs

```python
from artifex.generative_models.core.configuration import (
    VoxelConfig,
    VoxelNetworkConfig,
)

# Network config for voxel 3D CNN
voxel_network_config = VoxelNetworkConfig(
    name="voxel_network",
    hidden_dims=(128, 64),  # Required base config
    activation="relu",
    base_channels=64,  # Base number of 3D CNN channels
    num_layers=4,       # Number of 3D convolutional layers
)

# Voxel config with nested network
voxel_config = VoxelConfig(
    name="demo_voxel",
    network=voxel_network_config,
    resolution=16,
    use_conditioning=True,
    conditioning_dim=10,
    loss_type="focal",
    focal_gamma=2.0,
)
voxel_model = create_model(voxel_config, rngs=rngs)
```

**Use cases:**

- Medical imaging (CT, MRI scans)
- 3D scene understanding
- Volumetric shape generation

**Key parameters:**

- `resolution`: Grid resolution (16³ = 4,096 voxels)
- `channels`: Multi-scale architecture layers
- `use_conditioning`: Enable class-conditional generation
- `loss_type`: "focal" handles sparse voxel data
- `focal_gamma`: Focus on hard-to-classify voxels (2.0 is standard)

## Expected Output

```
Creating point cloud model...
Created model: PointCloudModel
Sample shape: (1, 512, 3)

Creating mesh model...
Created model: MeshModel

Creating voxel model with conditioning...
Created model: VoxelModel

Demo completed successfully!
```

## Code Walkthrough

### Step 1: Setup Random Number Generation

```python
rng = jax.random.PRNGKey(42)
rngs = nnx.Rngs(params=rng)
```

Initialize RNG using Flax NNX patterns for reproducible model creation.

### Step 2: Create Models via Factory

All three models use the unified `create_model()` factory pattern:

```python
model = create_model(config, rngs=rngs)
```

This abstracts away model-specific initialization details and provides a consistent API.

### Step 3: Generate Samples (Optional)

```python
sample = point_cloud_model.sample(1, rngs=rngs)
# shape: (1, 512, 3) - batch_size=1, num_points=512, xyz=3
```

## Choosing the Right Representation

| Representation | When to Use | Strengths | Limitations |
|---------------|-------------|-----------|-------------|
| **Point Cloud** | Raw sensor data, irregular shapes, molecular structures | Flexible, no topology required, permutation-invariant | No explicit surface, harder to render |
| **Mesh** | Graphics, animation, smooth surfaces | Explicit topology, efficient rendering, smooth surfaces | Requires consistent topology, harder to optimize |
| **Voxel** | Medical imaging, volumetric data, regular grids | Easy CNN processing, regular structure | Memory-intensive, discretization artifacts |

## Experiments to Try

1. **Change loss types** - Try different distance metrics for point clouds:

   ```python
   "loss_type": "emd"  # Earth Mover's Distance (slower but more accurate)
   ```

2. **Adjust mesh templates** - Experiment with different starting shapes:

   ```python
   "template_type": "cube"  # or "icosahedron", "octahedron"
   ```

3. **Scale voxel resolution** - Balance memory vs. detail:

   ```python
   "resolution": 32  # 32³ = 32,768 voxels (8x more memory)
   ```

4. **Conditional generation** - Create class-specific voxel shapes:

   ```python
   labels = jnp.array([0, 1, 5, 9])  # Different classes
   voxel_model.generate(labels, rngs=rngs)
   ```

## Next Steps

<div class="grid cards" markdown>

- :material-cube-outline: **Point Cloud Deep Dive**

    ---

    Learn advanced point cloud techniques with PointNet and Set Abstraction

    [:octicons-arrow-right-24: simple_point_cloud_example.py](simple-point-cloud-example.md)

- :material-atom: **Protein Modeling**

    ---

    Apply point clouds to protein structure prediction

    [:octicons-arrow-right-24: protein_point_cloud_example.py](../protein/protein-point-cloud-example.md)

- :material-chart-line: **Geometric Losses**

    ---

    Explore specialized losses for geometric data

    [:octicons-arrow-right-24: geometric_losses_demo.py](geometric-losses-demo.md)

- :material-flask: **Geometric Benchmarks**

    ---

    Comprehensive evaluation on geometric tasks

    [:octicons-arrow-right-24: geometric_benchmark_demo.py](geometric-benchmark-demo.md)

</div>

## Troubleshooting

### "Backend 'cuda' is not in the list of known backends"

**Solution**: JAX is looking for CUDA but can't find it. Run with CPU:

```bash
JAX_PLATFORMS=cpu python examples/generative_models/geometric/geometric_models_demo.py
```

### "ModuleNotFoundError: No module named 'artifex'"

**Solution**: Activate the environment first:

```bash
source activate.sh
python examples/generative_models/geometric/geometric_models_demo.py
```

### Memory errors with high resolution voxels

**Solution**: Reduce voxel resolution or use gradient checkpointing:

```python
"resolution": 8,  # Lower resolution (8³ = 512 voxels)
```

## Additional Resources

- [Configuration Reference](../../api/core/configuration.md)
- [Factory Pattern Guide](../../factory/index.md)
- [JAX Geometric Processing](https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html)
