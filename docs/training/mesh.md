# Device Mesh Management

**Module:** `artifex.generative_models.training.distributed.mesh`

**Source:** `src/artifex/generative_models/training/distributed/mesh.py`

## Overview

The `DeviceMeshManager` class provides utilities for creating and managing JAX device meshes for distributed training. It supports various parallelism strategies including data parallelism, model parallelism, and hybrid parallelism.

## Classes

### DeviceMeshManager

Manager for creating and configuring JAX device meshes.

```python
class DeviceMeshManager:
    """Manager for creating and configuring JAX device meshes.

    This class provides methods for creating device meshes with various
    configurations for data parallelism, model parallelism, and hybrid
    parallelism strategies.
    """
```

#### Constructor

```python
def __init__(self, devices: Sequence[Any] | None = None) -> None:
    """Initialize DeviceMeshManager.

    Args:
        devices: Optional list of devices to use. If None, uses all
            available devices from jax.devices().
    """
```

#### Methods

##### create_device_mesh

Create a device mesh with the specified shape.

```python
def create_device_mesh(
    self,
    mesh_shape: dict[str, int] | list[tuple[str, int]],
    devices: Sequence[Any] | None = None,
) -> Mesh:
    """Create a device mesh with the specified shape.

    Args:
        mesh_shape: Shape specification as either:
            - dict mapping axis names to sizes, e.g., {"data": 2, "model": 1}
            - list of (axis_name, size) tuples, e.g., [("data", 2), ("model", 1)]
        devices: Optional list of devices to use.

    Returns:
        A JAX Mesh with the specified configuration.

    Raises:
        ValueError: If mesh requires more devices than available.
    """
```

##### create_data_parallel_mesh

Create a mesh for data parallelism.

```python
def create_data_parallel_mesh(
    self,
    num_devices: int | None = None,
    axis_name: str = "data",
) -> Mesh:
    """Create a mesh for data parallelism.

    Args:
        num_devices: Number of devices to use. If None, uses all available.
        axis_name: Name of the data parallel axis.

    Returns:
        A JAX Mesh configured for data parallelism.
    """
```

##### create_model_parallel_mesh

Create a mesh for model parallelism.

```python
def create_model_parallel_mesh(
    self,
    num_devices: int | None = None,
    axis_name: str = "model",
) -> Mesh:
    """Create a mesh for model parallelism.

    Args:
        num_devices: Number of devices to use. If None, uses all available.
        axis_name: Name of the model parallel axis.

    Returns:
        A JAX Mesh configured for model parallelism.
    """
```

##### create_hybrid_mesh

Create a mesh for hybrid data and model parallelism.

```python
def create_hybrid_mesh(
    self,
    data_parallel_size: int = 1,
    model_parallel_size: int = 1,
    data_axis: str = "data",
    model_axis: str = "model",
) -> Mesh:
    """Create a mesh for hybrid data and model parallelism.

    Args:
        data_parallel_size: Number of devices for data parallelism.
        model_parallel_size: Number of devices for model parallelism.
        data_axis: Name of the data parallel axis.
        model_axis: Name of the model parallel axis.

    Returns:
        A JAX Mesh configured for hybrid parallelism.
    """
```

##### get_mesh_info

Get information about a device mesh.

```python
def get_mesh_info(self, mesh: Mesh) -> dict[str, Any]:
    """Get information about a device mesh.

    Args:
        mesh: The mesh to get information about.

    Returns:
        Dictionary containing mesh information with keys:
            - total_devices: Total number of devices in the mesh
            - axes: Dict mapping axis names to their sizes
    """
```

#### Properties

- `num_devices: int` - Number of available devices
- `devices: list[Any]` - List of available devices

## Usage Examples

### Basic Usage

```python
from artifex.generative_models.training.distributed import DeviceMeshManager

# Create manager (uses all available devices)
manager = DeviceMeshManager()
print(f"Available devices: {manager.num_devices}")

# Create a data-parallel mesh using all devices
mesh = manager.create_data_parallel_mesh()
```

### Data Parallelism

```python
# Create mesh for data parallelism with 4 devices
manager = DeviceMeshManager()
mesh = manager.create_data_parallel_mesh(num_devices=4, axis_name="batch")

# Use the mesh with JAX sharding
from jax.sharding import NamedSharding, PartitionSpec

# Shard data along batch dimension
data_sharding = NamedSharding(mesh, PartitionSpec("batch"))
```

### Model Parallelism

```python
# Create mesh for model parallelism
manager = DeviceMeshManager()
mesh = manager.create_model_parallel_mesh(num_devices=2, axis_name="model")

# Shard model parameters
param_sharding = NamedSharding(mesh, PartitionSpec(None, "model"))
```

### Hybrid Parallelism

```python
# Create 2D mesh: 2 devices for data, 2 for model (total 4 devices)
manager = DeviceMeshManager()
mesh = manager.create_hybrid_mesh(
    data_parallel_size=2,
    model_parallel_size=2,
    data_axis="data",
    model_axis="model",
)

# Get mesh info
info = manager.get_mesh_info(mesh)
print(f"Total devices: {info['total_devices']}")
print(f"Axes: {info['axes']}")  # {'data': 2, 'model': 2}
```

### Using Dict or List Specification

```python
manager = DeviceMeshManager()

# Using dict specification
mesh_dict = manager.create_device_mesh({"data": 2, "model": 1})

# Using list of tuples specification
mesh_list = manager.create_device_mesh([("data", 2), ("model", 1)])
```

## Module Statistics

- **Classes:** 1 (`DeviceMeshManager`)
- **Methods:** 5 public methods
- **Properties:** 2 (`num_devices`, `devices`)
