# Device Placement

**Module:** `artifex.generative_models.training.distributed.device_placement`

**Source:** `src/artifex/generative_models/training/distributed/device_placement.py`

## Overview

The `DevicePlacement` module provides utilities for explicit device placement of JAX arrays and PyTrees, enabling efficient data distribution across accelerators. It includes hardware-aware batch size recommendations based on JAX performance guidelines.

## Enums

### HardwareType

Enumeration of supported hardware types for batch size recommendations.

```python
class HardwareType(Enum):
    """Enumeration of supported hardware types."""
    TPU_V5E = "tpu_v5e"
    TPU_V5P = "tpu_v5p"
    TPU_V4 = "tpu_v4"
    H100 = "h100"
    A100 = "a100"
    V100 = "v100"
    CPU = "cpu"
    UNKNOWN = "unknown"
```

## Classes

### BatchSizeRecommendation

Hardware-specific batch size recommendations dataclass.

```python
@dataclass(frozen=True)
class BatchSizeRecommendation:
    """Hardware-specific batch size recommendations.

    Attributes:
        min_batch_size: Minimum batch size for reasonable efficiency.
        optimal_batch_size: Optimal batch size for peak throughput.
        critical_batch_size: Critical batch size for reaching roofline (per JAX guide).
        max_memory_batch_size: Maximum batch size before OOM (estimate).
        notes: Additional notes about the recommendation.
    """
    min_batch_size: int
    optimal_batch_size: int
    critical_batch_size: int
    max_memory_batch_size: int | None = None
    notes: str = ""
```

#### Hardware-Specific Values

| Hardware | Min Batch | Optimal | Critical | Notes |
|----------|-----------|---------|----------|-------|
| TPU v5e | 64 | 256 | 240 | Critical batch size for reaching roofline |
| TPU v5p | 128 | 512 | 480 | Higher throughput, needs larger batches |
| TPU v4 | 64 | 256 | 192 | Similar to v5e, slightly lower critical |
| H100 | 64 | 320 | 298 | Critical batch size for roofline |
| A100 | 32 | 256 | 240 | For 80GB variant |
| V100 | 16 | 128 | 96 | Memory-limited on 16GB variant |
| CPU | 1 | 32 | 16 | Memory-bandwidth bound |

### DevicePlacement

Utility class for explicit device placement of JAX arrays.

```python
class DevicePlacement:
    """Utility class for explicit device placement of JAX arrays.

    This class provides methods for placing arrays on specific devices,
    distributing batches across devices using sharding, and providing
    hardware-aware batch size recommendations.
    """
```

#### Constructor

```python
def __init__(self, default_device: Any | None = None) -> None:
    """Initialize DevicePlacement.

    Args:
        default_device: Default device to use when none is specified.
            If None, uses jax.devices()[0].
    """
```

#### Methods

##### place_on_device

Place data on a specific device.

```python
def place_on_device(
    self,
    data: Any,
    device: Any | None = None,
) -> Any:
    """Place data on a specific device.

    Args:
        data: PyTree of JAX arrays to place on device.
        device: Target device. If None, uses the default device.

    Returns:
        PyTree with arrays placed on the specified device.
    """
```

##### distribute_batch

Distribute data across devices using sharding.

```python
def distribute_batch(
    self,
    data: Any,
    sharding: Sharding,
) -> Any:
    """Distribute data across devices using the specified sharding.

    Args:
        data: PyTree of JAX arrays to distribute.
        sharding: JAX Sharding specification.

    Returns:
        PyTree with arrays distributed according to the sharding.
    """
```

##### replicate_across_devices

Replicate data across all devices.

```python
def replicate_across_devices(
    self,
    data: Any,
    devices: list[Any] | None = None,
) -> Any:
    """Replicate data across all specified devices.

    Args:
        data: PyTree of JAX arrays to replicate.
        devices: List of devices to replicate to. If None, uses all devices.

    Returns:
        PyTree with arrays replicated across devices.
    """
```

##### shard_batch_dim

Shard data along the batch dimension.

```python
def shard_batch_dim(
    self,
    data: Any,
    mesh: Mesh,
    batch_axis: int = 0,
    mesh_axis: str = "data",
) -> Any:
    """Shard data along the batch dimension.

    This is the most common sharding pattern for data-parallel training,
    where each device processes a slice of the batch.

    Args:
        data: PyTree of JAX arrays to shard.
        mesh: Device mesh to shard across.
        batch_axis: The axis index representing the batch dimension.
        mesh_axis: The mesh axis name to shard along.

    Returns:
        PyTree with arrays sharded along the batch dimension.
    """
```

##### prefetch_to_device

Create a prefetching wrapper for async data placement.

```python
def prefetch_to_device(
    self,
    data_iterator: Iterator[Any],
    device: Any | None = None,
    buffer_size: int = 2,
) -> Iterator[Any]:
    """Create a prefetching wrapper that places data on device asynchronously.

    This enables overlapping data transfer with computation for improved
    throughput in training loops.

    Args:
        data_iterator: Iterator yielding PyTrees of data.
        device: Target device for prefetching.
        buffer_size: Number of batches to prefetch.

    Returns:
        Iterator that yields device-placed data.
    """
```

##### get_batch_size_recommendation

Get hardware-specific batch size recommendations.

```python
def get_batch_size_recommendation(
    self,
    hardware_type: HardwareType | None = None,
) -> BatchSizeRecommendation:
    """Get batch size recommendation for the current hardware.

    Args:
        hardware_type: Override hardware type. If None, uses detected type.

    Returns:
        BatchSizeRecommendation with hardware-specific values.
    """
```

##### validate_batch_size

Validate batch size against hardware recommendations.

```python
def validate_batch_size(
    self,
    batch_size: int,
    warn_suboptimal: bool = True,
) -> tuple[bool, str]:
    """Validate batch size against hardware recommendations.

    Args:
        batch_size: The batch size to validate.
        warn_suboptimal: Whether to warn for suboptimal (but valid) sizes.

    Returns:
        Tuple of (is_valid, message).
    """
```

##### get_device_info

Get information about available devices.

```python
def get_device_info(self) -> dict[str, Any]:
    """Get information about available devices.

    Returns:
        Dictionary containing device information including:
        - num_devices: Number of available devices
        - hardware_type: Detected hardware type
        - platforms: List of unique platforms
        - device_kinds: List of device kinds
        - devices: Detailed list of device info
    """
```

#### Properties

- `hardware_type: HardwareType` - The detected hardware type
- `num_devices: int` - Number of available devices

## Convenience Functions

### place_on_device

```python
def place_on_device(data: Any, device: Any | None = None) -> Any:
    """Convenience function for placing data on a device.

    Args:
        data: PyTree of JAX arrays.
        device: Target device. If None, uses first available device.

    Returns:
        PyTree with arrays on the specified device.
    """
```

### distribute_batch

```python
def distribute_batch(data: Any, sharding: Sharding) -> Any:
    """Convenience function for distributing data across devices.

    Args:
        data: PyTree of JAX arrays.
        sharding: JAX Sharding specification.

    Returns:
        PyTree with arrays distributed according to sharding.
    """
```

### get_batch_size_recommendation

```python
def get_batch_size_recommendation(
    hardware_type: HardwareType | None = None,
) -> BatchSizeRecommendation:
    """Get batch size recommendation for current or specified hardware.

    Args:
        hardware_type: Hardware type to get recommendation for.

    Returns:
        BatchSizeRecommendation with hardware-specific values.
    """
```

## Usage Examples

### Basic Device Placement

```python
from artifex.generative_models.training.distributed import (
    DevicePlacement,
    place_on_device,
)
import jax.numpy as jnp

# Create placement utility
placement = DevicePlacement()
print(f"Detected hardware: {placement.hardware_type}")
print(f"Available devices: {placement.num_devices}")

# Place data on default device
data = jnp.ones((32, 784))
placed_data = placement.place_on_device(data)

# Or use convenience function
placed_data = place_on_device(data)
```

### Batch Size Validation

```python
from artifex.generative_models.training.distributed import (
    DevicePlacement,
    HardwareType,
    get_batch_size_recommendation,
)

# Get recommendation for current hardware
placement = DevicePlacement()
rec = placement.get_batch_size_recommendation()
print(f"Optimal batch size: {rec.optimal_batch_size}")
print(f"Critical batch size: {rec.critical_batch_size}")

# Get recommendation for specific hardware
h100_rec = get_batch_size_recommendation(HardwareType.H100)
print(f"H100 critical batch: {h100_rec.critical_batch_size}")  # 298

# Validate batch size
is_valid, message = placement.validate_batch_size(256)
print(f"Valid: {is_valid}, Message: {message}")
```

### Distributing Batches with Sharding

```python
from artifex.generative_models.training.distributed import (
    DevicePlacement,
    distribute_batch,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import jax
import numpy as np

# Create device mesh
devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=("data",))

# Create sharding for batch dimension
data_sharding = NamedSharding(mesh, PartitionSpec("data", None))

# Distribute data
placement = DevicePlacement()
batch = {"images": jnp.ones((8, 28, 28, 3)), "labels": jnp.zeros((8,))}
distributed = placement.distribute_batch(batch, data_sharding)

# Or use convenience function
distributed = distribute_batch(batch, data_sharding)
```

### Sharding Along Batch Dimension

```python
from artifex.generative_models.training.distributed import DevicePlacement
from jax.sharding import Mesh
import jax
import numpy as np

placement = DevicePlacement()

# Create mesh
devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=("data",))

# Shard batch along first dimension
batch = {
    "images": jnp.ones((16, 224, 224, 3)),
    "labels": jnp.zeros((16,), dtype=jnp.int32),
}
sharded_batch = placement.shard_batch_dim(batch, mesh)
```

### Prefetching Data to Device

```python
from artifex.generative_models.training.distributed import DevicePlacement

placement = DevicePlacement()

# Create a data iterator
def data_generator():
    for i in range(100):
        yield {"batch": jnp.ones((32, 784)) * i}

# Prefetch data to GPU with buffer of 2 batches
prefetched = placement.prefetch_to_device(
    data_generator(),
    buffer_size=2,
)

# Training loop with prefetched data
for batch in prefetched:
    # Data is already on GPU when we receive it
    process_batch(batch)
```

### Replicating Model Weights

```python
from artifex.generative_models.training.distributed import DevicePlacement

placement = DevicePlacement()

# Model weights to replicate
weights = {
    "layer1": jnp.ones((784, 256)),
    "layer2": jnp.ones((256, 10)),
}

# Replicate across all devices
replicated_weights = placement.replicate_across_devices(weights)

# Or replicate to specific devices
gpu_devices = jax.devices("gpu")[:2]
replicated_weights = placement.replicate_across_devices(weights, devices=gpu_devices)
```

### Getting Device Information

```python
from artifex.generative_models.training.distributed import DevicePlacement

placement = DevicePlacement()
info = placement.get_device_info()

print(f"Number of devices: {info['num_devices']}")
print(f"Hardware type: {info['hardware_type']}")
print(f"Platforms: {info['platforms']}")

for device in info['devices']:
    print(f"  Device {device['id']}: {device['platform']} ({device['device_kind']})")
```

## Module Statistics

- **Classes:** 2 (`DevicePlacement`, `BatchSizeRecommendation`)
- **Enums:** 1 (`HardwareType`)
- **Convenience Functions:** 3
- **Instance Methods:** 7
