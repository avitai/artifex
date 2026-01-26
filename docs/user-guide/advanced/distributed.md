# Distributed Training

Distributed training enables training large models across multiple GPUs or nodes by parallelizing computation and distributing data. Artifex provides configuration and utilities for distributed training using JAX's native parallelization capabilities.

<div class="grid cards" markdown>

- :material-chart-line:{ .lg .middle } **Data Parallelism**

    ---

    Distribute data batches across devices while replicating the model

    [:octicons-arrow-right-24: Learn more](#data-parallelism)

- :material-cube-outline:{ .lg .middle } **Model Parallelism**

    ---

    Split large models across devices when they don't fit in memory

    [:octicons-arrow-right-24: Learn more](#model-parallelism)

- :material-pipeline:{ .lg .middle } **Pipeline Parallelism**

    ---

    Split model layers across devices and pipeline batches

    [:octicons-arrow-right-24: Learn more](#pipeline-parallelism)

- :material-grid:{ .lg .middle } **Device Meshes**

    ---

    Organize devices with multi-dimensional parallelism strategies

    [:octicons-arrow-right-24: Learn more](#device-meshes)

</div>

## Overview

Artifex uses JAX's `jax.sharding` API and device meshes for distributed training, providing:

- **Automatic distribution**: JAX handles device communication
- **SPMD (Single Program Multiple Data)**: Same code runs on all devices
- **Flexible strategies**: Mix data, model, and pipeline parallelism
- **XLA optimization**: Automatic fusion and communication overlap

## Artifex Distributed Training Module

Artifex provides a high-level distributed training module that simplifies common distributed training patterns:

```python
from artifex.generative_models.training.distributed import (
    DeviceMeshManager,  # Device mesh creation and management
    DataParallel,       # Data parallel training utilities
    DevicePlacement,    # Explicit device placement
    DistributedMetrics, # Metrics aggregation across devices
    HardwareType,       # Hardware type enumeration
    BatchSizeRecommendation,  # Hardware-specific batch recommendations
)
```

### Quick Start with Artifex Utilities

```python
from artifex.generative_models.training.distributed import (
    DeviceMeshManager,
    DataParallel,
    DistributedMetrics,
)
import jax.numpy as jnp

# Create device mesh manager
manager = DeviceMeshManager()
print(f"Available devices: {manager.num_devices}")

# Create mesh for data parallelism
mesh = manager.create_data_parallel_mesh()

# Create data parallel utilities
dp = DataParallel()
dm = DistributedMetrics()

# Create sharding for batch data
sharding = dp.create_data_parallel_sharding(mesh)

# Shard batch data
batch = {"inputs": jnp.ones((32, 784)), "targets": jnp.zeros((32,))}
sharded_batch = dp.shard_batch(batch, sharding)

# In training step, aggregate gradients and metrics
# grads = dp.all_reduce_gradients(grads, reduce_type="mean")
# metrics = dm.reduce_mean(metrics)
```

For detailed documentation of each module, see:

- [Device Mesh Management](../../training/mesh.md) - `DeviceMeshManager` for creating device meshes
- [Data Parallel Training](../../training/data_parallel.md) - `DataParallel` for data parallelism
- [Device Placement](../../training/device_placement.md) - `DevicePlacement` for explicit device placement
- [Distributed Metrics](../../training/distributed_metrics.md) - `DistributedMetrics` for aggregating metrics

### Why Distributed Training?

Use distributed training when:

1. **Large Batches**: Need bigger batch sizes than fit on one GPU
2. **Large Models**: Model parameters exceed single device memory
3. **Faster Training**: Reduce wall-clock time with more compute
4. **Multi-Node**: Scale to cluster-level training

## Distributed Configuration

Artifex provides a comprehensive configuration system for distributed training:

```python
from artifex.configs.schema.distributed import DistributedConfig

# Basic distributed configuration
config = DistributedConfig(
    enabled=True,
    world_size=4,  # Total number of devices
    backend="nccl",  # Use NCCL for NVIDIA GPUs

    # Device mesh configuration
    mesh_shape=(2, 2),  # 2x2 device grid
    mesh_axis_names=("data", "model"),  # Axis semantics

    # Parallelism settings
    tensor_parallel_size=2,  # Tensor parallelism degree
    pipeline_parallel_size=1,  # Pipeline parallelism degree
)
```

### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `enabled` | `bool` | Enable distributed training |
| `world_size` | `int` | Total number of processes |
| `backend` | `str` | Backend: `nccl`, `gloo`, `mpi` |
| `rank` | `int` | Global rank of this process |
| `local_rank` | `int` | Local rank on this node |
| `num_nodes` | `int` | Number of nodes in cluster |
| `num_processes_per_node` | `int` | Processes per node |
| `master_addr` | `str` | Master node address |
| `master_port` | `int` | Communication port |
| `tensor_parallel_size` | `int` | Tensor parallelism group size |
| `pipeline_parallel_size` | `int` | Pipeline parallelism group size |
| `mesh_shape` | `tuple` | Device mesh dimensions |
| `mesh_axis_names` | `tuple` | Semantic names for mesh axes |
| `mixed_precision` | `str` | Mixed precision mode: `no`, `fp16`, `bf16` |

### Configuration Validation

The configuration includes automatic validation:

```python
# This configuration is validated automatically
config = DistributedConfig(
    enabled=True,
    world_size=8,
    num_nodes=2,
    num_processes_per_node=4,
    tensor_parallel_size=2,
    pipeline_parallel_size=2,
    # Automatically validates:
    # - world_size == num_nodes * num_processes_per_node
    # - tensor_parallel * pipeline_parallel <= world_size
    # - rank < world_size
)

# Get derived values
data_parallel_size = config.get_data_parallel_size()  # 2
is_main = config.is_main_process()  # True if rank == 0
is_local_main = config.is_local_main_process()  # True if local_rank == 0
```

## Data Parallelism

Data parallelism replicates the model on each device and processes different data batches in parallel.

### Basic Data Parallelism

```python
import jax
import jax.numpy as jnp
from flax import nnx

# Get available devices
devices = jax.devices()
print(f"Available devices: {len(devices)}")  # e.g., 4 GPUs

# Create model
model = create_vae_model(config, rngs=nnx.Rngs(0))

# Replicate model across devices
replicated_model = jax.device_put_replicated(
    nnx.state(model),
    devices
)

# Training step with pmap
@jax.pmap
def train_step(model_state, batch):
    """Training step replicated across devices."""
    # Reconstruct model from state
    model = nnx.merge(model_graphdef, model_state)

    # Forward pass
    def loss_fn(model):
        output = model(batch["data"])
        return output["loss"]

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Update parameters (NNX 0.11.0+ API)
    optimizer.update(model, grads)

    return nnx.state(model), loss

# Store model structure
model_graphdef, _ = nnx.split(model)

# Prepare batched data (one batch per device)
batch_per_device = {
    "data": jnp.array([...]),  # Shape: (num_devices, batch_size, ...)
}

# Run parallel training step
updated_state, losses = train_step(replicated_model, batch_per_device)

# Average losses across devices
mean_loss = jnp.mean(losses)
```

### Data Parallelism with Device Mesh

Modern approach using `jax.sharding`:

```python
import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx

# Create device mesh for data parallelism
devices = jax.devices()
mesh = Mesh(devices, axis_names=("data",))

# Define sharding for data (shard along batch dimension)
data_sharding = NamedSharding(mesh, P("data", None, None, None))

# Define sharding for model (replicate across all devices)
model_sharding = NamedSharding(mesh, P())

# Create model
model = create_vae_model(config, rngs=nnx.Rngs(0))
model_state = nnx.state(model)

# Shard model state (replicate)
sharded_model_state = jax.device_put(model_state, model_sharding)

# JIT-compiled training step with sharding
@jax.jit
def train_step(model_state, batch):
    """Training step with automatic distribution."""
    model = nnx.merge(model_graphdef, model_state)

    def loss_fn(model):
        output = model(batch["data"])
        return output["loss"]

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # NNX 0.11.0+ API

    return nnx.state(model), loss

# Store model structure
model_graphdef, _ = nnx.split(model)

# Training loop
for batch in dataloader:
    # Shard batch data across devices
    sharded_batch = jax.device_put(batch, data_sharding)

    # Training step (automatically distributed)
    sharded_model_state, loss = train_step(
        sharded_model_state,
        sharded_batch
    )

    print(f"Loss: {loss}")
```

### Gradient Aggregation

When using data parallelism, gradients are automatically aggregated:

```python
@jax.jit
def train_step_with_aggregation(model_state, batch):
    """Training step with explicit gradient aggregation."""
    model = nnx.merge(model_graphdef, model_state)

    def loss_fn(model):
        output = model(batch["data"])
        return output["loss"]

    # Compute gradients on this device's data
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Average gradients across devices (handled automatically by JAX)
    # When using jax.pmap, use jax.lax.pmean:
    # grads = jax.lax.pmean(grads, axis_name="batch")
    # loss = jax.lax.pmean(loss, axis_name="batch")

    # Update parameters (NNX 0.11.0+ API)
    optimizer.update(model, grads)

    return nnx.state(model), loss
```

## Model Parallelism

Model parallelism (tensor parallelism) splits model layers across devices, useful when models don't fit in single-device memory.

### Tensor Parallelism Basics

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flax import nnx

# Create 2D device mesh: (data_parallel, model_parallel)
devices = jax.devices()
mesh = Mesh(
    devices.reshape(2, 2),  # 2 data parallel, 2 model parallel
    axis_names=("data", "model")
)

# Define sharding for model parameters
# Shard weights along model axis, replicate bias
weight_sharding = NamedSharding(mesh, P(None, "model"))  # (in_features, out_features)
bias_sharding = NamedSharding(mesh, P("model"))  # (out_features,)

# Create model with sharded parameters
class ShardedLinear(nnx.Module):
    """Linear layer with sharded weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        super().__init__()

        # Create weight with sharding
        self.weight = nnx.Param(
            nnx.initializers.lecun_normal()(
                rngs.params(),
                (in_features, out_features)
            )
        )

        # Apply sharding
        self.weight = jax.device_put(
            self.weight,
            NamedSharding(mesh, P(None, "model"))
        )

        # Create bias with sharding
        self.bias = nnx.Param(
            jnp.zeros(out_features)
        )
        self.bias = jax.device_put(
            self.bias,
            NamedSharding(mesh, P("model"))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Computation automatically distributed
        return x @ self.weight + self.bias
```

### Multi-Layer Model Parallelism

```python
class ShardedVAEEncoder(nnx.Module):
    """VAE encoder with model parallelism."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        *,
        rngs: nnx.Rngs,
        mesh: Mesh,
    ):
        super().__init__()

        # First layer: replicated input, sharded output
        self.layer1 = ShardedLinear(
            input_dim, hidden_dim,
            rngs=rngs, mesh=mesh
        )

        # Second layer: sharded input, sharded output
        self.layer2 = ShardedLinear(
            hidden_dim, hidden_dim,
            rngs=rngs, mesh=mesh
        )

        # Output layers for mean and logvar
        self.mean_layer = ShardedLinear(
            hidden_dim, latent_dim,
            rngs=rngs, mesh=mesh
        )
        self.logvar_layer = ShardedLinear(
            hidden_dim, latent_dim,
            rngs=rngs, mesh=mesh
        )

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        # Forward pass with automatic communication
        h = nnx.relu(self.layer1(x))
        h = nnx.relu(self.layer2(h))

        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)

        return {"mean": mean, "logvar": logvar}

# Create model with mesh
devices = jax.devices()
mesh = Mesh(devices.reshape(2, 2), axis_names=("data", "model"))

# Initialize model
encoder = ShardedVAEEncoder(
    input_dim=784,
    hidden_dim=512,
    latent_dim=20,
    rngs=nnx.Rngs(0),
    mesh=mesh,
)

# Model parameters are automatically sharded
```

### Activation Sharding

Control how activations are sharded between layers:

```python
from jax.experimental import shard_map

@jax.jit
def sharded_forward(model_state, x):
    """Forward pass with explicit activation sharding."""
    model = nnx.merge(model_graphdef, model_state)

    # x shape: (batch, features)
    # Shard along batch dimension
    x_sharding = NamedSharding(mesh, P("data", None))
    x = jax.device_put(x, x_sharding)

    # Forward pass
    h1 = model.layer1(x)  # Output sharded along (data, model)
    h1 = nnx.relu(h1)

    # Collect along model axis before next layer
    h1 = jax.lax.all_gather(h1, "model", axis=1, tiled=True)

    h2 = model.layer2(h1)

    return h2
```

## Pipeline Parallelism

Pipeline parallelism splits model layers across devices and pipelines microbatches through stages.

### Pipeline Stage Definition

```python
from flax import nnx
import jax
import jax.numpy as jnp

class PipelineStage(nnx.Module):
    """A single stage in a pipeline parallel model."""

    def __init__(
        self,
        layer_specs: list,
        stage_id: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.stage_id = stage_id
        self.layers = []

        # Create layers for this stage
        for spec in layer_specs:
            layer = nnx.Linear(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                rngs=rngs,
            )
            self.layers.append(layer)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through this stage."""
        for layer in self.layers:
            x = nnx.relu(layer(x))
        return x

# Define 4-stage pipeline
stage_specs = [
    # Stage 0: Input layers
    [{"in_features": 784, "out_features": 512}],
    # Stage 1: Middle layers
    [{"in_features": 512, "out_features": 512}],
    # Stage 2: Middle layers
    [{"in_features": 512, "out_features": 256}],
    # Stage 3: Output layers
    [{"in_features": 256, "out_features": 10}],
]

# Create stages
stages = [
    PipelineStage(spec, stage_id=i, rngs=nnx.Rngs(i))
    for i, spec in enumerate(stage_specs)
]
```

### Microbatch Pipeline Execution

```python
def pipeline_forward(stages, inputs, num_microbatches):
    """Execute forward pass with pipeline parallelism."""
    # Split batch into microbatches
    microbatch_size = inputs.shape[0] // num_microbatches
    microbatches = [
        inputs[i * microbatch_size:(i + 1) * microbatch_size]
        for i in range(num_microbatches)
    ]

    num_stages = len(stages)

    # Pipeline state: activations at each stage
    stage_activations = [None] * num_stages
    outputs = []

    # Pipeline schedule: (time_step, stage_id, microbatch_id)
    for time_step in range(num_stages + num_microbatches - 1):
        for stage_id in range(num_stages):
            microbatch_id = time_step - stage_id

            # Check if this stage should process a microbatch
            if 0 <= microbatch_id < num_microbatches:
                if stage_id == 0:
                    # First stage: use input
                    stage_input = microbatches[microbatch_id]
                else:
                    # Other stages: use previous stage output
                    stage_input = stage_activations[stage_id - 1]

                # Process through this stage
                stage_output = stages[stage_id](stage_input)
                stage_activations[stage_id] = stage_output

                # If last stage, collect output
                if stage_id == num_stages - 1:
                    outputs.append(stage_output)

    # Concatenate outputs
    return jnp.concatenate(outputs, axis=0)

# Use pipeline
inputs = jnp.ones((32, 784))  # Batch of 32
output = pipeline_forward(stages, inputs, num_microbatches=4)
```

### GPipe-Style Pipeline

```python
class GPipePipeline(nnx.Module):
    """GPipe-style pipeline with gradient accumulation."""

    def __init__(
        self,
        num_stages: int,
        layers_per_stage: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.stages = []

        for i in range(num_stages):
            stage_layers = []
            for j in range(layers_per_stage):
                stage_layers.append(
                    nnx.Linear(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        rngs=rngs,
                    )
                )
            self.stages.append(stage_layers)

    def forward_stage(self, stage_id: int, x: jax.Array) -> jax.Array:
        """Forward pass through one stage."""
        for layer in self.stages[stage_id]:
            x = nnx.relu(layer(x))
        return x

    def __call__(
        self,
        x: jax.Array,
        num_microbatches: int = 1
    ) -> jax.Array:
        """Forward pass with microbatching."""
        if num_microbatches == 1:
            # No microbatching
            for stage_id in range(self.num_stages):
                x = self.forward_stage(stage_id, x)
            return x

        # Microbatched pipeline
        return pipeline_forward(
            [lambda x, i=i: self.forward_stage(i, x)
             for i in range(self.num_stages)],
            x,
            num_microbatches
        )
```

## Device Meshes

Device meshes organize devices with multi-dimensional parallelism.

### Creating Device Meshes

```python
import jax
from jax.sharding import Mesh

# Get available devices
devices = jax.devices()
print(f"Total devices: {len(devices)}")  # e.g., 8 GPUs

# 1D mesh (data parallelism only)
mesh_1d = Mesh(devices, axis_names=("data",))

# 2D mesh (data + model parallelism)
mesh_2d = Mesh(
    devices.reshape(4, 2),  # 4 data parallel, 2 model parallel
    axis_names=("data", "model")
)

# 3D mesh (data + model + pipeline parallelism)
mesh_3d = Mesh(
    devices.reshape(2, 2, 2),  # 2x2x2 grid
    axis_names=("data", "model", "pipeline")
)

# Check mesh properties
print(f"Mesh shape: {mesh_2d.shape}")  # (4, 2)
print(f"Mesh axis names: {mesh_2d.axis_names}")  # ('data', 'model')
```

### Using Mesh Context

```python
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

# Create mesh
devices = jax.devices()
mesh = Mesh(devices.reshape(2, 4), axis_names=("data", "model"))

# Use mesh context for automatic sharding
with mesh:
    # Create model
    model = create_vae_model(config, rngs=nnx.Rngs(0))

    # Define sharding strategies
    data_sharding = NamedSharding(mesh, P("data", None))
    param_sharding = NamedSharding(mesh, P(None, "model"))

    # Shard model parameters
    model_state = nnx.state(model)
    sharded_state = jax.tree.map(
        lambda x: jax.device_put(x, param_sharding),
        model_state
    )

    # Training loop
    for batch in dataloader:
        # Shard batch
        sharded_batch = jax.device_put(batch, data_sharding)

        # Training step (automatically uses mesh)
        sharded_state, loss = train_step(sharded_state, sharded_batch)
```

### Mesh Inspection

```python
# Inspect sharding of arrays
def inspect_sharding(array, name="array"):
    """Print sharding information for an array."""
    sharding = array.sharding
    print(f"{name}:")
    print(f"  Shape: {array.shape}")
    print(f"  Sharding: {sharding}")
    print(f"  Devices: {len(sharding.device_set)} devices")

# Check model parameter sharding
for name, param in nnx.state(model).items():
    inspect_sharding(param, name)

# Visualize mesh
def visualize_mesh(mesh):
    """Visualize device mesh layout."""
    print(f"Mesh shape: {mesh.shape}")
    print(f"Axis names: {mesh.axis_names}")
    print("\nDevice layout:")

    devices_array = mesh.devices
    for i in range(devices_array.shape[0]):
        for j in range(devices_array.shape[1]):
            device = devices_array[i, j]
            print(f"  [{i},{j}]: {device}")

visualize_mesh(mesh)
```

## Multi-Node Training

Scaling training to multiple nodes requires coordination across machines.

### Multi-Node Setup

```bash
# Node 0 (master)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8  # 2 nodes x 4 GPUs
export RANK=0
export LOCAL_RANK=0

python train.py --distributed

# Node 1 (worker)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4  # Ranks 4-7 on node 1
export LOCAL_RANK=0

python train.py --distributed
```

### JAX Multi-Node Initialization

```python
import jax
import os

def setup_multinode():
    """Initialize JAX for multi-node training."""
    # Get environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", 29500))

    # JAX automatically handles multi-host setup
    # Just need to ensure CUDA_VISIBLE_DEVICES is set correctly
    # and that the same code runs on all hosts

    print(f"Rank {rank}/{world_size} on {jax.device_count()} local devices")
    print(f"Total devices: {jax.device_count()} local, {jax.device_count() * world_size} global")

    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_master": rank == 0,
    }

# Setup distributed training
dist_info = setup_multinode()

# Create mesh across all nodes
devices = jax.devices()  # All devices across all hosts
mesh = Mesh(devices, axis_names=("data",))

# Training code identical to single-node
with mesh:
    # Your training code here
    pass
```

### Distributed Training Script

```python
from artifex.configs.schema.distributed import DistributedConfig
from artifex.generative_models.training.trainer import Trainer
import jax

def main():
    # Create distributed config
    dist_config = DistributedConfig(
        enabled=True,
        world_size=8,
        num_nodes=2,
        num_processes_per_node=4,
        master_addr=os.environ.get("MASTER_ADDR", "localhost"),
        master_port=int(os.environ.get("MASTER_PORT", 29500)),
        rank=int(os.environ.get("RANK", 0)),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        mesh_shape=(8,),  # Data parallel only
        mesh_axis_names=("data",),
    )

    # Create device mesh
    devices = jax.devices()
    mesh = Mesh(
        devices.reshape(dist_config.mesh_shape),
        axis_names=dist_config.mesh_axis_names
    )

    # Create model and training config
    model_config = create_model_config()
    training_config = create_training_config()

    # Create trainer
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        distributed_config=dist_config,
    )

    # Train with automatic distribution
    with mesh:
        trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
```

## Performance Optimization

Optimize distributed training for maximum efficiency.

### Communication Overlap

Overlap computation with communication:

```python
@jax.jit
def optimized_train_step(model_state, batch, optimizer_state):
    """Training step with computation-communication overlap."""
    model = nnx.merge(model_graphdef, model_state)

    # Forward pass
    def loss_fn(model):
        output = model(batch["data"])
        return output["loss"]

    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # JAX automatically overlaps:
    # 1. Gradient computation (backward pass)
    # 2. Gradient all-reduce (across devices)
    # 3. Parameter updates

    # Update optimizer
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    model_state = optax.apply_updates(model_state, updates)

    return model_state, optimizer_state, loss
```

### Gradient Accumulation

Accumulate gradients across microbatches:

```python
@jax.jit
def train_step_with_accumulation(
    model_state,
    batch,
    optimizer_state,
    num_microbatches: int = 4
):
    """Training step with gradient accumulation."""
    model = nnx.merge(model_graphdef, model_state)

    # Split batch into microbatches
    microbatch_size = batch["data"].shape[0] // num_microbatches

    # Initialize accumulated gradients
    accumulated_grads = jax.tree.map(jnp.zeros_like, nnx.state(model))
    total_loss = 0.0

    # Process microbatches
    for i in range(num_microbatches):
        start_idx = i * microbatch_size
        end_idx = (i + 1) * microbatch_size
        microbatch = {
            "data": batch["data"][start_idx:end_idx]
        }

        # Compute gradients for this microbatch
        def loss_fn(model):
            output = model(microbatch["data"])
            return output["loss"]

        loss, grads = nnx.value_and_grad(loss_fn)(model)

        # Accumulate gradients
        accumulated_grads = jax.tree.map(
            lambda acc, g: acc + g / num_microbatches,
            accumulated_grads,
            grads
        )
        total_loss += loss / num_microbatches

    # Single optimizer update with accumulated gradients
    updates, optimizer_state = optimizer.update(
        accumulated_grads,
        optimizer_state
    )
    model_state = optax.apply_updates(model_state, updates)

    return model_state, optimizer_state, total_loss

# Use with larger effective batch size
for batch in dataloader:  # batch_size = 32
    # Effective batch size = 32 * 4 = 128
    model_state, optimizer_state, loss = train_step_with_accumulation(
        model_state, batch, optimizer_state, num_microbatches=4
    )
```

!!! tip "GradientAccumulator Utility"
    Artifex provides a dedicated `GradientAccumulator` class for cleaner gradient accumulation with automatic normalization. See [Advanced Features](../training/advanced-features.md#gradient-accumulation) for details.

### Memory-Efficient Training

Reduce memory usage in distributed training:

```python
# Use mixed precision
from jax import numpy as jnp

@jax.jit
def mixed_precision_train_step(model_state, batch):
    """Training step with mixed precision (bfloat16)."""
    # Cast inputs to bfloat16
    batch_bf16 = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
        batch
    )

    # Forward and backward in bfloat16
    model = nnx.merge(model_graphdef, model_state)

    def loss_fn(model):
        output = model(batch_bf16["data"])
        return output["loss"]

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Cast gradients back to float32 for optimizer
    grads_fp32 = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        grads
    )

    # Update in float32
    optimizer.update(grads_fp32)

    return nnx.state(model), loss.astype(jnp.float32)
```

!!! tip "Dynamic Loss Scaling"
    For numerical stability in mixed-precision distributed training, use the `DynamicLossScaler` class which automatically adjusts loss scaling to prevent overflow/underflow. See [Advanced Features](../training/advanced-features.md#dynamic-loss-scaling) for details.

## Troubleshooting

Common issues and solutions in distributed training.

### Out of Memory (OOM)

**Problem**: Model doesn't fit in GPU memory even with distribution.

**Solutions**:

1. **Increase Model Parallelism**:

```python
# Use more model parallel devices
config = DistributedConfig(
    tensor_parallel_size=4,  # Increase from 2
    mesh_shape=(2, 4),  # 2 data, 4 model parallel
)
```

2. **Add Gradient Accumulation**:

```python
# Reduce microbatch size, accumulate gradients
train_step_with_accumulation(
    model_state, batch, optimizer_state,
    num_microbatches=8  # Smaller microbatches
)
```

3. **Use Gradient Checkpointing** (see [Checkpointing Guide](checkpointing.md))

### Slow Training

**Problem**: Training slower than expected with multiple devices.

**Solutions**:

1. **Check Device Utilization**:

```python
import jax.profiler

# Profile training step
jax.profiler.start_trace("/tmp/tensorboard")
train_step(model_state, batch)
jax.profiler.stop_trace()

# View in TensorBoard
# tensorboard --logdir=/tmp/tensorboard
```

2. **Optimize Batch Size**:

```python
# Increase batch size per device
# Optimal: batch_size * num_devices fills GPU memory ~80%
optimal_batch_size = 64  # Per device
total_batch_size = optimal_batch_size * num_devices
```

3. **Reduce Communication Overhead**:

```python
# Use larger microbatches in pipeline parallelism
pipeline_forward(stages, inputs, num_microbatches=2)  # Instead of 8

# Increase data parallelism, reduce model parallelism if possible
```

### Hanging or Deadlocks

**Problem**: Training hangs or deadlocks during execution.

**Solutions**:

1. **Check Collective Operations**:

```python
# Ensure all devices participate in collectives
@jax.jit
def train_step(model_state, batch):
    # Bad: Only some devices execute all-reduce
    if jax.process_index() == 0:
        grads = jax.lax.pmean(grads, "batch")  # Deadlock!

    # Good: All devices execute all-reduce
    grads = jax.lax.pmean(grads, "batch")  # OK

    return model_state, loss
```

2. **Verify World Size**:

```python
# Check all processes are launched
assert jax.device_count() == expected_devices
assert jax.process_count() == expected_processes
```

### Numerical Instability

**Problem**: Loss becomes NaN or diverges in distributed training.

**Solutions**:

1. **Check Gradient Aggregation**:

```python
# Ensure gradients are averaged, not summed
grads = jax.lax.pmean(grads, "batch")  # Mean
# grads = jax.lax.psum(grads, "batch")  # Sum (wrong!)
```

2. **Use Gradient Clipping**:

```python
import optax

# Clip gradients before update
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip to norm 1.0
    optax.adam(learning_rate=1e-4),
)
```

## Best Practices

### DO

- ✅ **Use jax.sharding** for modern distributed training
- ✅ **Profile before optimizing** - measure actual bottlenecks
- ✅ **Start with data parallelism** - simplest and most efficient
- ✅ **Use mixed precision (bfloat16)** for memory and speed
- ✅ **Test on single device first** before distributing
- ✅ **Monitor device utilization** with profiling tools
- ✅ **Use gradient accumulation** for large effective batch sizes
- ✅ **Validate mesh configuration** with DistributedConfig
- ✅ **Keep code identical across devices** (SPMD principle)
- ✅ **Log only on rank 0** to avoid cluttered output

### DON'T

- ❌ **Don't use different code on different devices** - breaks SPMD
- ❌ **Don't skip validation** - invalid configs cause cryptic errors
- ❌ **Don't over-shard** - communication overhead dominates
- ❌ **Don't ignore profiling** - assumptions often wrong
- ❌ **Don't use pmap for new code** - use jax.sharding instead
- ❌ **Don't assume linear scaling** - measure actual speedup
- ❌ **Don't mix parallelism strategies** without profiling
- ❌ **Don't forget gradient averaging** in data parallelism
- ❌ **Don't use model parallelism** if data parallelism works
- ❌ **Don't checkpoint on all ranks** - only rank 0 should save

## Summary

Distributed training in Artifex leverages JAX's native capabilities:

1. **Data Parallelism**: Replicate model, distribute data batches
2. **Model Parallelism**: Shard model parameters across devices
3. **Pipeline Parallelism**: Split model layers, pipeline microbatches
4. **Device Meshes**: Multi-dimensional parallelism strategies
5. **Automatic Distribution**: JAX handles communication with jax.sharding

Key APIs:

- `DistributedConfig`: Configuration with validation
- `jax.sharding.Mesh`: Multi-dimensional device organization
- `PartitionSpec`: Specify sharding strategies
- `NamedSharding`: Apply sharding to arrays
- `@jax.jit`: Automatic distribution with XLA

## Next Steps

<div class="grid cards" markdown>

- :material-book-open-variant:{ .lg .middle } **Model Parallelism**

    ---

    Deep dive into tensor and pipeline parallelism strategies

    [:octicons-arrow-right-24: Read the guide](parallelism.md)

- :material-content-save:{ .lg .middle } **Checkpointing**

    ---

    Learn about gradient and model checkpointing for memory efficiency

    [:octicons-arrow-right-24: Checkpointing guide](checkpointing.md)

- :material-cube-outline:{ .lg .middle } **Custom Architectures**

    ---

    Build custom distributed model architectures

    [:octicons-arrow-right-24: Architecture guide](architectures.md)

- :material-speedometer:{ .lg .middle } **Training Guide**

    ---

    Return to the comprehensive training documentation

    [:octicons-arrow-right-24: Training guide](../training/training-guide.md)

</div>
