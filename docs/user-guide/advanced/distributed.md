# Distributed Training

Distributed training in artifex is JAX's `jax.sharding` model: one program runs
on every device, arrays carry their sharding, and XLA inserts the collectives.
The helpers that build meshes, place batches and reduce gradients and metrics
are not artifex's. They live in [substrax](https://substrax.readthedocs.io), the
infrastructure package every Avitai library depends on, so a mesh built for a
datarax pipeline is the mesh an artifex trainer runs on.

## What substrax provides

| Module | Names | Purpose |
| --- | --- | --- |
| `substrax.devices` | `detect_devices`, `DeviceInfo`, `DeviceKind`, `DevicePlacement`, `HardwareType`, `BatchSizeRecommendation`, `place_on_device`, `distribute_batch`, `get_batch_size_recommendation` | Device identity, explicit placement and hardware-aware batch sizes |
| `substrax.mesh` | `DeviceMeshManager`, `MeshRules`, `data_parallel_rules`, `fsdp_rules`, `create_named_sharding`, `partition_spec_for_names`, `ShardingConfig`, `ParallelismConfig`, the `ShardingStrategy` classes | Meshes, partition rules and multi-dimensional parallelism plans |
| `substrax.spmd` | `create_data_parallel_sharding`, `place_batch_on_shards`, `place_nnx_state_on_shards`, `spmd_train_step`, `reduce_gradient_tree`, `reduce_mean`, `reduce_sum`, `reduce_max`, `reduce_min`, `reduce_custom`, `reduce_mean_collective`, `reduce_sum_collective`, `all_gather`, `collect_from_devices` | Data-parallel sharding, the SPMD step and cross-device reductions |

Every artifex trainer takes an `nnx.Optimizer` and a loss function, so
distribution is applied around the trainer, not inside it: build the mesh, shard
the batch, run the step under the mesh context.

## Data parallelism

Data parallelism keeps one replica of the model per device and splits every
batch along its leading axis.

```python
import jax
from flax import nnx
from substrax.mesh import DeviceMeshManager
from substrax.spmd import create_data_parallel_sharding, place_batch_on_shards, spmd_train_step

mesh = DeviceMeshManager.create_data_parallel_mesh()   # every visible device on one "data" axis
sharding = create_data_parallel_sharding(mesh)        # NamedSharding over the "data" axis

model = create_vae_model(config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


def loss_fn(model, batch):
    return model.loss_fn(batch, model(batch["images"], training=True))["total_loss"]


@nnx.jit
def train_step(model, optimizer, batch):
    return spmd_train_step(model, optimizer, loss_fn, batch)


with jax.set_mesh(mesh):
    for batch in train_loader():
        loss = train_step(model, optimizer, place_batch_on_shards(batch, sharding))
```

`spmd_train_step` differentiates with `nnx.value_and_grad` and updates the
optimizer; because the parameters are replicated and the batch is sharded, XLA
reduces the gradients across devices on its own. `DeviceMeshManager.get_mesh_info`
reports the shape and axis names of the mesh you built.

### Metrics across devices

Per-device metrics are reduced with the `substrax.spmd` collectives:

```python
from substrax.spmd import reduce_mean, collect_from_devices

metrics = reduce_mean({"loss": loss, "kl": kl})   # one value per metric
per_device = collect_from_devices({"loss": loss})  # every device's value, for inspection
```

`reduce_mean_collective` and `reduce_sum_collective` are the in-`jit` forms that
take an axis name.

## Model and pipeline parallelism

Larger models shard parameters instead of, or as well as, data. substrax
expresses the plan as strategies over named mesh axes:

```python
from substrax.mesh import (
    DataParallelStrategy,
    DeviceMeshManager,
    MultiDimensionalStrategy,
    ParallelismConfig,
    ShardingConfig,
    TensorParallelStrategy,
)

mesh = DeviceMeshManager.create_hybrid_mesh(data_parallel_size=2, model_parallel_size=4)
plan = MultiDimensionalStrategy(
    strategies={
        "data": DataParallelStrategy("data", 0),
        "tensor": TensorParallelStrategy("model", 1),
    },
    config=ParallelismConfig(
        mesh_shape=(2, 4),
        mesh_axis_names=("data", "model"),
        sharding_config=ShardingConfig(data_parallel_size=2, tensor_parallel_size=4),
    ),
)
```

`FSDPStrategy` shards parameters above `min_weight_size` across the data axis,
and `PipelineParallelStrategy` assigns `num_stages` to a pipeline axis. Model
state is placed with `place_nnx_state_on_shards(state, mesh, filter_sharding)`,
where the filter maps parameter paths to `PartitionSpec`s; `MeshRules`,
`data_parallel_rules` and `fsdp_rules` build those specs from parameter names.

`ProductionOptimizer` in `artifex.generative_models.inference.optimization`
accepts a `ParallelismConfig` and records it with the optimised model, so the
plan travels with the artifact.

## Device placement and batch sizes

`DevicePlacement` detects the hardware type (CPU, the GPU generation, the TPU
version) and places or splits pytrees explicitly:

```python
from substrax.devices import DevicePlacement, get_batch_size_recommendation

placement = DevicePlacement()
batch = placement.place_on_device(batch)                 # to the default device
recommendation = get_batch_size_recommendation(model_memory_gb=2.0)
print(placement.hardware_type, recommendation.batch_size)
```

`get_batch_size_recommendation` reads the hardware's memory and a per-generation
table to suggest a batch size; `validate_batch_size` checks a chosen one.

## Multi-process training

On several hosts, initialise JAX before touching devices and build the mesh
from `jax.devices()`, which then spans every process:

```python
import jax

jax.distributed.initialize()   # coordinator address and process index from the environment
mesh = DeviceMeshManager.create_data_parallel_mesh()
```

Each process feeds its own shard of the data; `place_batch_on_shards` with a
global sharding assembles the global batch. Checkpoints written through
`substrax.checkpoint.OrbaxCheckpointStore` are sharding-aware and can be
restored onto a different device count.

## Performance notes

- Keep the batch axis divisible by the number of devices on the `data` axis;
  `DevicePlacement.validate_batch_size` reports the largest valid size.
- Prefer `bfloat16` compute with `float32` parameters on TPU and recent GPUs, and
  pair it with dynamic loss scaling from the
  [Advanced Features guide](../training/advanced-features.md#dynamic-loss-scaling).
- Gradient accumulation composes with sharding through `optax.MultiSteps` in the
  optimizer; nothing in the step changes.
- Profile with `calibrax.profiling` to confirm the collectives overlap with
  compute before scaling out further.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ValueError: ... not divisible by ...` on `place_batch_on_shards` | batch axis not a multiple of the device count | pad the last batch or drop it in the loader |
| One device busy, the rest idle | the step ran outside `jax.set_mesh` or inputs were never sharded | wrap the loop in the mesh context and shard every batch |
| Out of memory on one replica | per-device batch too large | lower the per-device batch and raise `every_k_schedule` on the optimizer |
| Different loss per process | per-process RNG or data not aligned | seed from the process index and shard the dataset by it |

## Next steps

- [Advanced Features](../training/advanced-features.md): accumulation and loss scaling
- [Checkpointing](checkpointing.md): the store the trainers save through
- substrax API reference: <https://substrax.readthedocs.io>
