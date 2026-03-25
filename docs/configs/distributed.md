# Distributed Config

`DistributedConfig` describes multi-process topology, device-mesh shape, and
distributed runtime policy.

## Public Import

```python
from artifex.configs import DistributedBackend, DistributedConfig

config = DistributedConfig(
    name="multi_gpu",
    enabled=True,
    backend=DistributedBackend.NCCL,
    world_size=4,
    num_nodes=1,
    num_processes_per_node=4,
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
)
```

## Key Fields

- `enabled`
- `backend`
- `world_size`
- `rank`
- `local_rank`
- `num_nodes`
- `num_processes_per_node`
- `master_addr`
- `master_port`
- `tensor_parallel_size`
- `pipeline_parallel_size`
- `mesh_shape`
- `mesh_axis_names`
- `mixed_precision`

The dataclass validates cross-field consistency and auto-configures a mesh when
distributed mode is enabled without an explicit mesh.
