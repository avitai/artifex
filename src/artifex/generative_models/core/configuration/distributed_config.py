"""DistributedConfig frozen dataclass configuration.

Design:
- Frozen dataclass inheriting from BaseConfig
- All validation in __post_init__ using DRY utilities
- DistributedBackend as StrEnum (Python 3.11+)
- Auto-configure mesh logic uses object.__setattr__ for frozen mutation
- Helper methods for data parallelism, mesh config, and process identity
"""

import dataclasses
import enum
from typing import Literal

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.validation import (
    validate_non_negative_int,
    validate_positive_int,
    validate_range,
)


class DistributedBackend(enum.StrEnum):
    """Supported distributed training backends.

    Attributes:
        NCCL: NVIDIA GPU distributed training
        GLOO: CPU distributed training
        MPI: MPI distributed training
    """

    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DistributedConfig(BaseConfig):
    """Configuration for distributed training with robust validation.

    This config contains all distributed training settings including backend
    selection, process topology, model parallelism, mesh configuration,
    communication optimizations, and mixed precision.

    Auto-configures the device mesh for Flax NNX sharding when ``enabled``
    is True and no explicit ``mesh_shape`` is provided.

    Attributes:
        enabled: Whether to enable distributed training.
        backend: Distributed backend to use (nccl, gloo, mpi).
        world_size: Total number of processes (must be positive).
        rank: Global rank of this process (non-negative, < world_size).
        local_rank: Local rank of this process (non-negative, < num_processes_per_node).
        num_nodes: Number of nodes (must be positive).
        num_processes_per_node: Processes per node (must be positive).
        master_addr: Master node address for rendezvous.
        master_port: Port for distributed communication (1024-65535).
        tensor_parallel_size: Size of tensor parallelism group (must be positive).
        pipeline_parallel_size: Size of pipeline parallelism group (must be positive).
        mesh_shape: Device mesh shape for NNX sharding (auto-configured if None).
        mesh_axis_names: Axis names for device mesh (auto-configured if None).
        find_unused_parameters: Whether to find unused parameters in DDP.
        gradient_as_bucket_view: Whether to use gradient bucketing in DDP.
        broadcast_buffers: Whether to broadcast buffers in DDP.
        mixed_precision: Mixed precision training mode ("no", "fp16", or "bf16").
    """

    # Distributed training settings
    enabled: bool = False
    backend: DistributedBackend = DistributedBackend.NCCL
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    num_nodes: int = 1
    num_processes_per_node: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500

    # Model parallelism settings
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Flax NNX specific settings
    mesh_shape: tuple[int, ...] | None = None
    mesh_axis_names: tuple[str, ...] | None = None

    # Communication settings
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = True

    # Mixed precision settings
    mixed_precision: Literal["no", "fp16", "bf16"] = "no"

    def __post_init__(self) -> None:
        """Validate all fields and auto-configure mesh.

        Validation uses DRY utilities from validation.py where possible.
        Follows fail-fast principle -- raise on first error.
        Uses ``object.__setattr__`` for frozen field mutation during
        auto-configuration of mesh shape and axis names.
        """
        # Call parent validation first
        super(DistributedConfig, self).__post_init__()

        # --- Field-level validation ---

        # Positive integer fields
        validate_positive_int(self.world_size, "world_size")
        validate_positive_int(self.num_nodes, "num_nodes")
        validate_positive_int(self.num_processes_per_node, "num_processes_per_node")
        validate_positive_int(self.tensor_parallel_size, "tensor_parallel_size")
        validate_positive_int(self.pipeline_parallel_size, "pipeline_parallel_size")

        # Non-negative integer fields
        validate_non_negative_int(self.rank, "rank")
        validate_non_negative_int(self.local_rank, "local_rank")

        # Port range
        validate_range(self.master_port, "master_port", min_val=1024, max_val=65535)

        # --- Cross-field consistency checks ---

        # Rank must be less than world_size
        if self.rank >= self.world_size:
            raise ValueError(f"rank ({self.rank}) must be less than world_size ({self.world_size})")

        # Local rank must be less than num_processes_per_node
        if self.local_rank >= self.num_processes_per_node:
            raise ValueError(
                f"local_rank ({self.local_rank}) must be less than "
                f"num_processes_per_node ({self.num_processes_per_node})"
            )

        # World size must equal num_nodes * num_processes_per_node
        if self.world_size != self.num_nodes * self.num_processes_per_node:
            raise ValueError(
                f"world_size ({self.world_size}) must equal "
                f"num_nodes ({self.num_nodes}) * "
                f"num_processes_per_node ({self.num_processes_per_node})"
            )

        # Parallelism sizes must not exceed world size
        total_parallel = self.tensor_parallel_size * self.pipeline_parallel_size
        if total_parallel > self.world_size:
            raise ValueError(
                f"Product of parallel sizes ({total_parallel}) cannot exceed "
                f"world_size ({self.world_size})"
            )

        if self.world_size % total_parallel != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"product of parallel sizes ({total_parallel})"
            )

        # --- Mesh validation ---

        if self.mesh_shape is not None:
            mesh_size = 1
            for dim in self.mesh_shape:
                mesh_size *= dim
            if mesh_size != self.world_size:
                raise ValueError(
                    f"mesh_shape product ({mesh_size}) must equal world_size ({self.world_size})"
                )

        if self.mesh_axis_names is not None:
            if self.mesh_shape is None:
                raise ValueError("mesh_axis_names requires mesh_shape to be set")
            if len(self.mesh_axis_names) != len(self.mesh_shape):
                raise ValueError("mesh_axis_names length must match mesh_shape length")

        # --- Auto-configure mesh for NNX if not specified ---

        if self.enabled and self.mesh_shape is None:
            self._auto_configure_mesh()

    def _auto_configure_mesh(self) -> None:
        """Auto-configure mesh shape and axis names based on parallelism settings.

        Sets ``mesh_shape`` and ``mesh_axis_names`` using ``object.__setattr__``
        because this dataclass is frozen.
        """
        if self.world_size == 1:
            object.__setattr__(self, "mesh_shape", (1,))
            object.__setattr__(self, "mesh_axis_names", ("data",))
        elif self.tensor_parallel_size > 1 and self.pipeline_parallel_size > 1:
            # Both tensor and pipeline parallelism
            data_parallel = self.world_size // (
                self.tensor_parallel_size * self.pipeline_parallel_size
            )
            object.__setattr__(
                self,
                "mesh_shape",
                (data_parallel, self.tensor_parallel_size, self.pipeline_parallel_size),
            )
            object.__setattr__(self, "mesh_axis_names", ("data", "model", "pipeline"))
        elif self.tensor_parallel_size > 1:
            # Only tensor parallelism
            data_parallel = self.world_size // self.tensor_parallel_size
            object.__setattr__(self, "mesh_shape", (data_parallel, self.tensor_parallel_size))
            object.__setattr__(self, "mesh_axis_names", ("data", "model"))
        else:
            # Data parallelism only
            object.__setattr__(self, "mesh_shape", (self.world_size,))
            object.__setattr__(self, "mesh_axis_names", ("data",))

    def get_data_parallel_size(self) -> int:
        """Calculate the data parallel size.

        Returns:
            Number of data-parallel replicas, computed as world_size divided
            by the product of tensor and pipeline parallel sizes.
        """
        return self.world_size // (self.tensor_parallel_size * self.pipeline_parallel_size)

    def get_mesh_config(self) -> dict[str, tuple[int, ...] | tuple[str, ...]] | None:
        """Get mesh configuration for Flax NNX.

        Returns:
            Dictionary with ``mesh_shape`` and ``axis_names`` keys,
            or None if mesh is not configured.
        """
        if self.mesh_shape is None or self.mesh_axis_names is None:
            return None

        return {
            "mesh_shape": self.mesh_shape,
            "axis_names": self.mesh_axis_names,
        }

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0).

        Returns:
            True if this process has global rank 0.
        """
        return self.rank == 0

    def is_local_main_process(self) -> bool:
        """Check if this is the local main process (local_rank 0).

        Returns:
            True if this process has local rank 0.
        """
        return self.local_rank == 0
