"""Enhanced distributed training configuration schema with improved validation."""

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from artifex.configs.schema.base import BaseConfig


class DistributedBackend(str, Enum):
    """Supported distributed training backends."""

    NCCL = "nccl"  # NVIDIA GPU distributed training
    GLOO = "gloo"  # CPU distributed training
    MPI = "mpi"  # MPI distributed training


class DistributedConfig(BaseConfig):
    """Enhanced configuration for distributed training with robust validation."""

    # Default name
    name: str = Field(
        "distributed_config",
        description="Name for this distributed training configuration",
    )

    # Distributed training settings
    enabled: bool = Field(False, description="Whether to enable distributed training")
    backend: DistributedBackend = Field(
        DistributedBackend.NCCL, description="Distributed backend to use"
    )
    world_size: int = Field(1, description="Total number of processes")
    rank: int = Field(0, description="Global rank of this process")
    local_rank: int = Field(0, description="Local rank of this process")
    num_nodes: int = Field(1, description="Number of nodes")
    num_processes_per_node: int = Field(1, description="Processes per node")
    master_addr: str = Field("localhost", description="Master node address")
    master_port: int = Field(29500, description="Port for distributed communication")

    # Model parallelism settings
    tensor_parallel_size: int = Field(1, description="Size of tensor parallelism group")
    pipeline_parallel_size: int = Field(1, description="Size of pipeline parallelism group")

    # Flax NNX specific settings
    mesh_shape: tuple | None = Field(None, description="Device mesh shape for NNX sharding")
    mesh_axis_names: tuple | None = Field(None, description="Axis names for device mesh")

    # Communication settings
    find_unused_parameters: bool = Field(
        False, description="Whether to find unused parameters in DDP"
    )
    gradient_as_bucket_view: bool = Field(
        True, description="Whether to use gradient bucketing in DDP"
    )
    broadcast_buffers: bool = Field(True, description="Whether to broadcast buffers in DDP")

    # Mixed precision settings
    mixed_precision: Literal["no", "fp16", "bf16"] = Field(
        "no", description="Mixed precision training mode"
    )

    @field_validator(
        "world_size",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "num_nodes",
        "num_processes_per_node",
    )
    @classmethod
    def validate_positive_int(cls, v):
        """Validate that value is a positive integer."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("master_port")
    @classmethod
    def validate_port(cls, v):
        """Validate port number."""
        if v < 1024 or v > 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v

    @field_validator("rank", "local_rank")
    @classmethod
    def validate_rank_non_negative(cls, v):
        """Validate rank values are non-negative."""
        if v < 0:
            raise ValueError("Rank cannot be negative")
        return v

    @model_validator(mode="after")
    def validate_distributed_consistency(self):
        """Validate consistency across distributed configuration."""
        # Validate rank vs world_size
        if self.rank >= self.world_size:
            raise ValueError(f"Rank ({self.rank}) must be less than world_size ({self.world_size})")

        # Validate local_rank vs num_processes_per_node
        if self.local_rank >= self.num_processes_per_node:
            raise ValueError(
                f"Local rank ({self.local_rank}) must be less than "
                f"processes per node ({self.num_processes_per_node})"
            )

        # Validate world_size consistency
        if self.world_size != self.num_nodes * self.num_processes_per_node:
            raise ValueError(
                f"World size ({self.world_size}) must equal "
                f"num_nodes ({self.num_nodes}) * "
                f"num_processes_per_node ({self.num_processes_per_node})"
            )

        # Validate parallelism sizes
        total_parallel = self.tensor_parallel_size * self.pipeline_parallel_size
        if total_parallel > self.world_size:
            raise ValueError(
                f"Product of parallel sizes ({total_parallel}) cannot exceed "
                f"world_size ({self.world_size})"
            )

        if self.world_size % total_parallel != 0:
            raise ValueError(
                f"World size ({self.world_size}) must be divisible by "
                f"product of parallel sizes ({total_parallel})"
            )

        # Validate mesh shape if provided
        if self.mesh_shape is not None:
            mesh_size = 1
            for dim in self.mesh_shape:
                mesh_size *= dim
            if mesh_size != self.world_size:
                raise ValueError(
                    f"Mesh shape product ({mesh_size}) must equal world_size ({self.world_size})"
                )

        # Validate mesh axis names if provided
        if self.mesh_axis_names is not None:
            if self.mesh_shape is None:
                raise ValueError("mesh_axis_names requires mesh_shape to be set")
            if len(self.mesh_axis_names) != len(self.mesh_shape):
                raise ValueError("mesh_axis_names length must match mesh_shape length")

        # Auto-configure mesh for NNX if not specified
        if self.enabled and self.mesh_shape is None:
            if self.world_size == 1:
                self.mesh_shape = (1,)
                self.mesh_axis_names = ("data",)
            elif self.tensor_parallel_size > 1 and self.pipeline_parallel_size > 1:
                # Both tensor and pipeline parallelism
                data_parallel = self.world_size // (
                    self.tensor_parallel_size * self.pipeline_parallel_size
                )
                self.mesh_shape = (
                    data_parallel,
                    self.tensor_parallel_size,
                    self.pipeline_parallel_size,
                )
                self.mesh_axis_names = ("data", "model", "pipeline")
            elif self.tensor_parallel_size > 1:
                # Only tensor parallelism
                data_parallel = self.world_size // self.tensor_parallel_size
                self.mesh_shape = (data_parallel, self.tensor_parallel_size)
                self.mesh_axis_names = ("data", "model")
            else:
                # Data parallelism only
                self.mesh_shape = (self.world_size,)
                self.mesh_axis_names = ("data",)

        return self

    def get_data_parallel_size(self) -> int:
        """Calculate the data parallel size."""
        return self.world_size // (self.tensor_parallel_size * self.pipeline_parallel_size)

    def get_mesh_config(self) -> dict | None:
        """Get mesh configuration for Flax NNX."""
        if self.mesh_shape is None or self.mesh_axis_names is None:
            return None

        return {
            "mesh_shape": self.mesh_shape,
            "axis_names": self.mesh_axis_names,
        }

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def is_local_main_process(self) -> bool:
        """Check if this is the local main process (local_rank 0)."""
        return self.local_rank == 0
