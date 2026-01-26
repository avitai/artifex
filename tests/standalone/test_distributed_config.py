from enum import Enum

from pydantic import BaseModel, Field, model_validator


class BaseConfig(BaseModel):
    """Base configuration class."""

    name: str = Field("test_config", description="Configuration name")

    class Config:
        extra = "forbid"


class DistributedBackend(str, Enum):
    """Distributed training backend."""

    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


class DistributedConfig(BaseConfig):
    """Configuration for distributed training."""

    backend: DistributedBackend = Field(DistributedBackend.NCCL, description="Distributed backend")
    world_size: int = Field(1, description="Total number of processes")
    rank: int = Field(0, description="Process rank")
    local_rank: int = Field(0, description="Local process rank")
    use_distributed: bool = Field(False, description="Whether to use distributed training")
    distributed_port: int = Field(12355, description="Port for distributed communication")
    tensor_parallel_size: int = Field(1, description="Tensor parallelism size")
    pipeline_parallel_size: int = Field(1, description="Pipeline parallelism size")

    @model_validator(mode="after")
    def validate_ranks(self):
        """Validate rank configurations."""
        world_size = self.world_size
        rank = self.rank

        if world_size <= 0:
            raise ValueError("World size must be positive")

        if rank < 0 or rank >= world_size:
            raise ValueError(f"Rank must be between 0 and {world_size - 1}")

        # Validate tensor and pipeline parallel sizes
        tensor_parallel = self.tensor_parallel_size
        pipeline_parallel = self.pipeline_parallel_size

        if tensor_parallel > world_size:
            raise ValueError("Tensor parallel size cannot exceed world size")

        if pipeline_parallel > world_size:
            raise ValueError("Pipeline parallel size cannot exceed world size")

        if tensor_parallel * pipeline_parallel > world_size:
            raise ValueError(
                "Product of tensor and pipeline parallel sizes cannot exceed world size"
            )

        return self


def test_distributed_enum():
    """Test distributed backend enum values."""
    assert DistributedBackend.NCCL.value == "nccl"
    assert DistributedBackend.GLOO.value == "gloo"
    assert DistributedBackend.MPI.value == "mpi"


def test_distributed_config_instantiation():
    """Test that DistributedConfig can be instantiated with valid values."""
    config = DistributedConfig(
        backend=DistributedBackend.NCCL,
        world_size=8,
        rank=0,
        local_rank=0,
        use_distributed=True,
    )

    assert config.backend == DistributedBackend.NCCL
    assert config.world_size == 8
    assert config.rank == 0
    assert config.local_rank == 0
    assert config.use_distributed is True
