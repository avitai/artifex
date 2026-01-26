"""Simplified standalone test for distributed training configuration."""

from enum import Enum

import pytest
from pydantic import BaseModel, Field, model_validator


class ProcessGroup(str, Enum):
    """Process group types for distributed training."""

    DATA_PARALLEL = "data_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"


class DistributedSetup(BaseModel):
    """Simple distributed training setup configuration."""

    enabled: bool = Field(False, description="Whether distributed training is enabled")
    world_size: int = Field(1, description="Total number of processes")
    current_rank: int = Field(0, description="Current process rank")
    process_group: ProcessGroup = Field(
        ProcessGroup.DATA_PARALLEL, description="Process group type"
    )

    @model_validator(mode="after")
    def validate_ranks(self):
        """Validate rank configuration."""
        if not self.enabled:
            return self

        if self.world_size <= 0:
            raise ValueError("World size must be positive")

        if self.current_rank < 0 or self.current_rank >= self.world_size:
            raise ValueError(f"Rank must be between 0 and {self.world_size - 1}")

        return self


def test_process_group_enum():
    """Test process group enum values."""
    assert ProcessGroup.DATA_PARALLEL.value == "data_parallel"
    assert ProcessGroup.TENSOR_PARALLEL.value == "tensor_parallel"
    assert ProcessGroup.PIPELINE_PARALLEL.value == "pipeline_parallel"


def test_distributed_setup_defaults():
    """Test default values for distributed setup."""
    setup = DistributedSetup()
    assert setup.enabled is False
    assert setup.world_size == 1
    assert setup.current_rank == 0
    assert setup.process_group == ProcessGroup.DATA_PARALLEL


def test_distributed_setup_values():
    """Test custom values for distributed setup."""
    setup = DistributedSetup(
        enabled=True,
        world_size=4,
        current_rank=2,
        process_group=ProcessGroup.TENSOR_PARALLEL,
    )

    assert setup.enabled is True
    assert setup.world_size == 4
    assert setup.current_rank == 2
    assert setup.process_group == ProcessGroup.TENSOR_PARALLEL


def test_distributed_setup_validation():
    """Test validation rules for distributed setup."""
    # Invalid world size
    with pytest.raises(ValueError, match="World size must be positive"):
        DistributedSetup(enabled=True, world_size=0)

    with pytest.raises(ValueError, match="World size must be positive"):
        DistributedSetup(enabled=True, world_size=-1)

    # Invalid rank
    with pytest.raises(ValueError, match="Rank must be between"):
        DistributedSetup(enabled=True, world_size=4, current_rank=4)

    with pytest.raises(ValueError, match="Rank must be between"):
        DistributedSetup(enabled=True, world_size=4, current_rank=-1)

    # If distributed is disabled, validation should pass regardless of values
    try:
        DistributedSetup(enabled=False, world_size=-1, current_rank=100)
    except ValueError:
        pytest.fail("Validation should not run when distributed is disabled")


def test_process_group_compatibility():
    """Test compatibility between process groups and world size."""
    # Create a valid configuration with 4 processes
    setup = DistributedSetup(
        enabled=True,
        world_size=4,
        current_rank=0,
        process_group=ProcessGroup.DATA_PARALLEL,
    )
    assert setup.process_group == ProcessGroup.DATA_PARALLEL

    # Switch to tensor parallel
    setup.process_group = ProcessGroup.TENSOR_PARALLEL
    assert setup.process_group == ProcessGroup.TENSOR_PARALLEL

    # Switch to pipeline parallel
    setup.process_group = ProcessGroup.PIPELINE_PARALLEL
