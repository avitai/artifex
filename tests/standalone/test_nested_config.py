"""Tests for nested configuration objects."""

from enum import Enum

import pytest
from pydantic import BaseModel, Field, model_validator


class OptimizationType(str, Enum):
    """Type of optimization algorithm."""

    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMS_PROP = "rmsprop"


class DataType(str, Enum):
    """Data type for model parameters."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class OptimizerConfig(BaseModel):
    """Configuration for optimization algorithm."""

    type: OptimizationType = Field(OptimizationType.ADAM, description="Optimization algorithm")
    learning_rate: float = Field(0.001, description="Learning rate")
    weight_decay: float = Field(0.0, description="Weight decay coefficient")
    beta1: float | None = Field(None, description="Beta1 for Adam")
    beta2: float | None = Field(None, description="Beta2 for Adam")
    momentum: float | None = Field(None, description="Momentum for SGD")

    @model_validator(mode="after")
    def validate_optimizer_params(self):
        """Validate that the optimizer parameters are correct for the type."""
        if self.type in [OptimizationType.ADAM, OptimizationType.ADAMW]:
            if self.beta1 is None:
                self.beta1 = 0.9
            if self.beta2 is None:
                self.beta2 = 0.999
            if self.momentum is not None:
                raise ValueError(f"Momentum should not be set for {self.type} optimizer")
        elif self.type == OptimizationType.SGD:
            if self.momentum is None:
                self.momentum = 0.0
            if self.beta1 is not None or self.beta2 is not None:
                raise ValueError("Beta parameters should not be set for SGD optimizer")
        return self


class DataConfig(BaseModel):
    """Configuration for data processing."""

    batch_size: int = Field(32, description="Batch size for training")
    shuffle: bool = Field(True, description="Whether to shuffle data")
    num_workers: int = Field(4, description="Number of worker processes")
    pin_memory: bool = Field(True, description="Whether to pin memory")

    @model_validator(mode="after")
    def validate_batch_size(self):
        """Validate batch size is positive and a power of 2."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Check if power of 2
        if (self.batch_size & (self.batch_size - 1)) != 0:
            raise ValueError("Batch size should be a power of 2")

        return self


class TrainingConfig(BaseModel):
    """Configuration for model training."""

    epochs: int = Field(10, description="Number of training epochs")
    # Default parameters will be added by the model validator
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(
            type=OptimizationType.ADAM,
            learning_rate=0.001,
            weight_decay=0.0,
            beta1=None,
            beta2=None,
            momentum=None,
        ),
        description="Optimizer configuration",
    )
    # Default parameters will be added by the model validator
    data: DataConfig = Field(
        default_factory=lambda: DataConfig(
            batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        ),
        description="Data configuration",
    )
    precision: DataType = Field(DataType.FLOAT32, description="Precision for training")
    clip_grad_norm: float | None = Field(None, description="Gradient clipping norm")
    use_mixed_precision: bool = Field(False, description="Whether to use mixed precision training")

    @model_validator(mode="after")
    def validate_mixed_precision(self):
        """Validate mixed precision settings."""
        if self.use_mixed_precision and self.precision == DataType.FLOAT32:
            raise ValueError("When using mixed precision, precision should not be float32")
        return self

    class Config:
        """Pydantic config."""

        extra = "forbid"


def test_optimizer_config_defaults():
    """Test default values for optimizer configuration."""
    # Create a config but skip validation to check raw default values
    config = OptimizerConfig.model_construct()
    assert config.type == OptimizationType.ADAM
    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.0
    assert config.beta1 is None
    assert config.beta2 is None
    assert config.momentum is None

    # After validation, Adam should have beta values set
    validated_config = OptimizerConfig()
    assert validated_config.beta1 == 0.9
    assert validated_config.beta2 == 0.999
    assert validated_config.momentum is None


def test_optimizer_type_specific_params():
    """Test type-specific parameter validation."""
    # Adam optimizer
    adam_config = OptimizerConfig(type=OptimizationType.ADAM, beta1=0.8)
    assert adam_config.beta1 == 0.8
    assert adam_config.beta2 == 0.999

    # SGD optimizer
    sgd_config = OptimizerConfig(type=OptimizationType.SGD, momentum=0.9)
    assert sgd_config.momentum == 0.9
    assert sgd_config.beta1 is None

    # Test invalid parameter combinations
    with pytest.raises(ValueError):
        OptimizerConfig(type=OptimizationType.ADAM, momentum=0.9)

    with pytest.raises(ValueError):
        OptimizerConfig(type=OptimizationType.SGD, beta1=0.9)


def test_data_config_validation():
    """Test data configuration validation."""
    # Valid batch sizes (powers of 2)
    DataConfig(batch_size=32)
    DataConfig(batch_size=64)
    DataConfig(batch_size=128)

    # Invalid batch sizes
    with pytest.raises(ValueError, match="Batch size must be positive"):
        DataConfig(batch_size=0)

    with pytest.raises(ValueError, match="Batch size must be positive"):
        DataConfig(batch_size=-16)

    with pytest.raises(ValueError, match="Batch size should be a power of 2"):
        DataConfig(batch_size=24)

    with pytest.raises(ValueError, match="Batch size should be a power of 2"):
        DataConfig(batch_size=100)


def test_training_config_nesting():
    """Test nested configuration objects."""
    # Default initialization
    config = TrainingConfig()
    assert config.epochs == 10
    assert isinstance(config.optimizer, OptimizerConfig)
    assert isinstance(config.data, DataConfig)

    # Custom initialization
    config = TrainingConfig(
        epochs=20,
        optimizer=OptimizerConfig(
            type=OptimizationType.SGD,
            learning_rate=0.01,
        ),
        data=DataConfig(
            batch_size=64,
            shuffle=False,
        ),
    )

    assert config.epochs == 20
    assert config.optimizer.type == OptimizationType.SGD
    assert config.optimizer.learning_rate == 0.01
    assert config.data.batch_size == 64
    assert config.data.shuffle is False
    assert config.data.num_workers == 4  # Default value


def test_mixed_precision_validation():
    """Test mixed precision validation."""
    # Valid configurations
    TrainingConfig(use_mixed_precision=False, precision=DataType.FLOAT32)
    TrainingConfig(use_mixed_precision=True, precision=DataType.FLOAT16)
    TrainingConfig(use_mixed_precision=True, precision=DataType.BFLOAT16)

    # Invalid configuration
    with pytest.raises(ValueError, match="When using mixed precision"):
        TrainingConfig(use_mixed_precision=True, precision=DataType.FLOAT32)
