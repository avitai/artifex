from typing import Literal

from pydantic import Field, field_validator

from artifex.configs.schema.base import BaseConfig


class OptimizerConfig(BaseConfig):
    """Configuration for optimizers."""

    optimizer_type: Literal["adamw", "adam", "sgd"] = Field(
        "adamw", description="Type of optimizer to use"
    )
    learning_rate: float = Field(1e-4, description="Base learning rate")
    weight_decay: float = Field(1e-5, description="Weight decay for regularization")
    beta1: float = Field(0.9, description="Adam beta1 parameter")
    beta2: float = Field(0.999, description="Adam beta2 parameter")
    eps: float = Field(1e-8, description="Epsilon parameter for numerical stability")

    @field_validator("learning_rate", "weight_decay", "eps")
    @classmethod
    def validate_positive_float(cls, v):
        """Validate that value is a positive float."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("beta1", "beta2")
    @classmethod
    def validate_beta(cls, v):
        """Validate beta parameters."""
        if v < 0 or v >= 1:
            raise ValueError("Beta must be in range [0, 1)")
        return v


class SchedulerConfig(BaseConfig):
    """Configuration for learning rate schedulers."""

    scheduler_type: Literal["cosine", "linear", "step", "constant"] = Field(
        "cosine", description="Type of learning rate scheduler"
    )
    warmup_steps: int = Field(500, description="Number of warmup steps for learning rate")
    warmup_ratio: float = Field(0.1, description="Ratio of warmup steps to total steps")
    min_lr_ratio: float = Field(0.001, description="Minimum learning rate as ratio of initial lr")

    @field_validator("warmup_steps")
    @classmethod
    def validate_warmup_steps(cls, v):
        """Validate warmup steps."""
        if v < 0:
            raise ValueError("Warmup steps must be non-negative")
        return v

    @field_validator("warmup_ratio", "min_lr_ratio")
    @classmethod
    def validate_ratio(cls, v):
        """Validate ratio parameters."""
        if v < 0 or v > 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v


class TrainingConfig(BaseConfig):
    """Configuration for model training."""

    # Batch size and device settings
    batch_size: int = Field(32, description="Training batch size")
    eval_batch_size: int = Field(32, description="Evaluation batch size")
    gradient_accumulation_steps: int = Field(
        1, description="Number of steps to accumulate gradients"
    )
    use_effective_batch_size: bool = Field(
        True, description="Use effective batch size for scheduler calculations"
    )

    # Training duration
    num_epochs: int = Field(100, description="Number of training epochs")
    max_steps: int | None = Field(
        None, description="Maximum number of training steps (overrides epochs)"
    )

    # Optimizer and scheduler
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(
            name="default_optimizer",
            description="Default optimizer",
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=1e-5,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
        ),
        description="Optimizer configuration",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=lambda: SchedulerConfig(
            name="default_scheduler",
            description="Default scheduler",
            scheduler_type="cosine",
            warmup_steps=500,
            warmup_ratio=0.1,
            min_lr_ratio=0.001,
        ),
        description="Learning rate scheduler configuration",
    )

    # Checkpointing and logging
    log_freq: int = Field(20, description="Logging frequency (steps)")
    eval_freq: int = Field(200, description="Evaluation frequency (steps)")
    save_freq: int = Field(100, description="Checkpoint saving frequency (steps)")
    max_checkpoints: int = Field(5, description="Maximum number of checkpoints to keep")

    # Dataloader settings
    num_workers: int = Field(4, description="Number of dataloader workers")

    # Regularization
    grad_clip_norm: float | None = Field(1.0, description="Gradient norm clipping value")

    @field_validator(
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "log_freq",
        "eval_freq",
        "save_freq",
        "num_workers",
    )
    @classmethod
    def validate_positive_int(cls, v):
        """Validate that value is a positive integer."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, v):
        """Validate max_steps if provided."""
        if v is not None and v <= 0:
            raise ValueError("Max steps must be positive if provided")
        return v

    @field_validator("grad_clip_norm")
    @classmethod
    def validate_grad_clip_norm(cls, v):
        """Validate gradient clipping norm if provided."""
        if v is not None and v <= 0:
            raise ValueError("Gradient clipping norm must be positive if provided")
        return v
