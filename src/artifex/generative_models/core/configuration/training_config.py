"""TrainingConfig frozen dataclass configuration.

Replaces Pydantic TrainingConfig with frozen dataclass.

Design:
- Frozen dataclass inheriting from BaseConfig
- All validation in __post_init__ using DRY utilities
- Nested OptimizerConfig and optional SchedulerConfig
- Path support for checkpoint_dir
"""

import dataclasses
from pathlib import Path

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.optimizer_config import OptimizerConfig
from artifex.generative_models.core.configuration.scheduler_config import SchedulerConfig
from artifex.generative_models.core.configuration.validation import (
    validate_positive_float,
    validate_positive_int,
)


@dataclasses.dataclass(frozen=True)
class TrainingConfig(BaseConfig):
    """Configuration for training.

    This config contains all training-related settings including optimizer,
    scheduler, batch size, epochs, checkpointing, and logging.

    Attributes:
        optimizer: Optimizer configuration (required, OptimizerConfig)
        scheduler: Learning rate scheduler configuration (optional, SchedulerConfig)
        batch_size: Batch size for training (must be positive)
        num_epochs: Number of training epochs (must be positive)
        gradient_clip_norm: Gradient clipping by norm (optional, must be positive if set)
        checkpoint_dir: Directory for saving checkpoints
        save_frequency: Save checkpoint every N steps (must be positive)
        max_checkpoints: Maximum number of checkpoints to keep (must be positive)
        log_frequency: Log metrics every N steps (must be positive)
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name (optional, required if use_wandb=True)
    """

    # Required nested config (has dummy default for dataclass field ordering)
    optimizer: OptimizerConfig = None  # Will validate in __post_init__

    # Optional nested config
    scheduler: SchedulerConfig | None = None

    # Basic training parameters
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip_norm: float | None = 1.0

    # Checkpointing
    checkpoint_dir: Path = dataclasses.field(default_factory=lambda: Path("./checkpoints"))
    save_frequency: int = 1000
    max_checkpoints: int = 5

    # Logging
    log_frequency: int = 100
    use_wandb: bool = False
    wandb_project: str | None = None

    def __post_init__(self):
        """Validate all fields.

        Validation uses DRY utilities from validation.py where possible.
        Follows fail-fast principle - raise on first error.
        """
        # Call parent validation first
        super().__post_init__()

        # Validate required nested config
        if self.optimizer is None:
            raise ValueError("optimizer is required and cannot be None")

        # Validate batch_size (must be positive)
        validate_positive_int(self.batch_size, "batch_size")

        # Validate num_epochs (must be positive)
        validate_positive_int(self.num_epochs, "num_epochs")

        # Validate gradient_clip_norm (must be positive if set)
        if self.gradient_clip_norm is not None:
            validate_positive_float(self.gradient_clip_norm, "gradient_clip_norm")

        # Validate save_frequency (must be positive)
        validate_positive_int(self.save_frequency, "save_frequency")

        # Validate max_checkpoints (must be positive)
        validate_positive_int(self.max_checkpoints, "max_checkpoints")

        # Validate log_frequency (must be positive)
        validate_positive_int(self.log_frequency, "log_frequency")

        # Validate wandb_project if use_wandb is True (optional validation)
        # Note: We don't enforce this strictly, just document the expectation
