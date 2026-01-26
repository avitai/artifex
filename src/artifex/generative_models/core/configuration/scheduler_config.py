"""SchedulerConfig frozen dataclass configuration.

Replaces Pydantic SchedulerConfiguration with frozen dataclass.

Design:
- Frozen dataclass inheriting from BaseConfig
- All validation in __post_init__ using DRY utilities
- Support for all common scheduler types and their parameters
- Milestones as tuple (immutable)
"""

import dataclasses

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.validation import (
    validate_positive_int,
    validate_probability,
)


@dataclasses.dataclass(frozen=True)
class SchedulerConfig(BaseConfig):
    """Configuration for learning rate schedulers.

    Supports common scheduler types (cosine, linear, exponential, step, multistep, etc.)
    with their specific parameters.

    Attributes:
        scheduler_type: Type of scheduler (constant, linear, cosine, exponential, polynomial,
                       step, multistep, cyclic, one_cycle, none)
        warmup_steps: Number of warmup steps (>= 0)
        min_lr_ratio: Minimum learning rate ratio [0, 1]
        cycle_length: Cycle length for cosine scheduler (optional, must be positive if set)
        decay_rate: Decay rate for exponential scheduler (0, 1]
        decay_steps: Decay steps for exponential scheduler (must be positive)
        total_steps: Total steps for linear scheduler (optional, must be positive if set)
        step_size: Step size for step scheduler (must be positive)
        gamma: Gamma for step scheduler (0, 1]
        milestones: Milestones for multistep scheduler (tuple of positive ints)
    """

    # Required fields (have dummy defaults for dataclass field ordering)
    scheduler_type: str = ""  # Will validate in __post_init__

    # Common parameters - Optional with defaults
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    # Cosine scheduler
    cycle_length: int | None = None

    # Exponential scheduler
    decay_rate: float = 0.95
    decay_steps: int = 1000

    # Linear scheduler
    total_steps: int | None = None

    # Step scheduler
    step_size: int = 1000
    gamma: float = 0.1

    # MultiStep scheduler (use tuple for immutability)
    milestones: tuple[int, ...] = ()

    def __post_init__(self):
        """Validate all fields.

        Validation uses DRY utilities from validation.py where possible.
        Follows fail-fast principle - raise on first error.
        """
        # Call parent validation first
        super().__post_init__()

        # Validate required fields (they have dummy defaults for dataclass compatibility)
        if not self.scheduler_type:
            raise ValueError("scheduler_type is required and cannot be empty")

        # Validate scheduler_type
        self._validate_scheduler_type(self.scheduler_type)

        # Validate warmup_steps (must be non-negative)
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")

        # Validate min_lr_ratio (probability in [0, 1])
        validate_probability(self.min_lr_ratio, "min_lr_ratio")

        # Validate cycle_length (must be positive if set)
        if self.cycle_length is not None:
            validate_positive_int(self.cycle_length, "cycle_length")

        # Validate decay_rate (must be in (0, 1])
        if not (0 < self.decay_rate <= 1):
            raise ValueError(f"decay_rate must be in (0, 1], got {self.decay_rate}")

        # Validate decay_steps (must be positive)
        validate_positive_int(self.decay_steps, "decay_steps")

        # Validate total_steps (must be positive if set)
        if self.total_steps is not None:
            validate_positive_int(self.total_steps, "total_steps")

        # Validate step_size (must be positive)
        validate_positive_int(self.step_size, "step_size")

        # Validate gamma (must be in (0, 1])
        if not (0 < self.gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")

        # Validate milestones (all must be positive if provided)
        if self.milestones:
            for i, milestone in enumerate(self.milestones):
                if milestone <= 0:
                    raise ValueError(f"milestones[{i}] must be positive, got {milestone}")

    def _validate_scheduler_type(self, scheduler_type: str) -> None:
        """Validate scheduler type.

        Args:
            scheduler_type: Scheduler type to validate

        Raises:
            ValueError: If scheduler type is invalid
        """
        valid_schedulers = {
            "constant",
            "linear",
            "cosine",
            "exponential",
            "polynomial",
            "step",
            "multistep",
            "cyclic",
            "one_cycle",
            "none",
        }

        if scheduler_type not in valid_schedulers:
            raise ValueError(
                f"Unknown scheduler_type: {scheduler_type}. "
                f"Valid options: {sorted(valid_schedulers)}"
            )
