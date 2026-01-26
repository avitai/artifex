"""OptimizerConfig frozen dataclass configuration.

Replaces Pydantic OptimizerConfiguration with frozen dataclass.

Design:
- Frozen dataclass inheriting from BaseConfig
- All validation in __post_init__ using DRY utilities
- Support for all common optimizer types and their parameters
"""

import dataclasses

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.validation import (
    validate_learning_rate,
    validate_positive_float,
    validate_probability,
)


@dataclasses.dataclass(frozen=True)
class OptimizerConfig(BaseConfig):
    """Configuration for optimizers.

    Supports common optimizer types (Adam, AdamW, SGD, RMSProp, etc.) with their
    specific parameters.

    Attributes:
        optimizer_type: Type of optimizer (adam, adamw, sgd, rmsprop, adagrad, lamb, radam, nadam)
        learning_rate: Learning rate (must be positive)
        weight_decay: Weight decay for L2 regularization (>= 0)
        beta1: Beta1 parameter for Adam-like optimizers [0, 1]
        beta2: Beta2 parameter for Adam-like optimizers [0, 1]
        eps: Epsilon for numerical stability (must be positive)
        momentum: Momentum for SGD [0, 1]
        nesterov: Whether to use Nesterov momentum (SGD)
        initial_accumulator_value: Initial value for AdaGrad/RMSProp (>= 0)
        gradient_clip_norm: Gradient clipping by norm (optional, must be positive if set)
        gradient_clip_value: Gradient clipping by value (optional, must be positive if set)
    """

    # Required fields (have dummy defaults for dataclass field ordering)
    optimizer_type: str = ""  # Will validate in __post_init__
    learning_rate: float = 0.0  # Will validate in __post_init__

    # Common parameters - Optional with defaults
    weight_decay: float = 0.0

    # Adam/AdamW specific
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # SGD specific
    momentum: float = 0.0
    nesterov: bool = False

    # AdaGrad/RMSProp specific
    initial_accumulator_value: float = 0.1

    # Gradient clipping
    gradient_clip_norm: float | None = None
    gradient_clip_value: float | None = None

    def __post_init__(self):
        """Validate all fields.

        Validation uses DRY utilities from validation.py.
        Follows fail-fast principle - raise on first error.
        """
        # Call parent validation first
        super().__post_init__()

        # Validate required fields (they have dummy defaults for dataclass compatibility)
        if not self.optimizer_type:
            raise ValueError("optimizer_type is required and cannot be empty")

        # Validate optimizer_type
        self._validate_optimizer_type(self.optimizer_type)

        # Validate learning_rate
        validate_learning_rate(self.learning_rate)

        # Validate weight_decay (must be non-negative)
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")

        # Validate beta1 and beta2 (probabilities in [0, 1])
        validate_probability(self.beta1, "beta1")
        validate_probability(self.beta2, "beta2")

        # Validate eps (must be positive)
        validate_positive_float(self.eps, "eps")

        # Validate momentum (probability in [0, 1])
        validate_probability(self.momentum, "momentum")

        # Validate initial_accumulator_value (must be non-negative)
        if self.initial_accumulator_value < 0:
            raise ValueError(
                f"initial_accumulator_value must be non-negative, "
                f"got {self.initial_accumulator_value}"
            )

        # Validate gradient clipping values (must be positive if set)
        if self.gradient_clip_norm is not None:
            validate_positive_float(self.gradient_clip_norm, "gradient_clip_norm")

        if self.gradient_clip_value is not None:
            validate_positive_float(self.gradient_clip_value, "gradient_clip_value")

    def _validate_optimizer_type(self, optimizer_type: str) -> None:
        """Validate optimizer type.

        Args:
            optimizer_type: Optimizer type to validate

        Raises:
            ValueError: If optimizer type is invalid
        """
        valid_optimizers = {
            "adam",
            "adamw",
            "sgd",
            "rmsprop",
            "adagrad",
            "lamb",
            "radam",
            "nadam",
        }

        if optimizer_type not in valid_optimizers:
            raise ValueError(
                f"Unknown optimizer_type: {optimizer_type}. "
                f"Valid options: {sorted(valid_optimizers)}"
            )
