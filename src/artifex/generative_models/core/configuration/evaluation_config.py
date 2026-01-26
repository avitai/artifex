"""Evaluation configuration using frozen dataclasses.

This module provides a frozen dataclass-based configuration for evaluation
and metrics, replacing the Pydantic-based EvaluationConfig.
"""

import dataclasses
from pathlib import Path
from typing import Any

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.validation import validate_positive_int


@dataclasses.dataclass(frozen=True)
class EvaluationConfig(BaseConfig):
    """Configuration for evaluation and metrics.

    This dataclass provides a type-safe, immutable configuration for
    evaluation settings in generative models. It replaces the Pydantic
    EvaluationConfig.

    Attributes:
        metrics: Tuple of metrics to compute (e.g., "fid", "inception_score")
        metric_params: Per-metric parameters as dict of dicts
        eval_batch_size: Batch size for evaluation
        num_eval_samples: Number of samples to evaluate (None = all)
        save_predictions: Whether to save model predictions
        save_metrics: Whether to save computed metrics
        output_dir: Directory for evaluation outputs
    """

    # Required field with dummy default (validated in __post_init__)
    metrics: tuple[str, ...] = ()

    # Metric configuration
    metric_params: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    # Evaluation settings
    eval_batch_size: int = 32
    num_eval_samples: int | None = None

    # Output settings
    save_predictions: bool = False
    save_metrics: bool = True
    output_dir: Path = Path("./evaluation")

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate metrics (required field with dummy default)
        if not self.metrics:
            raise ValueError("metrics cannot be empty")

        # Validate eval_batch_size
        validate_positive_int(self.eval_batch_size, "eval_batch_size")

        # Validate num_eval_samples if provided
        if self.num_eval_samples is not None:
            validate_positive_int(self.num_eval_samples, "num_eval_samples")

        # Convert output_dir to Path if it's a string (for direct instantiation)
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
