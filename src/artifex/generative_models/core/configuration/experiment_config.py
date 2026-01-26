"""Experiment configuration using frozen dataclasses.

This module provides a frozen dataclass-based configuration for complete
experiments, replacing the Pydantic-based ExperimentConfiguration.
"""

import dataclasses
from pathlib import Path

from artifex.generative_models.core.configuration.base_dataclass import BaseConfig
from artifex.generative_models.core.configuration.data_config import DataConfig
from artifex.generative_models.core.configuration.evaluation_config import EvaluationConfig
from artifex.generative_models.core.configuration.model_config import ModelConfig
from artifex.generative_models.core.configuration.training_config import TrainingConfig


@dataclasses.dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    """Configuration for complete experiments.

    This dataclass provides a type-safe, immutable configuration for
    complete experiments, combining model, training, data, and evaluation
    configurations. It replaces the Pydantic ExperimentConfiguration.

    Attributes:
        model_cfg: Model configuration
        training_cfg: Training configuration
        data_cfg: Data configuration
        eval_cfg: Evaluation configuration (optional)
        seed: Random seed for reproducibility
        deterministic: Whether to use deterministic algorithms
        output_dir: Directory for experiment outputs
        track_carbon: Whether to track carbon emissions
        track_memory: Whether to track memory usage
    """

    # Required nested configs with dummy defaults (validated in __post_init__)
    model_cfg: ModelConfig | None = None
    training_cfg: TrainingConfig | None = None
    data_cfg: DataConfig | None = None

    # Optional nested config
    eval_cfg: EvaluationConfig | None = None

    # Experiment settings
    seed: int = 42
    deterministic: bool = True
    output_dir: Path = Path("./experiments")

    # Tracking
    track_carbon: bool = False
    track_memory: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If any validation fails
        """
        super().__post_init__()

        # Validate required nested configs
        if self.model_cfg is None:
            raise ValueError("model_cfg is required")
        if self.training_cfg is None:
            raise ValueError("training_cfg is required")
        if self.data_cfg is None:
            raise ValueError("data_cfg is required")

        # Convert output_dir to Path if it's a string (for direct instantiation)
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
