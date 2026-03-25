"""Experiment configuration using frozen dataclasses.

This module provides a frozen dataclass-based configuration for complete
experiments and retained experiment templates.
"""

import dataclasses
from pathlib import Path
from typing import Any

from artifex.generative_models.core.configuration.base_dataclass import (
    BaseConfig,
    ConfigDocument,
)
from artifex.generative_models.core.configuration.data_config import DataConfig
from artifex.generative_models.core.configuration.evaluation_config import EvaluationConfig
from artifex.generative_models.core.configuration.model_creation import (
    is_model_creation_config,
    materialize_model_creation_config,
    model_creation_error_message,
    ModelCreationConfig,
)
from artifex.generative_models.core.configuration.training_config import TrainingConfig


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentTemplateOverrides:
    """Typed override sections for retained experiment templates."""

    model: dict[str, Any] = dataclasses.field(default_factory=dict)
    data: dict[str, Any] = dataclasses.field(default_factory=dict)
    training: dict[str, Any] = dataclasses.field(default_factory=dict)
    inference: dict[str, Any] = dataclasses.field(default_factory=dict)
    distributed: dict[str, Any] = dataclasses.field(default_factory=dict)
    hyperparam: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentTemplateConfig(ConfigDocument):
    """Typed reference template for retained experiment YAML assets.

    This config models the retained experiment-template documents under
    ``src/artifex/configs/experiments``. These assets reference other config
    files plus optional override mappings; they are not direct
    ``ExperimentConfig`` documents and do not carry runtime ``BaseConfig``
    fields such as ``name``.
    """

    experiment_name: str = ""
    seed: int = 42
    output_dir: Path = Path("./experiments")

    model_config: str = ""
    data_config: str = ""
    training_config: str = ""
    inference_config: str | None = None
    distributed_config: str | None = None
    hyperparam_config: str | None = None

    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = ""
    overrides: ExperimentTemplateOverrides = dataclasses.field(
        default_factory=ExperimentTemplateOverrides
    )

    def __post_init__(self) -> None:
        """Validate the retained experiment template contract."""
        if not self.experiment_name.strip():
            raise ValueError("experiment_name cannot be empty")

        required_refs = {
            "model_config": self.model_config,
            "data_config": self.data_config,
            "training_config": self.training_config,
        }
        for field_name, value in required_refs.items():
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty")

        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        if not self.log_level.strip():
            raise ValueError("log_level cannot be empty")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig(BaseConfig):
    """Configuration for complete experiments.

    This dataclass provides a type-safe, immutable configuration for
    complete experiments, combining model, training, data, and evaluation
    configurations.

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
    model_cfg: ModelCreationConfig | None = None
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
        super(ExperimentConfig, self).__post_init__()

        # Validate required nested configs
        if self.model_cfg is None:
            raise ValueError("model_cfg is required")
        if not is_model_creation_config(self.model_cfg):
            raise TypeError(model_creation_error_message())
        if self.training_cfg is None:
            raise ValueError("training_cfg is required")
        if self.data_cfg is None:
            raise ValueError("data_cfg is required")

        # Convert output_dir to Path if it's a string (for direct instantiation)
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create an ExperimentConfig with explicit nested family-config selection."""
        payload = dict(data)

        model_cfg = payload.get("model_cfg")
        if isinstance(model_cfg, dict):
            payload["model_cfg"] = materialize_model_creation_config(model_cfg)

        training_cfg = payload.get("training_cfg")
        if isinstance(training_cfg, dict):
            payload["training_cfg"] = TrainingConfig.from_dict(training_cfg)

        data_cfg = payload.get("data_cfg")
        if isinstance(data_cfg, dict):
            payload["data_cfg"] = DataConfig.from_dict(data_cfg)

        eval_cfg = payload.get("eval_cfg")
        if isinstance(eval_cfg, dict):
            payload["eval_cfg"] = EvaluationConfig.from_dict(eval_cfg)

        return cls(**payload)
