"""Configuration schema definitions for generative models."""

from artifex.configs.schema.base import (
    BaseConfig,
    ExperimentConfig,
)
from artifex.configs.schema.data import (
    DataConfig,
    DatasetConfig,
    ProteinDatasetConfig,
)
from artifex.configs.schema.extensions import (
    ConstraintExtensionConfig,
    ExtensionConfig,
    ExtensionsConfig,
    ProteinBackboneConstraintConfig,
    ProteinExtensionConfig,
    ProteinMixinConfig,
)
from artifex.configs.schema.inference import (
    DiffusionInferenceConfig,
    InferenceConfig,
    ProteinDiffusionInferenceConfig,
)

# Import diffusion configs directly to avoid circular imports
# Model configs have been moved to unified configuration system
# Import from artifex.generative_models.core.configuration instead
from artifex.configs.schema.training import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


__all__ = [
    # Base
    "BaseConfig",
    "ExperimentConfig",
    # Data configs
    "DataConfig",
    "DatasetConfig",
    "ProteinDatasetConfig",
    # Training configs
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    # Inference configs
    "InferenceConfig",
    "DiffusionInferenceConfig",
    "ProteinDiffusionInferenceConfig",
    # Extension configs
    "ExtensionConfig",
    "ConstraintExtensionConfig",
    "ExtensionsConfig",
    "ProteinExtensionConfig",
    "ProteinBackboneConstraintConfig",
    "ProteinMixinConfig",
]
