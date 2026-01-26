"""Configuration system for generative models."""

# Base configs; Data configs; Model configs;
# Inference configs; Training configs
from artifex.configs.schema import (
    BaseConfig,
    DataConfig,
    DatasetConfig,
    DiffusionInferenceConfig,
    ExperimentConfig,
    InferenceConfig,
    OptimizerConfig,
    ProteinDatasetConfig,
    ProteinDiffusionInferenceConfig,
    SchedulerConfig,
    TrainingConfig,
)

# Import the new schema modules
from artifex.configs.schema.distributed import (
    DistributedBackend,
    DistributedConfig,
)
from artifex.configs.schema.hyperparam import (
    CategoricalDistribution,
    ChoiceDistribution,
    HyperparamSearchConfig,
    ParameterDistribution,
    SearchType,
    UniformDistribution,
)

# Environment-specific configs; Configuration versioning; Config loading
from artifex.configs.utils import (
    apply_env_overrides,
    compute_config_hash,
    ConfigTemplate,
    ConfigTemplateManager,
    ConfigVersion,
    ConfigVersionRegistry,
    create_config_from_yaml,
    DISTRIBUTED_TEMPLATE,
    get_config_path,
    get_data_config,
    get_env_config_path,
    get_env_name,
    get_inference_config,
    get_model_config,
    get_training_config,
    load_env_config,
    load_experiment_config,
    load_yaml_config,
    PROTEIN_DIFFUSION_TEMPLATE,
    SIMPLE_TRAINING_TEMPLATE,
    template_manager,
)


__all__ = [
    # Base configs
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
    # Distributed training
    "DistributedBackend",
    "DistributedConfig",
    # Hyperparameter search
    "SearchType",
    "ParameterDistribution",
    "CategoricalDistribution",
    "UniformDistribution",
    "ChoiceDistribution",
    "HyperparamSearchConfig",
    # Utility functions
    "create_config_from_yaml",
    "get_config_path",
    "get_data_config",
    "get_inference_config",
    "get_model_config",
    "get_training_config",
    "load_experiment_config",
    "load_yaml_config",
    # Environment-specific configs
    "apply_env_overrides",
    "get_env_config_path",
    "get_env_name",
    "load_env_config",
    # Configuration versioning
    "ConfigVersion",
    "ConfigVersionRegistry",
    "compute_config_hash",
    # Configuration templates
    "ConfigTemplate",
    "ConfigTemplateManager",
    "template_manager",
    "PROTEIN_DIFFUSION_TEMPLATE",
    "SIMPLE_TRAINING_TEMPLATE",
    "DISTRIBUTED_TEMPLATE",
]
