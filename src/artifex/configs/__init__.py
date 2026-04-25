"""Public convenience surface for the typed configuration runtime."""

# pyright: reportUnsupportedDunderAll=false

from importlib import import_module
from typing import Any


_CONFIGURATION_EXPORTS = {
    "ArchitectureExtensionConfig",
    "BaseConfig",
    "CategoricalDistribution",
    "ChoiceDistribution",
    "ConstraintExtensionConfig",
    "DataConfig",
    "DiffusionInferenceConfig",
    "DistributedBackend",
    "DistributedConfig",
    "EvaluationExtensionConfig",
    "ExperimentConfig",
    "ExperimentTemplateConfig",
    "ExperimentTemplateOverrides",
    "ExtensionConfig",
    "HyperparamSearchConfig",
    "InferenceConfig",
    "LossExtensionConfig",
    "ModalityExtensionConfig",
    "OptimizerConfig",
    "ParameterDistribution",
    "ProteinDiffusionInferenceConfig",
    "ProteinExtensionConfig",
    "ProteinExtensionsConfig",
    "ProteinMixinConfig",
    "SamplingExtensionConfig",
    "SchedulerConfig",
    "SearchType",
    "TrainingConfig",
    "UniformDistribution",
}

_CONFIG_LOADER_EXPORTS = {
    "get_config_path",
    "get_data_config",
    "get_inference_config",
    "get_protein_extensions_config",
    "get_training_config",
    "load_experiment_config",
}

_TEMPLATE_EXPORTS = {
    "ConfigTemplateManager",
    "DISTRIBUTED_TEMPLATE",
    "SIMPLE_TRAINING_TEMPLATE",
    "template_manager",
}

_VERSIONING_EXPORTS = {
    "compute_config_hash",
    "ConfigVersion",
    "ConfigVersionRegistry",
}

_MODULE_BY_EXPORT = {
    **{name: "artifex.generative_models.core.configuration" for name in _CONFIGURATION_EXPORTS},
    **{name: "artifex.configs.utils.config_loader" for name in _CONFIG_LOADER_EXPORTS},
    **{
        name: "artifex.generative_models.core.configuration.management.templates"
        for name in _TEMPLATE_EXPORTS
    },
    **{
        name: "artifex.generative_models.core.configuration.management.versioning"
        for name in _VERSIONING_EXPORTS
    },
}

__all__ = [
    # Base configs
    "BaseConfig",
    "ExperimentConfig",
    "ExperimentTemplateConfig",
    "ExperimentTemplateOverrides",
    # Data configs
    "DataConfig",
    # Training configs
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    # Extension configs
    "ExtensionConfig",
    "ConstraintExtensionConfig",
    "ProteinExtensionConfig",
    "ProteinExtensionsConfig",
    "ProteinMixinConfig",
    "ArchitectureExtensionConfig",
    "SamplingExtensionConfig",
    "LossExtensionConfig",
    "EvaluationExtensionConfig",
    "ModalityExtensionConfig",
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
    "get_config_path",
    "get_data_config",
    "get_inference_config",
    "get_protein_extensions_config",
    "get_training_config",
    "load_experiment_config",
    # Configuration versioning
    "ConfigVersion",
    "ConfigVersionRegistry",
    "compute_config_hash",
    # Configuration templates
    "ConfigTemplateManager",
    "template_manager",
    "SIMPLE_TRAINING_TEMPLATE",
    "DISTRIBUTED_TEMPLATE",
]


def __getattr__(name: str) -> Any:
    """Load exported config symbols lazily on first attribute access."""
    try:
        module_path = _MODULE_BY_EXPORT[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    """Keep introspection aligned with the documented export surface."""
    return sorted(__all__)
