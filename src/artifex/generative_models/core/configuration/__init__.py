"""Lazy public surface for typed Artifex configuration models.

This package exposes the supported frozen dataclass config types without
eagerly importing the full configuration tree. Internal foundations such as
``ConfigDocument`` and validation helpers stay in their concrete modules.
"""

from importlib import import_module
from typing import Any


_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "artifex.generative_models.core.configuration.autoregressive_config": (
        "AutoregressiveConfig",
        "PixelCNNConfig",
        "TransformerConfig",
        "TransformerNetworkConfig",
        "WaveNetConfig",
    ),
    "artifex.generative_models.core.configuration.backbone_config": (
        "BackboneConfig",
        "BackboneTypeLiteral",
        "DiTBackboneConfig",
        "UNet1DBackboneConfig",
        "UNet2DConditionBackboneConfig",
        "UNetBackboneConfig",
    ),
    "artifex.generative_models.core.configuration.base_dataclass": ("BaseConfig",),
    "artifex.generative_models.core.configuration.base_network": ("BaseNetworkConfig",),
    "artifex.generative_models.core.configuration.data_config": ("DataConfig",),
    "artifex.generative_models.core.configuration.diffusion_config": (
        "DDIMConfig",
        "DDPMConfig",
        "DiffusionConfig",
        "DiTConfig",
        "LatentDiffusionConfig",
        "NoiseScheduleConfig",
        "ScoreDiffusionConfig",
        "StableDiffusionConfig",
    ),
    "artifex.generative_models.core.configuration.distributed_config": (
        "DistributedBackend",
        "DistributedConfig",
    ),
    "artifex.generative_models.core.configuration.energy_config": (
        "DeepEBMConfig",
        "EBMConfig",
        "EnergyNetworkConfig",
        "MCMCConfig",
        "SampleBufferConfig",
    ),
    "artifex.generative_models.core.configuration.evaluation_config": ("EvaluationConfig",),
    "artifex.generative_models.core.configuration.experiment_config": (
        "ExperimentConfig",
        "ExperimentTemplateConfig",
        "ExperimentTemplateOverrides",
    ),
    "artifex.generative_models.core.configuration.extension_config": (
        "ArchitectureExtensionConfig",
        "AudioSpectralConfig",
        "AugmentationExtensionConfig",
        "CallbackExtensionConfig",
        "ChemicalConstraintConfig",
        "ClassifierFreeGuidanceConfig",
        "ConstrainedSamplingConfig",
        "ConstraintExtensionConfig",
        "EvaluationExtensionConfig",
        "ExtensionConfig",
        "ExtensionPipelineConfig",
        "ImageAugmentationConfig",
        "LossExtensionConfig",
        "ModalityExtensionConfig",
        "ProteinDihedralConfig",
        "ProteinExtensionConfig",
        "ProteinExtensionsConfig",
        "ProteinMixinConfig",
        "SamplingExtensionConfig",
        "TextEmbeddingConfig",
    ),
    "artifex.generative_models.core.configuration.flow_config": (
        "CouplingNetworkConfig",
        "FlowConfig",
        "ConditionalFlowConfig",
        "GlowConfig",
        "IAFConfig",
        "MAFConfig",
        "NeuralSplineConfig",
        "RealNVPConfig",
    ),
    "artifex.generative_models.core.configuration.gan_config": (
        "ConditionalGANConfig",
        "CycleGANConfig",
        "DCGANConfig",
        "GANConfig",
        "LSGANConfig",
        "WGANConfig",
    ),
    "artifex.generative_models.core.configuration.geometric_config": (
        "GeometricConfig",
        "GraphConfig",
        "GraphNetworkConfig",
        "MeshConfig",
        "MeshNetworkConfig",
        "PointCloudConfig",
        "PointCloudNetworkConfig",
        "ProteinGraphConfig",
        "ProteinPointCloudConfig",
        "VoxelConfig",
        "VoxelNetworkConfig",
    ),
    "artifex.generative_models.core.configuration.hyperparam_config": (
        "CategoricalDistribution",
        "ChoiceDistribution",
        "HyperparamSearchConfig",
        "LogUniformDistribution",
        "ParameterDistribution",
        "SearchType",
        "UniformDistribution",
    ),
    "artifex.generative_models.core.configuration.inference_config": (
        "DiffusionInferenceConfig",
        "InferenceConfig",
        "ProteinDiffusionInferenceConfig",
    ),
    "artifex.generative_models.core.configuration.modality_config": (
        "BaseModalityConfig",
        "ModalityConfig",
    ),
    "artifex.generative_models.core.configuration.network_configs": (
        "ConditionalDiscriminatorConfig",
        "ConditionalGeneratorConfig",
        "ConditionalParams",
        "ConvDiscriminatorConfig",
        "ConvGeneratorConfig",
        "CycleGANGeneratorConfig",
        "DecoderConfig",
        "DiscriminatorConfig",
        "EncoderConfig",
        "GeneratorConfig",
        "PatchGANDiscriminatorConfig",
    ),
    "artifex.generative_models.core.configuration.optimizer_config": ("OptimizerConfig",),
    "artifex.generative_models.core.configuration.scheduler_config": ("SchedulerConfig",),
    "artifex.generative_models.core.configuration.training_config": ("TrainingConfig",),
    "artifex.generative_models.core.configuration.vae_config": (
        "BetaVAEConfig",
        "BetaVAEWithCapacityConfig",
        "ConditionalVAEConfig",
        "VAEConfig",
        "VQVAEConfig",
    ),
}

_MODULE_BY_EXPORT = {
    export_name: module_path
    for module_path, export_names in _EXPORT_GROUPS.items()
    for export_name in export_names
}

__all__ = list(_MODULE_BY_EXPORT)  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> Any:
    """Load exported config symbols lazily on first attribute access."""
    try:
        module_path = _MODULE_BY_EXPORT[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    """Keep introspection aligned with the supported configuration surface."""
    return sorted(__all__)
