"""Unified configuration system for artifex generative models.

This module provides a centralized configuration management system that
replaces the fragmented configuration approaches across the codebase.
"""

# Frozen dataclass configs (NEW - migrating from Pydantic)
# Autoregressive configurations
from .autoregressive_config import (
    AutoregressiveConfig,
    PixelCNNConfig,
    TransformerConfig,
    TransformerNetworkConfig,
    WaveNetConfig,
)

# Backbone configurations (polymorphic backbone system for diffusion models)
from .backbone_config import (
    BackboneConfig,
    BackboneTypeLiteral,
    create_backbone,
    DiTBackboneConfig,
    get_backbone_config_type,
    UNet2DConditionBackboneConfig,
    UNetBackboneConfig,
    UViTBackboneConfig,
)
from .base_dataclass import BaseConfig
from .base_network import BaseNetworkConfig
from .data_config import DataConfig

# Diffusion configurations
from .diffusion_config import (
    DDIMConfig,
    DDPMConfig,
    DiffusionConfig,
    DiTConfig,
    LatentDiffusionConfig,
    NoiseScheduleConfig,
    ScoreDiffusionConfig,
    StableDiffusionConfig,
)

# Energy configurations
from .energy_config import (
    DeepEBMConfig,
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from .evaluation_config import EvaluationConfig
from .experiment_config import ExperimentConfig

# Extension configurations (frozen dataclass)
from .extension_config import (
    ArchitectureExtensionConfig,
    AudioSpectralConfig,
    AugmentationExtensionConfig,
    CallbackExtensionConfig,
    ChemicalConstraintConfig,
    ClassifierFreeGuidanceConfig,
    ConstrainedSamplingConfig,
    ConstraintExtensionConfig,
    EvaluationExtensionConfig,
    ExtensionConfig,
    ExtensionPipelineConfig,
    ImageAugmentationConfig,
    LossExtensionConfig,
    ModalityExtensionConfig,
    ProteinDihedralConfig,
    ProteinExtensionConfig,
    ProteinMixinConfig,
    SamplingExtensionConfig,
    TextEmbeddingConfig,
)

# Flow configurations
from .flow_config import (
    CouplingNetworkConfig,
    FlowConfig,
    GlowConfig,
    IAFConfig,
    MAFConfig,
    NeuralSplineConfig,
    RealNVPConfig,
)

# GAN configurations (frozen dataclass)
from .gan_config import (
    ConditionalGANConfig,
    CycleGANConfig,
    DCGANConfig,
    GANConfig,
    LSGANConfig,
    WGANConfig,
)

# Geometric configurations
from .geometric_config import (
    GeometricConfig,
    GraphConfig,
    GraphNetworkConfig,
    MeshConfig,
    MeshNetworkConfig,
    PointCloudConfig,
    PointCloudNetworkConfig,
    ProteinConstraintConfig,
    ProteinGraphConfig,
    ProteinPointCloudConfig,
    VoxelConfig,
    VoxelNetworkConfig,
)
from .modality_config import ModalityConfig
from .model_config import ModelConfig
from .network_configs import (
    ConditionalDiscriminatorConfig,
    ConditionalGeneratorConfig,
    ConditionalParams,
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    CycleGANGeneratorConfig,
    DecoderConfig,
    DiscriminatorConfig,
    EncoderConfig,
    GeneratorConfig,
    PatchGANDiscriminatorConfig,
)
from .optimizer_config import OptimizerConfig
from .scheduler_config import SchedulerConfig
from .training_config import TrainingConfig

# VAE configurations
from .vae_config import (
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
    ConditionalVAEConfig,
    VAEConfig,
    VQVAEConfig,
)
from .validation import (
    validate_activation,
    validate_dropout_rate,
    validate_learning_rate,
    validate_positive_float,
    validate_positive_int,
    validate_positive_tuple,
    validate_probability,
)


__all__ = [
    # Extension configurations (frozen dataclass)
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
    "ProteinMixinConfig",
    "SamplingExtensionConfig",
    "TextEmbeddingConfig",
    # Frozen dataclass configs (NEW - migrating from Pydantic)
    "AutoregressiveConfig",
    # Backbone configurations (polymorphic backbone system)
    "BackboneConfig",
    "BackboneTypeLiteral",
    "create_backbone",
    "DiTBackboneConfig",
    "get_backbone_config_type",
    "UNet2DConditionBackboneConfig",
    "UNetBackboneConfig",
    "UViTBackboneConfig",
    "BaseConfig",
    "BaseNetworkConfig",
    "BetaVAEConfig",
    "BetaVAEWithCapacityConfig",
    "ConditionalDiscriminatorConfig",
    "ConditionalGANConfig",
    "ConditionalGeneratorConfig",
    "ConditionalParams",
    "ConditionalVAEConfig",
    "ConvDiscriminatorConfig",
    "ConvGeneratorConfig",
    "CouplingNetworkConfig",
    "CycleGANConfig",
    "CycleGANGeneratorConfig",
    "DataConfig",
    "DCGANConfig",
    "DDIMConfig",
    "DDPMConfig",
    "DecoderConfig",
    "DeepEBMConfig",
    "DiffusionConfig",
    "DiscriminatorConfig",
    "DiTConfig",
    "EBMConfig",
    "EncoderConfig",
    "EnergyNetworkConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FlowConfig",
    "GANConfig",
    "GeometricConfig",
    "GlowConfig",
    "GeneratorConfig",
    "GraphConfig",
    "GraphNetworkConfig",
    "IAFConfig",
    "LatentDiffusionConfig",
    "LSGANConfig",
    "MAFConfig",
    "MCMCConfig",
    "MeshConfig",
    "MeshNetworkConfig",
    "ModalityConfig",
    "ModelConfig",
    "NeuralSplineConfig",
    "NoiseScheduleConfig",
    "OptimizerConfig",
    "PatchGANDiscriminatorConfig",
    "PixelCNNConfig",
    "PointCloudConfig",
    "PointCloudNetworkConfig",
    "ProteinConstraintConfig",
    "ProteinGraphConfig",
    "ProteinPointCloudConfig",
    "RealNVPConfig",
    "SampleBufferConfig",
    "SchedulerConfig",
    "ScoreDiffusionConfig",
    "StableDiffusionConfig",
    "TrainingConfig",
    "TransformerConfig",
    "TransformerNetworkConfig",
    "VAEConfig",
    "VoxelConfig",
    "VoxelNetworkConfig",
    "VQVAEConfig",
    "WaveNetConfig",
    "WGANConfig",
    # Validation functions (DRY utilities)
    "validate_positive_int",
    "validate_positive_float",
    "validate_positive_tuple",
    "validate_dropout_rate",
    "validate_probability",
    "validate_learning_rate",
    "validate_activation",
]
