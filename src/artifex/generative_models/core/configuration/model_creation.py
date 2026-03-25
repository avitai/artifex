"""Shared helpers for the supported model-creation configuration surface."""

from __future__ import annotations

from typing import Any, TypeAlias

from dacite import DaciteError

from artifex.generative_models.core.configuration.autoregressive_config import (
    AutoregressiveConfig,
    PixelCNNConfig,
    TransformerConfig,
    WaveNetConfig,
)
from artifex.generative_models.core.configuration.diffusion_config import (
    DDIMConfig,
    DDPMConfig,
    DiffusionConfig,
    LatentDiffusionConfig,
    ScoreDiffusionConfig,
    StableDiffusionConfig,
)
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
)
from artifex.generative_models.core.configuration.flow_config import (
    ConditionalFlowConfig,
    FlowConfig,
    GlowConfig,
    IAFConfig,
    MAFConfig,
    NeuralSplineConfig,
    RealNVPConfig,
)
from artifex.generative_models.core.configuration.gan_config import (
    ConditionalGANConfig,
    CycleGANConfig,
    DCGANConfig,
    GANConfig,
    LSGANConfig,
    WGANConfig,
)
from artifex.generative_models.core.configuration.geometric_config import (
    GeometricConfig,
    GraphConfig,
    MeshConfig,
    PointCloudConfig,
    ProteinGraphConfig,
    ProteinPointCloudConfig,
    VoxelConfig,
)
from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
    ConditionalVAEConfig,
    VAEConfig,
    VQVAEConfig,
)


ModelCreationConfig: TypeAlias = (
    VAEConfig
    | BetaVAEConfig
    | BetaVAEWithCapacityConfig
    | ConditionalVAEConfig
    | VQVAEConfig
    | GANConfig
    | WGANConfig
    | LSGANConfig
    | ConditionalGANConfig
    | CycleGANConfig
    | DCGANConfig
    | DiffusionConfig
    | DDPMConfig
    | DDIMConfig
    | ScoreDiffusionConfig
    | LatentDiffusionConfig
    | StableDiffusionConfig
    | EBMConfig
    | DeepEBMConfig
    | FlowConfig
    | ConditionalFlowConfig
    | RealNVPConfig
    | GlowConfig
    | MAFConfig
    | IAFConfig
    | NeuralSplineConfig
    | AutoregressiveConfig
    | TransformerConfig
    | PixelCNNConfig
    | WaveNetConfig
    | GeometricConfig
    | PointCloudConfig
    | ProteinPointCloudConfig
    | MeshConfig
    | VoxelConfig
    | GraphConfig
    | ProteinGraphConfig
)

MODEL_CREATION_CONFIG_TYPES = (
    VAEConfig,
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
    ConditionalVAEConfig,
    VQVAEConfig,
    GANConfig,
    WGANConfig,
    LSGANConfig,
    ConditionalGANConfig,
    CycleGANConfig,
    DCGANConfig,
    DiffusionConfig,
    DDPMConfig,
    DDIMConfig,
    ScoreDiffusionConfig,
    LatentDiffusionConfig,
    StableDiffusionConfig,
    EBMConfig,
    DeepEBMConfig,
    FlowConfig,
    ConditionalFlowConfig,
    RealNVPConfig,
    GlowConfig,
    MAFConfig,
    IAFConfig,
    NeuralSplineConfig,
    AutoregressiveConfig,
    TransformerConfig,
    PixelCNNConfig,
    WaveNetConfig,
    PointCloudConfig,
    ProteinPointCloudConfig,
    MeshConfig,
    VoxelConfig,
    GraphConfig,
    ProteinGraphConfig,
    GeometricConfig,
)

_MODEL_CREATION_CONFIG_CANDIDATES = (
    VAEConfig,
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
    ConditionalVAEConfig,
    VQVAEConfig,
    GANConfig,
    WGANConfig,
    LSGANConfig,
    ConditionalGANConfig,
    CycleGANConfig,
    DCGANConfig,
    DiffusionConfig,
    DDPMConfig,
    DDIMConfig,
    ScoreDiffusionConfig,
    LatentDiffusionConfig,
    StableDiffusionConfig,
    EBMConfig,
    DeepEBMConfig,
    FlowConfig,
    ConditionalFlowConfig,
    RealNVPConfig,
    GlowConfig,
    MAFConfig,
    IAFConfig,
    NeuralSplineConfig,
    AutoregressiveConfig,
    TransformerConfig,
    PixelCNNConfig,
    WaveNetConfig,
    PointCloudConfig,
    ProteinPointCloudConfig,
    MeshConfig,
    VoxelConfig,
    GraphConfig,
    ProteinGraphConfig,
    GeometricConfig,
)

_MODEL_CREATION_ERROR = (
    "model_cfg must be a supported family-specific typed model config "
    "(for example VAEConfig, DDPMConfig, FlowConfig, EBMConfig, or PointCloudConfig), "
    "not a generic model_class payload."
)


def is_model_creation_config(config: Any) -> bool:
    """Return whether a value is on the supported family-specific config surface."""
    return isinstance(config, MODEL_CREATION_CONFIG_TYPES)


def materialize_model_creation_config(config_dict: dict[str, Any]) -> ModelCreationConfig:
    """Materialize the narrowest supported model config from a raw mapping."""
    for config_class in _MODEL_CREATION_CONFIG_CANDIDATES:
        try:
            config = config_class.from_dict(config_dict)
        except (AttributeError, DaciteError, TypeError, ValueError):
            continue
        if is_model_creation_config(config):
            return config

    raise ValueError(_MODEL_CREATION_ERROR)


def model_creation_error_message() -> str:
    """Expose the shared validation message for public error boundaries."""
    return _MODEL_CREATION_ERROR
