"""Core factory implementation using dataclass configs.

This factory ONLY accepts dataclass-based configurations.
Model type is determined by config type, not a model_class string field.
"""

import dataclasses
from typing import Any, TypeVar, Union

from flax import nnx

from artifex.generative_models.core.configuration.autoregressive_config import (
    AutoregressiveConfig,
    PixelCNNConfig,
    TransformerConfig,
    WaveNetConfig,
)
from artifex.generative_models.core.configuration.diffusion_config import (
    DDPMConfig,
    DiffusionConfig,
    ScoreDiffusionConfig,
)
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
)
from artifex.generative_models.core.configuration.extension_config import ExtensionConfig
from artifex.generative_models.core.configuration.flow_config import (
    FlowConfig,
)
from artifex.generative_models.core.configuration.gan_config import (
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
    VoxelConfig,
)
from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    ConditionalVAEConfig,
    VAEConfig,
    VQVAEConfig,
)
from artifex.generative_models.extensions.base import ModelExtension
from artifex.generative_models.factory.registry import BuilderNotFoundError, ModelTypeRegistry


T = TypeVar("T")

# Type aliases for all supported dataclass configs
VAEConfigs = Union[VAEConfig, BetaVAEConfig, ConditionalVAEConfig, VQVAEConfig]
GANConfigs = Union[GANConfig, DCGANConfig, WGANConfig, LSGANConfig]
DiffusionConfigs = Union[DiffusionConfig, DDPMConfig, ScoreDiffusionConfig]
EBMConfigs = Union[EBMConfig, DeepEBMConfig]
FlowConfigs = FlowConfig
AutoregressiveConfigs = Union[
    AutoregressiveConfig, TransformerConfig, PixelCNNConfig, WaveNetConfig
]
GeometricConfigs = Union[GeometricConfig, PointCloudConfig, MeshConfig, VoxelConfig, GraphConfig]

# Union of all supported dataclass configs
DataclassConfig = Union[
    VAEConfigs,
    GANConfigs,
    DiffusionConfigs,
    EBMConfigs,
    FlowConfigs,
    AutoregressiveConfigs,
    GeometricConfigs,
]


class ModelFactory:
    """Centralized factory for all generative models.

    This factory accepts dataclass-based configurations. Model type is
    determined by config type, not by a model_class string field.

    Example:
        >>> from artifex.generative_models.factory import create_model
        >>> config = DDPMConfig(
        ...     name="my_model", backbone=..., noise_schedule=..., input_shape=(32, 32, 3)
        ... )
        >>> model = create_model(config, rngs=rngs)
    """

    def __init__(self):
        """Initialize the factory."""
        self.registry = ModelTypeRegistry()
        self._register_default_builders()

    def _register_default_builders(self):
        """Register all default builders."""
        # Import builders lazily to avoid circular imports
        from artifex.generative_models.factory.builders.diffusion import DiffusionBuilder
        from artifex.generative_models.factory.builders.flow import FlowBuilder
        from artifex.generative_models.factory.builders.gan import GANBuilder
        from artifex.generative_models.factory.builders.vae import VAEBuilder

        # Register the builders
        self.registry.register("vae", VAEBuilder())
        self.registry.register("gan", GANBuilder())
        self.registry.register("diffusion", DiffusionBuilder())
        self.registry.register("flow", FlowBuilder())

        # EBM builder
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        self.registry.register("ebm", EBMBuilder())

        # Autoregressive builder
        from artifex.generative_models.factory.builders.autoregressive import AutoregressiveBuilder

        self.registry.register("autoregressive", AutoregressiveBuilder())

        # Geometric builder
        from artifex.generative_models.factory.builders.geometric import GeometricBuilder

        self.registry.register("geometric", GeometricBuilder())

    def create(
        self,
        config: DataclassConfig,
        *,
        modality: str | None = None,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> Any:
        """Create a model from dataclass configuration.

        This is the single entry point for all model creation.

        Args:
            config: Dataclass model configuration (DDPMConfig, VAEConfig, etc.)
            modality: Optional modality for adaptation
            rngs: Random number generators
            **kwargs: Additional arguments

        Returns:
            Created model instance

        Raises:
            TypeError: If config is not a supported dataclass config
            ValueError: If builder not found for model type
        """
        # Validate configuration type
        self._validate_config(config)

        # Extract model type from config type
        model_type = self._extract_model_type(config)

        # Get builder for model type
        try:
            builder = self.registry.get_builder(model_type)
        except BuilderNotFoundError as e:
            raise ValueError(str(e)) from e

        # Build the model
        model = builder.build(config, rngs=rngs, **kwargs)

        # Apply modality if specified
        if modality:
            from artifex.generative_models.modalities.registry import get_modality

            modality_handler = get_modality(modality, rngs=rngs)
            if modality_handler is not None:
                adapter = modality_handler.get_adapter(model_type)
                if adapter is not None:
                    model = adapter.adapt(model, config)

        return model

    def _extract_model_type(self, config: DataclassConfig) -> str:
        """Extract model type from config type.

        Args:
            config: Dataclass config instance

        Returns:
            Model type identifier (vae, gan, diffusion, ebm, flow, autoregressive, geometric)
        """
        # VAE configs
        if isinstance(config, (VAEConfig, BetaVAEConfig, ConditionalVAEConfig, VQVAEConfig)):
            return "vae"

        # GAN configs
        if isinstance(config, (GANConfig, DCGANConfig, WGANConfig, LSGANConfig)):
            return "gan"

        # Diffusion configs
        if isinstance(config, (DiffusionConfig, DDPMConfig, ScoreDiffusionConfig)):
            return "diffusion"

        # EBM configs
        if isinstance(config, (EBMConfig, DeepEBMConfig)):
            return "ebm"

        # Flow configs
        if isinstance(config, FlowConfig):
            return "flow"

        # Autoregressive configs
        autoregressive_types = (
            AutoregressiveConfig,
            TransformerConfig,
            PixelCNNConfig,
            WaveNetConfig,
        )
        if isinstance(config, autoregressive_types):
            return "autoregressive"

        # Geometric configs
        geometric_types = (
            GeometricConfig,
            PointCloudConfig,
            MeshConfig,
            VoxelConfig,
            GraphConfig,
        )
        if isinstance(config, geometric_types):
            return "geometric"

        raise TypeError(
            f"Unknown config type: {type(config).__name__}. "
            f"Use a supported dataclass config (DDPMConfig, VAEConfig, PointCloudConfig, etc.)"
        )

    def _validate_config(self, config: DataclassConfig) -> None:
        """Validate that config is a supported dataclass config.

        Args:
            config: Config to validate

        Raises:
            TypeError: If config is not a supported type
        """
        # Reject None
        if config is None:
            raise TypeError("config cannot be None")

        # Reject dict configs
        if isinstance(config, dict):
            raise TypeError(
                "Expected dataclass config, got dict. "
                "Use a dataclass config (DDPMConfig, VAEConfig, PointCloudConfig, etc.) instead."
            )

        # Accept dataclass configs only
        if dataclasses.is_dataclass(config):
            return

        # Reject everything else
        raise TypeError(
            f"Expected dataclass config, got {type(config).__name__}. "
            "Use a dataclass config (DDPMConfig, VAEConfig, PointCloudConfig, etc.) instead."
        )


# Global factory instance
_factory = ModelFactory()


def create_model(
    config: DataclassConfig, *, modality: str | None = None, rngs: nnx.Rngs, **kwargs
) -> Any:
    """Create a model from configuration.

    This is the main public API for model creation.

    Args:
        config: Dataclass model configuration (DDPMConfig, VAEConfig, etc.)
        modality: Modality for adaptation
        rngs: Random number generators
        **kwargs: Additional arguments

    Returns:
        Created model instance

    Raises:
        TypeError: If config is not a supported dataclass config
    """
    return _factory.create(config, modality=modality, rngs=rngs, **kwargs)


def create_model_with_extensions(
    config: DataclassConfig,
    *,
    extensions_config: dict[str, ExtensionConfig] | None = None,
    modality: str | None = None,
    rngs: nnx.Rngs,
    **kwargs,
) -> tuple[Any, dict[str, ModelExtension]]:
    """Create a model and its extensions from configuration.

    This function creates both the model and any specified extensions,
    returning them as a tuple for use with trainers that support extensions.

    Args:
        config: Dataclass model configuration (DDPMConfig, VAEConfig, etc.)
        extensions_config: Dictionary mapping extension names to their configs.
            Extension names should match registered extensions in the registry.
        modality: Modality for adaptation
        rngs: Random number generators
        **kwargs: Additional arguments

    Returns:
        Tuple of (model, extensions_dict) where:
        - model: The created model instance
        - extensions_dict: Dictionary mapping extension names to extension instances

    Raises:
        TypeError: If config is not a supported dataclass config
        ValueError: If an extension name is not found in the registry

    Example:
        >>> config = VAEConfig(name="my_vae", latent_dim=32, input_shape=(32, 32, 3))
        >>> ext_configs = {
        ...     "image_augmentation": AugmentationExtensionConfig(name="aug"),
        ... }
        >>> model, extensions = create_model_with_extensions(
        ...     config, extensions_config=ext_configs, rngs=rngs
        ... )
        >>> trainer = VAETrainer(config=train_config, model=model, extensions=extensions)
    """
    # Create the model
    model = _factory.create(config, modality=modality, rngs=rngs, **kwargs)

    # Create extensions
    extensions: dict[str, ModelExtension] = {}
    if extensions_config:
        from artifex.generative_models.extensions.registry import get_extensions_registry

        registry = get_extensions_registry()

        for ext_name, ext_config in extensions_config.items():
            extension = registry.create_extension(ext_name, ext_config, rngs=rngs)
            extensions[ext_name] = extension

    return model, extensions
