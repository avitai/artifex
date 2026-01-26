"""Diffusion model builder with dataclass configs.

This builder accepts dataclass-based diffusion configs (DDPMConfig, DiffusionConfig, etc.)
and creates the appropriate diffusion model instances.

The builder follows Principle #4: Methods Take Configs, NOT Individual Parameters.
Model class is determined by config type, not a model_class string field.
"""

from typing import Any, Union

from flax import nnx

from artifex.generative_models.core.configuration.diffusion_config import (
    DDPMConfig,
    DiffusionConfig,
    ScoreDiffusionConfig,
)


# Type alias for all supported diffusion configs
DiffusionConfigTypes = Union[DiffusionConfig, DDPMConfig, ScoreDiffusionConfig]


class DiffusionBuilder:
    """Builder for diffusion models using dataclass configs.

    This builder accepts dataclass-based diffusion configs and creates
    the appropriate model instances. The model class is determined by
    the config type:

    - DiffusionConfig -> DiffusionModel
    - DDPMConfig -> DDPMModel
    - ScoreDiffusionConfig -> ScoreDiffusionModel

    All configs must have a polymorphic BackboneConfig (with backbone_type field).
    """

    def build(
        self,
        config: DiffusionConfigTypes,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> Any:
        """Build a diffusion model from config.

        Args:
            config: Dataclass diffusion config (DDPMConfig, DiffusionConfig, etc.)
            rngs: Random number generators
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated diffusion model

        Raises:
            TypeError: If config is not a supported diffusion config type
        """
        # Validate config type
        if config is None:
            raise TypeError("config cannot be None")

        if isinstance(config, dict):
            raise TypeError(
                "config must be a dataclass config (DDPMConfig, DiffusionConfig, etc.), "
                "not a dict. Use DDPMConfig(...) or similar to create the config."
            )

        # Check for old Pydantic ModelConfiguration
        if hasattr(config, "model_class"):
            raise TypeError(
                "config must be a dataclass config (DDPMConfig, DiffusionConfig, etc.), "
                "not a Pydantic ModelConfiguration. "
                "The builder no longer accepts ModelConfiguration."
            )

        # Get model class based on config type
        model_class = self._get_model_class(config)

        # Build and return the model
        return model_class(config=config, rngs=rngs, **kwargs)

    def _get_model_class(self, config: DiffusionConfigTypes) -> type:
        """Get the model class based on config type.

        Args:
            config: Diffusion config instance

        Returns:
            Model class to instantiate

        Raises:
            TypeError: If config type is not supported
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.diffusion.base import DiffusionModel
        from artifex.generative_models.models.diffusion.ddpm import DDPMModel
        from artifex.generative_models.models.diffusion.score import ScoreDiffusionModel

        # Map config types to model classes
        # Order matters - check more specific types first (DDPMConfig before DiffusionConfig)
        if isinstance(config, DDPMConfig):
            return DDPMModel
        elif isinstance(config, ScoreDiffusionConfig):
            return ScoreDiffusionModel
        elif isinstance(config, DiffusionConfig):
            return DiffusionModel
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                f"Expected one of: DDPMConfig, ScoreDiffusionConfig, DiffusionConfig"
            )
