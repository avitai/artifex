"""Energy-based model builder with dataclass configs.

This builder accepts dataclass-based EBM configs (EBMConfig, DeepEBMConfig)
and creates the appropriate EBM model instances.

The builder follows Principle #4: Methods Take Configs, NOT Individual Parameters.
Model class is determined by config type, not a model_class string field.
"""

from typing import Any, Union

from flax import nnx

from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
)


# Type alias for all supported EBM configs
EBMConfigTypes = Union[EBMConfig, DeepEBMConfig]


class EBMBuilder:
    """Builder for energy-based models using dataclass configs.

    This builder accepts dataclass-based EBM configs and creates
    the appropriate model instances. The model class is determined by
    the config type:

    - EBMConfig -> EBM
    - DeepEBMConfig -> DeepEBM

    All configs must have nested dataclass configs for energy_network,
    mcmc, and sample_buffer.
    """

    def build(
        self,
        config: EBMConfigTypes,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> Any:
        """Build an EBM model from config.

        Args:
            config: Dataclass EBM config (EBMConfig or DeepEBMConfig)
            rngs: Random number generators
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated EBM model

        Raises:
            TypeError: If config is not a supported EBM config type
        """
        # Validate config type
        if config is None:
            raise TypeError("config cannot be None")

        if isinstance(config, dict):
            raise TypeError(
                "config must be a dataclass config (EBMConfig, DeepEBMConfig), "
                "not a dict. Use EBMConfig(...) or DeepEBMConfig(...) to create the config."
            )

        # Check for old Pydantic ModelConfiguration
        if hasattr(config, "model_class"):
            raise TypeError(
                "config must be a dataclass config (EBMConfig, DeepEBMConfig), "
                "not a Pydantic ModelConfiguration. "
                "The builder no longer accepts ModelConfiguration."
            )

        # Get model class based on config type
        model_class = self._get_model_class(config)

        # Build and return the model
        return model_class(config=config, rngs=rngs, **kwargs)

    def _get_model_class(self, config: EBMConfigTypes) -> type:
        """Get the model class based on config type.

        Args:
            config: EBM config instance

        Returns:
            Model class to instantiate

        Raises:
            TypeError: If config type is not supported
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.energy.ebm import DeepEBM, EBM

        # Map config types to model classes
        # Order matters - check more specific types first (DeepEBMConfig before EBMConfig)
        if isinstance(config, DeepEBMConfig):
            return DeepEBM
        elif isinstance(config, EBMConfig):
            return EBM
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                f"Expected one of: DeepEBMConfig, EBMConfig"
            )
