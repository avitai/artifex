"""Autoregressive model builder with dataclass configs.

This builder accepts dataclass-based autoregressive configs (TransformerConfig,
PixelCNNConfig, WaveNetConfig) and creates the appropriate model instances.

The builder follows Principle #4: Methods Take Configs, NOT Individual Parameters.
Model class is determined by config type, not a model_class string field.
"""

from typing import Any, Union

from flax import nnx

from artifex.generative_models.core.configuration.autoregressive_config import (
    AutoregressiveConfig,
    PixelCNNConfig,
    TransformerConfig,
    WaveNetConfig,
)


# Type alias for all supported autoregressive configs
AutoregressiveConfigTypes = Union[
    AutoregressiveConfig, TransformerConfig, PixelCNNConfig, WaveNetConfig
]


class AutoregressiveBuilder:
    """Builder for autoregressive models using dataclass configs.

    This builder accepts dataclass-based autoregressive configs and creates
    the appropriate model instances. The model class is determined by
    the config type:

    - TransformerConfig -> TransformerAutoregressiveModel
    - PixelCNNConfig -> PixelCNN
    - WaveNetConfig -> WaveNet
    - AutoregressiveConfig -> AutoregressiveModel (base)

    TransformerConfig requires a nested TransformerNetworkConfig.
    PixelCNNConfig requires an image_shape tuple.
    WaveNetConfig requires vocab_size and sequence_length.
    """

    def build(
        self,
        config: AutoregressiveConfigTypes,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> Any:
        """Build an autoregressive model from config.

        Args:
            config: Dataclass autoregressive config (TransformerConfig, etc.)
            rngs: Random number generators
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated autoregressive model

        Raises:
            TypeError: If config is not a supported autoregressive config type
        """
        # Validate config type
        if config is None:
            raise TypeError("config cannot be None")

        if isinstance(config, dict):
            raise TypeError(
                "config must be a dataclass config (TransformerConfig, PixelCNNConfig, etc.), "
                "not a dict. Use TransformerConfig(...) or similar to create the config."
            )

        # Check for old Pydantic ModelConfiguration
        if hasattr(config, "model_class"):
            raise TypeError(
                "config must be a dataclass config (TransformerConfig, PixelCNNConfig, etc.), "
                "not a Pydantic ModelConfiguration. "
                "The builder no longer accepts ModelConfiguration."
            )

        # Get model class based on config type
        model_class = self._get_model_class(config)

        # Build and return the model
        return model_class(config=config, rngs=rngs, **kwargs)

    def _get_model_class(self, config: AutoregressiveConfigTypes) -> type:
        """Get the model class based on config type.

        Args:
            config: Autoregressive config instance

        Returns:
            Model class to instantiate

        Raises:
            TypeError: If config type is not supported
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.autoregressive.base import (
            AutoregressiveModel,
        )
        from artifex.generative_models.models.autoregressive.pixel_cnn import PixelCNN
        from artifex.generative_models.models.autoregressive.transformer import (
            TransformerAutoregressiveModel,
        )
        from artifex.generative_models.models.autoregressive.wavenet import WaveNet

        # Map config types to model classes
        # Order matters - check more specific types first (subclasses before base classes)
        if isinstance(config, TransformerConfig):
            return TransformerAutoregressiveModel
        elif isinstance(config, PixelCNNConfig):
            return PixelCNN
        elif isinstance(config, WaveNetConfig):
            return WaveNet
        elif isinstance(config, AutoregressiveConfig):
            return AutoregressiveModel
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                f"Expected one of: TransformerConfig, PixelCNNConfig, WaveNetConfig, "
                f"AutoregressiveConfig"
            )
