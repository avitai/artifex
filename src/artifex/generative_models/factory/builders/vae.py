"""VAE model builder with dataclass configs.

This builder accepts dataclass-based VAE configs (VAEConfig, BetaVAEConfig, etc.)
and creates the appropriate VAE model instances.

The builder follows Principle #4: Methods Take Configs, NOT Individual Parameters.
Model class is determined by config type, not a model_class string field.
"""

from typing import Any, Union

from flax import nnx

from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    ConditionalVAEConfig,
    VAEConfig,
    VQVAEConfig,
)


# Type alias for all supported VAE configs
VAEConfigTypes = Union[VAEConfig, BetaVAEConfig, ConditionalVAEConfig, VQVAEConfig]


class VAEBuilder:
    """Builder for VAE models using dataclass configs.

    This builder accepts dataclass-based VAE configs and creates
    the appropriate model instances. The model class is determined by
    the config type:

    - VAEConfig -> VAE
    - BetaVAEConfig -> BetaVAE
    - ConditionalVAEConfig -> ConditionalVAE
    - VQVAEConfig -> VQVAE

    All configs must have nested EncoderConfig and DecoderConfig.
    """

    def build(
        self,
        config: VAEConfigTypes,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> Any:
        """Build a VAE model from config.

        Args:
            config: Dataclass VAE config (VAEConfig, BetaVAEConfig, etc.)
            rngs: Random number generators
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated VAE model

        Raises:
            TypeError: If config is not a supported VAE config type
        """
        # Validate config type
        if config is None:
            raise TypeError("config cannot be None")

        if isinstance(config, dict):
            raise TypeError(
                "config must be a dataclass config (VAEConfig, BetaVAEConfig, etc.), "
                "not a dict. Use VAEConfig(...) or similar to create the config."
            )

        # Check for old Pydantic ModelConfiguration
        if hasattr(config, "model_class"):
            raise TypeError(
                "config must be a dataclass config (VAEConfig, BetaVAEConfig, etc.), "
                "not a Pydantic ModelConfiguration. "
                "The builder no longer accepts ModelConfiguration."
            )

        # Get model class based on config type
        model_class = self._get_model_class(config)

        # Build and return the model
        return model_class(config=config, rngs=rngs, **kwargs)

    def _get_model_class(self, config: VAEConfigTypes) -> type:
        """Get the model class based on config type.

        Args:
            config: VAE config instance

        Returns:
            Model class to instantiate

        Raises:
            TypeError: If config type is not supported
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.vae.base import VAE
        from artifex.generative_models.models.vae.beta_vae import BetaVAE
        from artifex.generative_models.models.vae.conditional import ConditionalVAE
        from artifex.generative_models.models.vae.vq_vae import VQVAE

        # Map config types to model classes
        # Order matters - check more specific types first (subclasses before base classes)
        if isinstance(config, VQVAEConfig):
            return VQVAE
        elif isinstance(config, ConditionalVAEConfig):
            return ConditionalVAE
        elif isinstance(config, BetaVAEConfig):
            return BetaVAE
        elif isinstance(config, VAEConfig):
            return VAE
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                f"Expected one of: VQVAEConfig, ConditionalVAEConfig, BetaVAEConfig, VAEConfig"
            )
