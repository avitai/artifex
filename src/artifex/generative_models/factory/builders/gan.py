"""GAN model builder with dataclass configs.

This builder accepts dataclass-based GAN configs (GANConfig, WGANConfig, etc.)
and creates the appropriate GAN model instances.

The builder follows Principle #4: Methods Take Configs, NOT Individual Parameters.
Model class is determined by config type, not a model_class string field.
"""

from typing import Any, Union

from flax import nnx

from artifex.generative_models.core.configuration.gan_config import (
    ConditionalGANConfig,
    CycleGANConfig,
    DCGANConfig,
    GANConfig,
    LSGANConfig,
    WGANConfig,
)


# Type alias for all supported GAN configs
GANConfigTypes = Union[
    GANConfig, WGANConfig, LSGANConfig, ConditionalGANConfig, CycleGANConfig, DCGANConfig
]


class GANBuilder:
    """Builder for GAN models using dataclass configs.

    This builder accepts dataclass-based GAN configs and creates
    the appropriate model instances. The model class is determined by
    the config type:

    - WGANConfig -> WGAN
    - LSGANConfig -> LSGAN
    - ConditionalGANConfig -> ConditionalGAN
    - CycleGANConfig -> CycleGAN
    - DCGANConfig -> DCGAN
    - GANConfig -> (base GAN, currently not implemented)

    All configs must have nested GeneratorConfig and DiscriminatorConfig.
    """

    def build(
        self,
        config: GANConfigTypes,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> Any:
        """Build a GAN model from config.

        Args:
            config: Dataclass GAN config (WGANConfig, LSGANConfig, etc.)
            rngs: Random number generators
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated GAN model

        Raises:
            TypeError: If config is not a supported GAN config type
        """
        # Validate config type
        if config is None:
            raise TypeError("config cannot be None")

        if isinstance(config, dict):
            raise TypeError(
                "config must be a dataclass config (WGANConfig, LSGANConfig, etc.), "
                "not a dict. Use WGANConfig(...) or similar to create the config."
            )

        # Check for old Pydantic ModelConfiguration
        if hasattr(config, "model_class"):
            raise TypeError(
                "config must be a dataclass config (WGANConfig, LSGANConfig, etc.), "
                "not a Pydantic ModelConfiguration. "
                "The builder no longer accepts ModelConfiguration."
            )

        # Get model class based on config type
        model_class = self._get_model_class(config)

        # Build and return the model
        return model_class(config=config, rngs=rngs, **kwargs)

    def _get_model_class(self, config: GANConfigTypes) -> type:
        """Get the model class based on config type.

        Args:
            config: GAN config instance

        Returns:
            Model class to instantiate

        Raises:
            TypeError: If config type is not supported
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.gan.conditional import ConditionalGAN
        from artifex.generative_models.models.gan.cyclegan import CycleGAN
        from artifex.generative_models.models.gan.dcgan import DCGAN
        from artifex.generative_models.models.gan.lsgan import LSGAN
        from artifex.generative_models.models.gan.wgan import WGAN

        # Map config types to model classes
        # Order matters - check more specific types first (subclasses before base classes)
        if isinstance(config, WGANConfig):
            return WGAN
        elif isinstance(config, LSGANConfig):
            return LSGAN
        elif isinstance(config, ConditionalGANConfig):
            return ConditionalGAN
        elif isinstance(config, CycleGANConfig):
            return CycleGAN
        elif isinstance(config, DCGANConfig):
            return DCGAN
        elif isinstance(config, GANConfig):
            # Base GANConfig - not directly instantiable
            raise TypeError(
                "Cannot build model from base GANConfig. "
                "Use a specific GAN config type: WGANConfig, LSGANConfig, "
                "ConditionalGANConfig, CycleGANConfig, or DCGANConfig."
            )
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                f"Expected one of: WGANConfig, LSGANConfig, ConditionalGANConfig, "
                f"CycleGANConfig, DCGANConfig"
            )
