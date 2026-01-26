"""Flow model builder with dataclass configs.

This builder accepts dataclass-based flow configs (FlowConfig, RealNVPConfig, etc.)
and creates the appropriate normalizing flow model instances.

The builder follows Principle #4: Methods Take Configs, NOT Individual Parameters.
Model class is determined by config type, not a model_class string field.
"""

from typing import Any, Union

from flax import nnx

from artifex.generative_models.core.configuration.flow_config import (
    FlowConfig,
    GlowConfig,
    IAFConfig,
    MAFConfig,
    NeuralSplineConfig,
    RealNVPConfig,
)


# Type alias for all supported flow configs
FlowConfigTypes = Union[
    FlowConfig, RealNVPConfig, GlowConfig, MAFConfig, IAFConfig, NeuralSplineConfig
]


class FlowBuilder:
    """Builder for normalizing flow models using dataclass configs.

    This builder accepts dataclass-based flow configs and creates
    the appropriate model instances. The model class is determined by
    the config type:

    - RealNVPConfig -> RealNVP
    - GlowConfig -> Glow
    - MAFConfig -> MAF
    - IAFConfig -> IAF
    - NeuralSplineConfig -> NeuralSplineFlow
    - FlowConfig -> NormalizingFlow (base)

    All configs must have a nested CouplingNetworkConfig.
    """

    def build(
        self,
        config: FlowConfigTypes,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> Any:
        """Build a flow model from config.

        Args:
            config: Dataclass flow config (RealNVPConfig, GlowConfig, etc.)
            rngs: Random number generators
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated flow model

        Raises:
            TypeError: If config is not a supported flow config type
        """
        # Validate config type
        if config is None:
            raise TypeError("config cannot be None")

        if isinstance(config, dict):
            raise TypeError(
                "config must be a dataclass config (RealNVPConfig, GlowConfig, etc.), "
                "not a dict. Use RealNVPConfig(...) or similar to create the config."
            )

        # Check for old Pydantic ModelConfiguration
        if hasattr(config, "model_class"):
            raise TypeError(
                "config must be a dataclass config (RealNVPConfig, GlowConfig, etc.), "
                "not a Pydantic ModelConfiguration. "
                "The builder no longer accepts ModelConfiguration."
            )

        # Get model class based on config type
        model_class = self._get_model_class(config)

        # Build and return the model
        return model_class(config=config, rngs=rngs, **kwargs)

    def _get_model_class(self, config: FlowConfigTypes) -> type:
        """Get the model class based on config type.

        Args:
            config: Flow config instance

        Returns:
            Model class to instantiate

        Raises:
            TypeError: If config type is not supported
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.flow.base import NormalizingFlow
        from artifex.generative_models.models.flow.glow import Glow
        from artifex.generative_models.models.flow.iaf import IAF
        from artifex.generative_models.models.flow.maf import MAF
        from artifex.generative_models.models.flow.neural_spline import NeuralSplineFlow
        from artifex.generative_models.models.flow.real_nvp import RealNVP

        # Map config types to model classes
        # Order matters - check more specific types first (subclasses before base classes)
        if isinstance(config, RealNVPConfig):
            return RealNVP
        elif isinstance(config, GlowConfig):
            return Glow
        elif isinstance(config, MAFConfig):
            return MAF
        elif isinstance(config, IAFConfig):
            return IAF
        elif isinstance(config, NeuralSplineConfig):
            return NeuralSplineFlow
        elif isinstance(config, FlowConfig):
            return NormalizingFlow
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                f"Expected one of: RealNVPConfig, GlowConfig, MAFConfig, IAFConfig, "
                f"NeuralSplineConfig, FlowConfig"
            )
