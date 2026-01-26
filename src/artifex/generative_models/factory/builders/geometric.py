"""Geometric model builder with dataclass configs.

This builder accepts dataclass-based geometric configs (PointCloudConfig,
MeshConfig, VoxelConfig, GraphConfig) and creates the appropriate model instances.

The builder follows Principle #4: Methods Take Configs, NOT Individual Parameters.
Model class is determined by config type, not a model_class string field.
"""

from typing import Any, Union

from flax import nnx

from artifex.generative_models.core.configuration.geometric_config import (
    GeometricConfig,
    GraphConfig,
    MeshConfig,
    PointCloudConfig,
    VoxelConfig,
)


# Type alias for all supported geometric configs
GeometricConfigTypes = Union[
    GeometricConfig, PointCloudConfig, MeshConfig, VoxelConfig, GraphConfig
]


class GeometricBuilder:
    """Builder for geometric models using dataclass configs.

    This builder accepts dataclass-based geometric configs and creates
    the appropriate model instances. The model class is determined by
    the config type:

    - PointCloudConfig -> PointCloudModel
    - MeshConfig -> MeshModel
    - VoxelConfig -> VoxelModel
    - GraphConfig -> GraphModel
    - GeometricConfig -> GeometricModel (base)

    All configs must have a nested network configuration specific to
    the model type (PointCloudNetworkConfig, MeshNetworkConfig, etc.).
    """

    def build(
        self,
        config: GeometricConfigTypes,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any,
    ) -> Any:
        """Build a geometric model from config.

        Args:
            config: Dataclass geometric config (PointCloudConfig, MeshConfig, etc.)
            rngs: Random number generators
            **kwargs: Additional keyword arguments passed to model constructor

        Returns:
            Instantiated geometric model

        Raises:
            TypeError: If config is not a supported geometric config type
        """
        # Validate config type
        if config is None:
            raise TypeError("config cannot be None")

        if isinstance(config, dict):
            raise TypeError(
                "config must be a dataclass config (PointCloudConfig, MeshConfig, etc.), "
                "not a dict. Use PointCloudConfig(...) or similar to create the config."
            )

        # Check for old Pydantic ModelConfiguration
        if hasattr(config, "model_class"):
            raise TypeError(
                "config must be a dataclass config (PointCloudConfig, MeshConfig, etc.), "
                "not a Pydantic ModelConfiguration. "
                "The builder no longer accepts ModelConfiguration."
            )

        # Get model class based on config type
        model_class = self._get_model_class(config)

        # Build and return the model
        return model_class(config=config, rngs=rngs, **kwargs)

    def _get_model_class(self, config: GeometricConfigTypes) -> type:
        """Get the model class based on config type.

        Args:
            config: Geometric config instance

        Returns:
            Model class to instantiate

        Raises:
            TypeError: If config type is not supported
        """
        # Import here to avoid circular imports
        from artifex.generative_models.models.geometric.base import GeometricModel
        from artifex.generative_models.models.geometric.graph import GraphModel
        from artifex.generative_models.models.geometric.mesh import MeshModel
        from artifex.generative_models.models.geometric.point_cloud import (
            PointCloudModel,
        )
        from artifex.generative_models.models.geometric.voxel import VoxelModel

        # Map config types to model classes
        # Order matters - check more specific types first (subclasses before base classes)
        if isinstance(config, PointCloudConfig):
            return PointCloudModel
        elif isinstance(config, MeshConfig):
            return MeshModel
        elif isinstance(config, VoxelConfig):
            return VoxelModel
        elif isinstance(config, GraphConfig):
            return GraphModel
        elif isinstance(config, GeometricConfig):
            return GeometricModel
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                f"Expected one of: PointCloudConfig, MeshConfig, VoxelConfig, "
                f"GraphConfig, GeometricConfig"
            )
