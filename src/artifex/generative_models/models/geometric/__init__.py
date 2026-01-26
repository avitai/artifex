"""Geometric generative models for 3D structures.

This module provides models for generating 3D geometric structures
like point clouds, meshes, voxels, and graphs.
"""

from artifex.generative_models.extensions.protein import (
    BondAngleExtension,
    BondLengthExtension,
    create_protein_extensions,
    ProteinMixinExtension,
)
from artifex.generative_models.models.geometric.base import (
    GeometricModel,
)

# Factory functions have been moved to the centralized factory
# from artifex.generative_models.factory import create_model
from artifex.generative_models.models.geometric.graph import GraphModel
from artifex.generative_models.models.geometric.mesh import MeshModel
from artifex.generative_models.models.geometric.point_cloud import (
    PointCloudModel,
)

# Protein-specific geometric models
from artifex.generative_models.models.geometric.protein_graph import ProteinGraphModel
from artifex.generative_models.models.geometric.protein_point_cloud import ProteinPointCloudModel
from artifex.generative_models.models.geometric.voxel import VoxelModel


__all__ = [
    # Base models
    "GeometricModel",
    "PointCloudModel",
    "MeshModel",
    "VoxelModel",
    "GraphModel",
    # Protein-specific geometric models
    "ProteinGraphModel",
    "ProteinPointCloudModel",
    # Extension compatibility
    "BondAngleExtension",
    "BondLengthExtension",
    "ProteinMixinExtension",
    "create_protein_extensions",
]

# Deprecation warning
import warnings


warnings.warn(
    "Protein-specific constraints have been moved to artifex.generative_models.extensions.protein.",
    DeprecationWarning,
    stacklevel=2,
)
