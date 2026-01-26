"""Tests for the geometric model module initialization."""

from artifex.generative_models.models.geometric import (
    GeometricModel,
    MeshModel,
    PointCloudModel,
    VoxelModel,
)


class TestGeometricModelsInit:
    """Tests for the geometric models module initialization."""

    def test_all_models_exported(self):
        """Test that all model classes are properly exported."""
        # Check that all expected classes are imported correctly
        assert GeometricModel is not None
        assert PointCloudModel is not None
        assert MeshModel is not None
        assert VoxelModel is not None

    def test_inheritance(self):
        """Test that all models properly inherit from the base class."""
        assert issubclass(PointCloudModel, GeometricModel)
        assert issubclass(MeshModel, GeometricModel)
        assert issubclass(VoxelModel, GeometricModel)

    def test_distinct_classes(self):
        """Test that all model classes are distinct."""
        assert PointCloudModel != MeshModel
        assert PointCloudModel != VoxelModel
        assert MeshModel != VoxelModel
