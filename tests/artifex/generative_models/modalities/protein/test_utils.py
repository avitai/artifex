"""Tests for protein modality utilities.

This module contains tests for utility functions in the protein modality.
"""

from artifex.generative_models.modalities.protein.utils import (
    get_protein_adapter,
)


class MockModel:
    """Mock model for testing."""

    __name__ = "MockModel"


class MockGeometricModel:
    """Mock geometric model for testing."""

    __name__ = "GeometricModel"


class MockPointCloudModel:
    """Mock point cloud model for testing."""

    __name__ = "PointCloudModel"


class MockGraphModel:
    """Mock graph model for testing."""

    __name__ = "GraphModel"


class MockDiffusionModel:
    """Mock diffusion model for testing."""

    __name__ = "DiffusionModel"


def test_get_protein_adapter_geometric():
    """Test getting a protein adapter for a geometric model."""
    adapter = get_protein_adapter(MockGeometricModel)
    assert adapter.__class__.__name__ == "ProteinGeometricAdapter"
    assert adapter.model_cls == MockGeometricModel


def test_get_protein_adapter_point_cloud():
    """Test getting a protein adapter for a point cloud model."""
    adapter = get_protein_adapter(MockPointCloudModel)
    assert adapter.__class__.__name__ == "ProteinGeometricAdapter"
    assert adapter.model_cls == MockPointCloudModel


def test_get_protein_adapter_graph():
    """Test getting a protein adapter for a graph model."""
    adapter = get_protein_adapter(MockGraphModel)
    assert adapter.__class__.__name__ == "ProteinGeometricAdapter"
    assert adapter.model_cls == MockGraphModel


def test_get_protein_adapter_diffusion():
    """Test getting a protein adapter for a diffusion model."""
    adapter = get_protein_adapter(MockDiffusionModel)
    assert adapter.__class__.__name__ == "ProteinDiffusionAdapter"
    assert adapter.model_cls == MockDiffusionModel


def test_get_protein_adapter_generic():
    """Test getting a protein adapter for a generic model."""
    adapter = get_protein_adapter(MockModel)
    assert adapter.__class__.__name__ == "ProteinModelAdapter"
    assert adapter.model_cls == MockModel
