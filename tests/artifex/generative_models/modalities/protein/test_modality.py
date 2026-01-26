"""Tests for protein modality implementation.

This module tests the protein modality class that provides protein-specific
functionality to generative models.
"""

from collections.abc import Mapping

import pytest

# Import registry functionality and modality classes
from artifex.generative_models.modalities.protein import ProteinModality
from artifex.generative_models.modalities.protein.config import register_protein_modality
from artifex.generative_models.modalities.registry import _MODALITY_REGISTRY, clear_modalities


# Clear registry before tests to prevent duplicate registration errors
clear_modalities()

# Register protein modality manually for tests
register_protein_modality(force_register=True)


@pytest.fixture(autouse=True)
def save_registry():
    """Save and restore the modality registry."""
    # Save registry state
    saved = dict(_MODALITY_REGISTRY)
    try:
        yield
    finally:
        # Restore registry state
        _MODALITY_REGISTRY.clear()
        _MODALITY_REGISTRY.update(saved)


def test_protein_modality_init():
    """Test initializing a ProteinModality."""
    modality = ProteinModality()

    # Check basic properties
    assert modality.name == "protein"
    assert hasattr(modality, "get_adapter")
    assert hasattr(modality, "get_extensions")


def test_protein_modality_get_adapter():
    """Test getting a protein adapter."""
    modality = ProteinModality()

    # Get default adapter
    adapter = modality.get_adapter()
    assert adapter.modality == "protein"
    assert adapter.name == "protein_model"

    # Get geometric adapter
    geometric_adapter = modality.get_adapter("geometric")
    assert geometric_adapter.modality == "protein"
    assert geometric_adapter.name == "protein_geometric"

    # Get diffusion adapter
    diffusion_adapter = modality.get_adapter("diffusion")
    assert diffusion_adapter.modality == "protein"
    assert diffusion_adapter.name == "protein_diffusion"


def test_protein_modality_get_adapter_unknown():
    """Test getting an unknown adapter from protein modality."""
    modality = ProteinModality()

    with pytest.raises(ValueError):
        modality.get_adapter("unknown_adapter")


def test_protein_modality_get_extensions():
    """Test getting extensions from protein modality."""
    modality = ProteinModality()

    # Get extensions with default config
    extensions = modality.get_extensions({})
    assert isinstance(extensions, Mapping)  # nnx.Dict is a Mapping, not dict
    assert len(extensions) == 0  # Empty config = no extensions

    # Get extensions with backbone constraints
    config = {
        "extensions": {
            "use_backbone_constraints": True,
            "bond_length_weight": 2.0,
            "bond_angle_weight": 1.5,
        }
    }
    extensions = modality.get_extensions(config)
    assert isinstance(extensions, Mapping)
    assert "bond_length" in extensions
    assert "bond_angle" in extensions
    assert extensions["bond_length"].weight == 2.0
    assert extensions["bond_angle"].weight == 1.5

    # Get extensions with protein mixin
    config = {
        "extensions": {
            "use_protein_mixin": True,
            "aa_embedding_dim": 32,
        }
    }
    extensions = modality.get_extensions(config)
    assert isinstance(extensions, Mapping)
    assert "protein_mixin" in extensions
    assert extensions["protein_mixin"].embedding_dim == 32


def test_protein_modality_get_extensions_all():
    """Test getting all extensions from protein modality."""
    modality = ProteinModality()

    # Get all extensions
    config = {
        "extensions": {
            "use_backbone_constraints": True,
            "bond_length_weight": 2.0,
            "bond_angle_weight": 1.5,
            "use_protein_mixin": True,
            "aa_embedding_dim": 32,
        }
    }
    extensions = modality.get_extensions(config)

    # Verify extensions
    assert isinstance(extensions, Mapping)  # nnx.Dict is a Mapping, not dict
    assert "bond_length" in extensions
    assert "bond_angle" in extensions
    assert "protein_mixin" in extensions
