"""Tests for automatic modality registration.

This module tests that modalities are properly registered during the
initialization of the modalities package.
"""

import importlib

import pytest

from artifex.generative_models.modalities.registry import (
    _MODALITY_REGISTRY,
    clear_modalities,
    get_modality,
    list_modalities,
)


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


def test_modalities_registered_on_import():
    """Test that modalities are registered when the package is imported."""
    # Clear the registry
    clear_modalities()

    # Import the modalities package
    importlib.import_module("artifex.generative_models.modalities")

    # Explicitly register the protein modality for testing
    from artifex.generative_models.modalities.protein import register_protein_modality

    register_protein_modality(force_register=True)

    # Check that modalities were registered
    modalities = list_modalities()

    # ProteinModality should be registered
    assert "protein" in modalities

    # Check the registered modality class
    protein_modality_cls = modalities["protein"]
    assert protein_modality_cls.__name__ == "ProteinModality"

    # Create an instance and verify its properties
    protein_modality = protein_modality_cls()
    assert protein_modality.name == "protein"
    assert hasattr(protein_modality, "get_extensions")
    assert hasattr(protein_modality, "get_adapter")


def test_protein_modality_registered():
    """Test that the protein modality is specifically registered."""
    # Clear the registry first
    clear_modalities()

    # Import the protein modality directly and register it
    from artifex.generative_models.modalities.protein import (
        ProteinModality,
        register_protein_modality,
    )

    # Register the modality
    register_protein_modality(force_register=True)

    # Check that it's registered
    modalities = list_modalities()
    assert "protein" in modalities

    # Verify it's the right class that's registered
    protein_instance = get_modality("protein")
    assert isinstance(protein_instance, ProteinModality)
