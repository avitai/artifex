"""Tests for the modality registry.

This module contains tests for the modality registry functionality.
"""

import pytest

from artifex.generative_models.modalities.base import Modality, ModelAdapter
from artifex.generative_models.modalities.registry import (
    _MODALITY_REGISTRY,
    get_modality,
    list_modalities,
    register_modality,
)


class MockGenerativeModel:
    """Mock generative model for testing."""

    def __call__(self, *args, **kwargs):
        """Mock forward pass."""
        return {}

    def generate(self, *args, **kwargs):
        """Mock generate method."""
        import jax.numpy as jnp

        return jnp.zeros((1, 1))

    def loss_fn(self, *args, **kwargs):
        """Mock loss function."""
        return {}


@pytest.fixture
def model_adapter_factory():
    """Create a test model adapter factory."""

    def _create_adapter():
        class _TestModelAdapter(ModelAdapter):
            """Test adapter implementation for testing."""

            def create(self, config, *, rngs, **kwargs):
                """Create a test model."""
                return MockGenerativeModel()

        return _TestModelAdapter()

    return _create_adapter


@pytest.fixture
def modality_factory(model_adapter_factory):
    """Create a test modality factory."""

    def _create_modality():
        class _TestModality(Modality):
            """Test modality implementation for testing."""

            name = "test_modality"

            def get_extensions(self, config, *, rngs):
                """Get extensions for testing."""
                return {}

            def get_adapter(self, model_cls):
                """Get adapter for testing."""
                return model_adapter_factory()

        return _TestModality()

    return _create_modality


# Make sure registry is clean before each test
@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the modality registry before each test."""
    # Save existing entries
    saved = dict(_MODALITY_REGISTRY)
    # Clear the registry
    _MODALITY_REGISTRY.clear()
    # Run the test
    yield
    # Restore the registry
    _MODALITY_REGISTRY.clear()
    _MODALITY_REGISTRY.update(saved)


def test_register_modality(modality_factory):
    """Test registering a modality."""
    # Register a modality
    modality_class = modality_factory().__class__
    register_modality("test", modality_class)

    # Check that it was registered
    assert "test" in _MODALITY_REGISTRY
    assert _MODALITY_REGISTRY["test"] == modality_class


def test_register_modality_duplicate(modality_factory):
    """Test registering a duplicate modality."""
    # Register a modality
    modality_class = modality_factory().__class__
    register_modality("test", modality_class)

    # Try to register it again
    with pytest.raises(ValueError):
        register_modality("test", modality_class)


def test_get_modality(modality_factory):
    """Test getting a modality."""
    # Register a modality
    modality_class = modality_factory().__class__
    register_modality("test", modality_class)

    # Get the modality
    modality = get_modality("test")

    # Check that it's an instance of the right class
    assert isinstance(modality, modality_class)


def test_get_modality_unknown():
    """Test getting an unknown modality."""
    # Try to get an unknown modality
    with pytest.raises(ValueError):
        get_modality("unknown")


def test_list_modalities(modality_factory):
    """Test listing modalities."""
    # Register some modalities
    modality_class = modality_factory().__class__
    register_modality("test1", modality_class)
    register_modality("test2", modality_class)

    # List modalities
    modalities = list_modalities()

    # Check that they're all there
    assert "test1" in modalities
    assert "test2" in modalities
    assert modalities["test1"] == modality_class
    assert modalities["test2"] == modality_class


def test_list_modalities_empty():
    """Test listing modalities when none are registered."""
    # List modalities
    modalities = list_modalities()

    # Check that it's empty
    assert len(modalities) == 0
