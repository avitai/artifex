"""Tests for protein modality adapters.

This module tests the adapters that allow protein-specific models to
interface with different model architectures.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.modalities.base import ModelAdapter
from artifex.generative_models.modalities.protein.adapters import (
    ProteinDiffusionAdapter,
    ProteinGeometricAdapter,
    ProteinModelAdapter,
)
from artifex.generative_models.modalities.protein.config import (
    register_protein_modality,
)
from artifex.generative_models.modalities.protein.utils import (
    get_protein_adapter,
)
from artifex.generative_models.modalities.registry import (
    _MODALITY_REGISTRY,
)


@pytest.fixture(autouse=True)
def save_registry():
    """Isolate protein-modality registry setup per test."""
    saved = dict(_MODALITY_REGISTRY)
    _MODALITY_REGISTRY.clear()
    register_protein_modality(force_register=True)
    try:
        yield
    finally:
        _MODALITY_REGISTRY.clear()
        _MODALITY_REGISTRY.update(saved)


def test_protein_model_adapter_init():
    """Test initializing a ProteinModelAdapter."""
    adapter = ProteinModelAdapter()

    # Check basic properties
    assert adapter.name == "protein_model"
    assert adapter.modality == "protein"
    assert issubclass(adapter.__class__, ModelAdapter)


def test_protein_model_adapter_create_keeps_generic_model_family():
    """The exported default adapter should preserve the factory-selected model family."""
    adapter = ProteinModelAdapter()
    config = PointCloudConfig(
        name="protein_point_cloud",
        network=PointCloudNetworkConfig(
            name="protein_point_cloud_network",
            hidden_dims=(64,),
            activation="relu",
            embed_dim=64,
            num_heads=4,
            num_layers=2,
        ),
        num_points=32,
        point_dim=3,
    )

    model = adapter.create(config, rngs=nnx.Rngs(params=0))

    assert model.__class__.__name__ == "PointCloudModel"


def test_protein_geometric_adapter_adapt_is_explicit_no_op():
    """The protein geometric adapter should currently be an explicit no-op."""
    adapter = ProteinGeometricAdapter()
    model = object()

    assert adapter.adapt(model, config={}) is model


def test_protein_geometric_adapter_init():
    """Test initializing a ProteinGeometricAdapter."""
    adapter = ProteinGeometricAdapter()

    # Check basic properties
    assert adapter.name == "protein_geometric"
    assert adapter.modality == "protein"
    assert issubclass(adapter.__class__, ModelAdapter)


def test_protein_diffusion_adapter_init():
    """Test initializing a ProteinDiffusionAdapter."""
    adapter = ProteinDiffusionAdapter()

    # Check basic properties
    assert adapter.name == "protein_diffusion"
    assert adapter.modality == "protein"
    assert issubclass(adapter.__class__, ModelAdapter)


def test_protein_model_adapter_adapt_inputs():
    """Test adapting inputs with ProteinModelAdapter."""
    adapter = ProteinModelAdapter()

    # Create test inputs
    inputs = {
        "sequence": jnp.ones((8, 21)),  # One-hot encoded sequence
        "coordinates": jnp.ones((8, 3, 3)),  # Atom coordinates
    }

    # Adapt inputs
    adapted = adapter.adapt_inputs(inputs)

    # Verify adaptation
    assert isinstance(adapted, dict)
    assert "sequence" in adapted
    assert "coordinates" in adapted


def test_protein_geometric_adapter_adapt_inputs():
    """Test adapting inputs with ProteinGeometricAdapter."""
    adapter = ProteinGeometricAdapter()

    # Create test inputs
    inputs = {
        "sequence": jnp.ones((8, 21)),  # One-hot encoded sequence
        "coordinates": jnp.ones((8, 3, 3)),  # Atom coordinates
    }

    # Adapt inputs
    adapted = adapter.adapt_inputs(inputs)

    # Verify adaptation
    assert isinstance(adapted, dict)
    assert "positions" in adapted  # Should transform to positions


def test_protein_diffusion_adapter_adapt_inputs():
    """Test adapting inputs with ProteinDiffusionAdapter."""
    adapter = ProteinDiffusionAdapter()

    # Create test inputs
    inputs = {
        "sequence": jnp.ones((8, 21)),  # One-hot encoded sequence
        "coordinates": jnp.ones((8, 3, 3)),  # Atom coordinates
    }

    # Adapt inputs
    adapted = adapter.adapt_inputs(inputs)

    # Verify adaptation
    assert isinstance(adapted, dict)
    # Should have sequence and noise
    assert "sequence" in adapted
    assert "noise" in adapted


def test_get_protein_adapter():
    """Test getting a protein adapter by name."""
    # Get default adapter
    adapter = get_protein_adapter()
    assert isinstance(adapter, ProteinModelAdapter)

    # Get specific adapters
    geometric_adapter = get_protein_adapter("geometric")
    assert isinstance(geometric_adapter, ProteinGeometricAdapter)

    diffusion_adapter = get_protein_adapter("diffusion")
    assert isinstance(diffusion_adapter, ProteinDiffusionAdapter)


def test_get_protein_adapter_unknown():
    """Test getting an unknown protein adapter."""
    with pytest.raises(ValueError):
        get_protein_adapter("unknown_adapter")
