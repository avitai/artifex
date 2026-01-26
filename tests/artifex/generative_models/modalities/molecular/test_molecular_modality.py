"""Tests for MolecularModality class.

This module tests the molecular modality functionality, including
extension creation with the updated API (extensions instead of extension_configs).
"""

import jax
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ChemicalConstraintConfig,
    ExtensionConfig,
    ModalityConfig,
)
from artifex.generative_models.modalities.molecular import MolecularModality


@pytest.fixture
def rngs():
    """Create random number generators for tests."""
    key = jax.random.PRNGKey(42)
    return nnx.Rngs(params=key)


@pytest.fixture
def molecular_modality(rngs):
    """Create a MolecularModality instance for testing."""
    return MolecularModality(rngs=rngs)


def test_molecular_modality_init(molecular_modality):
    """Test that MolecularModality initializes correctly."""
    assert molecular_modality is not None
    assert molecular_modality.name == "molecular"
    assert hasattr(molecular_modality, "rngs")


def test_get_extensions_empty_config(molecular_modality, rngs):
    """Test getting extensions with an empty configuration."""
    config = ModalityConfig(
        name="molecular_config",
        modality_name="molecular",
    )

    extensions = molecular_modality.get_extensions(config, rngs=rngs)

    # Should return nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    # With empty extensions config, should be empty
    assert len(extensions) == 0


def test_get_extensions_with_chemical_constraints(molecular_modality, rngs):
    """Test getting extensions with chemical constraint configuration."""
    # Use ChemicalConstraintConfig for chemical constraints
    chemical_config = ChemicalConstraintConfig(
        name="chemical",
        weight=1.0,
        enabled=True,
        enforce_valence=True,
        enforce_bond_lengths=True,
    )

    config = ModalityConfig(
        name="molecular_config",
        modality_name="molecular",
        extensions={"chemical": chemical_config},
    )

    extensions = molecular_modality.get_extensions(config, rngs=rngs)

    # Should return nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    assert "chemical" in extensions
    # Verify the extension was created
    assert extensions["chemical"] is not None


def test_get_extensions_with_pharmacophore(molecular_modality, rngs):
    """Test getting extensions with pharmacophore configuration."""
    # Use base ExtensionConfig for pharmacophore (uses default feature types)
    pharma_config = ExtensionConfig(
        name="pharmacophore",
        weight=1.0,
        enabled=True,
    )

    config = ModalityConfig(
        name="molecular_config",
        modality_name="molecular",
        extensions={"pharmacophore": pharma_config},
    )

    extensions = molecular_modality.get_extensions(config, rngs=rngs)

    # Should return nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    assert "pharmacophore" in extensions
    # Verify the extension was created
    assert extensions["pharmacophore"] is not None


def test_get_extensions_with_both(molecular_modality, rngs):
    """Test getting extensions with both chemical and pharmacophore configs."""
    # Use ChemicalConstraintConfig for chemical constraints
    chemical_config = ChemicalConstraintConfig(
        name="chemical",
        weight=1.0,
        enabled=True,
    )

    # Use base ExtensionConfig for pharmacophore
    pharma_config = ExtensionConfig(
        name="pharmacophore",
        weight=0.8,
        enabled=True,
    )

    config = ModalityConfig(
        name="molecular_config",
        modality_name="molecular",
        extensions={"chemical": chemical_config, "pharmacophore": pharma_config},
    )

    extensions = molecular_modality.get_extensions(config, rngs=rngs)

    # Should return nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    assert "chemical" in extensions
    assert "pharmacophore" in extensions
    assert len(extensions) == 2


def test_get_extensions_returns_nnx_dict(molecular_modality, rngs):
    """Test that get_extensions returns nnx.Dict for Flax NNX 0.12.0+ compatibility."""
    # Use ChemicalConstraintConfig for chemical extension
    chemical_config = ChemicalConstraintConfig(
        name="chemical",
        weight=1.0,
        enabled=True,
    )

    config = ModalityConfig(
        name="molecular_config",
        modality_name="molecular",
        extensions={"chemical": chemical_config},
    )

    extensions = molecular_modality.get_extensions(config, rngs=rngs)

    # Verify it's an nnx.Dict (not just a plain dict)
    assert isinstance(extensions, nnx.Dict)
    # But should still work like a dict
    assert "chemical" in extensions
    assert len(extensions) == 1


def test_get_adapter_default(molecular_modality):
    """Test getting the default adapter."""
    adapter = molecular_modality.get_adapter("default")

    assert adapter is not None
    # Adapter should have the required ModelAdapter interface methods
    assert hasattr(adapter, "create")
    assert hasattr(adapter, "adapt_input")
    assert hasattr(adapter, "adapt_output")
    assert hasattr(adapter, "adapt_loss")


def test_get_adapter_diffusion(molecular_modality):
    """Test getting the diffusion adapter."""
    adapter = molecular_modality.get_adapter("diffusion")

    assert adapter is not None
    # Should be a subclass of MolecularAdapter with adapter methods
    assert hasattr(adapter, "create")
    assert hasattr(adapter, "adapt_input")
    assert hasattr(adapter, "adapt_output")
    assert hasattr(adapter, "adapt_loss")


def test_get_adapter_geometric(molecular_modality):
    """Test getting the geometric adapter."""
    adapter = molecular_modality.get_adapter("geometric")

    assert adapter is not None
    # Should be a subclass of MolecularAdapter with adapter methods
    assert hasattr(adapter, "create")
    assert hasattr(adapter, "adapt_input")
    assert hasattr(adapter, "adapt_output")
    assert hasattr(adapter, "adapt_loss")


def test_config_validation_wrong_type(molecular_modality, rngs):
    """Test that config validation catches wrong configuration type."""
    # Passing a dict instead of a proper config object should fail.
    # The method will raise AttributeError when trying to access .extensions on a dict.
    with pytest.raises((TypeError, ValueError, AttributeError)):
        molecular_modality.get_extensions({"name": "test"}, rngs=rngs)
