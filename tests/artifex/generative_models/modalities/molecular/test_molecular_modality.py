"""Tests for MolecularModality class.

This module tests the molecular modality functionality, including
extension creation with the updated API (extensions instead of extension_configs).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ChemicalConstraintConfig,
    ExtensionConfig,
    ModalityConfig,
)
from artifex.generative_models.extensions.chemical.constraints import ChemicalConstraints
from artifex.generative_models.extensions.registry import ExtensionsRegistry
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
    assert isinstance(extensions["chemical"], ChemicalConstraints)


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
    assert hasattr(adapter, "adapt")
    assert hasattr(adapter, "adapt_input")
    assert hasattr(adapter, "adapt_output")
    assert hasattr(adapter, "adapt_loss")


def test_get_adapter_accepts_model_family_keys(molecular_modality):
    """The molecular modality should accept the same model-family keys as the factory."""
    adapter = molecular_modality.get_adapter("vae")
    model = object()

    assert adapter.adapt(model, object()) is model


def test_get_adapter_diffusion(molecular_modality):
    """Test getting the diffusion adapter."""
    adapter = molecular_modality.get_adapter("diffusion")

    assert adapter is not None
    # Should be a subclass of MolecularAdapter with adapter methods
    assert hasattr(adapter, "create")
    assert hasattr(adapter, "adapt")
    assert hasattr(adapter, "adapt_input")
    assert hasattr(adapter, "adapt_output")
    assert hasattr(adapter, "adapt_loss")


def test_get_adapter_geometric(molecular_modality):
    """Test getting the geometric adapter."""
    adapter = molecular_modality.get_adapter("geometric")

    assert adapter is not None
    # Should be a subclass of MolecularAdapter with adapter methods
    assert hasattr(adapter, "create")
    assert hasattr(adapter, "adapt")
    assert hasattr(adapter, "adapt_input")
    assert hasattr(adapter, "adapt_output")
    assert hasattr(adapter, "adapt_loss")


def test_extensions_registry_creates_typed_chemical_constraint_config(rngs):
    """The registry default config for chemical constraints should be typed and live."""
    extension = ExtensionsRegistry().create_extension("chemical_constraints", rngs=rngs)

    assert isinstance(extension, ChemicalConstraints)
    assert isinstance(extension.config, ChemicalConstraintConfig)
    assert extension.enforce_valence is True
    assert extension.enforce_bond_lengths is True


def test_chemical_constraints_follow_typed_config_flags(rngs):
    """Chemical validity checks should follow the typed chemical constraint config."""
    config = ChemicalConstraintConfig(
        name="chemical",
        enforce_valence=False,
        enforce_bond_lengths=True,
        enforce_ring_closure=False,
    )
    extension = ChemicalConstraints(config, rngs=rngs)
    coordinates = jnp.zeros((4, 3))
    atom_types = jnp.zeros((4,), dtype=jnp.int32)
    bonds = jnp.eye(4)

    results = extension.validate_molecular_structure(coordinates, atom_types, bonds)

    assert "bond_length_validity" in results
    assert "valence_validity" not in results
    assert "ring_strain_validity" not in results


def test_diffusion_adapter_adapt_loss_preserves_canonical_loss_dict(molecular_modality):
    """Diffusion adapter should add physics penalties without breaking the loss contract."""
    adapter = molecular_modality.get_adapter("diffusion")
    adapter._compute_physics_penalty = lambda batch, outputs: jnp.array(2.0)

    def base_loss_fn(batch, outputs, **kwargs):
        return {
            "total_loss": jnp.array(1.0),
            "reconstruction_loss": jnp.array(1.0),
        }

    adapted_loss_fn = adapter.adapt_loss(base_loss_fn)
    result = adapted_loss_fn(
        {"coordinates": jnp.zeros((1, 4, 3))},
        {"coordinates": jnp.zeros((1, 4, 3))},
    )

    assert jnp.isclose(result["reconstruction_loss"], 1.0)
    assert jnp.isclose(result["physics_penalty"], 2.0)
    assert jnp.isclose(result["total_loss"], 1.2)


def test_geometric_adapter_adapt_loss_preserves_canonical_loss_dict(molecular_modality):
    """Geometric adapter should add geometry penalties into total_loss explicitly."""
    adapter = molecular_modality.get_adapter("geometric")
    adapter._compute_geometry_penalty = lambda batch, outputs: jnp.array(3.0)

    def base_loss_fn(batch, outputs, **kwargs):
        return {
            "total_loss": jnp.array(0.5),
            "coord_loss": jnp.array(0.5),
        }

    adapted_loss_fn = adapter.adapt_loss(base_loss_fn)
    result = adapted_loss_fn(
        {"coordinates": jnp.zeros((1, 4, 3))},
        {"coordinates": jnp.zeros((1, 4, 3))},
    )

    assert jnp.isclose(result["coord_loss"], 0.5)
    assert jnp.isclose(result["geometry_penalty"], 3.0)
    assert jnp.isclose(result["total_loss"], 0.8)


def test_adapters_require_total_loss_in_base_loss_dict(molecular_modality):
    """Adapters should reject ambiguous loss dicts without a canonical total_loss."""
    diffusion_adapter = molecular_modality.get_adapter("diffusion")

    def invalid_base_loss_fn(batch, outputs, **kwargs):
        return {"reconstruction_loss": jnp.array(1.0)}

    adapted_loss_fn = diffusion_adapter.adapt_loss(invalid_base_loss_fn)

    with pytest.raises(ValueError, match="total_loss"):
        adapted_loss_fn(
            {"coordinates": jnp.zeros((1, 4, 3))},
            {"coordinates": jnp.zeros((1, 4, 3))},
        )


def test_config_validation_wrong_type(molecular_modality, rngs):
    """Test that config validation catches wrong configuration type."""
    # Passing a dict instead of a proper config object should fail.
    # The method will raise AttributeError when trying to access .extensions on a dict.
    with pytest.raises((TypeError, ValueError, AttributeError)):
        molecular_modality.get_extensions({"name": "test"}, rngs=rngs)
