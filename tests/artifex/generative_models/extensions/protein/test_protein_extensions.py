"""Tests for protein-specific extensions.

This module contains tests for protein-specific extensions that add
protein-oriented functionality to generative models.

This file has been merged with tests/protein/test_protein_extensions.py
to consolidate all protein extension tests in one place.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    PointCloudConfig,
    PointCloudNetworkConfig,
    ProteinDihedralConfig,
    ProteinExtensionConfig,
    ProteinExtensionsConfig,
    ProteinMixinConfig,
)
from artifex.generative_models.extensions.protein import (
    BondAngleExtension,
    BondLengthExtension,
    create_protein_extensions,
    ProteinBackboneConstraint,
    ProteinDihedralConstraint,
    ProteinMixinExtension,
)
from artifex.generative_models.models.geometric import PointCloudModel


@pytest.fixture
def mock_rngs():
    """Create mock random number generator keys."""
    return nnx.Rngs(0)


@pytest.fixture
def rngs():
    """Initialize random number generators for tests (alternative fixture)."""
    rng_key = jax.random.PRNGKey(42)
    params_key, dropout_key = jax.random.split(rng_key)
    return nnx.Rngs(params=params_key, dropout=dropout_key)


@pytest.fixture
def bond_length_config():
    """Create a basic configuration for BondLengthExtension."""
    return ProteinExtensionConfig(
        name="bond_length",
        weight=2.0,
        enabled=True,
        bond_length_weight=2.0,
    )


@pytest.fixture
def bond_angle_config():
    """Create a basic configuration for BondAngleExtension."""
    return ProteinExtensionConfig(
        name="bond_angle",
        weight=1.5,
        enabled=True,
        bond_angle_weight=1.5,
    )


@pytest.fixture
def protein_mixin_config():
    """Create a basic configuration for ProteinMixinExtension."""
    return ProteinMixinConfig(
        name="protein_mixin",
        weight=1.0,
        enabled=True,
    )


@pytest.fixture
def protein_atom_positions():
    """Create explicit protein-shaped coordinates for extension tests."""
    return jnp.linspace(0.0, 1.0, num=2 * 5 * 4 * 3, dtype=jnp.float32).reshape(2, 5, 4, 3)


@pytest.fixture
def protein_atom_mask(protein_atom_positions):
    """Create an all-valid atom mask matching the protein coordinate fixture."""
    return jnp.ones(protein_atom_positions.shape[:-1], dtype=jnp.float32)


def test_bond_length_extension_init(bond_length_config, mock_rngs):
    """Test initializing a BondLengthExtension."""
    extension = BondLengthExtension(bond_length_config, rngs=mock_rngs)

    assert extension.weight == 2.0
    assert extension.enabled is True

    # Check for presence of expected attributes based on implementation
    assert hasattr(extension, "config")
    assert isinstance(extension.config, ProteinExtensionConfig)


def test_bond_angle_extension_init(bond_angle_config, mock_rngs):
    """Test initializing a BondAngleExtension."""
    extension = BondAngleExtension(bond_angle_config, rngs=mock_rngs)

    assert extension.weight == 1.5
    assert extension.enabled is True

    # Check for presence of expected attributes based on implementation
    assert hasattr(extension, "config")
    assert isinstance(extension.config, ProteinExtensionConfig)


def test_protein_mixin_extension_init(protein_mixin_config, mock_rngs):
    """Test initializing a ProteinMixinExtension."""
    extension = ProteinMixinExtension(protein_mixin_config, rngs=mock_rngs)

    assert extension.weight == 1.0
    assert extension.enabled is True

    # Check for presence of expected attributes based on implementation
    assert hasattr(extension, "config")
    assert isinstance(extension.config, ProteinMixinConfig)
    assert hasattr(extension, "embedding_dim")
    assert extension.embedding_dim == 16


def test_bond_length_extension_call(
    bond_length_config,
    mock_rngs,
    protein_atom_positions,
    protein_atom_mask,
):
    """Test calling a BondLengthExtension with explicit protein coordinates."""
    extension = BondLengthExtension(bond_length_config, rngs=mock_rngs)

    inputs = {"atom_mask": protein_atom_mask}
    model_outputs = {"atom_positions": protein_atom_positions}

    result = extension(inputs, model_outputs)

    assert isinstance(result, dict)
    assert result["extension_type"] == "bond_length"


def test_bond_angle_extension_call(
    bond_angle_config,
    mock_rngs,
    protein_atom_positions,
    protein_atom_mask,
):
    """Test calling a BondAngleExtension with explicit protein coordinates."""
    extension = BondAngleExtension(bond_angle_config, rngs=mock_rngs)

    inputs = {"atom_mask": protein_atom_mask}
    model_outputs = {"atom_positions": protein_atom_positions}

    result = extension(inputs, model_outputs)

    assert isinstance(result, dict)
    assert result["extension_type"] == "bond_angle"


def test_protein_mixin_extension_call(protein_mixin_config, mock_rngs):
    """Test calling a ProteinMixinExtension."""
    extension = ProteinMixinExtension(protein_mixin_config, rngs=mock_rngs)

    # Create minimal inputs and outputs for testing
    inputs = {"sequence": jnp.zeros((10, 21))}
    model_outputs = {"embeddings": jnp.ones((10, 64))}

    result = extension(inputs, model_outputs)

    # The result should be a dictionary with extension outputs
    assert isinstance(result, dict)


def test_bond_length_extension_loss(
    bond_length_config,
    mock_rngs,
    protein_atom_positions,
    protein_atom_mask,
):
    """Test calculating loss with a BondLengthExtension."""
    extension = BondLengthExtension(bond_length_config, rngs=mock_rngs)

    batch = {"atom_mask": protein_atom_mask}
    model_outputs = {"atom_positions": protein_atom_positions}

    loss = extension.loss_fn(batch, model_outputs)

    assert isinstance(loss, jax.Array)
    assert loss.ndim == 0


def test_bond_angle_extension_loss(
    bond_angle_config,
    mock_rngs,
    protein_atom_positions,
    protein_atom_mask,
):
    """Test calculating loss with a BondAngleExtension."""
    extension = BondAngleExtension(bond_angle_config, rngs=mock_rngs)

    batch = {"atom_mask": protein_atom_mask}
    model_outputs = {"atom_positions": protein_atom_positions}

    loss = extension.loss_fn(batch, model_outputs)

    assert isinstance(loss, jax.Array)
    assert loss.ndim == 0


def test_create_protein_extensions_empty(mock_rngs):
    """Test creating protein extensions with empty config."""
    extensions = create_protein_extensions(ProteinExtensionsConfig(name="empty"), rngs=mock_rngs)

    # The result should be an nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    assert len(extensions) == 0


def test_create_protein_extensions_contains_nonexistent_key(mock_rngs):
    """Test that extensions dict properly handles 'in' operator for non-existent keys.

    This tests for a bug where nnx.Dict's __contains__ raises AttributeError
    when checking for missing keys instead of returning False.
    """
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(name="bond_length"),
        bond_angle=ProteinExtensionConfig(name="bond_angle"),
    )
    extensions = create_protein_extensions(config, rngs=mock_rngs)

    # These should work without raising AttributeError
    assert "bond_length" in extensions  # Exists
    assert "bond_angle" in extensions  # Exists
    assert "nonexistent_key" not in extensions  # Does NOT exist - this was failing
    assert "protein_quality" not in extensions  # Does NOT exist - this was failing


def test_create_protein_extensions_backbone(mock_rngs):
    """Test creating protein extensions with backbone constraints."""
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(
            name="bond_length",
            weight=2.0,
            bond_length_weight=2.0,
        ),
        bond_angle=ProteinExtensionConfig(
            name="bond_angle",
            weight=1.5,
            bond_angle_weight=1.5,
        ),
    )

    extensions = create_protein_extensions(config, rngs=mock_rngs)

    # The result should be nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    assert "bond_length" in extensions
    assert "bond_angle" in extensions

    # Check extension types
    assert isinstance(extensions["bond_length"], BondLengthExtension)
    assert isinstance(extensions["bond_angle"], BondAngleExtension)

    # Check weights
    assert extensions["bond_length"].weight == 2.0
    assert extensions["bond_angle"].weight == 1.5


def test_create_protein_extensions_mixin(mock_rngs):
    """Test creating protein extensions with protein mixin."""
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        mixin=ProteinMixinConfig(
            name="protein_mixin",
            embedding_dim=32,
            use_one_hot=False,
            num_aa_types=22,
        ),
    )

    extensions = create_protein_extensions(config, rngs=mock_rngs)

    # The result should be nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    assert "protein_mixin" in extensions

    # Check extension type
    assert isinstance(extensions["protein_mixin"], ProteinMixinExtension)

    # Check configuration values from typed config were applied
    assert extensions["protein_mixin"].embedding_dim == 32
    assert extensions["protein_mixin"].use_one_hot is False
    assert extensions["protein_mixin"].num_aa_types == 22


def test_create_protein_extensions_all(mock_rngs):
    """Test creating all protein extensions."""
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(name="bond_length", weight=2.0, bond_length_weight=2.0),
        bond_angle=ProteinExtensionConfig(name="bond_angle", weight=1.5, bond_angle_weight=1.5),
        backbone=ProteinExtensionConfig(name="backbone_constraint", weight=1.0),
        dihedral=ProteinDihedralConfig(name="dihedral_constraint", weight=0.7),
        mixin=ProteinMixinConfig(name="protein_mixin", embedding_dim=32),
    )

    extensions = create_protein_extensions(config, rngs=mock_rngs)

    # The result should be nnx.Dict (compatible with dict interface)
    assert isinstance(extensions, (dict, nnx.Dict))
    assert "bond_length" in extensions
    assert "bond_angle" in extensions
    assert "backbone" in extensions
    assert "dihedral" in extensions
    assert "protein_mixin" in extensions
    assert isinstance(extensions["backbone"], ProteinBackboneConstraint)
    assert isinstance(extensions["dihedral"], ProteinDihedralConstraint)


def test_bond_length_extension_rejects_ambiguous_flattened_dict_payload(
    bond_length_config,
    mock_rngs,
    protein_atom_positions,
):
    """Bond-length helpers should not silently coerce flat dict payloads."""
    extension = BondLengthExtension(bond_length_config, rngs=mock_rngs)
    flattened_positions = protein_atom_positions.reshape(protein_atom_positions.shape[0], -1, 3)

    with pytest.raises(ValueError, match="atom_positions"):
        extension({}, {"positions": flattened_positions})

    with pytest.raises(ValueError, match="atom_positions"):
        extension({}, {"predicted_coordinates": flattened_positions})


def test_bond_angle_extension_rejects_ambiguous_flattened_dict_payload(
    bond_angle_config,
    mock_rngs,
    protein_atom_positions,
):
    """Bond-angle helpers should not silently coerce flat dict payloads."""
    extension = BondAngleExtension(bond_angle_config, rngs=mock_rngs)
    flattened_positions = protein_atom_positions.reshape(protein_atom_positions.shape[0], -1, 3)

    with pytest.raises(ValueError, match="atom_positions"):
        extension({}, {"positions": flattened_positions})

    with pytest.raises(ValueError, match="atom_positions"):
        extension({}, {"predicted_coordinates": flattened_positions})


def test_create_protein_extensions_coordinate_contract_is_consistent(
    mock_rngs,
    protein_atom_positions,
    protein_atom_mask,
):
    """Coordinate-consuming protein extensions should agree on one payload contract."""
    config = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(name="bond_length"),
        bond_angle=ProteinExtensionConfig(name="bond_angle"),
        backbone=ProteinExtensionConfig(name="backbone_constraint"),
        dihedral=ProteinDihedralConfig(name="dihedral_constraint"),
    )
    extensions = create_protein_extensions(config, rngs=mock_rngs)

    explicit_inputs = {"atom_mask": protein_atom_mask}
    explicit_outputs = {"atom_positions": protein_atom_positions}
    for extension_name in ("bond_length", "bond_angle", "backbone", "dihedral"):
        result = extensions[extension_name](explicit_inputs, explicit_outputs)
        assert isinstance(result, dict)

    flattened_outputs = {
        "positions": protein_atom_positions.reshape(protein_atom_positions.shape[0], -1, 3)
    }
    for extension_name in ("bond_length", "bond_angle", "backbone", "dihedral"):
        with pytest.raises(ValueError, match="atom_positions"):
            extensions[extension_name](explicit_inputs, flattened_outputs)


def test_point_cloud_model_preserves_explicit_atom_positions_for_protein_extensions(mock_rngs):
    """Generic point-cloud models should keep explicit protein coordinates available to extensions."""
    extension_bundle = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(name="bond_length"),
        bond_angle=ProteinExtensionConfig(name="bond_angle"),
    )
    extensions = create_protein_extensions(extension_bundle, rngs=mock_rngs)
    model_config = PointCloudConfig(
        name="protein_point_cloud_with_extensions",
        network=PointCloudNetworkConfig(
            name="protein_network",
            hidden_dims=(16, 16),
            embed_dim=16,
            num_heads=2,
            num_layers=1,
            dropout_rate=0.0,
            activation="gelu",
        ),
        num_points=8,
        dropout_rate=0.0,
    )
    model = PointCloudModel(model_config, extensions=extensions, rngs=mock_rngs)

    batch = {
        "atom_positions": jnp.ones((1, 2, 4, 3), dtype=jnp.float32),
        "atom_mask": jnp.ones((1, 2, 4), dtype=jnp.float32),
    }
    outputs = model(batch)

    assert outputs["positions"].shape == (1, 8, 3)
    assert outputs["atom_positions"].shape == (1, 2, 4, 3)
    assert set(outputs["extension_outputs"]) == {"bond_length", "bond_angle"}


# ========================================================================
# Tests merged from tests/protein/test_protein_extensions.py
# ========================================================================


@pytest.fixture
def model_config():
    """Create a model configuration for tests."""
    num_residues = 10
    num_atoms_per_residue = 4
    num_points = num_residues * num_atoms_per_residue

    return {
        "num_points": num_points,
        "model_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "num_residues": num_residues,
        "num_atoms_per_residue": num_atoms_per_residue,
    }


@pytest.fixture
def test_batch(model_config):
    """Create a test batch of data."""
    batch_size = 2
    num_residues = model_config["num_residues"]
    num_points = model_config["num_points"]

    # Initialize RNG
    rng_key = jax.random.PRNGKey(123)

    # Create amino acid type inputs
    aatype_key, coords_key = jax.random.split(rng_key)
    aatype = jax.random.randint(aatype_key, (batch_size, num_residues), 0, 20)

    # Create random coordinates
    coords = jax.random.normal(coords_key, (batch_size, num_points, 3)) * 5.0

    # Create mask (all valid)
    mask = jnp.ones((batch_size, num_points))

    return {
        "aatype": aatype,
        "positions": coords,
        "mask": mask,
    }


def test_create_extensions_alternative_config(rngs):
    """Test creating individual protein extensions with alternative config format."""
    # Test bond length extension
    bond_length_config = ProteinExtensionConfig(
        name="bond_length",
        weight=1.0,
        enabled=True,
        bond_length_weight=1.0,
    )
    bond_length_ext = BondLengthExtension(
        bond_length_config,
        rngs=rngs,
    )
    assert isinstance(bond_length_ext, BondLengthExtension)

    # Test bond angle extension
    bond_angle_config = ProteinExtensionConfig(
        name="bond_angle",
        weight=0.5,
        enabled=True,
        bond_angle_weight=0.5,
    )
    bond_angle_ext = BondAngleExtension(
        bond_angle_config,
        rngs=rngs,
    )
    assert isinstance(bond_angle_ext, BondAngleExtension)

    # Test protein mixin extension
    protein_mixin_config = ProteinMixinConfig(
        name="protein_mixin",
        weight=1.0,
        enabled=True,
    )
    protein_mixin_ext = ProteinMixinExtension(
        protein_mixin_config,
        rngs=rngs,
    )
    assert isinstance(protein_mixin_ext, ProteinMixinExtension)


def test_create_protein_extensions_with_backbone_and_mixin_bundle(rngs):
    """Test the canonical typed protein bundle can express the common helper case."""
    protein_config = ProteinExtensionsConfig(
        name="protein_extensions",
        bond_length=ProteinExtensionConfig(name="bond_length"),
        bond_angle=ProteinExtensionConfig(name="bond_angle"),
        mixin=ProteinMixinConfig(name="protein_mixin", embedding_dim=64),
    )

    extensions = create_protein_extensions(protein_config, rngs=rngs)

    assert "bond_length" in extensions
    assert "bond_angle" in extensions
    assert "protein_mixin" in extensions
    assert isinstance(extensions["bond_length"], BondLengthExtension)
    assert isinstance(extensions["bond_angle"], BondAngleExtension)
    assert isinstance(extensions["protein_mixin"], ProteinMixinExtension)
