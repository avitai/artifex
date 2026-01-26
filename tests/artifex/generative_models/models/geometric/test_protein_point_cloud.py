"""Tests for ProteinPointCloudModel."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ProteinConstraintConfig,
    ProteinPointCloudConfig,
)
from artifex.generative_models.core.configuration.geometric_config import (
    PointCloudNetworkConfig,
)
from artifex.generative_models.models.geometric.protein_point_cloud import (
    ProteinPointCloudModel,
)


@pytest.fixture
def model_config():
    """Fixture for model configuration."""
    network = PointCloudNetworkConfig(
        name="protein_pc_network",
        hidden_dims=(32,),
        embed_dim=32,  # Must be divisible by num_heads
        num_heads=4,
        num_layers=2,
        use_positional_encoding=True,
        activation="relu",
        dropout_rate=0.1,
    )
    constraint_config = ProteinConstraintConfig(
        backbone_weight=1.0,
        bond_weight=1.0,
        angle_weight=0.5,
        dihedral_weight=0.3,
    )
    return ProteinPointCloudConfig(
        name="protein_point_cloud",
        network=network,
        num_points=32,  # 8 residues * 4 atoms
        point_dim=3,
        num_residues=8,
        num_atoms_per_residue=4,
        backbone_indices=(0, 1, 2, 3),
        use_constraints=True,
        constraint_config=constraint_config,
    )


@pytest.fixture
def protein_data():
    """Fixture for synthetic protein data."""
    batch_size = 2
    num_residues = 8
    num_atoms = 4

    # Use jax random for consistency
    key = jax.random.PRNGKey(42)
    pos_key, mask_key, aa_key = jax.random.split(key, 3)

    # Generate random positions
    shape = (batch_size, num_residues, num_atoms, 3)
    positions = jax.random.normal(pos_key, shape=shape)

    # All atoms are valid
    atom_mask = jnp.ones((batch_size, num_residues, num_atoms))

    # Random amino acid types (0-19 for 20 standard amino acids)
    aatype = jax.random.randint(aa_key, shape=(batch_size, num_residues), minval=0, maxval=20)

    return {
        "atom_positions": positions,
        "atom_mask": atom_mask,
        "aatype": aatype,
    }


def test_protein_point_cloud_init(model_config):
    """Test model initialization."""
    # Create RNG keys for initialization
    key = jax.random.PRNGKey(0)
    key, dropout_key = jax.random.split(key)
    rngs = nnx.Rngs(params=key, dropout=dropout_key)

    # Initialize model
    model = ProteinPointCloudModel(model_config, rngs=rngs)

    # Verify attributes were set correctly
    assert model.num_residues == model_config.num_residues
    assert model.num_atoms == model_config.num_atoms_per_residue
    assert tuple(model.backbone_indices) == model_config.backbone_indices

    # Verify the num_points was set to match the protein dimensions
    expected_points = model_config.num_residues * model_config.num_atoms_per_residue
    assert model.num_points == expected_points

    # Check that constraints were created
    assert hasattr(model, "backbone_constraint")
    assert hasattr(model, "dihedral_constraint")
    assert model.backbone_constraint is not None
    assert model.dihedral_constraint is not None


def test_protein_point_cloud_call(model_config, protein_data):
    """Test forward pass through the model."""
    # Create RNG keys
    key = jax.random.PRNGKey(0)
    key, dropout_key = jax.random.split(key)
    rngs = nnx.Rngs(params=key, dropout=dropout_key)

    # Initialize model
    model = ProteinPointCloudModel(model_config, rngs=rngs)

    # Run forward pass with deterministic=True to avoid dropout
    outputs = model(protein_data, deterministic=True)

    # Check that outputs have the expected structure
    assert "positions" in outputs
    assert "embeddings" in outputs

    # Check shapes
    batch_size = protein_data["atom_positions"].shape[0]
    expected_shape = (batch_size, model.num_residues, model.num_atoms, 3)
    assert outputs["positions"].shape == expected_shape

    # Check that extension outputs are included
    assert "extension_outputs" in outputs


def test_protein_point_cloud_sample(model_config):
    """Test sampling from the model."""
    # Create RNG keys
    key = jax.random.PRNGKey(0)
    key, dropout_key, sample_key = jax.random.split(key, 3)
    # Include params and sample keys for complete RNG setup
    rngs = nnx.Rngs(params=key, dropout=dropout_key, sample=sample_key)

    # Initialize model
    model = ProteinPointCloudModel(model_config, rngs=rngs)

    # Test sample method
    n_samples = 3
    samples = model.sample(n_samples=n_samples, rngs=rngs)

    # Check shape
    assert samples.shape == (n_samples, model.num_residues, model.num_atoms, 3)

    # Test with a different RNG key to ensure it's used
    new_key = jax.random.PRNGKey(42)
    # Include both params and sample in the new RNG
    new_rngs = nnx.Rngs(params=new_key, sample=new_key)

    # Sample again with new RNG
    new_samples = model.sample(n_samples=n_samples, rngs=new_rngs)

    # Verify different samples are generated with different keys
    assert not jnp.allclose(samples, new_samples)


def test_protein_point_cloud_generate(model_config):
    """Test the generate method, which should be an alias for sample."""
    # Create RNG keys
    key = jax.random.PRNGKey(0)
    key, dropout_key, generate_key = jax.random.split(key, 3)
    # Include params key for the model to use
    rngs = nnx.Rngs(params=key, dropout=dropout_key, sample=generate_key)

    # Initialize model
    model = ProteinPointCloudModel(model_config, rngs=rngs)

    # Test generate method
    batch_size = 4
    generated = model.generate(batch_size=batch_size, rngs=rngs)

    # Check shape
    assert generated.shape == (batch_size, model.num_residues, model.num_atoms, 3)

    # Verify generate method's output shape matches sample method's output shape
    # with the same batch size
    gen_batch_size = 2
    gen_output = model.generate(batch_size=gen_batch_size, rngs=rngs)
    sample_output = model.sample(n_samples=gen_batch_size, rngs=rngs)

    # Verify both outputs have the same shape
    assert gen_output.shape == sample_output.shape

    # Verify shape matches expectations
    expected_shape = (gen_batch_size, model.num_residues, model.num_atoms, 3)
    assert gen_output.shape == expected_shape
    assert sample_output.shape == expected_shape


def test_amino_acid_encoding(model_config):
    """Test the amino acid encoding functionality."""
    # Create RNG keys
    key = jax.random.PRNGKey(0)
    key, dropout_key = jax.random.split(key)
    rngs = nnx.Rngs(params=key, dropout=dropout_key)

    # Initialize model
    model = ProteinPointCloudModel(model_config, rngs=rngs)

    # Create mock amino acid data
    batch_size = 2
    aatype = jax.random.randint(key, shape=(batch_size, model.num_residues), minval=0, maxval=20)

    # Encode amino acids
    encoded = model._encode_aatype(aatype)

    # Check shape
    assert encoded.shape == (batch_size, model.num_residues, model.embed_dim)

    # Feed amino acid data through the model
    inputs = {"aatype": aatype}
    outputs = model(inputs, deterministic=True)

    # Verify output structure includes positions
    assert "positions" in outputs
    # Get actual batch size from outputs
    actual_batch_size = outputs["positions"].shape[0]
    shape = (actual_batch_size, model.num_residues, model.num_atoms, 3)
    assert outputs["positions"].shape == shape


def test_loss_function(model_config, protein_data):
    """Test the loss function."""
    # Create RNG keys
    key = jax.random.PRNGKey(0)
    key, dropout_key = jax.random.split(key)
    rngs = nnx.Rngs(params=key, dropout=dropout_key)

    # Initialize model
    model = ProteinPointCloudModel(model_config, rngs=rngs)

    # Get outputs
    outputs = model(protein_data, deterministic=True)

    # Make sure the output and input shapes match for the loss function
    # Because the test fixture uses atom_positions but model might produce positions
    if "positions" in outputs and "atom_positions" in protein_data:
        # Make a copy to avoid modifying the original
        test_data = dict(protein_data)
        test_data["positions"] = test_data.pop("atom_positions")
        loss_dict = model.loss_fn(test_data, outputs)
    else:
        loss_dict = model.loss_fn(protein_data, outputs)

    # Verify loss structure
    assert "mse_loss" in loss_dict
    assert "total_loss" in loss_dict

    # Check if constraints are included in the loss
    if model.backbone_constraint:
        constraint_names = ["proteinbackboneconstraint", "protein_backbone"]
        assert any(name in loss_dict for name in constraint_names)

    if model.dihedral_constraint:
        constraint_names = ["proteindihedralconstraint", "protein_dihedral"]
        assert any(name in loss_dict for name in constraint_names)
