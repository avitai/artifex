"""Tests for protein-specific loss functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from artifex.generative_models.modalities.protein.losses import (
    CompositeLoss,
    create_backbone_loss,
    create_dihedral_loss,
    create_protein_structure_loss,
    create_rmsd_loss,
    LossRegistry,
)


@pytest.fixture
def protein_batch():
    """Fixture for synthetic protein batch data."""
    batch_size = 2
    num_residues = 10
    num_atoms = 4

    # Create random atom positions
    rng = np.random.RandomState(42)
    atom_positions = rng.normal(size=(batch_size, num_residues, num_atoms, 3)).astype(np.float32)

    # All atoms are valid
    atom_mask = np.ones((batch_size, num_residues, num_atoms))

    # Create batch data
    batch = {
        "atom_positions": jnp.array(atom_positions),
        "atom_mask": jnp.array(atom_mask),
    }

    # Create model outputs (slightly perturbed positions)
    outputs = {
        "atom_positions": jnp.array(atom_positions)
        + jnp.array(rng.normal(scale=0.1, size=atom_positions.shape).astype(np.float32)),
    }

    return batch, outputs


def test_rmsd_loss(protein_batch):
    """Test RMSD loss function."""
    batch, outputs = protein_batch

    # Create RMSD loss function
    rmsd_loss_fn = create_rmsd_loss()

    # Calculate loss
    loss = rmsd_loss_fn(batch, outputs)

    # Check that loss value is reasonable
    assert jnp.isfinite(loss)
    assert loss >= 0.0
    assert loss.ndim == 0  # scalar

    # Test with missing atom positions
    empty_batch = {}
    empty_outputs = {}

    # Should return zero if positions not available
    assert rmsd_loss_fn(empty_batch, empty_outputs) == 0.0


def test_backbone_loss(protein_batch):
    """Test backbone geometry loss function."""
    batch, outputs = protein_batch

    # Create backbone loss function
    backbone_loss_fn = create_backbone_loss()

    # Calculate loss
    loss = backbone_loss_fn(batch, outputs)

    # Check that loss value is reasonable
    assert jnp.isfinite(loss)
    assert loss >= 0.0
    assert loss.ndim == 0  # scalar

    # Test with missing atom positions
    empty_batch = {}
    empty_outputs = {}

    # Should return zero if positions not available
    assert backbone_loss_fn(empty_batch, empty_outputs) == 0.0


def test_dihedral_loss(protein_batch):
    """Test dihedral angle loss function."""
    batch, outputs = protein_batch

    # Create dihedral loss function
    dihedral_loss_fn = create_dihedral_loss()

    # Calculate loss
    loss = dihedral_loss_fn(batch, outputs)

    # Check that loss value is reasonable
    assert jnp.isfinite(loss)
    assert loss >= 0.0
    assert loss.ndim == 0  # scalar


def test_composite_loss(protein_batch):
    """Test composite loss function."""
    batch, outputs = protein_batch

    # Create individual loss functions
    rmsd_loss_fn = create_rmsd_loss()
    backbone_loss_fn = create_backbone_loss()

    # Create composite loss
    loss_terms = {
        "rmsd": (rmsd_loss_fn, 1.0),
        "backbone": (backbone_loss_fn, 0.5),
    }

    composite_loss = CompositeLoss(loss_terms)

    # Calculate losses
    losses = composite_loss(batch, outputs)

    # Check that losses have the expected structure
    assert "rmsd" in losses
    assert "backbone" in losses
    assert "total" in losses

    # Check that loss values are reasonable
    for name, loss in losses.items():
        assert jnp.isfinite(loss)
        assert loss >= 0.0
        assert loss.ndim == 0  # scalar

    # Check that total loss is weighted sum of individual losses
    expected_total = losses["rmsd"] + losses["backbone"]
    assert jnp.allclose(losses["total"], expected_total)


def test_create_protein_structure_loss():
    """Test creation of protein structure loss."""
    # Create protein structure loss with default weights
    loss = create_protein_structure_loss()

    # Check that it's a CompositeLoss instance
    assert isinstance(loss, CompositeLoss)

    # Check that it has the expected loss terms
    assert "rmsd" in loss.loss_terms
    assert "backbone" in loss.loss_terms
    assert "dihedral" in loss.loss_terms

    # Check weights
    assert loss.loss_terms["rmsd"][1] == 1.0
    assert loss.loss_terms["backbone"][1] == 0.5
    assert loss.loss_terms["dihedral"][1] == 0.3

    # Create with custom weights
    custom_loss = create_protein_structure_loss(
        rmsd_weight=2.0,
        backbone_weight=1.0,
        dihedral_weight=0.5,
    )

    # Check custom weights
    assert custom_loss.loss_terms["rmsd"][1] == 2.0
    assert custom_loss.loss_terms["backbone"][1] == 1.0
    assert custom_loss.loss_terms["dihedral"][1] == 0.5


def test_loss_registry():
    """Test the loss registry."""
    # Get losses from registry
    rmsd_loss = LossRegistry.get_loss("rmsd")
    backbone_loss = LossRegistry.get_loss("backbone")
    dihedral_loss = LossRegistry.get_loss("dihedral")
    protein_structure_loss = LossRegistry.get_loss("protein_structure")

    # Check types
    assert callable(rmsd_loss)
    assert callable(backbone_loss)
    assert callable(dihedral_loss)
    assert isinstance(protein_structure_loss, CompositeLoss)

    # Register a new loss function
    def dummy_loss_fn(batch, outputs, **kwargs):
        return jnp.array(1.0)

    LossRegistry.register_loss("dummy", lambda: dummy_loss_fn)

    # Get the registered loss
    dummy_loss = LossRegistry.get_loss("dummy")
    assert callable(dummy_loss)

    # Check that it returns the expected value
    assert dummy_loss({}, {}) == 1.0

    # Register a composite loss
    def create_dummy_composite():
        return CompositeLoss({"dummy": (dummy_loss_fn, 1.0)})

    LossRegistry.register_composite_loss("dummy_composite", create_dummy_composite)

    # Get the registered composite loss
    dummy_composite = LossRegistry.get_loss("dummy_composite")
    assert isinstance(dummy_composite, CompositeLoss)

    # Test with unknown loss name
    with pytest.raises(ValueError):
        LossRegistry.get_loss("unknown_loss")
