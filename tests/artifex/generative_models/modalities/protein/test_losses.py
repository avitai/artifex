"""Tests for protein-specific loss functions."""

import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from artifex.generative_models.modalities.protein.losses import (
    create_backbone_loss,
    create_dihedral_loss,
    create_protein_structure_loss,
    create_rmsd_loss,
)


@pytest.fixture
def protein_batch():
    """Fixture for synthetic protein batch data."""
    batch_size = 2
    num_residues = 10
    num_atoms = 4

    rng = np.random.RandomState(42)
    atom_positions = rng.normal(size=(batch_size, num_residues, num_atoms, 3)).astype(np.float32)
    atom_mask = np.ones((batch_size, num_residues, num_atoms))

    batch = {
        "atom_positions": jnp.array(atom_positions),
        "atom_mask": jnp.array(atom_mask),
    }
    outputs = {
        "atom_positions": jnp.array(atom_positions)
        + jnp.array(rng.normal(scale=0.1, size=atom_positions.shape).astype(np.float32)),
    }
    return batch, outputs


def test_rmsd_loss(protein_batch):
    """RMSD loss should return a finite scalar."""
    batch, outputs = protein_batch
    rmsd_loss_fn = create_rmsd_loss()

    loss = rmsd_loss_fn(batch, outputs)

    assert jnp.isfinite(loss)
    assert loss >= 0.0
    assert loss.ndim == 0


def test_rmsd_loss_requires_atom_positions():
    """Protein losses should fail loudly when required coordinates are missing."""
    rmsd_loss_fn = create_rmsd_loss()

    with pytest.raises(ValueError, match="atom positions"):
        rmsd_loss_fn({}, {})


def test_backbone_loss(protein_batch):
    """Backbone geometry loss should return a finite scalar."""
    batch, outputs = protein_batch
    backbone_loss_fn = create_backbone_loss()

    loss = backbone_loss_fn(batch, outputs)

    assert jnp.isfinite(loss)
    assert loss >= 0.0
    assert loss.ndim == 0


def test_backbone_loss_requires_atom_positions():
    """Backbone geometry loss requires protein coordinates."""
    backbone_loss_fn = create_backbone_loss()

    with pytest.raises(ValueError, match="atom positions"):
        backbone_loss_fn({}, {})


def test_dihedral_loss(protein_batch):
    """Dihedral loss should return a finite scalar."""
    batch, outputs = protein_batch
    dihedral_loss_fn = create_dihedral_loss()

    loss = dihedral_loss_fn(batch, outputs)

    assert jnp.isfinite(loss)
    assert loss >= 0.0
    assert loss.ndim == 0


def test_dihedral_loss_requires_atom_positions():
    """Dihedral loss requires protein coordinates."""
    dihedral_loss_fn = create_dihedral_loss()

    with pytest.raises(ValueError, match="atom positions"):
        dihedral_loss_fn({}, {})


def test_create_protein_structure_loss(protein_batch):
    """Protein structure loss should expose a canonical loss dict."""
    batch, outputs = protein_batch
    loss_fn = create_protein_structure_loss(
        rmsd_weight=2.0,
        backbone_weight=1.0,
        dihedral_weight=0.5,
    )

    losses = loss_fn(batch, outputs)

    assert set(losses) == {"total_loss", "rmsd_loss", "backbone_loss", "dihedral_loss"}
    for loss in losses.values():
        assert jnp.isfinite(loss)
        assert loss.ndim == 0

    expected_total = (
        2.0 * losses["rmsd_loss"] + 1.0 * losses["backbone_loss"] + 0.5 * losses["dihedral_loss"]
    )
    assert jnp.allclose(losses["total_loss"], expected_total)


def test_protein_losses_do_not_export_local_registry_or_composite_framework():
    """Protein losses should not duplicate the shared composition framework."""
    losses_module = importlib.import_module("artifex.generative_models.modalities.protein.losses")

    assert not hasattr(losses_module, "CompositeLoss")
    assert not hasattr(losses_module, "LossRegistry")
