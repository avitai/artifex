"""Tests for protein visualization utilities."""

import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib.figure import Figure

from artifex.visualization.protein_viz import ProteinVisualizer


@pytest.fixture
def protein_structure():
    """Fixture for synthetic protein structure data."""
    num_residues = 10
    num_atoms = 4

    # Create random atom positions with a helical pattern
    positions = np.zeros((num_residues, num_atoms, 3))

    # Create a basic helical pattern for the backbone
    for i in range(num_residues):
        # Backbone atoms (N, CA, C, O) following a rough helical pattern
        t = i * 2 * np.pi / 3.6  # ~3.6 residues per turn
        r = 2.0  # Helix radius
        z = i * 1.5  # Rise per residue

        # Place N atom
        positions[i, 0] = [r * np.cos(t), r * np.sin(t), z]

        # Place CA atom slightly offset
        positions[i, 1] = [r * np.cos(t) + 0.5, r * np.sin(t) + 0.5, z + 0.5]

        # Place C atom
        positions[i, 2] = [r * np.cos(t + 0.5), r * np.sin(t + 0.5), z + 1.0]

        # Place O atom
        positions[i, 3] = [r * np.cos(t + 0.5) + 0.5, r * np.sin(t + 0.5) + 0.5, z + 1.0]

    # All atoms are valid
    mask = np.ones((num_residues, num_atoms))

    # Set amino acid types (all alanine)
    aatype = np.zeros(num_residues, dtype=np.int32)

    return {
        "atom_positions": positions,
        "atom_mask": mask,
        "aatype": aatype,
    }


def test_to_pdb_string(protein_structure):
    """Test conversion of atom positions to PDB string."""
    # Convert structure to PDB string
    pdb_string = ProteinVisualizer.to_pdb_string(
        atom_positions=protein_structure["atom_positions"],
        atom_mask=protein_structure["atom_mask"],
        aatype=protein_structure["aatype"],
    )

    # Check that the PDB string is valid
    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0

    # Check for expected PDB format
    assert pdb_string.startswith("MODEL     1")
    assert "ATOM  " in pdb_string
    assert pdb_string.endswith("END")

    # Check number of atoms
    num_atoms = (
        protein_structure["atom_positions"].shape[0] * protein_structure["atom_positions"].shape[1]
    )
    atom_lines = [line for line in pdb_string.split("\n") if line.startswith("ATOM")]
    assert len(atom_lines) == num_atoms


def test_calculate_dihedral_angles(protein_structure):
    """Test calculation of dihedral angles."""
    # Calculate dihedral angles
    phi, psi = ProteinVisualizer.calculate_dihedral_angles(protein_structure["atom_positions"])

    # Check shapes
    num_residues = protein_structure["atom_positions"].shape[0]
    assert phi.shape == (num_residues,)
    assert psi.shape == (num_residues,)

    # First residue should have NaN phi angle
    assert np.isnan(phi[0])

    # Last residue should have NaN psi angle
    assert np.isnan(psi[-1])

    # Middle residues should have valid angles
    for i in range(1, num_residues - 1):
        assert not np.isnan(phi[i])
        assert not np.isnan(psi[i])
        assert -np.pi <= phi[i] <= np.pi
        assert -np.pi <= psi[i] <= np.pi

    # Test with JAX array
    jax_positions = jnp.array(protein_structure["atom_positions"])
    phi_jax, psi_jax = ProteinVisualizer.calculate_dihedral_angles(jax_positions)

    # Results should be the same (within numerical precision)
    np.testing.assert_allclose(
        np.array(phi_jax)[~np.isnan(phi)],
        phi[~np.isnan(phi)],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(psi_jax)[~np.isnan(psi)],
        psi[~np.isnan(psi)],
        rtol=1e-5,
    )


def test_plot_ramachandran():
    """Test creation of Ramachandran plot."""
    # Create some synthetic phi/psi angles
    num_residues = 20
    rng = np.random.RandomState(42)

    # Generate angles in typical regions
    phi = rng.uniform(-np.pi, 0, num_residues).astype(np.float32)
    psi = rng.uniform(-np.pi / 2, np.pi / 2, num_residues).astype(np.float32)

    # Create plot
    fig = ProteinVisualizer.plot_ramachandran(phi, psi)

    # Check that the plot is valid
    assert isinstance(fig, Figure)

    # Test with highlight_outliers=True
    fig_with_outliers = ProteinVisualizer.plot_ramachandran(phi, psi, highlight_outliers=True)
    assert isinstance(fig_with_outliers, Figure)

    # Test with residue_indices
    residue_indices = np.arange(num_residues)
    fig_with_indices = ProteinVisualizer.plot_ramachandran(
        phi, psi, residue_indices=residue_indices
    )
    assert isinstance(fig_with_indices, Figure)


def test_visualize_protein_structure(protein_structure):
    """Test 2D visualization of protein structure."""
    # Create visualization
    fig = ProteinVisualizer.visualize_protein_structure(protein_structure["atom_positions"])

    # Check that the plot is valid
    assert isinstance(fig, Figure)

    # Check that the figure has 4 axes (X-Y, X-Z, Y-Z projections + colorbar)
    assert len(fig.axes) == 4

    # Test with JAX array
    jax_positions = jnp.array(protein_structure["atom_positions"])
    fig_jax = ProteinVisualizer.visualize_protein_structure(jax_positions)
    assert isinstance(fig_jax, Figure)
