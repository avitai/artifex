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
    positions = np.zeros((num_residues, num_atoms, 3))

    for i in range(num_residues):
        t = i * 2 * np.pi / 3.6
        r = 2.0
        z = i * 1.5
        positions[i, 0] = [r * np.cos(t), r * np.sin(t), z]
        positions[i, 1] = [r * np.cos(t) + 0.5, r * np.sin(t) + 0.5, z + 0.5]
        positions[i, 2] = [r * np.cos(t + 0.5), r * np.sin(t + 0.5), z + 1.0]
        positions[i, 3] = [
            r * np.cos(t + 0.5) + 0.5,
            r * np.sin(t + 0.5) + 0.5,
            z + 1.0,
        ]

    mask = np.ones((num_residues, num_atoms))
    aatype = np.zeros(num_residues, dtype=np.int32)

    return {
        "atom_positions": positions,
        "atom_mask": mask,
        "aatype": aatype,
    }


@pytest.fixture
def batched_protein_structure(protein_structure):
    """Fixture for a single-item batched protein structure."""
    return {
        "atom_positions": protein_structure["atom_positions"][None, ...],
        "atom_mask": protein_structure["atom_mask"][None, ...],
        "aatype": protein_structure["aatype"][None, ...],
    }


def test_to_pdb_string(protein_structure):
    """Test conversion of atom positions to PDB string."""
    pdb_string = ProteinVisualizer.to_pdb_string(
        atom_positions=protein_structure["atom_positions"],
        atom_mask=protein_structure["atom_mask"],
        aatype=protein_structure["aatype"],
    )

    assert isinstance(pdb_string, str)
    assert len(pdb_string) > 0
    assert pdb_string.startswith("MODEL     1")
    assert "ATOM  " in pdb_string
    assert pdb_string.endswith("END")

    num_atoms = (
        protein_structure["atom_positions"].shape[0] * protein_structure["atom_positions"].shape[1]
    )
    atom_lines = [line for line in pdb_string.split("\n") if line.startswith("ATOM")]
    assert len(atom_lines) == num_atoms


def test_to_pdb_string_accepts_batched_inputs(protein_structure, batched_protein_structure):
    """Top-level visualizer should normalize batched protein payloads."""
    unbatched = ProteinVisualizer.to_pdb_string(
        atom_positions=protein_structure["atom_positions"],
        atom_mask=protein_structure["atom_mask"],
        aatype=protein_structure["aatype"],
    )
    batched = ProteinVisualizer.to_pdb_string(
        atom_positions=batched_protein_structure["atom_positions"],
        atom_mask=batched_protein_structure["atom_mask"],
        aatype=batched_protein_structure["aatype"],
    )

    assert batched == unbatched


def test_coords_to_pdb_alias(protein_structure):
    """Compatibility alias should route to the canonical PDB export path."""
    pdb_string = ProteinVisualizer.coords_to_pdb(
        protein_structure["atom_positions"],
        protein_structure["atom_mask"],
        protein_structure["aatype"],
    )

    assert pdb_string.startswith("MODEL     1")
    assert "ATOM  " in pdb_string


def test_export_to_pdb(protein_structure, tmp_path):
    """ProteinVisualizer should export PDB files from structure dict payloads."""
    output_path = tmp_path / "protein_structure.pdb"

    ProteinVisualizer.export_to_pdb(protein_structure, str(output_path))

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("MODEL     1")
    assert content.endswith("END")


def test_calculate_dihedral_angles(protein_structure, batched_protein_structure):
    """Test calculation of dihedral angles."""
    phi, psi = ProteinVisualizer.calculate_dihedral_angles(protein_structure["atom_positions"])

    num_residues = protein_structure["atom_positions"].shape[0]
    assert phi.shape == (num_residues,)
    assert psi.shape == (num_residues,)
    assert np.isnan(phi[0])
    assert np.isnan(psi[-1])

    for i in range(1, num_residues - 1):
        assert not np.isnan(phi[i])
        assert not np.isnan(psi[i])
        assert -np.pi <= phi[i] <= np.pi
        assert -np.pi <= psi[i] <= np.pi

    jax_positions = jnp.array(protein_structure["atom_positions"])
    phi_jax, psi_jax = ProteinVisualizer.calculate_dihedral_angles(jax_positions)
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

    phi_batched, psi_batched = ProteinVisualizer.calculate_dihedral_angles(
        batched_protein_structure["atom_positions"],
        batched_protein_structure["atom_mask"],
    )
    np.testing.assert_allclose(
        phi_batched[~np.isnan(phi)],
        phi[~np.isnan(phi)],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        psi_batched[~np.isnan(psi)],
        psi[~np.isnan(psi)],
        rtol=1e-5,
    )


def test_plot_ramachandran():
    """Test creation of Ramachandran plot."""
    num_residues = 20
    rng = np.random.RandomState(42)
    phi = rng.uniform(-np.pi, 0, num_residues).astype(np.float32)
    psi = rng.uniform(-np.pi / 2, np.pi / 2, num_residues).astype(np.float32)

    fig = ProteinVisualizer.plot_ramachandran(phi, psi)
    assert isinstance(fig, Figure)

    fig_with_outliers = ProteinVisualizer.plot_ramachandran(phi, psi, highlight_outliers=True)
    assert isinstance(fig_with_outliers, Figure)

    residue_indices = np.arange(num_residues)
    fig_with_indices = ProteinVisualizer.plot_ramachandran(
        phi,
        psi,
        residue_indices=residue_indices,
    )
    assert isinstance(fig_with_indices, Figure)


def test_visualize_protein_structure(protein_structure, batched_protein_structure):
    """Test 2D visualization of protein structure."""
    fig = ProteinVisualizer.visualize_protein_structure(protein_structure["atom_positions"])
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4

    jax_positions = jnp.array(protein_structure["atom_positions"])
    fig_jax = ProteinVisualizer.visualize_protein_structure(jax_positions)
    assert isinstance(fig_jax, Figure)

    fig_batched = ProteinVisualizer.visualize_protein_structure(
        batched_protein_structure["atom_positions"]
    )
    assert isinstance(fig_batched, Figure)
