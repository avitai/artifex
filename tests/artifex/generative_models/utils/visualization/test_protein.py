"""Tests for the protein visualization utilities."""

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from artifex.generative_models.utils.visualization.protein import (
    ProteinVisualizer,
)


@pytest.fixture
def dummy_protein_data():
    """Generate dummy protein data for visualization testing."""
    # Create a small protein structure with 3 residues and 4 backbone atoms
    batch_size = 1
    n_residues = 3
    n_atoms = 4  # N, CA, C, O

    # Create coordinates in a simple line
    coords = np.zeros((batch_size, n_residues, n_atoms, 3))
    for i in range(n_residues):
        # N atom
        coords[0, i, 0] = np.array([i * 3.8, 0.0, 0.0])
        # CA atom, 1.46Å from N
        coords[0, i, 1] = np.array([i * 3.8 + 1.46, 0.0, 0.0])
        # C atom, 1.52Å from CA
        coords[0, i, 2] = np.array([i * 3.8 + 1.46 + 1.52, 0.0, 0.0])
        # O atom, 1.23Å from C at ~120°
        coords[0, i, 3] = np.array([i * 3.8 + 1.46 + 1.52, 1.23, 0.0])

    # Create atom mask (all atoms present)
    mask = np.ones((batch_size, n_residues, n_atoms))

    # Create amino acid types (ALA, CYS, GLY)
    aatype = np.array([[1, 3, 7]])

    return {
        "coords": coords,
        "mask": mask,
        "aatype": aatype,
    }


@pytest.fixture
def jax_protein_data(dummy_protein_data):
    """Convert dummy protein data to JAX arrays."""
    return {
        "coords": jnp.array(dummy_protein_data["coords"]),
        "mask": jnp.array(dummy_protein_data["mask"]),
        "aatype": jnp.array(dummy_protein_data["aatype"]),
    }


class MockPy3Dmol:
    """Mock class for py3Dmol."""

    # Add VDW attribute for surface test
    VDW = "vdw"

    def view(self, width=None, height=None):
        """Mock view method."""
        mock_view = MagicMock()
        mock_view.addModel = MagicMock(return_value=mock_view)
        mock_view.setStyle = MagicMock(return_value=mock_view)
        mock_view.setBackgroundColor = MagicMock(return_value=mock_view)
        mock_view.zoomTo = MagicMock(return_value=mock_view)
        mock_view.addSurface = MagicMock(return_value=mock_view)
        return mock_view


class MockNGLView:
    """Mock class for nglview."""

    @staticmethod
    def show_file(filename):
        """Mock show_file method."""
        mock_view = MagicMock()
        mock_view.clear_representations = MagicMock()
        mock_view.add_representation = MagicMock()
        mock_view._remote_call = MagicMock()
        return mock_view


class TestProteinVisualizer:
    """Tests for the ProteinVisualizer class."""

    def test_initialization_py3dmol(self):
        """Test initialization with py3dmol backend."""
        with patch.dict("sys.modules", {"py3Dmol": MockPy3Dmol()}):
            visualizer = ProteinVisualizer(backend="py3dmol")
            assert visualizer.backend == "py3dmol"
            assert hasattr(visualizer, "_py3Dmol")

    def test_initialization_nglview(self):
        """Test initialization with nglview backend."""
        with patch.dict("sys.modules", {"nglview": MockNGLView()}):
            visualizer = ProteinVisualizer(backend="nglview")
            assert visualizer.backend == "nglview"
            assert hasattr(visualizer, "_nglview")

    def test_initialization_invalid_backend(self):
        """Test initialization with invalid backend."""
        with pytest.raises(ValueError):
            ProteinVisualizer(backend="invalid_backend")

    def test_coords_to_pdb_numpy(self, dummy_protein_data):
        """Test converting numpy coordinates to PDB string."""
        visualizer = ProteinVisualizer()

        # Extract coordinates and mask
        coords = dummy_protein_data["coords"]
        mask = dummy_protein_data["mask"]
        aatype = dummy_protein_data["aatype"]

        # Convert to PDB
        pdb_str = visualizer.coords_to_pdb(coords, mask, aatype)

        # Verify PDB string
        assert isinstance(pdb_str, str)
        assert "ATOM" in pdb_str
        assert "MODEL" in pdb_str
        assert "END" in pdb_str

        # Check number of atom lines
        atom_lines = [line for line in pdb_str.split("\n") if line.startswith("ATOM")]
        expected_atom_count = np.sum(mask).astype(int)
        assert len(atom_lines) == expected_atom_count

    def test_coords_to_pdb_jax(self, jax_protein_data):
        """Test converting JAX coordinates to PDB string."""
        visualizer = ProteinVisualizer()

        # Extract coordinates and mask
        coords = jax_protein_data["coords"]
        mask = jax_protein_data["mask"]
        aatype = jax_protein_data["aatype"]

        # Convert to PDB
        pdb_str = visualizer.coords_to_pdb(coords, mask, aatype)

        # Verify PDB string
        assert isinstance(pdb_str, str)
        assert "ATOM" in pdb_str
        assert "MODEL" in pdb_str
        assert "END" in pdb_str

    def test_coords_to_pdb_no_mask(self, dummy_protein_data):
        """Test converting coordinates to PDB string without mask."""
        visualizer = ProteinVisualizer()

        # Extract coordinates only
        coords = dummy_protein_data["coords"]

        # Convert to PDB
        pdb_str = visualizer.coords_to_pdb(coords)

        # Verify PDB string
        assert isinstance(pdb_str, str)
        assert "ATOM" in pdb_str
        assert "MODEL" in pdb_str
        assert "END" in pdb_str

    def test_coords_to_pdb_no_aatype(self, dummy_protein_data):
        """Test converting coordinates to PDB string without AA types."""
        visualizer = ProteinVisualizer()

        # Extract coordinates and mask
        coords = dummy_protein_data["coords"]
        mask = dummy_protein_data["mask"]

        # Convert to PDB
        pdb_str = visualizer.coords_to_pdb(coords, mask)

        # Verify PDB string
        assert isinstance(pdb_str, str)
        assert "ATOM" in pdb_str
        assert "MODEL" in pdb_str
        assert "END" in pdb_str

        # All residues should be GLY
        for line in pdb_str.split("\n"):
            if line.startswith("ATOM"):
                assert "GLY" in line

    def test_coords_to_pdb_batch_dimension(self, dummy_protein_data):
        """Test converting multi-batch coordinates to PDB string."""
        visualizer = ProteinVisualizer()

        # Extract coordinates and duplicate to create multiple batches
        coords = dummy_protein_data["coords"]
        # Shape (2, n_res, n_atoms, 3)
        multi_batch_coords = np.vstack([coords, coords])

        # Convert to PDB (should use first batch only)
        pdb_str = visualizer.coords_to_pdb(multi_batch_coords)

        # Verify PDB string has same number of atoms as single batch
        atom_lines = [line for line in pdb_str.split("\n") if line.startswith("ATOM")]
        # n_res * n_atoms
        expected_atom_count = coords.shape[1] * coords.shape[2]
        assert len(atom_lines) == expected_atom_count

    def test_export_to_pdb(self, dummy_protein_data, tmp_path):
        """Test exporting coordinates to PDB file."""
        visualizer = ProteinVisualizer()

        # Create output path in temporary directory
        output_path = tmp_path / "test_protein.pdb"

        # Export to PDB
        visualizer.export_to_pdb(dummy_protein_data, output_path)

        # Verify file was created
        assert output_path.exists()

        # Verify file content
        with open(output_path, "r") as f:
            content = f.read()
            assert "ATOM" in content
            assert "MODEL" in content
            assert "END" in content

    @patch(
        "artifex.generative_models.utils.visualization.protein.ProteinVisualizer._visualize_py3dmol"
    )
    def test_visualize_py3dmol(self, mock_visualize, dummy_protein_data):
        """Test visualize method with py3dmol backend."""
        # Setup mock return value
        mock_view = MagicMock()
        mock_visualize.return_value = mock_view

        # Create visualizer
        visualizer = ProteinVisualizer(backend="py3dmol")

        # Call visualize
        result = visualizer.visualize(
            dummy_protein_data,
            style="cartoon",
            color_scheme="chainname",
            width=800,
            height=600,
        )

        # Verify mock was called correctly
        mock_visualize.assert_called_once_with(
            dummy_protein_data,
            "cartoon",
            "chainname",
            800,
            600,
        )

        # Verify result
        assert result == mock_view

    @patch(
        "artifex.generative_models.utils.visualization.protein.ProteinVisualizer._visualize_nglview"
    )
    def test_visualize_nglview(self, mock_visualize, dummy_protein_data):
        """Test visualize method with nglview backend."""
        # Setup mock return value
        mock_view = MagicMock()
        mock_visualize.return_value = mock_view

        # Create visualizer
        visualizer = ProteinVisualizer(backend="nglview")

        # Call visualize
        result = visualizer.visualize(
            dummy_protein_data,
            style="ball_and_stick",
            color_scheme="element",
            width=600,
            height=400,
        )

        # Verify mock was called correctly
        mock_visualize.assert_called_once_with(
            dummy_protein_data,
            "ball_and_stick",
            "element",
            600,
            400,
        )

        # Verify result
        assert result == mock_view

    def test_visualize_invalid_backend(self, dummy_protein_data):
        """Test visualize method with invalid backend."""
        # Create visualizer with invalid backend attribute
        visualizer = ProteinVisualizer(backend="py3dmol")
        visualizer.backend = "invalid_backend"  # Force invalid backend

        # Call visualize
        with pytest.raises(ValueError):
            visualizer.visualize(dummy_protein_data)

    def test_visualize_py3dmol_implementation(self, dummy_protein_data):
        """Test _visualize_py3dmol implementation."""
        # Create the mocks directly
        mock_view = MagicMock()
        mock_view.addModel = MagicMock(return_value=mock_view)
        mock_view.setStyle = MagicMock(return_value=mock_view)
        mock_view.setBackgroundColor = MagicMock(return_value=mock_view)
        mock_view.zoomTo = MagicMock(return_value=mock_view)
        mock_view.addSurface = MagicMock(return_value=mock_view)

        # Create a mock py3Dmol class that returns our mock view
        mock_py3dmol = MagicMock()
        mock_py3dmol.view = MagicMock(return_value=mock_view)
        mock_py3dmol.VDW = "vdw"

        # Create visualizer and inject our mock
        visualizer = ProteinVisualizer()
        visualizer._py3Dmol = mock_py3dmol
        visualizer.backend = "py3dmol"

        # Call _visualize_py3dmol
        view = visualizer._visualize_py3dmol(
            dummy_protein_data,
            style="cartoon",
            color_scheme="chainname",
        )

        # Verify view methods were called correctly
        mock_view.addModel.assert_called_once()
        mock_view.setStyle.assert_called_once()
        mock_view.setBackgroundColor.assert_called_once()
        mock_view.zoomTo.assert_called_once()

        # Verify returned view is the mock view we created
        assert view == mock_view

    def test_visualize_py3dmol_with_surface(self, dummy_protein_data):
        """Test _visualize_py3dmol with surface option."""
        # Create the mocks directly
        mock_view = MagicMock()
        mock_view.addModel = MagicMock(return_value=mock_view)
        mock_view.setStyle = MagicMock(return_value=mock_view)
        mock_view.setBackgroundColor = MagicMock(return_value=mock_view)
        mock_view.zoomTo = MagicMock(return_value=mock_view)
        mock_view.addSurface = MagicMock(return_value=mock_view)

        # Create a mock py3Dmol class that returns our mock view
        mock_py3dmol = MagicMock()
        mock_py3dmol.view = MagicMock(return_value=mock_view)
        mock_py3dmol.VDW = "vdw"

        # Create visualizer and inject our mock
        visualizer = ProteinVisualizer()
        visualizer._py3Dmol = mock_py3dmol
        visualizer.backend = "py3dmol"

        # Call _visualize_py3dmol with surface=True
        view = visualizer._visualize_py3dmol(
            dummy_protein_data,
            style="cartoon",
            color_scheme="chainname",
            surface=True,
        )

        # Verify addSurface method was called
        mock_view.addSurface.assert_called_once()

        # Verify returned view is the mock view we created
        assert view == mock_view

    @patch("tempfile.NamedTemporaryFile")
    def test_visualize_nglview_implementation(self, mock_tempfile, dummy_protein_data):
        """Test _visualize_nglview implementation."""
        # Setup mock temporary file
        mock_tmp = MagicMock()
        mock_tmp.name = "temp.pdb"
        mock_tempfile.return_value.__enter__.return_value = mock_tmp

        # Create mock view and methods
        mock_view = MagicMock()
        mock_view.clear_representations = MagicMock()
        mock_view.add_representation = MagicMock()
        mock_view._remote_call = MagicMock()

        # Create mock nglview module
        mock_nglview = MagicMock()
        mock_nglview.show_file = MagicMock(return_value=mock_view)

        # Create visualizer and inject our mock
        visualizer = ProteinVisualizer()
        visualizer._nglview = mock_nglview
        visualizer.backend = "nglview"

        # Call _visualize_nglview
        result = visualizer._visualize_nglview(
            dummy_protein_data,
            style="cartoon",
            color_scheme="chainname",
            width=800,
            height=600,
        )

        # Verify temporary file was written
        mock_tmp.write.assert_called_once()

        # Verify view methods were called correctly
        mock_view.clear_representations.assert_called_once()
        mock_view.add_representation.assert_called_once()
        mock_view._remote_call.assert_called_once()

        # Verify result
        assert result == mock_view
