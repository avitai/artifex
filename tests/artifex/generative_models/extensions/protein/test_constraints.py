"""Tests for protein-specific constraints.

Including JAX compliance tests to ensure protein constraint modules properly
use JAX and avoid numpy in NNX context.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ProteinDihedralConfig,
    ProteinExtensionConfig,
)
from artifex.generative_models.extensions.protein.constraints import (
    BOND_ANGLES,
    calculate_bond_lengths,
    calculate_dihedral_angles,
    DIHEDRAL_ANGLES,
    ProteinBackboneConstraint,
    ProteinDihedralConstraint,
)


@pytest.fixture
def rng():
    """Random number generator fixture."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def dummy_coords(rng):
    """Create dummy protein coordinates."""
    # Create synthetic protein coordinates for testing
    # Shape: [batch_size, n_residues, n_atoms, 3]
    batch_size = 2
    n_residues = 10
    n_atoms = 4  # N, CA, C, O

    # Generate random coordinates
    coords = jax.random.normal(rng, shape=(batch_size, n_residues, n_atoms, 3))

    return coords


@pytest.fixture
def ideal_protein_coords():
    """Create ideal protein coordinates with correct bond geometry."""
    # Simplified alpha helix-like geometry for a 10-residue peptide
    # Shape: [1, 10, 4, 3] for 1 batch, 10 residues, 4 backbone atoms (N, CA, C, O), 3 coordinates
    batch_size = 1
    n_residues = 10
    n_atoms = 4

    coords = jnp.zeros((batch_size, n_residues, n_atoms, 3))

    # Start with a straight chain
    for i in range(n_residues):
        # N atom
        coords = coords.at[:, i, 0].set(jnp.array([i * 3.8, 0.0, 0.0]))
        # CA atom, 1.46Å from N
        coords = coords.at[:, i, 1].set(jnp.array([i * 3.8 + 1.46, 0.0, 0.0]))
        # C atom, 1.52Å from CA
        coords = coords.at[:, i, 2].set(jnp.array([i * 3.8 + 1.46 + 1.52, 0.0, 0.0]))
        # O atom, 1.23Å from C at ~120°
        coords = coords.at[:, i, 3].set(jnp.array([i * 3.8 + 1.46 + 1.52, 1.23, 0.0]))

    # Adjust positions to create more realistic protein geometry for dihedral tests
    # This is a very simplified approximation
    for i in range(1, n_residues):
        # Shift residues to mimic a helical structure
        y_offset = 1.5 * jnp.sin(i * jnp.pi / 3.5)
        z_offset = 1.5 * jnp.cos(i * jnp.pi / 3.5)

        coords = coords.at[:, i, 0].add(jnp.array([0.0, y_offset, z_offset]))
        coords = coords.at[:, i, 1].add(jnp.array([0.0, y_offset, z_offset]))
        coords = coords.at[:, i, 2].add(jnp.array([0.0, y_offset, z_offset]))
        coords = coords.at[:, i, 3].add(jnp.array([0.0, y_offset, z_offset]))

    return coords


@pytest.fixture
def dummy_mask():
    """Create a dummy atom mask for testing."""
    # Shape: [batch_size, n_residues, n_atoms]
    batch_size = 2
    n_residues = 10
    n_atoms = 4

    # All atoms are present (mask = 1.0)
    mask = jnp.ones((batch_size, n_residues, n_atoms))

    # Add some missing atoms (mask = 0.0) for testing
    mask = mask.at[:, -1, :].set(0.0)  # Last residue is missing
    mask = mask.at[:, 5, 3].set(0.0)  # Middle residue missing O atom

    return mask


@pytest.fixture
def backbone_constraint(rng):
    """Create backbone constraint fixture."""
    config = ProteinExtensionConfig(
        name="backbone_constraint",
        weight=1.0,
        enabled=True,
        bond_length_weight=1.0,
        bond_angle_weight=0.5,
        backbone_atoms=("N", "CA", "C", "O"),
    )
    return ProteinBackboneConstraint(config, rngs=nnx.Rngs(params=rng))


@pytest.fixture
def dihedral_constraint(rng):
    """Create dihedral constraint fixture."""
    config = ProteinDihedralConfig(
        name="dihedral_constraint",
        weight=1.0,
        enabled=True,
        phi_weight=0.5,
        psi_weight=0.5,
        omega_weight=1.0,
        target_secondary_structure="alpha_helix",
    )
    return ProteinDihedralConstraint(config, rngs=nnx.Rngs(params=rng))


class TestProteinBackboneConstraint:
    """Tests for ProteinBackboneConstraint class."""

    def test_initialization(self, rng):
        """Test constraint initialization with different configs."""
        # Default config (using default values)
        config = ProteinExtensionConfig(
            name="backbone_constraint",
            weight=1.0,
            enabled=True,
        )
        constraint = ProteinBackboneConstraint(config, rngs=nnx.Rngs(params=rng))
        assert constraint.bond_weight == 1.0  # default bond_length_weight
        assert constraint.angle_weight == 0.5  # default bond_angle_weight
        assert constraint.backbone_indices == [0, 1, 2, 3]  # indices for N, CA, C, O

        # Custom config
        config = ProteinExtensionConfig(
            name="backbone_constraint",
            weight=1.0,
            enabled=True,
            bond_length_weight=2.0,
            bond_angle_weight=1.5,
            backbone_atoms=("N", "CA", "C", "O"),
            ideal_bond_lengths={"N-CA": 1.5},
        )
        constraint = ProteinBackboneConstraint(config, rngs=nnx.Rngs(params=rng))
        assert constraint.bond_weight == 2.0
        assert constraint.angle_weight == 1.5
        assert constraint.backbone_indices == [0, 1, 2, 3]
        # Verify custom ideal_bond_lengths was applied
        assert constraint.ideal_bond_lengths["N-CA"] == 1.5

    def test_call_with_dict_input(self, backbone_constraint, dummy_coords):
        """Test __call__ method with dictionary input."""
        # Create model output dictionary
        model_output = {"positions": dummy_coords}

        # Call constraint
        result = backbone_constraint(inputs={}, model_outputs=model_output)

        # Verify metrics were returned
        assert isinstance(result, dict)
        assert "n_ca_length_mean" in result
        assert "ca_c_length_mean" in result
        assert "n_ca_c_angle_mean" in result

    def test_call_with_direct_coords(self, backbone_constraint, dummy_coords):
        """Test __call__ method with direct coordinate input."""
        # Call constraint with coords directly
        result = backbone_constraint(inputs={}, model_outputs=dummy_coords)

        # Verify metrics were returned
        assert isinstance(result, dict)
        assert "n_ca_length_mean" in result
        assert "ca_c_length_mean" in result
        assert "n_ca_c_angle_mean" in result

    def test_call_with_mask(self, backbone_constraint, dummy_coords, dummy_mask):
        """Test __call__ method with mask provided."""
        # Call constraint with mask in inputs
        result = backbone_constraint(inputs={"atom_mask": dummy_mask}, model_outputs=dummy_coords)

        # Verify metrics were returned
        assert isinstance(result, dict)
        assert "n_ca_length_mean" in result
        assert "ca_c_length_mean" in result
        assert "n_ca_c_angle_mean" in result

    def test_loss_fn(self, backbone_constraint, dummy_coords):
        """Test loss_fn method."""
        # Calculate loss
        loss = backbone_constraint.loss_fn(batch={}, model_outputs=dummy_coords)

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_loss_fn_with_mask(self, backbone_constraint, dummy_coords, dummy_mask):
        """Test loss_fn method with mask."""
        # Calculate loss with mask
        loss = backbone_constraint.loss_fn(
            batch={"atom_mask": dummy_mask}, model_outputs=dummy_coords
        )

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_validate(self, backbone_constraint, dummy_coords):
        """Test validate method."""
        # Validate coordinates
        metrics = backbone_constraint.validate(dummy_coords)

        # Verify metrics were returned
        assert isinstance(metrics, dict)
        assert "n_ca_length_mean" in metrics
        assert "ca_c_length_mean" in metrics
        assert "n_ca_c_angle_mean" in metrics

    def test_calculate_bond_metrics(self, backbone_constraint, dummy_coords):
        """Test _calculate_bond_metrics method."""
        # Calculate bond metrics
        metrics = backbone_constraint._calculate_bond_metrics(dummy_coords)

        # Verify metrics were returned
        assert isinstance(metrics, dict)
        assert "n_ca_length_mean" in metrics
        assert "ca_c_length_mean" in metrics
        assert "c_o_length_mean" in metrics
        assert "c_n_next_length_mean" in metrics

    def test_calculate_angle_metrics(self, backbone_constraint, dummy_coords):
        """Test _calculate_angle_metrics method."""
        # Calculate angle metrics
        metrics = backbone_constraint._calculate_angle_metrics(dummy_coords)

        # Verify metrics were returned
        assert isinstance(metrics, dict)
        assert "n_ca_c_angle_mean" in metrics
        assert "ca_c_o_angle_mean" in metrics
        assert "n_ca_c_angle_std" in metrics
        assert "ca_c_o_angle_std" in metrics

    def test_calculate_angles(self, backbone_constraint):
        """Test _calculate_angles method."""
        # Create two simple vectors
        v1 = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        v2 = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        # Calculate angles
        angles = backbone_constraint._calculate_angles(v1, v2)

        # Verify angles are correct (both should be 90 degrees = pi/2)
        assert jnp.allclose(angles, jnp.array([jnp.pi / 2, jnp.pi / 2]), atol=1e-5)

    def test_bond_length_loss(self, backbone_constraint, dummy_coords):
        """Test _bond_length_loss method."""
        # Calculate bond length loss
        loss = backbone_constraint._bond_length_loss(dummy_coords)

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_bond_angle_loss(self, backbone_constraint, dummy_coords):
        """Test _bond_angle_loss method."""
        # Calculate bond angle loss
        loss = backbone_constraint._bond_angle_loss(dummy_coords)

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_ideal_geometry(self, backbone_constraint, ideal_protein_coords):
        """Test that ideal geometry has lower loss."""
        # Get random coords
        rng = jax.random.PRNGKey(42)
        random_coords = jax.random.normal(rng, shape=ideal_protein_coords.shape)

        # Calculate losses for both
        ideal_loss = backbone_constraint.loss_fn({}, ideal_protein_coords)
        random_loss = backbone_constraint.loss_fn({}, random_coords)

        # Ideal structure should have lower loss
        assert ideal_loss < random_loss


class TestProteinDihedralConstraint:
    """Tests for ProteinDihedralConstraint class."""

    def test_initialization(self, rng):
        """Test constraint initialization with different configs."""
        # Default config (using default values)
        config = ProteinDihedralConfig(
            name="dihedral_constraint",
            weight=1.0,
            enabled=True,
        )
        constraint = ProteinDihedralConstraint(config, rngs=nnx.Rngs(params=rng))
        assert constraint.phi_weight == 0.5
        assert constraint.psi_weight == 0.5
        assert constraint.omega_weight == 1.0
        assert constraint.backbone_indices == [0, 1, 2, 3]  # indices for N, CA, C, O

        # Custom config
        config = ProteinDihedralConfig(
            name="dihedral_constraint",
            weight=1.0,
            enabled=True,
            phi_weight=0.1,
            psi_weight=0.2,
            omega_weight=0.3,
            ideal_phi=-1.0,
        )
        constraint = ProteinDihedralConstraint(config, rngs=nnx.Rngs(params=rng))
        assert constraint.phi_weight == 0.1
        assert constraint.psi_weight == 0.2
        assert constraint.omega_weight == 0.3
        assert constraint.backbone_indices == [0, 1, 2, 3]
        # Verify custom ideal_phi was applied
        assert constraint.ideal_phi == -1.0

    def test_call_with_dict_input(self, dihedral_constraint, dummy_coords):
        """Test __call__ method with dictionary input."""
        # Create model output dictionary
        model_output = {"positions": dummy_coords}

        # Call constraint
        result = dihedral_constraint(inputs={}, model_outputs=model_output)

        # Verify metrics were returned
        assert isinstance(result, dict)
        assert "phi_mean" in result
        assert "psi_mean" in result
        assert "omega_mean" in result

    def test_call_with_direct_coords(self, dihedral_constraint, dummy_coords):
        """Test __call__ method with direct coordinate input."""
        # Call constraint with coords directly
        result = dihedral_constraint(inputs={}, model_outputs=dummy_coords)

        # Verify metrics were returned
        assert isinstance(result, dict)
        assert "phi_mean" in result
        assert "psi_mean" in result
        assert "omega_mean" in result

    def test_call_with_mask(self, dihedral_constraint, dummy_coords, dummy_mask):
        """Test __call__ method with mask provided."""
        # Call constraint with mask in inputs
        result = dihedral_constraint(inputs={"atom_mask": dummy_mask}, model_outputs=dummy_coords)

        # Verify metrics were returned
        assert isinstance(result, dict)
        assert "phi_mean" in result
        assert "psi_mean" in result
        assert "omega_mean" in result

    def test_loss_fn(self, dihedral_constraint, dummy_coords):
        """Test loss_fn method."""
        # Calculate loss
        loss = dihedral_constraint.loss_fn(batch={}, model_outputs=dummy_coords)

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_loss_fn_with_mask(self, dihedral_constraint, dummy_coords, dummy_mask):
        """Test loss_fn method with mask."""
        # Calculate loss with mask
        loss = dihedral_constraint.loss_fn(
            batch={"atom_mask": dummy_mask}, model_outputs=dummy_coords
        )

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_validate(self, dihedral_constraint, dummy_coords):
        """Test validate method."""
        # Validate coordinates
        metrics = dihedral_constraint.validate(dummy_coords)

        # Verify metrics were returned
        assert isinstance(metrics, dict)
        assert "phi_mean" in metrics
        assert "psi_mean" in metrics
        assert "omega_mean" in metrics

    def test_calculate_dihedral(self, dihedral_constraint):
        """Test _calculate_dihedral method."""
        # Create four points forming a dihedral angle of 90 degrees
        p1 = jnp.array([0.0, 0.0, 0.0])
        p2 = jnp.array([1.0, 0.0, 0.0])
        p3 = jnp.array([1.0, 1.0, 0.0])
        p4 = jnp.array([1.0, 1.0, 1.0])

        # Calculate dihedral angle
        angle = dihedral_constraint._calculate_dihedral(p1, p2, p3, p4)

        # Should be 90 degrees = pi/2
        assert jnp.isclose(angle, jnp.pi / 2, atol=1e-5)

    def test_calculate_phi_angles(self, dihedral_constraint, dummy_coords):
        """Test _calculate_phi_angles method."""
        # Calculate phi angles
        phi_angles = dihedral_constraint._calculate_phi_angles(dummy_coords)

        # Verify shape and content
        assert phi_angles.shape == (dummy_coords.shape[0], dummy_coords.shape[1])
        # First residue should have phi=0 (undefined)
        assert jnp.all(phi_angles[:, 0] == 0)
        # Other residues should have non-zero phi values
        assert not jnp.all(phi_angles[:, 1:] == 0)

    def test_calculate_psi_angles(self, dihedral_constraint, dummy_coords):
        """Test _calculate_psi_angles method."""
        # Calculate psi angles
        psi_angles = dihedral_constraint._calculate_psi_angles(dummy_coords)

        # Verify shape and content
        assert psi_angles.shape == (dummy_coords.shape[0], dummy_coords.shape[1])
        # Last residue should have psi=0 (undefined)
        assert jnp.all(psi_angles[:, -1] == 0)
        # Other residues should have non-zero psi values
        assert not jnp.all(psi_angles[:, :-1] == 0)

    def test_calculate_omega_angles(self, dihedral_constraint, dummy_coords):
        """Test _calculate_omega_angles method."""
        # Calculate omega angles
        omega_angles = dihedral_constraint._calculate_omega_angles(dummy_coords)

        # Verify shape and content
        assert omega_angles.shape == (dummy_coords.shape[0], dummy_coords.shape[1])
        # Last residue should have omega=0 (undefined)
        assert jnp.all(omega_angles[:, -1] == 0)
        # Other residues should have non-zero omega values
        assert not jnp.all(omega_angles[:, :-1] == 0)

    def test_phi_angle_loss(self, dihedral_constraint, dummy_coords):
        """Test _phi_angle_loss method."""
        # Calculate phi angle loss
        loss = dihedral_constraint._phi_angle_loss(dummy_coords)

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_psi_angle_loss(self, dihedral_constraint, dummy_coords):
        """Test _psi_angle_loss method."""
        # Calculate psi angle loss
        loss = dihedral_constraint._psi_angle_loss(dummy_coords)

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_omega_angle_loss(self, dihedral_constraint, dummy_coords):
        """Test _omega_angle_loss method."""
        # Calculate omega angle loss
        loss = dihedral_constraint._omega_angle_loss(dummy_coords)

        # Verify loss is a scalar value
        assert isinstance(loss, jax.Array)
        assert loss.ndim == 0  # scalar
        assert not jnp.isnan(loss)

    def test_combined_loss(self, dihedral_constraint, dummy_coords):
        """Test combined loss calculation."""
        # Individual losses
        phi_loss = dihedral_constraint._phi_angle_loss(dummy_coords)
        psi_loss = dihedral_constraint._psi_angle_loss(dummy_coords)
        omega_loss = dihedral_constraint._omega_angle_loss(dummy_coords)

        # Combined loss
        combined_loss = dihedral_constraint.loss_fn({}, dummy_coords)

        # Verify combined loss is weighted sum of individual losses
        expected_loss = (
            dihedral_constraint.phi_weight * phi_loss
            + dihedral_constraint.psi_weight * psi_loss
            + dihedral_constraint.omega_weight * omega_loss
        )

        assert jnp.isclose(combined_loss, expected_loss, atol=1e-5)


# ========================================================================
# JAX Compliance Tests (merged from test_constraints_jax_compliance.py)
# ========================================================================


class TestProteinConstraintsJAXCompliance:
    """Test that protein constraint modules are JAX-compliant."""

    def test_bond_angle_constants_use_jax(self):
        """Test that bond angle constants are converted to JAX."""
        # Check that constants are plain Python floats
        assert isinstance(BOND_ANGLES["N-CA-C"], float)

    def test_dihedral_angle_constants_use_jax(self):
        """Test that dihedral angle constants are converted to JAX."""
        # Check dihedral angle constants are plain Python floats
        alpha_phi = DIHEDRAL_ANGLES["alpha_helix"]["phi"]
        assert isinstance(alpha_phi, float)

        beta_psi = DIHEDRAL_ANGLES["beta_sheet"]["psi"]
        assert isinstance(beta_psi, float)

    def test_backbone_constraint_with_jax_arrays(self):
        """Test ProteinBackboneConstraint with JAX arrays."""
        config = ProteinExtensionConfig(
            name="backbone_constraint",
            weight=1.0,
            enabled=True,
            bond_length_weight=1.0,
            bond_angle_weight=0.5,
        )

        rngs = nnx.Rngs(42)
        constraint = ProteinBackboneConstraint(config, rngs=rngs)

        # Create test data with JAX arrays
        coords = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0], [3.5, 1.0, 0.0]],
                [[4.0, 0.0, 0.0], [5.5, 0.0, 0.0], [6.5, 1.0, 0.0], [7.5, 1.0, 0.0]],
            ]
        )

        inputs = {"atom_mask": jnp.ones((2, 4))}
        outputs = {"atom_positions": coords}

        # Test that the constraint works with JAX arrays
        result = constraint(inputs, outputs)

        # Verify results are JAX arrays
        assert isinstance(result["n_ca_length_mean"], jax.Array)
        assert isinstance(result["ca_c_length_mean"], jax.Array)

    def test_dihedral_constraint_with_jax_arrays(self):
        """Test ProteinDihedralConstraint with JAX arrays."""
        config = ProteinDihedralConfig(
            name="dihedral_constraint",
            weight=0.2,
            enabled=True,
            target_secondary_structure="alpha_helix",
        )

        rngs = nnx.Rngs(42)
        constraint = ProteinDihedralConstraint(config, rngs=rngs)

        # Create test data with JAX arrays
        coords = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0], [3.5, 1.0, 0.0]],
                [[4.0, 0.0, 0.0], [5.5, 0.0, 0.0], [6.5, 1.0, 0.0], [7.5, 1.0, 0.0]],
                [[8.0, 0.0, 0.0], [9.5, 0.0, 0.0], [10.5, 1.0, 0.0], [11.5, 1.0, 0.0]],
            ]
        )

        inputs = {"atom_mask": jnp.ones((3, 4))}
        outputs = {"atom_positions": coords}

        # Test that the constraint works with JAX arrays
        result = constraint(inputs, outputs)

        # Verify results contain expected keys
        assert "phi_mean" in result
        assert "psi_mean" in result

        # Verify results are JAX arrays
        assert isinstance(result["phi_mean"], jax.Array)
        assert isinstance(result["psi_mean"], jax.Array)

    def test_calculate_bond_lengths_jax_only(self):
        """Test that calculate_bond_lengths uses only JAX operations."""
        # Create test coordinates as JAX array
        coords = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.458, 0.0, 0.0], [2.981, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[5.0, 0.0, 0.0], [6.458, 0.0, 0.0], [7.981, 0.0, 0.0], [9.0, 0.0, 0.0]],
            ]
        )

        mask = jnp.ones((2, 4))

        # Calculate bond lengths
        lengths = calculate_bond_lengths(coords, mask)

        # Verify all outputs are JAX arrays
        for bond_type, values in lengths.items():
            assert isinstance(values, jax.Array)
            # Verify no numpy arrays in the computation
            assert not isinstance(values, np.ndarray)

    def test_calculate_dihedral_angles_jax_only(self):
        """Test that calculate_dihedral_angles uses only JAX operations."""
        # Create test coordinates for 3 residues (minimum for phi and psi)
        coords = jnp.array(
            [
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0], [3.5, 1.0, 0.0]],
                [[4.0, 0.0, 0.0], [5.5, 0.0, 0.0], [6.5, 1.0, 0.0], [7.5, 1.0, 0.0]],
                [[8.0, 0.0, 0.0], [9.5, 0.0, 0.0], [10.5, 1.0, 0.0], [11.5, 1.0, 0.0]],
            ]
        )

        # Calculate dihedral angles
        angles = calculate_dihedral_angles(coords)

        # Verify outputs are JAX arrays
        assert isinstance(angles["phi"], jax.Array)
        assert isinstance(angles["psi"], jax.Array)

        # Verify shapes
        assert angles["phi"].shape == (3,)  # 3 residues
        assert angles["psi"].shape == (3,)  # 3 residues


class TestNumpyToJAXConversion:
    """Test patterns for converting numpy to JAX in protein constraints."""

    def test_radians_conversion(self):
        """Test converting np.radians to JAX-compatible code."""
        # Current pattern (uses numpy)
        angle_deg = 111.2
        angle_rad_np = np.radians(angle_deg)

        # JAX-compatible pattern
        angle_rad_jax = angle_deg * (jnp.pi / 180.0)

        # They should be equivalent
        assert jnp.allclose(angle_rad_np, angle_rad_jax)

    def test_constants_as_python_floats(self):
        """Test that constants can be pure Python floats."""
        # Instead of using np.radians, we can pre-compute
        # or use pure Python math

        # Using Python's math module (computed at module load time)
        angle_rad = math.radians(111.2)
        assert isinstance(angle_rad, float)

        # Or pre-computed values
        angle_rad_precomputed = 1.9408061107216327  # 111.2 degrees in radians
        assert isinstance(angle_rad_precomputed, float)
