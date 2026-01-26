"""Tests for molecular feature computation and extraction."""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.chemical.features import MolecularFeatures


class TestMolecularFeaturesInit:
    """Tests for MolecularFeatures initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        config = ExtensionConfig(name="test_molecular_features")
        rngs = nnx.Rngs(seed=42)

        features = MolecularFeatures(config, rngs=rngs)

        assert features.config == config
        assert len(features.feature_types) > 0
        assert features.include_3d_features is True

    def test_init_uses_default_feature_types(self):
        """Test that default feature types are used."""
        config = ExtensionConfig(name="test_default_features")
        rngs = nnx.Rngs(seed=42)

        features = MolecularFeatures(config, rngs=rngs)

        # Default feature types from the class
        expected_types = [
            "molecular_weight",
            "lipophilicity",
            "hydrogen_bonds",
            "polar_surface_area",
            "rotatable_bonds",
            "aromatic_rings",
        ]
        assert features.feature_types == expected_types

    def test_init_invalid_config_type(self):
        """Test initialization with invalid config type raises error."""
        rngs = nnx.Rngs(seed=42)

        with pytest.raises(TypeError, match="config must be ExtensionConfig"):
            MolecularFeatures({"invalid": "config"}, rngs=rngs)

    def test_init_atomic_properties_initialized(self):
        """Test that atomic properties are initialized."""
        config = ExtensionConfig(name="test_atomic_props")
        rngs = nnx.Rngs(seed=42)

        features = MolecularFeatures(config, rngs=rngs)

        # Check atomic masses exist
        assert 1 in features.atomic_masses  # Hydrogen
        assert 6 in features.atomic_masses  # Carbon
        assert 8 in features.atomic_masses  # Oxygen

        # Check VDW radii exist
        assert 1 in features.vdw_radii
        assert 6 in features.vdw_radii

        # Check electronegativities exist
        assert 1 in features.electronegativities
        assert 6 in features.electronegativities

    def test_init_vdw_radii_values(self):
        """Test Van der Waals radii are physically reasonable."""
        config = ExtensionConfig(name="test_vdw")
        rngs = nnx.Rngs(seed=42)

        features = MolecularFeatures(config, rngs=rngs)

        # VDW radii should be between 1.0 and 3.0 Angstroms for common atoms
        for radius in features.vdw_radii.values():
            assert 1.0 <= radius <= 3.0

    def test_init_electronegativity_values(self):
        """Test electronegativity values are in Pauling scale range."""
        config = ExtensionConfig(name="test_en")
        rngs = nnx.Rngs(seed=42)

        features = MolecularFeatures(config, rngs=rngs)

        # Pauling scale ranges from about 0.7 to 4.0
        for en in features.electronegativities.values():
            assert 0.5 <= en <= 4.5


class TestMolecularWeight:
    """Tests for molecular weight computation."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_mol_weight")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_molecular_weight_water(self, features):
        """Test molecular weight computation for water (H2O)."""
        # H2O: 2 hydrogen + 1 oxygen
        atom_types = jnp.array([1, 1, 8])  # H, H, O

        weight = features._compute_molecular_weight(atom_types)

        # Expected: 2 * 1.008 + 15.999 = 18.015
        assert abs(weight - 18.015) < 0.01

    def test_molecular_weight_methane(self, features):
        """Test molecular weight computation for methane (CH4)."""
        # CH4: 1 carbon + 4 hydrogen
        atom_types = jnp.array([6, 1, 1, 1, 1])  # C, H, H, H, H

        weight = features._compute_molecular_weight(atom_types)

        # Expected: 12.011 + 4 * 1.008 = 16.043
        assert abs(weight - 16.043) < 0.01

    def test_molecular_weight_unknown_atom(self, features):
        """Test that unknown atoms default to carbon mass."""
        # Unknown atom type (99)
        atom_types = jnp.array([99])

        weight = features._compute_molecular_weight(atom_types)

        # Should default to carbon: 12.011
        assert abs(weight - 12.011) < 0.01

    def test_molecular_weight_empty_molecule(self, features):
        """Test molecular weight for empty molecule."""
        atom_types = jnp.array([], dtype=jnp.int32)

        weight = features._compute_molecular_weight(atom_types)

        assert weight == 0.0


class TestLipophilicity:
    """Tests for lipophilicity (LogP) computation."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_logp")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_logp_returns_float(self, features):
        """Test that LogP returns a float value."""
        atom_types = jnp.array([6, 1, 1, 1, 1])

        logp = features._compute_logp(atom_types)

        assert isinstance(logp, float)

    def test_logp_polar_atoms_negative(self, features):
        """Test that polar atoms contribute negative LogP."""
        # Nitrogen and oxygen are polar
        atom_types = jnp.array([7, 8])

        logp = features._compute_logp(atom_types)

        assert logp < 0  # Should be negative (hydrophilic)

    def test_logp_hydrophobic_atoms_positive(self, features):
        """Test that hydrophobic atoms contribute positive LogP."""
        # Many carbons without polar groups
        atom_types = jnp.array([6, 6, 6, 6, 6, 6])

        logp = features._compute_logp(atom_types)

        assert logp > 0  # Should be positive (lipophilic)


class TestHydrogenBonds:
    """Tests for hydrogen bond counting."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_hbonds")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_hydrogen_bonds_water(self, features):
        """Test hydrogen bond counting for water."""
        atom_types = jnp.array([1, 1, 8])  # H, H, O

        hb = features._count_hydrogen_bonds(atom_types)

        assert "hydrogen_bond_donors" in hb
        assert "hydrogen_bond_acceptors" in hb
        assert hb["hydrogen_bond_donors"] >= 1  # At least oxygen
        assert hb["hydrogen_bond_acceptors"] >= 1

    def test_hydrogen_bonds_no_polar_atoms(self, features):
        """Test hydrogen bond counting with no polar atoms."""
        atom_types = jnp.array([6, 6, 1, 1])  # Carbon and hydrogen only

        hb = features._count_hydrogen_bonds(atom_types)

        assert hb["hydrogen_bond_donors"] == 0
        assert hb["hydrogen_bond_acceptors"] == 0


class TestPolarSurfaceArea:
    """Tests for polar surface area computation."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_tpsa")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_tpsa_with_polar_atoms(self, features):
        """Test TPSA computation with polar atoms."""
        atom_types = jnp.array([7, 8])  # N and O

        tpsa = features._compute_polar_surface_area(atom_types)

        assert tpsa > 0

    def test_tpsa_no_polar_atoms(self, features):
        """Test TPSA computation with no polar atoms."""
        atom_types = jnp.array([6, 6, 1, 1])  # C and H only

        tpsa = features._compute_polar_surface_area(atom_types)

        assert tpsa == 0.0


class TestRotatableBonds:
    """Tests for rotatable bond counting."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_rot_bonds")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_rotatable_bonds_no_bonds_matrix(self, features):
        """Test rotatable bonds with no bond matrix."""
        atom_types = jnp.array([6, 6, 6])

        count = features._count_rotatable_bonds(atom_types, None)

        assert count == 0.0

    def test_rotatable_bonds_with_bonds(self, features):
        """Test rotatable bond counting with bond matrix."""
        # Linear chain of 3 carbons: C-C-C
        atom_types = jnp.array([6, 6, 6])
        bonds = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        count = features._count_rotatable_bonds(atom_types, bonds)

        # Middle carbon has >1 connection on both sides
        assert isinstance(count, float)


class TestAromaticRings:
    """Tests for aromatic ring counting."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_aromatic")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_aromatic_rings_no_bonds(self, features):
        """Test aromatic ring counting with no bond matrix."""
        atom_types = jnp.array([6, 6, 6, 6, 6, 6])

        count = features._count_aromatic_rings(atom_types, None)

        assert count == 0.0

    def test_aromatic_rings_benzene(self, features):
        """Test aromatic ring counting for benzene-like structure."""
        # 6 carbons
        atom_types = jnp.array([6, 6, 6, 6, 6, 6])
        bonds = jnp.ones((6, 6)) - jnp.eye(6)  # All connected

        count = features._count_aromatic_rings(atom_types, bonds)

        # Should detect at least 1 aromatic ring
        assert count >= 1.0


class TestComputeDescriptors:
    """Tests for the main compute_descriptors method."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_descriptors")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    @pytest.fixture
    def molecule_data(self):
        """Create sample molecule data."""
        return {
            "coordinates": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.87, 0.0]]),
            "atom_types": jnp.array([8, 1, 1]),  # O, H, H (water)
        }

    def test_compute_descriptors_returns_dict(self, features, molecule_data):
        """Test that compute_descriptors returns a dictionary."""
        descriptors = features.compute_descriptors(molecule_data)

        assert isinstance(descriptors, dict)
        assert len(descriptors) > 0

    def test_compute_descriptors_includes_molecular_weight(self, features, molecule_data):
        """Test that molecular weight is included."""
        descriptors = features.compute_descriptors(molecule_data)

        assert "molecular_weight" in descriptors

    def test_compute_descriptors_includes_3d_features(self, features, molecule_data):
        """Test that 3D features are included when enabled."""
        descriptors = features.compute_descriptors(molecule_data)

        assert "molecular_volume" in descriptors
        assert "molecular_surface_area" in descriptors


class Test3DDescriptors:
    """Tests for 3D geometry descriptors."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_3d")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_molecular_volume_small_molecule(self, features):
        """Test molecular volume for small coordinate set."""
        # Less than 4 atoms
        coordinates = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        volume = features._compute_molecular_volume(coordinates)

        assert volume == 0.0  # Not enough atoms

    def test_molecular_volume_cube(self, features):
        """Test molecular volume for cube-shaped coordinates."""
        coordinates = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )

        volume = features._compute_molecular_volume(coordinates)

        assert volume > 0

    def test_surface_area_positive(self, features):
        """Test surface area is positive for valid coordinates."""
        coordinates = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        surface_area = features._compute_surface_area(coordinates)

        assert surface_area > 0

    def test_surface_area_insufficient_atoms(self, features):
        """Test surface area with insufficient atoms."""
        coordinates = jnp.array([[0.0, 0.0, 0.0]])

        surface_area = features._compute_surface_area(coordinates)

        assert surface_area == 0.0

    def test_gyration_radius_positive(self, features):
        """Test gyration radius is positive for spread-out coordinates."""
        coordinates = jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        )

        rg = features._compute_gyration_radius(coordinates)

        assert rg > 0

    def test_gyration_radius_insufficient_atoms(self, features):
        """Test gyration radius with insufficient atoms."""
        coordinates = jnp.array([[0.0, 0.0, 0.0]])

        rg = features._compute_gyration_radius(coordinates)

        assert rg == 0.0


class TestShapeDescriptors:
    """Tests for shape descriptor computation."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_shape")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_shape_descriptors_returns_dict(self, features):
        """Test shape descriptors returns dictionary."""
        coordinates = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        shape = features._compute_shape_descriptors(coordinates)

        assert "asphericity" in shape
        assert "eccentricity" in shape

    def test_shape_descriptors_insufficient_atoms(self, features):
        """Test shape descriptors with insufficient atoms."""
        coordinates = jnp.array([[0.0, 0.0, 0.0]])

        shape = features._compute_shape_descriptors(coordinates)

        assert shape["asphericity"] == 0.0
        assert shape["eccentricity"] == 0.0


class TestDrugLikenessScore:
    """Tests for drug-likeness score computation."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_drug_likeness")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_drug_likeness_perfect_score(self, features):
        """Test drug-likeness with no violations."""
        descriptors = {
            "molecular_weight": 300,  # < 500
            "lipophilicity": 2,  # < 5
            "hydrogen_bond_donors": 2,  # < 5
            "hydrogen_bond_acceptors": 5,  # < 10
        }

        score = features.compute_drug_likeness_score(descriptors)

        assert score == 1.0

    def test_drug_likeness_all_violations(self, features):
        """Test drug-likeness with all violations."""
        descriptors = {
            "molecular_weight": 600,  # > 500
            "lipophilicity": 6,  # > 5
            "hydrogen_bond_donors": 6,  # > 5
            "hydrogen_bond_acceptors": 12,  # > 10
        }

        score = features.compute_drug_likeness_score(descriptors)

        assert score == 0.0

    def test_drug_likeness_partial_violations(self, features):
        """Test drug-likeness with some violations."""
        descriptors = {
            "molecular_weight": 600,  # > 500 (violation)
            "lipophilicity": 2,  # < 5 (ok)
            "hydrogen_bond_donors": 6,  # > 5 (violation)
            "hydrogen_bond_acceptors": 5,  # < 10 (ok)
        }

        score = features.compute_drug_likeness_score(descriptors)

        # 2 violations = 1 - (2/4) = 0.5
        assert abs(score - 0.5) < 0.01


class TestFingerprint:
    """Tests for molecular fingerprint extraction."""

    @pytest.fixture
    def features(self):
        """Create MolecularFeatures instance."""
        config = ExtensionConfig(name="test_fingerprint")
        rngs = nnx.Rngs(seed=42)
        return MolecularFeatures(config, rngs=rngs)

    def test_fingerprint_shape(self, features):
        """Test fingerprint has correct shape."""
        molecule_data = {
            "coordinates": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "atom_types": jnp.array([6, 1]),  # C, H
        }

        fingerprint = features.extract_fingerprint(molecule_data, fingerprint_size=256)

        assert fingerprint.shape == (256,)

    def test_fingerprint_default_size(self, features):
        """Test fingerprint with default size."""
        molecule_data = {
            "coordinates": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "atom_types": jnp.array([6, 1]),
        }

        fingerprint = features.extract_fingerprint(molecule_data)

        assert fingerprint.shape == (1024,)

    def test_fingerprint_contains_atom_counts(self, features):
        """Test fingerprint contains atom type information."""
        molecule_data = {
            "coordinates": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "atom_types": jnp.array([6, 6]),  # Two carbons
        }

        fingerprint = features.extract_fingerprint(molecule_data)

        # Carbon (atomic number 6) should have count > 0 at index 6
        assert fingerprint[6] >= 1

    def test_fingerprint_with_bonds(self, features):
        """Test fingerprint with bond information."""
        molecule_data = {
            "coordinates": jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            "atom_types": jnp.array([6, 6]),
            "bonds": jnp.array([[0, 1], [1, 0]]),
        }

        fingerprint = features.extract_fingerprint(molecule_data, fingerprint_size=256)

        # Bond information should be at index 120
        assert fingerprint[120] >= 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_molecule(self):
        """Test handling of empty molecule data."""
        config = ExtensionConfig(name="test_empty")
        rngs = nnx.Rngs(seed=42)
        features = MolecularFeatures(config, rngs=rngs)

        molecule_data = {"coordinates": jnp.zeros((0, 3)), "atom_types": jnp.array([])}

        # Should not raise errors
        descriptors = features.compute_descriptors(molecule_data)
        assert isinstance(descriptors, dict)

    def test_single_atom(self):
        """Test handling of single atom molecule."""
        config = ExtensionConfig(name="test_single")
        rngs = nnx.Rngs(seed=42)
        features = MolecularFeatures(config, rngs=rngs)

        molecule_data = {
            "coordinates": jnp.array([[0.0, 0.0, 0.0]]),
            "atom_types": jnp.array([6]),  # Single carbon
        }

        descriptors = features.compute_descriptors(molecule_data)

        assert "molecular_weight" in descriptors
        assert abs(descriptors["molecular_weight"] - 12.011) < 0.01
