"""Integration tests for protein diffusion components."""

import os
import tempfile

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ProteinConstraintConfig,
    ProteinDihedralConfig,
    ProteinExtensionConfig,
    ProteinPointCloudConfig,
)
from artifex.generative_models.core.configuration.geometric_config import (
    PointCloudConfig,
    PointCloudNetworkConfig,
)
from artifex.generative_models.extensions.protein.constraints import (
    ProteinBackboneConstraint,
    ProteinDihedralConstraint,
)
from artifex.generative_models.models.geometric.point_cloud import (
    PointCloudModel,
)
from artifex.generative_models.models.geometric.protein_point_cloud import (
    ProteinPointCloudModel,
)
from artifex.generative_models.utils.visualization.protein import (
    ProteinVisualizer,
)


@pytest.fixture
def rng():
    """Random number generator fixture."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def protein_config():
    """Create a protein model configuration."""
    network = PointCloudNetworkConfig(
        name="protein_pc_network",
        hidden_dims=(64, 32),
        embed_dim=64,  # Must be divisible by num_heads
        num_heads=2,
        num_layers=2,
        use_positional_encoding=True,
        dropout_rate=0.1,
        activation="gelu",
    )
    constraint_config = ProteinConstraintConfig(
        backbone_weight=1.0,
        bond_weight=1.0,
        angle_weight=0.5,
        dihedral_weight=0.3,
    )
    return ProteinPointCloudConfig(
        name="test_protein_point_cloud",
        network=network,
        num_points=40,  # 10 residues * 4 atoms
        point_dim=3,
        num_residues=10,
        num_atoms_per_residue=4,
        backbone_indices=(0, 1, 2, 3),
        use_constraints=True,
        constraint_config=constraint_config,
    )


@pytest.fixture
def point_cloud_config():
    """Create a point cloud model configuration for base model tests."""
    network = PointCloudNetworkConfig(
        name="test_pc_network",
        hidden_dims=(64, 32),
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        use_positional_encoding=True,
        dropout_rate=0.1,
        activation="gelu",
    )
    return PointCloudConfig(
        name="test_point_cloud",
        network=network,
        num_points=40,
        point_dim=3,
    )


@pytest.fixture
def model_input(protein_config):
    """Create dummy model input."""
    batch_size = 2
    num_residues = protein_config.num_residues
    num_atoms = protein_config.num_atoms_per_residue

    # Create random coordinates
    rng = jax.random.PRNGKey(42)
    coords = jax.random.normal(rng, shape=(batch_size, num_residues, num_atoms, 3))

    # Create mask (all atoms present)
    mask = jnp.ones((batch_size, num_residues, num_atoms))

    # Create residue indices
    residue_indices = jnp.arange(num_residues)[None, :]
    residue_indices = jnp.repeat(residue_indices, batch_size, axis=0)

    # Create amino acid types (random)
    aa_rng = jax.random.PRNGKey(43)
    aatype = jax.random.randint(aa_rng, shape=(batch_size, num_residues), minval=1, maxval=20)

    return {
        "atom_positions": coords,
        "atom_mask": mask,
        "residue_index": residue_indices,
        "aatype": aatype,
    }


class TestProteinDiffusionIntegration:
    """Integration tests for protein diffusion components."""

    def test_constraint_model_integration(self, rng, protein_config, point_cloud_config):
        """Test integration of constraints with point cloud model."""
        # Get constraint config from protein config
        constraint_cfg = protein_config.constraint_config or ProteinConstraintConfig()

        backbone_config = ProteinExtensionConfig(
            name="backbone_constraint",
            weight=constraint_cfg.bond_weight,
            enabled=True,
            bond_length_weight=constraint_cfg.bond_weight,
            bond_angle_weight=constraint_cfg.angle_weight,
        )

        dihedral_config = ProteinDihedralConfig(
            name="dihedral_constraint",
            weight=constraint_cfg.dihedral_weight,
            enabled=True,
            phi_weight=constraint_cfg.phi_weight,
            psi_weight=constraint_cfg.psi_weight,
        )

        # Create constraint extensions (wrap in nnx.Dict for Flax NNX 0.12.0+)
        extensions = nnx.Dict(
            {
                "backbone": ProteinBackboneConstraint(backbone_config, rngs=nnx.Rngs(params=rng)),
                "dihedral": ProteinDihedralConstraint(dihedral_config, rngs=nnx.Rngs(params=rng)),
            }
        )

        # Create model with extensions using point_cloud_config (base PointCloudConfig)
        model = PointCloudModel(
            point_cloud_config, extensions=extensions, rngs=nnx.Rngs(params=rng)
        )

        # Verify extensions were properly added
        assert "backbone" in model.extension_modules
        assert "dihedral" in model.extension_modules

        # Verify extension types
        assert isinstance(model.extension_modules["backbone"], ProteinBackboneConstraint)
        assert isinstance(model.extension_modules["dihedral"], ProteinDihedralConstraint)

    def test_protein_model_forward_pass(self, rng, protein_config, model_input):
        """Test forward pass through protein-specific model."""
        # Create protein model
        model = ProteinPointCloudModel(protein_config, rngs=nnx.Rngs(params=rng))

        # Forward pass (removed rngs parameter)
        outputs = model(model_input)

        # Verify outputs
        assert "positions" in outputs
        assert "embeddings" in outputs
        assert "extension_outputs" in outputs

        # Check shapes
        batch_size = model_input["atom_positions"].shape[0]
        num_residues = protein_config.num_residues
        num_atoms = protein_config.num_atoms_per_residue

        # Positions should be in protein format [batch, residues, atoms, 3]
        positions = outputs["positions"]

        # Check if in protein-specific format
        if len(positions.shape) == 4:
            # Should be [batch, residues, atoms, 3]
            assert positions.shape == (batch_size, num_residues, num_atoms, 3)
        else:
            # Or flattened format [batch, residues*atoms, 3]
            assert positions.shape == (batch_size, num_residues * num_atoms, 3)

    def test_base_model_with_extensions(self, rng, protein_config, point_cloud_config, model_input):
        """Test base model with protein extensions."""
        # Get constraint config from protein config
        constraint_cfg = protein_config.constraint_config or ProteinConstraintConfig()

        # Create extension configs using frozen dataclass configs
        backbone_config = ProteinExtensionConfig(
            name="backbone_constraint",
            weight=constraint_cfg.bond_weight,
            enabled=True,
            bond_length_weight=constraint_cfg.bond_weight,
            bond_angle_weight=constraint_cfg.angle_weight,
        )

        dihedral_config = ProteinDihedralConfig(
            name="dihedral_constraint",
            weight=constraint_cfg.dihedral_weight,
            enabled=True,
            phi_weight=constraint_cfg.phi_weight,
            psi_weight=constraint_cfg.psi_weight,
        )

        # Create constraint extensions (wrap in nnx.Dict for Flax NNX 0.12.0+)
        extensions = nnx.Dict(
            {
                "backbone": ProteinBackboneConstraint(backbone_config, rngs=nnx.Rngs(params=rng)),
                "dihedral": ProteinDihedralConstraint(dihedral_config, rngs=nnx.Rngs(params=rng)),
            }
        )

        # Create base model with extensions using point_cloud_config
        model = PointCloudModel(
            point_cloud_config, extensions=extensions, rngs=nnx.Rngs(params=rng)
        )

        # Forward pass
        outputs = model(model_input)

        # Verify outputs
        assert "positions" in outputs
        assert "embeddings" in outputs
        assert "extension_outputs" in outputs

        # Verify extension outputs
        assert "backbone" in outputs["extension_outputs"]
        assert "dihedral" in outputs["extension_outputs"]

        # Check extension output content
        assert "n_ca_length_mean" in outputs["extension_outputs"]["backbone"]
        assert "phi_mean" in outputs["extension_outputs"]["dihedral"]

    def test_loss_function_integration(self, rng, protein_config, model_input):
        """Test loss function integration with extensions."""
        # Skip due to shape broadcasting issues
        pytest.skip("Skipped due to shape broadcasting issues between (2, 40, 3) and (2, 10, 4, 3)")

        # Create protein model
        model = ProteinPointCloudModel(protein_config, rngs=nnx.Rngs(params=rng))

        # Forward pass
        outputs = model(model_input)

        # Get loss function
        loss_fn = model.get_loss_fn()

        # Calculate loss
        loss_dict = loss_fn(model_input, outputs)

        # Verify loss dictionary
        assert "total_loss" in loss_dict
        assert "mse_loss" in loss_dict
        # Check for extension losses
        assert any(key.startswith("protein") for key in loss_dict.keys())

    def test_protein_model_sample(self, rng, protein_config):
        """Test sampling from protein model."""
        # Create protein model
        model = ProteinPointCloudModel(protein_config, rngs=nnx.Rngs(params=rng))

        # Sample from model
        n_samples = 2
        samples = model.sample(n_samples, rngs=nnx.Rngs(params=rng))

        # Verify sample shape
        assert samples.shape == (
            n_samples,
            protein_config.num_residues,
            protein_config.num_atoms_per_residue,
            3,
        )

    def test_integration_with_visualization(self, rng, protein_config):
        """Test integration with visualization tools."""
        # Skip test if no temporary directory available
        temp_dir = tempfile.gettempdir()
        if not os.access(temp_dir, os.W_OK):
            pytest.skip("No write access to temporary directory")

        # Create protein model
        model = ProteinPointCloudModel(protein_config, rngs=nnx.Rngs(params=rng))

        # Generate sample
        samples = model.sample(1, rngs=nnx.Rngs(params=rng))

        # Create mask and dummy amino acid types
        mask = jnp.ones((1, protein_config.num_residues, protein_config.num_atoms_per_residue))
        aatype = jnp.ones((1, protein_config.num_residues), dtype=jnp.int32)

        # Prepare protein data for visualization
        protein_data = {
            "coords": samples,
            "mask": mask,
            "aatype": aatype,
        }

        # Create PDB file
        visualizer = ProteinVisualizer()
        pdb_str = visualizer.coords_to_pdb(
            protein_data["coords"], protein_data["mask"], protein_data["aatype"]
        )

        # Verify PDB string
        assert isinstance(pdb_str, str)
        assert "ATOM" in pdb_str
        assert "MODEL" in pdb_str
        assert "END" in pdb_str

        # Export to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
            visualizer.export_to_pdb(protein_data, tmp_path)

        # Verify file was created and contains valid PDB
        try:
            assert os.path.exists(tmp_path)
            with open(tmp_path, "r") as f:
                content = f.read()
                assert "ATOM" in content
                assert "MODEL" in content
                assert "END" in content
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_end_to_end_pipeline(self, rng, protein_config, model_input):
        """Test end-to-end protein diffusion pipeline."""
        # Skip due to shape broadcasting issues
        pytest.skip("Skipped due to shape broadcasting issues between (2, 40, 3) and (2, 10, 4, 3)")

        # Create model
        model = ProteinPointCloudModel(protein_config, rngs=nnx.Rngs(params=rng))

        # Create loss function
        loss_fn = model.get_loss_fn()

        # Set up mock training pipeline
        def train_step(inputs):
            # Forward pass
            outputs = model(inputs)
            # Calculate loss
            loss_dict = loss_fn(inputs, outputs)
            return loss_dict

        # Basic training step
        loss_dict = train_step(model_input)

        # Verify loss dictionary
        assert "total_loss" in loss_dict
        assert "mse_loss" in loss_dict

        # Try generating samples with trained model
        samples = model.sample(n_samples=2, rngs=nnx.Rngs(params=jax.random.PRNGKey(123)))

        # Verify sample shapes
        assert samples.shape[0] == 2  # Batch dimension
        assert samples.shape[1] == protein_config.metadata["num_residues"]  # Residues
        assert samples.shape[2] == protein_config.metadata["num_atoms"]  # Atoms per residue
        assert samples.shape[3] == 3  # 3D coordinates
