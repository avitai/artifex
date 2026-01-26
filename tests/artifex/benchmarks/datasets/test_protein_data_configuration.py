"""Tests for protein datasets using DataConfig."""

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from artifex.benchmarks.datasets.protein_dataset import SyntheticProteinDataset
from artifex.generative_models.core.configuration import DataConfig


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def protein_config():
    """Create a DataConfig for protein testing."""
    return DataConfig(
        name="protein_test_config",
        dataset_name="synthetic_protein",
        data_dir=Path("./test_protein_data"),
        split="train",
        num_workers=2,
        metadata={
            "num_samples": 100,
            "num_residues": 10,
            "num_atoms": 4,
            "seed": 42,
            "batch_size": 8,
            "type": "synthetic_protein",
        },
    )


class TestSyntheticProteinDatasetWithDataConfig:
    """Test SyntheticProteinDataset with DataConfig."""

    def test_protein_init_with_data_configuration(self, protein_config, rngs):
        """Test creating protein dataset with DataConfig."""
        dataset = SyntheticProteinDataset(
            data_path=str(protein_config.data_dir), config=protein_config, rngs=rngs
        )

        assert isinstance(dataset.config, DataConfig)
        assert dataset.config.dataset_name == "synthetic_protein"
        assert dataset.num_samples == 100
        assert dataset.num_residues == 10
        assert dataset.num_atoms == 4

    def test_protein_rejects_dict_config(self, rngs):
        """Test that SyntheticProteinDataset rejects dict config."""
        dict_config = {"dataset_name": "synthetic_protein", "num_samples": 100, "num_residues": 10}

        with pytest.raises(TypeError, match="config must be DataConfig"):
            SyntheticProteinDataset(data_path="./test_protein_data", config=dict_config, rngs=rngs)

    def test_protein_get_batch_uses_config(self, protein_config, rngs):
        """Test that get_batch uses DataConfig metadata."""
        dataset = SyntheticProteinDataset(
            data_path=str(protein_config.data_dir), config=protein_config, rngs=rngs
        )

        batch = dataset.get_batch()

        # Check batch structure
        assert "coordinates" in batch
        assert "flat_coordinates" in batch
        assert "aatype" in batch

        # Check batch size from config
        assert batch["coordinates"].shape[0] == 8  # batch_size from metadata
        assert batch["coordinates"].shape[1] == 10  # num_residues from metadata
        assert batch["coordinates"].shape[2] == 4  # num_atoms from metadata
        assert batch["coordinates"].shape[3] == 3  # 3D coordinates

        # Check flattened coordinates
        assert batch["flat_coordinates"].shape[0] == 8
        assert batch["flat_coordinates"].shape[1] == 10 * 4 * 3  # residues * atoms * 3

    def test_protein_dataset_info(self, protein_config, rngs):
        """Test dataset info with DataConfig."""
        dataset = SyntheticProteinDataset(
            data_path=str(protein_config.data_dir), config=protein_config, rngs=rngs
        )

        info = dataset.get_dataset_info()

        assert info["name"] == "synthetic_protein"
        assert info["modality"] == "protein"
        assert info["num_samples"] == 100
        assert info["num_residues"] == 10
        assert info["num_atoms"] == 4
        assert info["split"] == "train"
        assert "data_shape" in info

    def test_protein_batch_consistency(self, protein_config, rngs):
        """Test that batches maintain consistent structure."""
        dataset = SyntheticProteinDataset(
            data_path=str(protein_config.data_dir), config=protein_config, rngs=rngs
        )

        # Get multiple batches
        batches = []
        for _ in range(3):
            batch = dataset.get_batch(batch_size=4)
            batches.append(batch)

        # Check consistency
        for batch in batches:
            assert batch["coordinates"].shape == (4, 10, 4, 3)
            assert batch["flat_coordinates"].shape == (4, 120)  # 10 * 4 * 3
            assert batch["aatype"].shape == (4, 10)

            # Check value ranges
            assert jnp.all((batch["aatype"] >= 0) & (batch["aatype"] < 20))
            assert jnp.isfinite(batch["coordinates"]).all()
            assert jnp.isfinite(batch["flat_coordinates"]).all()


class TestProteinDatasetWithDifferentConfigs:
    """Test protein dataset with various configurations."""

    def test_small_protein_config(self, rngs):
        """Test with small protein configuration."""
        config = DataConfig(
            name="small_protein_config",
            dataset_name="small_protein",
            data_dir=Path("./test_data"),
            split="train",
            metadata={"num_samples": 50, "num_residues": 5, "num_atoms": 3, "batch_size": 4},
        )

        dataset = SyntheticProteinDataset(data_path=str(config.data_dir), config=config, rngs=rngs)

        batch = dataset.get_batch()
        assert batch["coordinates"].shape == (4, 5, 3, 3)
        assert batch["flat_coordinates"].shape == (4, 45)  # 5 * 3 * 3

    def test_large_protein_config(self, rngs):
        """Test with large protein configuration."""
        config = DataConfig(
            name="large_protein_config",
            dataset_name="large_protein",
            data_dir=Path("./test_data"),
            split="train",
            metadata={"num_samples": 200, "num_residues": 20, "num_atoms": 5, "batch_size": 16},
        )

        dataset = SyntheticProteinDataset(data_path=str(config.data_dir), config=config, rngs=rngs)

        batch = dataset.get_batch()
        assert batch["coordinates"].shape == (16, 20, 5, 3)
        assert batch["flat_coordinates"].shape == (16, 300)  # 20 * 5 * 3
