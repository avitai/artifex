"""Tests for molecular datasets using DataConfig."""

from pathlib import Path

import flax.nnx as nnx
import pytest

from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset
from artifex.benchmarks.datasets.qm9 import QM9Dataset
from artifex.generative_models.core.configuration import DataConfig


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def qm9_config():
    """Create a DataConfig for QM9 testing."""
    return DataConfig(
        name="qm9_test_config",
        dataset_name="qm9",
        data_dir=Path("./test_qm9_data"),
        split="train",
        num_workers=2,
        metadata={"max_atoms": 29, "batch_size": 8, "num_conformations": 10, "type": "qm9"},
    )


@pytest.fixture
def crossdocked_config():
    """Create a DataConfig for CrossDocked testing."""
    return DataConfig(
        name="crossdocked_test_config",
        dataset_name="crossdocked2020",
        data_dir=Path("./test_crossdocked_data"),
        split="train",
        num_workers=2,
        metadata={
            "max_protein_atoms": 500,
            "max_ligand_atoms": 30,
            "pocket_radius": 8.0,
            "num_samples": 100,
            "batch_size": 4,
            "type": "crossdocked",
        },
    )


class TestQM9DatasetWithDataConfig:
    """Test QM9Dataset with DataConfig."""

    def test_qm9_init_with_data_configuration(self, qm9_config, rngs):
        """Test creating QM9 dataset with DataConfig."""
        dataset = QM9Dataset(data_path=str(qm9_config.data_dir), config=qm9_config, rngs=rngs)

        assert isinstance(dataset.config, DataConfig)
        assert dataset.config.dataset_name == "qm9"
        assert dataset.max_atoms == 29
        assert dataset.batch_size == 8
        assert dataset.num_conformations == 10

    def test_qm9_rejects_dict_config(self, rngs):
        """Test that QM9Dataset rejects dict config."""
        dict_config = {"dataset_name": "qm9", "batch_size": 32, "max_atoms": 29}

        with pytest.raises(TypeError, match="config must be DataConfig"):
            QM9Dataset(data_path="./test_qm9_data", config=dict_config, rngs=rngs)

    def test_qm9_get_batch_uses_config(self, qm9_config, rngs):
        """Test that get_batch uses DataConfig metadata."""
        dataset = QM9Dataset(data_path=str(qm9_config.data_dir), config=qm9_config, rngs=rngs)

        batch = dataset.get_batch(batch_size=8)

        # Check batch structure
        assert "coordinates" in batch
        assert "atom_types" in batch
        assert "atom_mask" in batch

        # Check batch size from config
        assert batch["coordinates"].shape[0] == 8  # batch_size from metadata
        assert batch["coordinates"].shape[1] == 29  # max_atoms from metadata
        assert batch["coordinates"].shape[2] == 3  # 3D coordinates

    def test_qm9_dataset_info(self, qm9_config, rngs):
        """Test dataset info with DataConfig."""
        dataset = QM9Dataset(data_path=str(qm9_config.data_dir), config=qm9_config, rngs=rngs)

        info = dataset.get_dataset_info()

        assert info["name"] == "qm9"
        assert info["modality"] == "molecular"
        assert info["max_atoms"] == 29
        assert info["split"] == "train"
        assert "n_samples" in info


class TestCrossDockedDatasetWithDataConfig:
    """Test CrossDockedDataset with DataConfig."""

    def test_crossdocked_init_with_data_configuration(self, crossdocked_config, rngs):
        """Test creating CrossDocked dataset with DataConfig."""
        dataset = CrossDockedDataset(
            data_path=str(crossdocked_config.data_dir), config=crossdocked_config, rngs=rngs
        )

        assert isinstance(dataset.config, DataConfig)
        assert dataset.config.dataset_name == "crossdocked2020"
        assert dataset.max_protein_atoms == 500
        assert dataset.max_ligand_atoms == 30
        assert dataset.pocket_radius == 8.0
        assert dataset.num_samples == 100

    def test_crossdocked_rejects_dict_config(self, rngs):
        """Test that CrossDockedDataset rejects dict config."""
        dict_config = {"dataset_name": "crossdocked", "batch_size": 32, "max_protein_atoms": 1000}

        with pytest.raises(TypeError, match="config must be DataConfig"):
            CrossDockedDataset(data_path="./test_crossdocked_data", config=dict_config, rngs=rngs)

    def test_crossdocked_get_batch_uses_config(self, crossdocked_config, rngs):
        """Test that get_batch uses DataConfig metadata."""
        dataset = CrossDockedDataset(
            data_path=str(crossdocked_config.data_dir), config=crossdocked_config, rngs=rngs
        )

        batch = dataset.get_batch(batch_size=4)

        # Check batch structure
        assert "protein_coords" in batch
        assert "protein_types" in batch
        assert "protein_masks" in batch
        assert "ligand_coords" in batch
        assert "ligand_types" in batch
        assert "ligand_masks" in batch
        assert "binding_affinities" in batch

        # Check batch size from config
        assert batch["binding_affinities"].shape[0] == 4  # batch_size from metadata
        assert batch["protein_coords"].shape == (4, 500, 3)  # batch_size, max_protein_atoms, 3
        assert batch["ligand_coords"].shape == (4, 30, 3)  # batch_size, max_ligand_atoms, 3

    def test_crossdocked_dataset_info(self, crossdocked_config, rngs):
        """Test dataset info with DataConfig."""
        dataset = CrossDockedDataset(
            data_path=str(crossdocked_config.data_dir), config=crossdocked_config, rngs=rngs
        )

        info = dataset.get_dataset_info()

        assert info["name"] == "crossdocked2020"
        assert info["modality"] == "molecular"
        assert info["max_protein_atoms"] == 500
        assert info["max_ligand_atoms"] == 30
        assert info["pocket_radius"] == 8.0
        assert info["split"] == "train"
        assert info["n_samples"] == 100
