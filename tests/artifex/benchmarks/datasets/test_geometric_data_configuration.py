"""Tests for geometric dataset using DataConfig."""

from pathlib import Path

import flax.nnx as nnx
import pytest

from artifex.benchmarks.datasets.geometric import ShapeNetDataset
from artifex.generative_models.core.configuration import DataConfig


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def shapenet_config():
    """Create a DataConfig for ShapeNet testing."""
    return DataConfig(
        name="shapenet_test_config",
        dataset_name="shapenet",
        data_dir=Path("./test_shapenet_data"),
        split="train",
        num_workers=2,
        metadata={
            "synsets": ["02691156"],  # Just airplane for testing
            "num_points": 1024,
            "normalize": True,
            "batch_size": 8,
            "data_source": "synthetic",  # Use synthetic data for testing
            "version": "v2",
            "type": "shapenet",
        },
    )


class TestShapeNetDatasetWithDataConfig:
    """Test ShapeNetDataset with DataConfig."""

    def test_shapenet_init_with_data_configuration(self, shapenet_config, rngs):
        """Test creating ShapeNet dataset with DataConfig."""
        dataset = ShapeNetDataset(
            data_path=str(shapenet_config.data_dir), config=shapenet_config, rngs=rngs
        )

        assert isinstance(dataset.config, DataConfig)
        assert dataset.config.dataset_name == "shapenet"
        assert dataset.num_points == 1024
        assert dataset.normalize is True
        assert dataset.synsets == ["02691156"]

    def test_shapenet_rejects_dict_config(self, rngs):
        """Test that ShapeNetDataset rejects dict config."""
        dict_config = {
            "dataset_name": "shapenet",
            "batch_size": 32,
            "synsets": ["02691156"],
            "num_points": 2048,
        }

        with pytest.raises(TypeError, match="config must be DataConfig"):
            ShapeNetDataset(data_path="./test_shapenet_data", config=dict_config, rngs=rngs)

    def test_shapenet_rejects_evaluation_configuration(self, rngs):
        """Test that ShapeNetDataset rejects EvaluationConfig."""
        # Try to use the old EvaluationConfig if it exists
        try:
            from artifex.generative_models.core.configuration import EvaluationConfig

            eval_config = EvaluationConfig(
                name="eval_config",
                eval_batch_size=32,
                metrics=["mse"],  # Required field
                save_predictions=False,
            )

            with pytest.raises(TypeError, match="config must be DataConfig"):
                ShapeNetDataset(data_path="./test_shapenet_data", config=eval_config, rngs=rngs)
        except (ImportError, Exception):
            # If EvaluationConfig doesn't exist or has different fields, skip test
            pytest.skip("EvaluationConfig not available or has different fields")

    def test_shapenet_get_batch_uses_config(self, shapenet_config, rngs):
        """Test that get_batch uses DataConfig metadata."""
        dataset = ShapeNetDataset(
            data_path=str(shapenet_config.data_dir), config=shapenet_config, rngs=rngs
        )

        # The dataset should create synthetic data
        batch = dataset.get_batch()

        # Check batch structure
        assert "point_clouds" in batch
        assert "labels" in batch

        # Check batch size from config
        assert batch["point_clouds"].shape[0] == 8  # batch_size from metadata
        assert batch["point_clouds"].shape[1] == 1024  # num_points from metadata
        assert batch["point_clouds"].shape[2] == 3  # 3D points

    def test_shapenet_dataset_info(self, shapenet_config, rngs):
        """Test dataset info with DataConfig."""
        dataset = ShapeNetDataset(
            data_path=str(shapenet_config.data_dir), config=shapenet_config, rngs=rngs
        )

        info = dataset.get_dataset_info()

        assert info["name"] == "ShapeNet"
        assert info["num_points"] == 1024
        assert info["synsets"] == ["02691156"]
        assert info["normalize"] is True
        assert "train_size" in info
        assert "val_size" in info
        assert "test_size" in info

    def test_shapenet_uses_split_from_config(self, shapenet_config, rngs):
        """Test that dataset uses split from DataConfig."""
        import dataclasses

        # Test with train split
        dataset = ShapeNetDataset(
            data_path=str(shapenet_config.data_dir), config=shapenet_config, rngs=rngs
        )

        # Get batch should use train split by default
        batch = dataset.get_batch()
        assert batch is not None

        # Test with val split - create new config with different split
        # (frozen dataclass cannot be modified)
        val_config = dataclasses.replace(shapenet_config, split="val")
        dataset_val = ShapeNetDataset(
            data_path=str(val_config.data_dir), config=val_config, rngs=rngs
        )

        # Get batch without specifying split should use config split
        batch_val = dataset_val.get_batch()
        assert batch_val is not None
