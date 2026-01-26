"""Tests for ExperimentConfig frozen dataclass.

This module tests the ExperimentConfig frozen dataclass which replaces the
Pydantic-based ExperimentConfiguration. Tests verify:
1. Basic instantiation and frozen behavior
2. Field validation (types, ranges, required fields)
3. Nested configuration handling
4. Default values
5. Serialization (to_dict, from_dict)
6. Edge cases and error handling
"""

import dataclasses
from pathlib import Path

import pytest

from artifex.generative_models.core.configuration.data_config import DataConfig
from artifex.generative_models.core.configuration.evaluation_config import EvaluationConfig
from artifex.generative_models.core.configuration.experiment_config import ExperimentConfig
from artifex.generative_models.core.configuration.model_config import ModelConfig
from artifex.generative_models.core.configuration.optimizer_config import OptimizerConfig
from artifex.generative_models.core.configuration.training_config import TrainingConfig


class TestExperimentConfigBasics:
    """Test basic ExperimentConfig functionality."""

    def test_create_minimal(self):
        """Test creating ExperimentConfig with minimal required fields."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="experiment",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        assert config.name == "experiment"
        assert config.model_cfg == model_cfg
        assert config.training_cfg == training_cfg
        assert config.data_cfg == data_cfg
        assert config.eval_cfg is None  # default
        assert config.seed == 42  # default
        assert config.deterministic is True  # default
        assert config.output_dir == Path("./experiments")  # default

    def test_create_full(self):
        """Test creating ExperimentConfig with all fields."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="cifar10")
        eval_cfg = EvaluationConfig(name="eval", metrics=("fid", "ssim"))

        config = ExperimentConfig(
            name="full-experiment",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            eval_cfg=eval_cfg,
            seed=123,
            deterministic=False,
            output_dir=Path("/path/to/experiments"),
            track_carbon=True,
            track_memory=True,
            description="Full experiment configuration",
            tags=("gan", "vision"),
        )
        assert config.name == "full-experiment"
        assert config.model_cfg == model_cfg
        assert config.training_cfg == training_cfg
        assert config.data_cfg == data_cfg
        assert config.eval_cfg == eval_cfg
        assert config.seed == 123
        assert config.deterministic is False
        assert config.output_dir == Path("/path/to/experiments")
        assert config.track_carbon is True
        assert config.track_memory is True
        assert config.description == "Full experiment configuration"
        assert config.tags == ("gan", "vision")

    def test_frozen(self):
        """Test that ExperimentConfig is frozen (immutable)."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.seed = 100

    def test_hash(self):
        """Test that ExperimentConfig instances are not hashable due to nested configs."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        # Configs with nested unhashable configs are not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config)


class TestExperimentConfigValidation:
    """Test ExperimentConfig validation."""

    def test_model_cfg_required(self):
        """Test that model_cfg is required."""
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        with pytest.raises(ValueError, match="model_cfg is required"):
            ExperimentConfig(
                name="test",
                training_cfg=training_cfg,
                data_cfg=data_cfg,
            )

    def test_training_cfg_required(self):
        """Test that training_cfg is required."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        with pytest.raises(ValueError, match="training_cfg is required"):
            ExperimentConfig(
                name="test",
                model_cfg=model_cfg,
                data_cfg=data_cfg,
            )

    def test_data_cfg_required(self):
        """Test that data_cfg is required."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)

        with pytest.raises(ValueError, match="data_cfg is required"):
            ExperimentConfig(
                name="test",
                model_cfg=model_cfg,
                training_cfg=training_cfg,
            )

    def test_eval_cfg_optional(self):
        """Test that eval_cfg is optional."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            eval_cfg=None,
        )
        assert config.eval_cfg is None

    def test_seed_negative_valid(self):
        """Test that negative seed is valid."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            seed=-1,
        )
        assert config.seed == -1

    def test_seed_zero_valid(self):
        """Test that zero seed is valid."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            seed=0,
        )
        assert config.seed == 0


class TestExperimentConfigDefaults:
    """Test ExperimentConfig default values."""

    def test_default_eval_cfg(self):
        """Test default eval_cfg is None."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        assert config.eval_cfg is None

    def test_default_seed(self):
        """Test default seed is 42."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        assert config.seed == 42

    def test_default_deterministic(self):
        """Test default deterministic is True."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        assert config.deterministic is True

    def test_default_output_dir(self):
        """Test default output_dir is ./experiments."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        assert config.output_dir == Path("./experiments")

    def test_default_track_carbon(self):
        """Test default track_carbon is False."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        assert config.track_carbon is False

    def test_default_track_memory(self):
        """Test default track_memory is False."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        assert config.track_memory is False


class TestExperimentConfigSerialization:
    """Test ExperimentConfig serialization."""

    def test_to_dict_minimal(self):
        """Test converting minimal config to dict."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
        )
        data = config.to_dict()
        assert data["name"] == "test"
        assert isinstance(data["model_cfg"], dict)
        assert isinstance(data["training_cfg"], dict)
        assert isinstance(data["data_cfg"], dict)
        assert data["eval_cfg"] is None
        assert data["seed"] == 42

    def test_to_dict_full(self):
        """Test converting full config to dict."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")
        eval_cfg = EvaluationConfig(name="eval", metrics=("fid", "ssim"))

        config = ExperimentConfig(
            name="full-experiment",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            eval_cfg=eval_cfg,
            seed=123,
            deterministic=False,
            output_dir=Path("/path/to/experiments"),
        )
        data = config.to_dict()
        assert data["name"] == "full-experiment"
        assert isinstance(data["model_cfg"], dict)
        assert isinstance(data["training_cfg"], dict)
        assert isinstance(data["data_cfg"], dict)
        assert isinstance(data["eval_cfg"], dict)
        assert data["seed"] == 123
        assert data["deterministic"] is False

    def test_from_dict_minimal(self):
        """Test creating config from minimal dict."""
        data = {
            "name": "test",
            "model_cfg": {
                "name": "model",
                "model_class": "MLP",
                "input_dim": 784,
                "output_dim": 10,
            },
            "training_cfg": {
                "name": "training",
                "optimizer": {
                    "name": "adam_opt",
                    "optimizer_type": "adam",
                    "learning_rate": 0.001,
                },
            },
            "data_cfg": {"name": "data", "dataset_name": "mnist"},
        }

        config = ExperimentConfig.from_dict(data)
        assert config.name == "test"
        assert config.model_cfg.model_class == "MLP"
        assert config.training_cfg.optimizer.optimizer_type == "adam"
        assert config.data_cfg.dataset_name == "mnist"
        assert config.seed == 42  # default

    def test_from_dict_with_nested_configs(self):
        """Test creating config from dict with all nested configs."""
        data = {
            "name": "full-experiment",
            "model_cfg": {
                "name": "model",
                "model_class": "CNN",
                "input_dim": 784,
                "output_dim": 10,
            },
            "training_cfg": {
                "name": "training",
                "optimizer": {
                    "name": "adamw_opt",
                    "optimizer_type": "adamw",
                    "learning_rate": 0.0001,
                    "weight_decay": 0.01,
                },
                "batch_size": 64,
            },
            "data_cfg": {"name": "data", "dataset_name": "cifar10"},
            "eval_cfg": {"name": "eval", "metrics": ["fid", "ssim"]},
            "seed": 123,
            "deterministic": False,
        }

        config = ExperimentConfig.from_dict(data)
        assert config.name == "full-experiment"
        assert config.model_cfg.model_class == "CNN"
        assert config.training_cfg.optimizer.optimizer_type == "adamw"
        assert config.training_cfg.batch_size == 64
        assert config.data_cfg.dataset_name == "cifar10"
        assert config.eval_cfg.metrics == ("fid", "ssim")
        assert config.seed == 123
        assert config.deterministic is False

    def test_roundtrip(self):
        """Test that to_dict -> from_dict roundtrip works."""
        model_cfg = ModelConfig(name="model", model_class="ResNet", input_dim=224, output_dim=1000)
        optimizer_cfg = OptimizerConfig(
            name="adam_opt", optimizer_type="adam", learning_rate=0.001, beta1=0.9, beta2=0.999
        )
        training_cfg = TrainingConfig(
            name="training", optimizer=optimizer_cfg, batch_size=128, num_epochs=50
        )
        data_cfg = DataConfig(name="data", dataset_name="imagenet")
        eval_cfg = EvaluationConfig(name="eval", metrics=("fid", "inception_score"))

        original = ExperimentConfig(
            name="roundtrip-test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            eval_cfg=eval_cfg,
            seed=42,
            deterministic=True,
        )

        data = original.to_dict()
        restored = ExperimentConfig.from_dict(data)

        assert restored.name == original.name
        assert restored.model_cfg.model_class == original.model_cfg.model_class
        assert (
            restored.training_cfg.optimizer.learning_rate
            == original.training_cfg.optimizer.learning_rate
        )
        assert restored.data_cfg.dataset_name == original.data_cfg.dataset_name
        assert restored.eval_cfg.metrics == original.eval_cfg.metrics
        assert restored.seed == original.seed
        assert restored.deterministic == original.deterministic


class TestExperimentConfigEdgeCases:
    """Test ExperimentConfig edge cases."""

    def test_output_dir_as_string(self):
        """Test that output_dir accepts string and converts to Path."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            output_dir="/path/to/experiments",
        )
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/path/to/experiments")

    def test_output_dir_as_path(self):
        """Test that output_dir accepts Path objects."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            output_dir=Path("/path/to/experiments"),
        )
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("/path/to/experiments")

    def test_all_tracking_enabled(self):
        """Test that all tracking flags can be enabled."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            track_carbon=True,
            track_memory=True,
        )
        assert config.track_carbon is True
        assert config.track_memory is True

    def test_deterministic_false(self):
        """Test that deterministic can be False."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            deterministic=False,
        )
        assert config.deterministic is False

    def test_large_seed(self):
        """Test that large seed values are valid."""
        model_cfg = ModelConfig(name="model", model_class="MLP", input_dim=784, output_dim=10)
        optimizer_cfg = OptimizerConfig(name="adam_opt", optimizer_type="adam", learning_rate=0.001)
        training_cfg = TrainingConfig(name="training", optimizer=optimizer_cfg)
        data_cfg = DataConfig(name="data", dataset_name="mnist")

        config = ExperimentConfig(
            name="test",
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            seed=2**31 - 1,  # Max 32-bit signed int
        )
        assert config.seed == 2**31 - 1
