"""Tests for TrainingConfig dataclass.

Following TDD approach: These tests define the expected behavior of TrainingConfig
before implementation. All tests should pass once implementation is complete.
"""

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from artifex.generative_models.core.configuration.optimizer_config import OptimizerConfig
from artifex.generative_models.core.configuration.scheduler_config import SchedulerConfig
from artifex.generative_models.core.configuration.training_config import TrainingConfig


class TestTrainingConfigBasics:
    """Test basic TrainingConfig functionality."""

    def test_create_minimal_config(self):
        """Test creating a minimal valid TrainingConfig."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        config = TrainingConfig(
            name="test_training",
            optimizer=optimizer,
        )

        assert config.name == "test_training"
        assert config.optimizer == optimizer
        # Check defaults
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.gradient_clip_norm == 1.0
        assert config.scheduler is None
        assert config.checkpoint_dir == Path("./checkpoints")
        assert config.save_frequency == 1000
        assert config.max_checkpoints == 5
        assert config.log_frequency == 100
        assert config.use_wandb is False
        assert config.wandb_project is None

    def test_create_full_config(self):
        """Test creating a fully specified TrainingConfig."""
        optimizer = OptimizerConfig(
            name="adamw_opt",
            optimizer_type="adamw",
            learning_rate=0.0001,
            weight_decay=0.01,
        )

        scheduler = SchedulerConfig(
            name="cosine_sched",
            scheduler_type="cosine",
            warmup_steps=1000,
            min_lr_ratio=0.01,
        )

        config = TrainingConfig(
            name="full_training",
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=64,
            num_epochs=200,
            gradient_clip_norm=2.0,
            checkpoint_dir=Path("./my_checkpoints"),
            save_frequency=500,
            max_checkpoints=10,
            log_frequency=50,
            use_wandb=True,
            wandb_project="my_project",
            metadata={"experiment": "test_001"},
        )

        assert config.name == "full_training"
        assert config.optimizer == optimizer
        assert config.scheduler == scheduler
        assert config.batch_size == 64
        assert config.num_epochs == 200
        assert config.gradient_clip_norm == 2.0
        assert config.checkpoint_dir == Path("./my_checkpoints")
        assert config.save_frequency == 500
        assert config.max_checkpoints == 10
        assert config.log_frequency == 50
        assert config.use_wandb is True
        assert config.wandb_project == "my_project"
        assert config.metadata == {"experiment": "test_001"}

    def test_create_without_scheduler(self):
        """Test creating TrainingConfig without scheduler."""
        optimizer = OptimizerConfig(
            name="sgd_opt",
            optimizer_type="sgd",
            learning_rate=0.01,
            momentum=0.9,
        )

        config = TrainingConfig(
            name="no_scheduler_training",
            optimizer=optimizer,
        )

        assert config.scheduler is None

    def test_is_frozen(self):
        """Test that TrainingConfig is immutable (frozen)."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        config = TrainingConfig(
            name="test",
            optimizer=optimizer,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.batch_size = 64

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.num_epochs = 200

    def test_equality(self):
        """Test equality comparison between configs."""
        optimizer1 = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )
        optimizer2 = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )
        optimizer3 = OptimizerConfig(
            name="sgd_opt",
            optimizer_type="sgd",
            learning_rate=0.01,
        )

        config1 = TrainingConfig(
            name="test",
            optimizer=optimizer1,
            batch_size=32,
        )
        config2 = TrainingConfig(
            name="test",
            optimizer=optimizer2,
            batch_size=32,
        )
        config3 = TrainingConfig(
            name="test",
            optimizer=optimizer3,
            batch_size=32,
        )

        assert config1 == config2
        assert config1 != config3


class TestTrainingConfigFromDict:
    """Test from_dict() class method."""

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        data = {
            "name": "test_training",
            "optimizer": {
                "name": "adam_opt",
                "optimizer_type": "adam",
                "learning_rate": 0.001,
            },
        }

        config = TrainingConfig.from_dict(data)

        assert config.name == "test_training"
        assert config.optimizer.optimizer_type == "adam"
        assert config.optimizer.learning_rate == 0.001

    def test_from_dict_with_scheduler(self):
        """Test from_dict with nested scheduler config."""
        data = {
            "name": "test_training",
            "optimizer": {
                "name": "adamw_opt",
                "optimizer_type": "adamw",
                "learning_rate": 0.0001,
            },
            "scheduler": {
                "name": "cosine_sched",
                "scheduler_type": "cosine",
                "warmup_steps": 1000,
            },
        }

        config = TrainingConfig.from_dict(data)

        assert config.optimizer.optimizer_type == "adamw"
        assert config.scheduler is not None
        assert config.scheduler.scheduler_type == "cosine"
        assert config.scheduler.warmup_steps == 1000

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "name": "full_training",
            "optimizer": {
                "name": "adam_opt",
                "optimizer_type": "adam",
                "learning_rate": 0.001,
            },
            "scheduler": {
                "name": "linear_sched",
                "scheduler_type": "linear",
                "total_steps": 100000,
            },
            "batch_size": 64,
            "num_epochs": 200,
            "gradient_clip_norm": 2.0,
            "checkpoint_dir": "./my_checkpoints",
            "save_frequency": 500,
            "max_checkpoints": 10,
            "log_frequency": 50,
            "use_wandb": True,
            "wandb_project": "my_project",
        }

        config = TrainingConfig.from_dict(data)

        assert config.batch_size == 64
        assert config.num_epochs == 200
        assert config.checkpoint_dir == Path("./my_checkpoints")
        assert config.use_wandb is True

    def test_from_dict_strict_mode(self):
        """Test from_dict rejects unknown fields in strict mode."""
        data = {
            "name": "test",
            "optimizer": {
                "name": "adam_opt",
                "optimizer_type": "adam",
                "learning_rate": 0.001,
            },
            "unknown_field": "should_fail",
        }

        with pytest.raises(Exception):  # dacite raises for unknown fields
            TrainingConfig.from_dict(data)


class TestTrainingConfigValidation:
    """Test validation in __post_init__."""

    def test_negative_batch_size_raises(self):
        """Test that negative or zero batch_size raises error."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                batch_size=-1,
            )

        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                batch_size=0,
            )

    def test_valid_batch_sizes(self):
        """Test that valid batch_size values are accepted."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        valid_values = [1, 16, 32, 64, 128, 256]

        for batch_size in valid_values:
            config = TrainingConfig(
                name="test",
                optimizer=optimizer,
                batch_size=batch_size,
            )
            assert config.batch_size == batch_size

    def test_negative_num_epochs_raises(self):
        """Test that negative or zero num_epochs raises error."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        with pytest.raises(ValueError, match="num_epochs"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                num_epochs=-1,
            )

        with pytest.raises(ValueError, match="num_epochs"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                num_epochs=0,
            )

    def test_negative_gradient_clip_norm_raises(self):
        """Test that negative gradient_clip_norm raises error."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        with pytest.raises(ValueError, match="gradient_clip_norm"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                gradient_clip_norm=-1.0,
            )

    def test_gradient_clip_norm_none_allowed(self):
        """Test that gradient_clip_norm=None is allowed."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        config = TrainingConfig(
            name="test",
            optimizer=optimizer,
            gradient_clip_norm=None,
        )

        assert config.gradient_clip_norm is None

    def test_negative_save_frequency_raises(self):
        """Test that negative or zero save_frequency raises error."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        with pytest.raises(ValueError, match="save_frequency"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                save_frequency=-100,
            )

        with pytest.raises(ValueError, match="save_frequency"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                save_frequency=0,
            )

    def test_negative_max_checkpoints_raises(self):
        """Test that negative or zero max_checkpoints raises error."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        with pytest.raises(ValueError, match="max_checkpoints"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                max_checkpoints=-1,
            )

        with pytest.raises(ValueError, match="max_checkpoints"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                max_checkpoints=0,
            )

    def test_negative_log_frequency_raises(self):
        """Test that negative or zero log_frequency raises error."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        with pytest.raises(ValueError, match="log_frequency"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                log_frequency=-10,
            )

        with pytest.raises(ValueError, match="log_frequency"):
            TrainingConfig(
                name="test",
                optimizer=optimizer,
                log_frequency=0,
            )


class TestTrainingConfigSerialization:
    """Test YAML serialization."""

    def test_to_yaml(self):
        """Test saving config to YAML."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        config = TrainingConfig(
            name="test_training",
            optimizer=optimizer,
            batch_size=64,
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            assert yaml_path.exists()

            # Load and verify
            loaded_config = TrainingConfig.from_yaml(yaml_path)
            assert loaded_config == config

    def test_from_yaml_with_nested_configs(self):
        """Test loading config from YAML with nested optimizer and scheduler."""
        optimizer = OptimizerConfig(
            name="adamw_opt",
            optimizer_type="adamw",
            learning_rate=0.0001,
            weight_decay=0.01,
        )

        scheduler = SchedulerConfig(
            name="cosine_sched",
            scheduler_type="cosine",
            warmup_steps=1000,
        )

        config = TrainingConfig(
            name="yaml_training",
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=128,
            num_epochs=500,
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(yaml_path)

            loaded_config = TrainingConfig.from_yaml(yaml_path)

            assert loaded_config.name == "yaml_training"
            assert loaded_config.optimizer.optimizer_type == "adamw"
            assert loaded_config.scheduler.scheduler_type == "cosine"
            assert loaded_config.batch_size == 128
            assert loaded_config.num_epochs == 500

    def test_yaml_roundtrip_preserves_nested_configs(self):
        """Test that YAML roundtrip preserves nested config objects."""
        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
            beta1=0.9,
        )

        scheduler = SchedulerConfig(
            name="exp_sched",
            scheduler_type="exponential",
            decay_rate=0.96,
        )

        original = TrainingConfig(
            name="roundtrip",
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=32,
            use_wandb=True,
            wandb_project="test_project",
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "roundtrip.yaml"
            original.to_yaml(yaml_path)
            loaded = TrainingConfig.from_yaml(yaml_path)

            # Verify nested configs preserved
            assert loaded.optimizer.learning_rate == 0.001
            assert loaded.optimizer.beta1 == 0.9
            assert loaded.scheduler.decay_rate == 0.96
            assert loaded.wandb_project == "test_project"


class TestTrainingConfigInheritance:
    """Test that TrainingConfig can be inherited."""

    def test_can_inherit_from_training_config(self):
        """Test creating a specialized config by inheriting."""

        @dataclasses.dataclass(frozen=True)
        class CustomTrainingConfig(TrainingConfig):
            """Specialized config for custom training."""

            custom_param: float = 1.0

            def __post_init__(self):
                """Validate custom fields."""
                # Call parent validation first
                super().__post_init__()

                # Validate custom fields
                if self.custom_param < 0:
                    raise ValueError("custom_param must be non-negative")

        optimizer = OptimizerConfig(
            name="adam_opt",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        # Create custom config
        custom_config = CustomTrainingConfig(
            name="custom_train",
            optimizer=optimizer,
            custom_param=2.0,
        )

        assert custom_config.name == "custom_train"
        assert custom_config.custom_param == 2.0

        # Test inheritance validation works
        with pytest.raises(ValueError, match="custom_param"):
            CustomTrainingConfig(
                name="custom_train",
                optimizer=optimizer,
                custom_param=-1.0,
            )
