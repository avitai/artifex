"""Tests for OptimizerConfig dataclass.

Following TDD approach: These tests define the expected behavior of OptimizerConfig
before implementation. All tests should pass once implementation is complete.
"""

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from artifex.generative_models.core.configuration.optimizer_config import OptimizerConfig


class TestOptimizerConfigBasics:
    """Test basic OptimizerConfig functionality."""

    def test_create_minimal_config(self):
        """Test creating a minimal valid OptimizerConfig."""
        config = OptimizerConfig(
            name="test_optimizer",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        assert config.name == "test_optimizer"
        assert config.optimizer_type == "adam"
        assert config.learning_rate == 0.001
        # Check defaults
        assert config.weight_decay == 0.0
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999
        assert config.eps == 1e-8
        assert config.momentum == 0.0
        assert config.nesterov is False
        assert config.initial_accumulator_value == 0.1
        assert config.gradient_clip_norm is None
        assert config.gradient_clip_value is None

    def test_create_full_adam_config(self):
        """Test creating a fully specified Adam OptimizerConfig."""
        config = OptimizerConfig(
            name="adam_optimizer",
            optimizer_type="adamw",
            learning_rate=0.0001,
            weight_decay=0.01,
            beta1=0.95,
            beta2=0.9995,
            eps=1e-7,
            gradient_clip_norm=1.0,
            metadata={"experiment": "test_001"},
        )

        assert config.name == "adam_optimizer"
        assert config.optimizer_type == "adamw"
        assert config.learning_rate == 0.0001
        assert config.weight_decay == 0.01
        assert config.beta1 == 0.95
        assert config.beta2 == 0.9995
        assert config.eps == 1e-7
        assert config.gradient_clip_norm == 1.0
        assert config.metadata == {"experiment": "test_001"}

    def test_create_sgd_config(self):
        """Test creating SGD OptimizerConfig."""
        config = OptimizerConfig(
            name="sgd_optimizer",
            optimizer_type="sgd",
            learning_rate=0.01,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0001,
        )

        assert config.optimizer_type == "sgd"
        assert config.momentum == 0.9
        assert config.nesterov is True
        assert config.weight_decay == 0.0001

    def test_is_frozen(self):
        """Test that OptimizerConfig is immutable (frozen)."""
        config = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=0.001,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.learning_rate = 0.01

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.optimizer_type = "sgd"

    def test_equality(self):
        """Test equality comparison between configs."""
        config1 = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=0.001,
        )
        config2 = OptimizerConfig(
            name="test",
            optimizer_type="adam",
            learning_rate=0.001,
        )
        config3 = OptimizerConfig(
            name="test",
            optimizer_type="sgd",
            learning_rate=0.001,
        )

        assert config1 == config2
        assert config1 != config3


class TestOptimizerConfigFromDict:
    """Test from_dict() class method."""

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        data = {
            "name": "test_optimizer",
            "optimizer_type": "adam",
            "learning_rate": 0.001,
        }

        config = OptimizerConfig.from_dict(data)

        assert config.name == "test_optimizer"
        assert config.optimizer_type == "adam"
        assert config.learning_rate == 0.001

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "name": "full_optimizer",
            "optimizer_type": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "beta1": 0.95,
            "beta2": 0.9995,
            "eps": 1e-7,
            "momentum": 0.0,
            "nesterov": False,
            "initial_accumulator_value": 0.1,
            "gradient_clip_norm": 1.0,
            "gradient_clip_value": None,
        }

        config = OptimizerConfig.from_dict(data)

        assert config.name == "full_optimizer"
        assert config.learning_rate == 0.0001
        assert config.gradient_clip_norm == 1.0

    def test_from_dict_strict_mode(self):
        """Test from_dict rejects unknown fields in strict mode."""
        data = {
            "name": "test",
            "optimizer_type": "adam",
            "learning_rate": 0.001,
            "unknown_field": "should_fail",
        }

        with pytest.raises(Exception):  # dacite raises for unknown fields
            OptimizerConfig.from_dict(data)


class TestOptimizerConfigValidation:
    """Test validation in __post_init__."""

    def test_invalid_optimizer_type_raises(self):
        """Test that invalid optimizer type raises error."""
        with pytest.raises(ValueError, match="optimizer_type"):
            OptimizerConfig(
                name="test",
                optimizer_type="invalid_optimizer",
                learning_rate=0.001,
            )

    def test_valid_optimizer_types(self):
        """Test that all valid optimizer types are accepted."""
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad", "lamb", "radam", "nadam"]

        for opt_type in valid_optimizers:
            config = OptimizerConfig(
                name="test",
                optimizer_type=opt_type,
                learning_rate=0.001,
            )
            assert config.optimizer_type == opt_type

    def test_negative_learning_rate_raises(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=-0.001,
            )

    def test_zero_learning_rate_raises(self):
        """Test that zero learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.0,
            )

    def test_negative_weight_decay_raises(self):
        """Test that negative weight decay raises error."""
        with pytest.raises(ValueError, match="weight_decay"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                weight_decay=-0.01,
            )

    def test_valid_weight_decay(self):
        """Test that valid weight decay values are accepted."""
        valid_values = [0.0, 0.01, 0.1, 1.0]

        for wd in valid_values:
            config = OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                weight_decay=wd,
            )
            assert config.weight_decay == wd

    def test_beta1_out_of_range_raises(self):
        """Test that beta1 outside [0, 1] raises error."""
        # Negative beta1
        with pytest.raises(ValueError, match="beta1"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                beta1=-0.1,
            )

        # beta1 > 1
        with pytest.raises(ValueError, match="beta1"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                beta1=1.5,
            )

    def test_beta2_out_of_range_raises(self):
        """Test that beta2 outside [0, 1] raises error."""
        # Negative beta2
        with pytest.raises(ValueError, match="beta2"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                beta2=-0.1,
            )

        # beta2 > 1
        with pytest.raises(ValueError, match="beta2"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                beta2=1.5,
            )

    def test_negative_eps_raises(self):
        """Test that negative or zero eps raises error."""
        with pytest.raises(ValueError, match="eps"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                eps=-1e-8,
            )

        with pytest.raises(ValueError, match="eps"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                eps=0.0,
            )

    def test_momentum_out_of_range_raises(self):
        """Test that momentum outside [0, 1] raises error."""
        # Negative momentum
        with pytest.raises(ValueError, match="momentum"):
            OptimizerConfig(
                name="test",
                optimizer_type="sgd",
                learning_rate=0.01,
                momentum=-0.1,
            )

        # momentum > 1
        with pytest.raises(ValueError, match="momentum"):
            OptimizerConfig(
                name="test",
                optimizer_type="sgd",
                learning_rate=0.01,
                momentum=1.5,
            )

    def test_negative_initial_accumulator_value_raises(self):
        """Test that negative initial_accumulator_value raises error."""
        with pytest.raises(ValueError, match="initial_accumulator_value"):
            OptimizerConfig(
                name="test",
                optimizer_type="adagrad",
                learning_rate=0.01,
                initial_accumulator_value=-0.1,
            )

    def test_negative_gradient_clip_norm_raises(self):
        """Test that negative gradient_clip_norm raises error."""
        with pytest.raises(ValueError, match="gradient_clip_norm"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                gradient_clip_norm=-1.0,
            )

    def test_negative_gradient_clip_value_raises(self):
        """Test that negative gradient_clip_value raises error."""
        with pytest.raises(ValueError, match="gradient_clip_value"):
            OptimizerConfig(
                name="test",
                optimizer_type="adam",
                learning_rate=0.001,
                gradient_clip_value=-1.0,
            )


class TestOptimizerConfigSerialization:
    """Test YAML serialization."""

    def test_to_yaml(self):
        """Test saving config to YAML."""
        config = OptimizerConfig(
            name="test_optimizer",
            optimizer_type="adamw",
            learning_rate=0.0001,
            weight_decay=0.01,
            beta1=0.95,
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            assert yaml_path.exists()

            # Load and verify
            loaded_config = OptimizerConfig.from_yaml(yaml_path)
            assert loaded_config == config

    def test_from_yaml(self):
        """Test loading config from YAML."""
        config = OptimizerConfig(
            name="yaml_optimizer",
            optimizer_type="sgd",
            learning_rate=0.01,
            momentum=0.9,
            nesterov=True,
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(yaml_path)

            loaded_config = OptimizerConfig.from_yaml(yaml_path)

            assert loaded_config.name == "yaml_optimizer"
            assert loaded_config.optimizer_type == "sgd"
            assert loaded_config.learning_rate == 0.01
            assert loaded_config.momentum == 0.9
            assert loaded_config.nesterov is True

    def test_yaml_roundtrip_preserves_values(self):
        """Test that YAML roundtrip preserves all values."""
        original = OptimizerConfig(
            name="roundtrip",
            optimizer_type="adam",
            learning_rate=0.001,
            gradient_clip_norm=1.0,
            metadata={"key": "value"},
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "roundtrip.yaml"
            original.to_yaml(yaml_path)
            loaded = OptimizerConfig.from_yaml(yaml_path)

            # Verify all fields preserved
            assert loaded.learning_rate == 0.001
            assert loaded.gradient_clip_norm == 1.0
            assert loaded.metadata == {"key": "value"}


class TestOptimizerConfigInheritance:
    """Test that OptimizerConfig can be inherited."""

    def test_can_inherit_from_optimizer_config(self):
        """Test creating a specialized config by inheriting."""

        @dataclasses.dataclass(frozen=True)
        class CustomOptimizerConfig(OptimizerConfig):
            """Specialized config for custom optimizer."""

            custom_param: float = 1.0

            def __post_init__(self):
                """Validate custom fields."""
                # Call parent validation first
                super().__post_init__()

                # Validate custom fields
                if self.custom_param < 0:
                    raise ValueError("custom_param must be non-negative")

        # Create custom config
        custom_config = CustomOptimizerConfig(
            name="custom_opt",
            optimizer_type="adam",
            learning_rate=0.001,
            custom_param=2.0,
        )

        assert custom_config.name == "custom_opt"
        assert custom_config.custom_param == 2.0

        # Test inheritance validation works
        with pytest.raises(ValueError, match="custom_param"):
            CustomOptimizerConfig(
                name="custom_opt",
                optimizer_type="adam",
                learning_rate=0.001,
                custom_param=-1.0,
            )
