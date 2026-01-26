"""Tests for SchedulerConfig dataclass.

Following TDD approach: These tests define the expected behavior of SchedulerConfig
before implementation. All tests should pass once implementation is complete.
"""

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from artifex.generative_models.core.configuration.scheduler_config import SchedulerConfig


class TestSchedulerConfigBasics:
    """Test basic SchedulerConfig functionality."""

    def test_create_minimal_config(self):
        """Test creating a minimal valid SchedulerConfig."""
        config = SchedulerConfig(
            name="test_scheduler",
            scheduler_type="cosine",
        )

        assert config.name == "test_scheduler"
        assert config.scheduler_type == "cosine"
        # Check defaults
        assert config.warmup_steps == 0
        assert config.min_lr_ratio == 0.0
        assert config.cycle_length is None
        assert config.decay_rate == 0.95
        assert config.decay_steps == 1000
        assert config.total_steps is None
        assert config.step_size == 1000
        assert config.gamma == 0.1
        assert config.milestones == ()

    def test_create_cosine_scheduler(self):
        """Test creating a cosine scheduler config."""
        config = SchedulerConfig(
            name="cosine_scheduler",
            scheduler_type="cosine",
            warmup_steps=1000,
            min_lr_ratio=0.01,
            cycle_length=50000,
        )

        assert config.scheduler_type == "cosine"
        assert config.warmup_steps == 1000
        assert config.min_lr_ratio == 0.01
        assert config.cycle_length == 50000

    def test_create_exponential_scheduler(self):
        """Test creating an exponential scheduler config."""
        config = SchedulerConfig(
            name="exp_scheduler",
            scheduler_type="exponential",
            decay_rate=0.96,
            decay_steps=2000,
        )

        assert config.scheduler_type == "exponential"
        assert config.decay_rate == 0.96
        assert config.decay_steps == 2000

    def test_create_linear_scheduler(self):
        """Test creating a linear scheduler config."""
        config = SchedulerConfig(
            name="linear_scheduler",
            scheduler_type="linear",
            warmup_steps=500,
            total_steps=100000,
        )

        assert config.scheduler_type == "linear"
        assert config.warmup_steps == 500
        assert config.total_steps == 100000

    def test_create_multistep_scheduler(self):
        """Test creating a multistep scheduler config."""
        config = SchedulerConfig(
            name="multistep_scheduler",
            scheduler_type="multistep",
            milestones=(10000, 20000, 30000),
            gamma=0.1,
        )

        assert config.scheduler_type == "multistep"
        assert config.milestones == (10000, 20000, 30000)
        assert config.gamma == 0.1

    def test_is_frozen(self):
        """Test that SchedulerConfig is immutable (frozen)."""
        config = SchedulerConfig(
            name="test",
            scheduler_type="cosine",
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.scheduler_type = "linear"

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.warmup_steps = 1000

    def test_equality(self):
        """Test equality comparison between configs."""
        config1 = SchedulerConfig(
            name="test",
            scheduler_type="cosine",
            warmup_steps=1000,
        )
        config2 = SchedulerConfig(
            name="test",
            scheduler_type="cosine",
            warmup_steps=1000,
        )
        config3 = SchedulerConfig(
            name="test",
            scheduler_type="linear",
            warmup_steps=1000,
        )

        assert config1 == config2
        assert config1 != config3


class TestSchedulerConfigFromDict:
    """Test from_dict() class method."""

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        data = {
            "name": "test_scheduler",
            "scheduler_type": "cosine",
        }

        config = SchedulerConfig.from_dict(data)

        assert config.name == "test_scheduler"
        assert config.scheduler_type == "cosine"

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "name": "full_scheduler",
            "scheduler_type": "multistep",
            "warmup_steps": 500,
            "min_lr_ratio": 0.01,
            "cycle_length": None,
            "decay_rate": 0.95,
            "decay_steps": 1000,
            "total_steps": 100000,
            "step_size": 1000,
            "gamma": 0.1,
            "milestones": [10000, 20000, 30000],  # List should convert to tuple
        }

        config = SchedulerConfig.from_dict(data)

        assert config.name == "full_scheduler"
        assert config.milestones == (10000, 20000, 30000)  # Converted to tuple
        assert isinstance(config.milestones, tuple)

    def test_from_dict_converts_lists_to_tuples(self):
        """Test that from_dict auto-converts lists to tuples."""
        data = {
            "name": "test",
            "scheduler_type": "multistep",
            "milestones": [5000, 10000],  # List
        }

        config = SchedulerConfig.from_dict(data)

        assert config.milestones == (5000, 10000)
        assert isinstance(config.milestones, tuple)

    def test_from_dict_strict_mode(self):
        """Test from_dict rejects unknown fields in strict mode."""
        data = {
            "name": "test",
            "scheduler_type": "cosine",
            "unknown_field": "should_fail",
        }

        with pytest.raises(Exception):  # dacite raises for unknown fields
            SchedulerConfig.from_dict(data)


class TestSchedulerConfigValidation:
    """Test validation in __post_init__."""

    def test_invalid_scheduler_type_raises(self):
        """Test that invalid scheduler type raises error."""
        with pytest.raises(ValueError, match="scheduler_type"):
            SchedulerConfig(
                name="test",
                scheduler_type="invalid_scheduler",
            )

    def test_valid_scheduler_types(self):
        """Test that all valid scheduler types are accepted."""
        valid_schedulers = [
            "constant",
            "linear",
            "cosine",
            "exponential",
            "polynomial",
            "step",
            "multistep",
            "cyclic",
            "one_cycle",
            "none",
        ]

        for sched_type in valid_schedulers:
            config = SchedulerConfig(
                name="test",
                scheduler_type=sched_type,
            )
            assert config.scheduler_type == sched_type

    def test_negative_warmup_steps_raises(self):
        """Test that negative warmup_steps raises error."""
        with pytest.raises(ValueError, match="warmup_steps"):
            SchedulerConfig(
                name="test",
                scheduler_type="cosine",
                warmup_steps=-100,
            )

    def test_valid_warmup_steps(self):
        """Test that valid warmup_steps values are accepted."""
        valid_values = [0, 100, 1000, 10000]

        for warmup in valid_values:
            config = SchedulerConfig(
                name="test",
                scheduler_type="cosine",
                warmup_steps=warmup,
            )
            assert config.warmup_steps == warmup

    def test_min_lr_ratio_out_of_range_raises(self):
        """Test that min_lr_ratio outside [0, 1] raises error."""
        # Negative min_lr_ratio
        with pytest.raises(ValueError, match="min_lr_ratio"):
            SchedulerConfig(
                name="test",
                scheduler_type="cosine",
                min_lr_ratio=-0.1,
            )

        # min_lr_ratio > 1
        with pytest.raises(ValueError, match="min_lr_ratio"):
            SchedulerConfig(
                name="test",
                scheduler_type="cosine",
                min_lr_ratio=1.5,
            )

    def test_valid_min_lr_ratio(self):
        """Test that valid min_lr_ratio values are accepted."""
        valid_values = [0.0, 0.01, 0.5, 1.0]

        for ratio in valid_values:
            config = SchedulerConfig(
                name="test",
                scheduler_type="cosine",
                min_lr_ratio=ratio,
            )
            assert config.min_lr_ratio == ratio

    def test_negative_cycle_length_raises(self):
        """Test that negative cycle_length raises error."""
        with pytest.raises(ValueError, match="cycle_length"):
            SchedulerConfig(
                name="test",
                scheduler_type="cosine",
                cycle_length=-1000,
            )

    def test_zero_cycle_length_raises(self):
        """Test that zero cycle_length raises error."""
        with pytest.raises(ValueError, match="cycle_length"):
            SchedulerConfig(
                name="test",
                scheduler_type="cosine",
                cycle_length=0,
            )

    def test_decay_rate_out_of_range_raises(self):
        """Test that decay_rate outside (0, 1] raises error."""
        # Zero decay_rate
        with pytest.raises(ValueError, match="decay_rate"):
            SchedulerConfig(
                name="test",
                scheduler_type="exponential",
                decay_rate=0.0,
            )

        # Negative decay_rate
        with pytest.raises(ValueError, match="decay_rate"):
            SchedulerConfig(
                name="test",
                scheduler_type="exponential",
                decay_rate=-0.5,
            )

        # decay_rate > 1
        with pytest.raises(ValueError, match="decay_rate"):
            SchedulerConfig(
                name="test",
                scheduler_type="exponential",
                decay_rate=1.5,
            )

    def test_negative_decay_steps_raises(self):
        """Test that negative or zero decay_steps raises error."""
        with pytest.raises(ValueError, match="decay_steps"):
            SchedulerConfig(
                name="test",
                scheduler_type="exponential",
                decay_steps=-100,
            )

        with pytest.raises(ValueError, match="decay_steps"):
            SchedulerConfig(
                name="test",
                scheduler_type="exponential",
                decay_steps=0,
            )

    def test_negative_total_steps_raises(self):
        """Test that negative or zero total_steps raises error."""
        with pytest.raises(ValueError, match="total_steps"):
            SchedulerConfig(
                name="test",
                scheduler_type="linear",
                total_steps=-1000,
            )

        with pytest.raises(ValueError, match="total_steps"):
            SchedulerConfig(
                name="test",
                scheduler_type="linear",
                total_steps=0,
            )

    def test_negative_step_size_raises(self):
        """Test that negative or zero step_size raises error."""
        with pytest.raises(ValueError, match="step_size"):
            SchedulerConfig(
                name="test",
                scheduler_type="step",
                step_size=-100,
            )

        with pytest.raises(ValueError, match="step_size"):
            SchedulerConfig(
                name="test",
                scheduler_type="step",
                step_size=0,
            )

    def test_gamma_out_of_range_raises(self):
        """Test that gamma outside (0, 1] raises error."""
        # Zero gamma
        with pytest.raises(ValueError, match="gamma"):
            SchedulerConfig(
                name="test",
                scheduler_type="step",
                gamma=0.0,
            )

        # Negative gamma
        with pytest.raises(ValueError, match="gamma"):
            SchedulerConfig(
                name="test",
                scheduler_type="step",
                gamma=-0.1,
            )

        # gamma > 1
        with pytest.raises(ValueError, match="gamma"):
            SchedulerConfig(
                name="test",
                scheduler_type="step",
                gamma=1.5,
            )


class TestSchedulerConfigSerialization:
    """Test YAML serialization."""

    def test_to_yaml(self):
        """Test saving config to YAML."""
        config = SchedulerConfig(
            name="test_scheduler",
            scheduler_type="cosine",
            warmup_steps=1000,
            min_lr_ratio=0.01,
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            assert yaml_path.exists()

            # Load and verify
            loaded_config = SchedulerConfig.from_yaml(yaml_path)
            assert loaded_config == config

    def test_from_yaml(self):
        """Test loading config from YAML."""
        config = SchedulerConfig(
            name="yaml_scheduler",
            scheduler_type="multistep",
            milestones=(10000, 20000, 30000),
            gamma=0.1,
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(yaml_path)

            loaded_config = SchedulerConfig.from_yaml(yaml_path)

            assert loaded_config.name == "yaml_scheduler"
            assert loaded_config.scheduler_type == "multistep"
            assert loaded_config.milestones == (10000, 20000, 30000)
            assert loaded_config.gamma == 0.1

    def test_yaml_roundtrip_preserves_types(self):
        """Test that YAML roundtrip preserves all types."""
        original = SchedulerConfig(
            name="roundtrip",
            scheduler_type="linear",
            warmup_steps=500,
            total_steps=100000,
            milestones=(5000, 10000),
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "roundtrip.yaml"
            original.to_yaml(yaml_path)
            loaded = SchedulerConfig.from_yaml(yaml_path)

            # Verify types preserved
            assert isinstance(loaded.warmup_steps, int)
            assert isinstance(loaded.total_steps, int)
            assert isinstance(loaded.milestones, tuple)
            assert loaded.milestones == (5000, 10000)


class TestSchedulerConfigInheritance:
    """Test that SchedulerConfig can be inherited."""

    def test_can_inherit_from_scheduler_config(self):
        """Test creating a specialized config by inheriting."""

        @dataclasses.dataclass(frozen=True)
        class CustomSchedulerConfig(SchedulerConfig):
            """Specialized config for custom scheduler."""

            custom_param: float = 1.0

            def __post_init__(self):
                """Validate custom fields."""
                # Call parent validation first
                super().__post_init__()

                # Validate custom fields
                if self.custom_param < 0:
                    raise ValueError("custom_param must be non-negative")

        # Create custom config
        custom_config = CustomSchedulerConfig(
            name="custom_sched",
            scheduler_type="cosine",
            custom_param=2.0,
        )

        assert custom_config.name == "custom_sched"
        assert custom_config.custom_param == 2.0

        # Test inheritance validation works
        with pytest.raises(ValueError, match="custom_param"):
            CustomSchedulerConfig(
                name="custom_sched",
                scheduler_type="cosine",
                custom_param=-1.0,
            )
