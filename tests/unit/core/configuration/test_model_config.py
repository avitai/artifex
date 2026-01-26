"""Tests for ModelConfig dataclass.

Following TDD approach: These tests define the expected behavior of ModelConfig
before implementation. All tests should pass once implementation is complete.
"""

import dataclasses
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from artifex.generative_models.core.configuration.model_config import ModelConfig


class TestModelConfigBasics:
    """Test basic ModelConfig functionality."""

    def test_create_minimal_config(self):
        """Test creating a minimal valid ModelConfig."""
        config = ModelConfig(
            name="test_model",
            model_class="artifex.models.TestModel",
            input_dim=(28, 28, 1),
        )

        assert config.name == "test_model"
        assert config.model_class == "artifex.models.TestModel"
        assert config.input_dim == (28, 28, 1)
        # Check defaults
        assert config.hidden_dims == (128, 256, 512)
        assert config.output_dim is None
        assert config.activation == "gelu"
        assert config.dropout_rate == 0.1
        assert config.use_batch_norm is True

    def test_create_full_config(self):
        """Test creating a fully specified ModelConfig."""
        config = ModelConfig(
            name="full_model",
            model_class="artifex.models.FullModel",
            input_dim=(32, 32, 3),
            hidden_dims=(64, 128, 256),
            output_dim=10,
            activation="relu",
            dropout_rate=0.2,
            use_batch_norm=False,
            rngs_seeds={"params": 42, "dropout": 24},
            parameters={"beta": 1.0, "kl_weight": 0.5},
            metadata={"experiment_id": "exp_001"},
            tags=("vision", "classification"),
        )

        assert config.name == "full_model"
        assert config.model_class == "artifex.models.FullModel"
        assert config.input_dim == (32, 32, 3)
        assert config.hidden_dims == (64, 128, 256)
        assert config.output_dim == 10
        assert config.activation == "relu"
        assert config.dropout_rate == 0.2
        assert config.use_batch_norm is False
        assert config.rngs_seeds == {"params": 42, "dropout": 24}
        assert config.parameters == {"beta": 1.0, "kl_weight": 0.5}
        assert config.metadata == {"experiment_id": "exp_001"}
        assert config.tags == ("vision", "classification")

    def test_is_frozen(self):
        """Test that ModelConfig is immutable (frozen)."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.name = "new_name"

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.dropout_rate = 0.5

    def test_equality(self):
        """Test equality comparison between configs."""
        config1 = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
            hidden_dims=(128, 256),
        )
        config2 = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
            hidden_dims=(128, 256),
        )
        config3 = ModelConfig(
            name="different",
            model_class="test.Model",
            input_dim=(28, 28),
        )

        assert config1 == config2
        assert config1 != config3

    def test_asdict(self):
        """Test conversion to dictionary."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(32, 32),
            hidden_dims=(64, 128),
            tags=("test",),
        )

        config_dict = dataclasses.asdict(config)

        assert config_dict["name"] == "test"
        assert config_dict["model_class"] == "test.Model"
        assert config_dict["input_dim"] == (32, 32)
        assert config_dict["hidden_dims"] == (64, 128)
        assert config_dict["tags"] == ("test",)


class TestModelConfigFromDict:
    """Test from_dict() class method."""

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        data = {
            "name": "test_model",
            "model_class": "test.Model",
            "input_dim": [28, 28, 1],  # List should convert to tuple
        }

        config = ModelConfig.from_dict(data)

        assert config.name == "test_model"
        assert config.model_class == "test.Model"
        assert config.input_dim == (28, 28, 1)  # Converted to tuple
        assert isinstance(config.input_dim, tuple)

    def test_from_dict_converts_lists_to_tuples(self):
        """Test that from_dict auto-converts lists to tuples."""
        data = {
            "name": "test",
            "model_class": "test.Model",
            "input_dim": [32, 32, 3],  # List
            "hidden_dims": [64, 128, 256],  # List
            "tags": ["vision", "test"],  # List
        }

        config = ModelConfig.from_dict(data)

        assert config.input_dim == (32, 32, 3)
        assert isinstance(config.input_dim, tuple)
        assert config.hidden_dims == (64, 128, 256)
        assert isinstance(config.hidden_dims, tuple)
        assert config.tags == ("vision", "test")
        assert isinstance(config.tags, tuple)

    def test_from_dict_with_nested_dicts(self):
        """Test from_dict with nested dictionaries."""
        data = {
            "name": "test",
            "model_class": "test.Model",
            "input_dim": [28, 28],
            "rngs_seeds": {"params": 42, "dropout": 24},
            "parameters": {"beta": 1.0, "kl_weight": 0.5},
            "metadata": {"experiment_id": "exp_001", "notes": "test"},
        }

        config = ModelConfig.from_dict(data)

        assert config.rngs_seeds == {"params": 42, "dropout": 24}
        assert config.parameters == {"beta": 1.0, "kl_weight": 0.5}
        assert config.metadata == {"experiment_id": "exp_001", "notes": "test"}

    def test_from_dict_strict_mode(self):
        """Test from_dict rejects unknown fields in strict mode (default)."""
        data = {
            "name": "test",
            "model_class": "test.Model",
            "input_dim": [28, 28],
            "unknown_field": "should_fail",
        }

        with pytest.raises(Exception):  # dacite raises for unknown fields
            ModelConfig.from_dict(data)


class TestModelConfigValidation:
    """Test validation in __post_init__."""

    def test_invalid_activation_raises(self):
        """Test that invalid activation function raises error."""
        with pytest.raises(ValueError, match="activation"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                activation="invalid_activation",
            )

    def test_valid_activations(self):
        """Test that all valid activations are accepted."""
        # Use the same list as in validation.py
        valid_activations = [
            "relu",
            "gelu",
            "swish",
            "silu",
            "tanh",
            "sigmoid",
            "elu",
            "leaky_relu",
            "relu6",
            "celu",
            "selu",
            "glu",
            "hard_tanh",
            "softplus",
            "softsign",
        ]

        for activation in valid_activations:
            config = ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                activation=activation,
            )
            assert config.activation == activation

    def test_invalid_dropout_rate_raises(self):
        """Test that invalid dropout rates raise errors."""
        # Negative dropout
        with pytest.raises(ValueError, match="dropout_rate"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                dropout_rate=-0.1,
            )

        # Dropout > 1.0
        with pytest.raises(ValueError, match="dropout_rate"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                dropout_rate=1.5,
            )

    def test_valid_dropout_rates(self):
        """Test that valid dropout rates are accepted."""
        valid_rates = [0.0, 0.1, 0.5, 1.0]

        for rate in valid_rates:
            config = ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                dropout_rate=rate,
            )
            assert config.dropout_rate == rate

    def test_empty_hidden_dims_raises(self):
        """Test that empty hidden_dims raises error."""
        with pytest.raises(ValueError, match="hidden_dims"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                hidden_dims=(),
            )

    def test_negative_hidden_dim_raises(self):
        """Test that negative values in hidden_dims raise error."""
        with pytest.raises(ValueError, match="hidden_dims"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                hidden_dims=(128, -256, 512),
            )

    def test_zero_hidden_dim_raises(self):
        """Test that zero values in hidden_dims raise error."""
        with pytest.raises(ValueError, match="hidden_dims"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                hidden_dims=(128, 0, 512),
            )


class TestModelConfigInputDimensions:
    """Test input_dim handling (can be int or tuple)."""

    def test_scalar_input_dim(self):
        """Test with scalar input_dim."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=784,  # Scalar
        )

        assert config.input_dim == 784
        assert isinstance(config.input_dim, int)

    def test_tuple_input_dim(self):
        """Test with tuple input_dim."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28, 1),  # Tuple
        )

        assert config.input_dim == (28, 28, 1)
        assert isinstance(config.input_dim, tuple)

    def test_negative_input_dim_raises(self):
        """Test that negative input_dim raises error."""
        with pytest.raises(ValueError, match="input_dim"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=-784,
            )

    def test_negative_in_tuple_input_dim_raises(self):
        """Test that negative values in tuple input_dim raise error."""
        with pytest.raises(ValueError, match="input_dim"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, -28, 1),
            )

    def test_zero_input_dim_raises(self):
        """Test that zero input_dim raises error."""
        with pytest.raises(ValueError, match="input_dim"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=0,
            )


class TestModelConfigOutputDimensions:
    """Test output_dim handling (can be None, int, or tuple)."""

    def test_output_dim_none(self):
        """Test with output_dim=None (default)."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
        )

        assert config.output_dim is None

    def test_output_dim_scalar(self):
        """Test with scalar output_dim."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
            output_dim=10,
        )

        assert config.output_dim == 10
        assert isinstance(config.output_dim, int)

    def test_output_dim_tuple(self):
        """Test with tuple output_dim."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
            output_dim=(10, 10),
        )

        assert config.output_dim == (10, 10)
        assert isinstance(config.output_dim, tuple)

    def test_negative_output_dim_raises(self):
        """Test that negative output_dim raises error."""
        with pytest.raises(ValueError, match="output_dim"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                output_dim=-10,
            )

    def test_zero_output_dim_raises(self):
        """Test that zero output_dim raises error."""
        with pytest.raises(ValueError, match="output_dim"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                output_dim=0,
            )


class TestModelConfigRngsSeeds:
    """Test rngs_seeds field."""

    def test_default_rngs_seeds(self):
        """Test default rngs_seeds value."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
        )

        assert config.rngs_seeds == {"params": 0, "dropout": 1}

    def test_custom_rngs_seeds(self):
        """Test custom rngs_seeds."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
            rngs_seeds={"params": 42, "dropout": 24, "sample": 99},
        )

        assert config.rngs_seeds == {"params": 42, "dropout": 24, "sample": 99}

    def test_empty_rngs_seeds_raises(self):
        """Test that empty rngs_seeds raises error."""
        with pytest.raises(ValueError, match="rngs_seeds"):
            ModelConfig(
                name="test",
                model_class="test.Model",
                input_dim=(28, 28),
                rngs_seeds={},
            )


class TestModelConfigParameters:
    """Test parameters field (model-specific functional parameters)."""

    def test_default_parameters_empty(self):
        """Test default parameters is empty dict."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
        )

        assert config.parameters == {}

    def test_custom_parameters(self):
        """Test custom parameters."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
            parameters={"beta": 1.0, "kl_weight": 0.5, "noise_steps": 1000},
        )

        assert config.parameters == {"beta": 1.0, "kl_weight": 0.5, "noise_steps": 1000}

    def test_parameters_immutable(self):
        """Test that parameters dict is not modifiable after creation."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28),
            parameters={"beta": 1.0},
        )

        # Config is frozen, can't reassign
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.parameters = {"new": "value"}


class TestModelConfigSerialization:
    """Test YAML serialization."""

    def test_to_yaml(self):
        """Test saving config to YAML."""
        config = ModelConfig(
            name="test",
            model_class="test.Model",
            input_dim=(28, 28, 1),
            hidden_dims=(64, 128),
            activation="relu",
            dropout_rate=0.2,
            tags=("test", "vision"),
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            config.to_yaml(yaml_path)

            assert yaml_path.exists()

            # Load and verify
            loaded_config = ModelConfig.from_yaml(yaml_path)
            assert loaded_config == config

    def test_from_yaml(self):
        """Test loading config from YAML."""
        config = ModelConfig(
            name="yaml_test",
            model_class="test.YamlModel",
            input_dim=(32, 32, 3),
            hidden_dims=(128, 256, 512),
            parameters={"beta": 1.5},
            metadata={"experiment": "test_001"},
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(yaml_path)

            loaded_config = ModelConfig.from_yaml(yaml_path)

            assert loaded_config.name == "yaml_test"
            assert loaded_config.model_class == "test.YamlModel"
            assert loaded_config.input_dim == (32, 32, 3)
            assert loaded_config.hidden_dims == (128, 256, 512)
            assert loaded_config.parameters == {"beta": 1.5}
            assert loaded_config.metadata == {"experiment": "test_001"}

    def test_yaml_roundtrip_preserves_types(self):
        """Test that YAML roundtrip preserves all types."""
        original = ModelConfig(
            name="roundtrip",
            model_class="test.Model",
            input_dim=(28, 28, 1),
            hidden_dims=(64, 128, 256),
            output_dim=10,
            tags=("a", "b", "c"),
        )

        with TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "roundtrip.yaml"
            original.to_yaml(yaml_path)
            loaded = ModelConfig.from_yaml(yaml_path)

            # Verify types are preserved
            assert isinstance(loaded.input_dim, tuple)
            assert isinstance(loaded.hidden_dims, tuple)
            assert isinstance(loaded.output_dim, int)
            assert isinstance(loaded.tags, tuple)


class TestModelConfigInheritance:
    """Test that ModelConfig can be inherited."""

    def test_can_inherit_from_model_config(self):
        """Test creating a specialized config by inheriting."""

        @dataclasses.dataclass(frozen=True)
        class VAEConfig(ModelConfig):
            """Specialized config for VAE models."""

            beta: float = 1.0
            kl_weight: float = 0.5

            def __post_init__(self):
                """Validate VAE-specific fields."""
                # Call parent validation first
                super().__post_init__()

                # Validate VAE-specific fields
                if self.beta < 0:
                    raise ValueError("beta must be non-negative")

        # Create VAE config
        vae_config = VAEConfig(
            name="vae_test",
            model_class="test.VAE",
            input_dim=(28, 28),
            beta=1.5,
            kl_weight=0.3,
        )

        assert vae_config.name == "vae_test"
        assert vae_config.beta == 1.5
        assert vae_config.kl_weight == 0.3

        # Test inheritance validation works
        with pytest.raises(ValueError, match="beta"):
            VAEConfig(
                name="vae_test",
                model_class="test.VAE",
                input_dim=(28, 28),
                beta=-1.0,
            )
