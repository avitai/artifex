"""Tests for GeneratorConfig frozen dataclass.

Following TDD principles: Write tests first, then implement GeneratorConfig.
"""

import pytest

from artifex.generative_models.core.configuration.network_configs import GeneratorConfig


class TestGeneratorConfigBasics:
    """Test basic GeneratorConfig creation and properties."""

    def test_create_minimal(self) -> None:
        """Test creating GeneratorConfig with minimal required fields."""
        config = GeneratorConfig(
            name="test_gen",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256, 512),
            activation="relu",  # Required from BaseNetworkConfig
        )
        assert config.name == "test_gen"
        assert config.latent_dim == 100
        assert config.output_shape == (28, 28, 1)
        assert config.hidden_dims == (256, 512)
        assert config.activation == "relu"

    def test_create_full(self) -> None:
        """Test creating GeneratorConfig with all fields."""
        config = GeneratorConfig(
            name="full_gen",
            latent_dim=128,
            output_shape=(32, 32, 3),
            hidden_dims=(128, 256, 512),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.2,
            use_bias=False,
            output_activation="tanh",
        )
        assert config.name == "full_gen"
        assert config.latent_dim == 128
        assert config.output_shape == (32, 32, 3)
        assert config.hidden_dims == (128, 256, 512)
        assert config.activation == "relu"
        assert config.batch_norm is True
        assert config.dropout_rate == 0.2
        assert config.use_bias is False
        assert config.output_activation == "tanh"

    def test_frozen(self) -> None:
        """Test that GeneratorConfig is frozen (immutable)."""
        config = GeneratorConfig(
            name="frozen_test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        with pytest.raises(AttributeError):
            config.latent_dim = 200  # type: ignore

    def test_hash(self) -> None:
        """Test that GeneratorConfig is NOT hashable due to metadata dict field."""
        config = GeneratorConfig(
            name="hash_test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        # Cannot be hashed due to dict field (metadata)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(config)


class TestGeneratorConfigValidation:
    """Test GeneratorConfig validation rules."""

    def test_latent_dim_zero(self) -> None:
        """Test that latent_dim=0 raises ValueError."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            GeneratorConfig(
                name="test",
                latent_dim=0,
                output_shape=(28, 28, 1),
                hidden_dims=(256,),
                activation="relu",
            )

    def test_latent_dim_negative(self) -> None:
        """Test that negative latent_dim raises ValueError."""
        with pytest.raises(ValueError, match="latent_dim must be positive"):
            GeneratorConfig(
                name="test",
                latent_dim=-100,
                output_shape=(28, 28, 1),
                hidden_dims=(256,),
                activation="relu",
            )

    def test_output_shape_empty(self) -> None:
        """Test that empty output_shape raises ValueError."""
        with pytest.raises(ValueError, match="output_shape cannot be empty"):
            GeneratorConfig(
                name="test",
                latent_dim=100,
                output_shape=(),
                hidden_dims=(256,),
                activation="relu",
            )

    def test_output_shape_non_positive_dimension(self) -> None:
        """Test that output_shape with non-positive dimension raises ValueError."""
        with pytest.raises(ValueError, match="All output_shape must be positive"):
            GeneratorConfig(
                name="test",
                latent_dim=100,
                output_shape=(28, 0, 1),
                hidden_dims=(256,),
                activation="relu",
            )

    def test_output_activation_invalid(self) -> None:
        """Test that invalid output_activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation function"):
            GeneratorConfig(
                name="test",
                latent_dim=100,
                output_shape=(28, 28, 1),
                hidden_dims=(256,),
                activation="relu",
                output_activation="invalid_activation",
            )

    def test_output_activation_none_valid(self) -> None:
        """Test that output_activation=None is valid."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
            output_activation=None,
        )
        assert config.output_activation is None


class TestGeneratorConfigDefaults:
    """Test GeneratorConfig default values."""

    def test_default_activation(self) -> None:
        """Test that activation is required (no default)."""
        # activation is required from BaseNetworkConfig, so this should fail
        with pytest.raises(ValueError, match="activation is required"):
            GeneratorConfig(
                name="test",
                latent_dim=100,
                output_shape=(28, 28, 1),
                hidden_dims=(256,),
            )

    def test_default_batch_norm(self) -> None:
        """Test default batch_norm is False (from BaseNetworkConfig)."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.batch_norm is False

    def test_default_dropout_rate(self) -> None:
        """Test default dropout_rate is 0.0."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.dropout_rate == 0.0

    def test_default_use_bias(self) -> None:
        """Test default use_bias is True."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.use_bias is True

    def test_default_output_activation(self) -> None:
        """Test default output_activation is 'tanh'."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.output_activation == "tanh"


class TestGeneratorConfigSerialization:
    """Test GeneratorConfig serialization (to_dict/from_dict)."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict with minimal config."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        data = config.to_dict()
        assert data["name"] == "test"
        assert data["latent_dim"] == 100
        # Note: to_dict preserves tuples, doesn't convert to lists
        assert data["output_shape"] == (28, 28, 1)
        assert data["hidden_dims"] == (256,)

    def test_to_dict_full(self) -> None:
        """Test to_dict with all fields."""
        config = GeneratorConfig(
            name="full",
            latent_dim=128,
            output_shape=(32, 32, 3),
            hidden_dims=(128, 256),
            activation="gelu",
            batch_norm=False,
            dropout_rate=0.3,
            use_bias=False,
            output_activation="sigmoid",
        )
        data = config.to_dict()
        assert data["activation"] == "gelu"
        assert data["batch_norm"] is False
        assert data["dropout_rate"] == 0.3
        assert data["use_bias"] is False
        assert data["output_activation"] == "sigmoid"

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal data."""
        data = {
            "name": "test",
            "latent_dim": 100,
            "output_shape": [28, 28, 1],
            "hidden_dims": [256],
            "activation": "relu",  # Required from BaseNetworkConfig
        }
        config = GeneratorConfig.from_dict(data)
        assert config.name == "test"
        assert config.latent_dim == 100
        assert config.output_shape == (28, 28, 1)
        assert config.hidden_dims == (256,)
        assert config.activation == "relu"

    def test_from_dict_full(self) -> None:
        """Test from_dict with all fields."""
        data = {
            "name": "full",
            "latent_dim": 128,
            "output_shape": [32, 32, 3],
            "hidden_dims": [128, 256],
            "activation": "gelu",
            "batch_norm": False,
            "dropout_rate": 0.3,
            "use_bias": False,
            "output_activation": "sigmoid",
        }
        config = GeneratorConfig.from_dict(data)
        assert config.activation == "gelu"
        assert config.batch_norm is False
        assert config.dropout_rate == 0.3
        assert config.use_bias is False
        assert config.output_activation == "sigmoid"

    def test_roundtrip(self) -> None:
        """Test that to_dict → from_dict preserves config."""
        original = GeneratorConfig(
            name="roundtrip",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256, 512),
            activation="relu",
            dropout_rate=0.2,
        )
        data = original.to_dict()
        restored = GeneratorConfig.from_dict(data)
        assert restored == original


class TestGeneratorConfigEdgeCases:
    """Test GeneratorConfig edge cases."""

    def test_single_hidden_dim(self) -> None:
        """Test with single hidden dimension."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.hidden_dims == (256,)

    def test_many_hidden_dims(self) -> None:
        """Test with many hidden dimensions."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(64, 128, 256, 512, 1024),
            activation="relu",
        )
        assert len(config.hidden_dims) == 5

    def test_1d_output_shape(self) -> None:
        """Test with 1D output shape (e.g., for audio)."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(1000,),
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.output_shape == (1000,)

    def test_4d_output_shape(self) -> None:
        """Test with 4D output shape (e.g., video frames)."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(16, 64, 64, 3),  # 16 frames, 64x64, 3 channels
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.output_shape == (16, 64, 64, 3)

    def test_large_latent_dim(self) -> None:
        """Test with large latent dimension."""
        config = GeneratorConfig(
            name="test",
            latent_dim=2048,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
        )
        assert config.latent_dim == 2048

    def test_no_dropout(self) -> None:
        """Test with dropout_rate=0.0 (no dropout)."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
            dropout_rate=0.0,
        )
        assert config.dropout_rate == 0.0

    def test_high_dropout(self) -> None:
        """Test with high dropout rate."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
            dropout_rate=0.9,
        )
        assert config.dropout_rate == 0.9

    def test_no_batch_norm(self) -> None:
        """Test with batch_norm=False."""
        config = GeneratorConfig(
            name="test",
            latent_dim=100,
            output_shape=(28, 28, 1),
            hidden_dims=(256,),
            activation="relu",
            batch_norm=False,
        )
        assert config.batch_norm is False
