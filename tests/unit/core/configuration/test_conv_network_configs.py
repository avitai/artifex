"""Tests for convolutional network configuration classes.

Tests ConvGeneratorConfig and ConvDiscriminatorConfig which extend the base
network configs with convolutional layer parameters (kernel_size, stride, padding).
"""

import pytest

from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    DiscriminatorConfig,
    GeneratorConfig,
)


class TestConvGeneratorConfigBasics:
    """Test basic ConvGeneratorConfig functionality."""

    def test_instantiation_with_all_required_fields(self):
        """Test that ConvGeneratorConfig can be instantiated with required fields."""
        config = ConvGeneratorConfig(
            name="test_conv_gen",
            latent_dim=100,
            output_shape=(3, 64, 64),
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert config.name == "test_conv_gen"
        assert config.latent_dim == 100
        assert config.output_shape == (3, 64, 64)
        assert config.kernel_size == (4, 4)
        assert config.stride == (2, 2)
        assert config.padding == "SAME"

    def test_is_frozen_dataclass(self):
        """Test that ConvGeneratorConfig is immutable."""
        config = ConvGeneratorConfig(
            name="frozen_test",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(256, 128),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        with pytest.raises(AttributeError):
            config.kernel_size = (5, 5)

    def test_inherits_from_generator_config(self):
        """Test that ConvGeneratorConfig inherits from GeneratorConfig."""
        config = ConvGeneratorConfig(
            name="inheritance_test",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(256, 128),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert isinstance(config, GeneratorConfig)


class TestConvGeneratorConfigDefaults:
    """Test default values for ConvGeneratorConfig."""

    def test_batch_norm_momentum_default(self):
        """Test default value for batch_norm_momentum."""
        config = ConvGeneratorConfig(
            name="defaults_test",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(256, 128),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert config.batch_norm_momentum == 0.9

    def test_batch_norm_use_running_avg_default(self):
        """Test default value for batch_norm_use_running_avg."""
        config = ConvGeneratorConfig(
            name="defaults_test",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(256, 128),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert config.batch_norm_use_running_avg is False

    def test_can_override_defaults(self):
        """Test that default values can be overridden."""
        config = ConvGeneratorConfig(
            name="override_test",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(256, 128),
            activation="relu",
            kernel_size=(5, 5),
            stride=(1, 1),
            padding="VALID",
            batch_norm_momentum=0.99,
            batch_norm_use_running_avg=True,
        )
        assert config.kernel_size == (5, 5)
        assert config.stride == (1, 1)
        assert config.padding == "VALID"
        assert config.batch_norm_momentum == 0.99
        assert config.batch_norm_use_running_avg is True


class TestConvGeneratorConfigValidation:
    """Test validation for ConvGeneratorConfig."""

    def test_kernel_size_must_be_positive(self):
        """Test that kernel_size must have positive dimensions."""
        with pytest.raises(ValueError, match="kernel_size"):
            ConvGeneratorConfig(
                name="invalid_kernel",
                latent_dim=100,
                output_shape=(3, 32, 32),
                hidden_dims=(256, 128),
                activation="relu",
                kernel_size=(0, 4),
                stride=(2, 2),
                padding="SAME",
            )

    def test_kernel_size_must_be_tuple(self):
        """Test that kernel_size must be a tuple of 2 ints."""
        with pytest.raises((ValueError, TypeError)):
            ConvGeneratorConfig(
                name="invalid_kernel",
                latent_dim=100,
                output_shape=(3, 32, 32),
                hidden_dims=(256, 128),
                activation="relu",
                kernel_size=4,  # Should be tuple
                stride=(2, 2),
                padding="SAME",
            )

    def test_stride_must_be_positive(self):
        """Test that stride must have positive dimensions."""
        with pytest.raises(ValueError, match="stride"):
            ConvGeneratorConfig(
                name="invalid_stride",
                latent_dim=100,
                output_shape=(3, 32, 32),
                hidden_dims=(256, 128),
                activation="relu",
                kernel_size=(4, 4),
                stride=(0, 2),
                padding="SAME",
            )

    def test_padding_must_be_valid_string(self):
        """Test that padding must be a valid string."""
        with pytest.raises(ValueError, match="padding"):
            ConvGeneratorConfig(
                name="invalid_padding",
                latent_dim=100,
                output_shape=(3, 32, 32),
                hidden_dims=(256, 128),
                activation="relu",
                kernel_size=(4, 4),
                stride=(2, 2),
                padding="INVALID_PADDING",
            )

    def test_valid_padding_values(self):
        """Test that valid padding values are accepted."""
        for padding in ["SAME", "VALID"]:
            config = ConvGeneratorConfig(
                name=f"padding_{padding}",
                latent_dim=100,
                output_shape=(3, 32, 32),
                hidden_dims=(256, 128),
                activation="relu",
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=padding,
            )
            assert config.padding == padding

    def test_batch_norm_momentum_must_be_in_range(self):
        """Test that batch_norm_momentum must be in (0, 1)."""
        with pytest.raises(ValueError, match="batch_norm_momentum"):
            ConvGeneratorConfig(
                name="invalid_momentum",
                latent_dim=100,
                output_shape=(3, 32, 32),
                hidden_dims=(256, 128),
                activation="relu",
                kernel_size=(4, 4),
                stride=(2, 2),
                padding="SAME",
                batch_norm_momentum=1.5,
            )


class TestConvGeneratorConfigSerialization:
    """Test serialization for ConvGeneratorConfig."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ConvGeneratorConfig(
            name="serialization_test",
            latent_dim=100,
            output_shape=(3, 64, 64),
            hidden_dims=(512, 256),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        data = config.to_dict()
        assert data["name"] == "serialization_test"
        assert data["latent_dim"] == 100
        assert data["kernel_size"] == (4, 4)
        assert data["stride"] == (2, 2)
        assert data["padding"] == "SAME"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "from_dict_test",
            "latent_dim": 100,
            "output_shape": (3, 64, 64),
            "hidden_dims": (512, 256),
            "activation": "relu",
            "kernel_size": (4, 4),
            "stride": (2, 2),
            "padding": "SAME",
        }
        config = ConvGeneratorConfig.from_dict(data)
        assert config.name == "from_dict_test"
        assert config.kernel_size == (4, 4)
        assert config.stride == (2, 2)

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves all values."""
        original = ConvGeneratorConfig(
            name="roundtrip_test",
            latent_dim=128,
            output_shape=(3, 128, 128),
            hidden_dims=(1024, 512, 256, 128),
            activation="gelu",
            kernel_size=(5, 5),
            stride=(2, 2),
            padding="VALID",
            batch_norm=True,
            batch_norm_momentum=0.95,
            batch_norm_use_running_avg=True,
        )
        data = original.to_dict()
        restored = ConvGeneratorConfig.from_dict(data)
        assert restored == original


class TestConvDiscriminatorConfigBasics:
    """Test basic ConvDiscriminatorConfig functionality."""

    def test_instantiation_with_all_required_fields(self):
        """Test that ConvDiscriminatorConfig can be instantiated with required fields."""
        config = ConvDiscriminatorConfig(
            name="test_conv_disc",
            input_shape=(3, 64, 64),
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert config.name == "test_conv_disc"
        assert config.input_shape == (3, 64, 64)
        assert config.kernel_size == (4, 4)
        assert config.stride == (2, 2)
        assert config.padding == "SAME"

    def test_is_frozen_dataclass(self):
        """Test that ConvDiscriminatorConfig is immutable."""
        config = ConvDiscriminatorConfig(
            name="frozen_test",
            input_shape=(3, 32, 32),
            hidden_dims=(64, 128),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        with pytest.raises(AttributeError):
            config.kernel_size = (5, 5)

    def test_inherits_from_discriminator_config(self):
        """Test that ConvDiscriminatorConfig inherits from DiscriminatorConfig."""
        config = ConvDiscriminatorConfig(
            name="inheritance_test",
            input_shape=(3, 32, 32),
            hidden_dims=(64, 128),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert isinstance(config, DiscriminatorConfig)


class TestConvDiscriminatorConfigDefaults:
    """Test default values for ConvDiscriminatorConfig."""

    def test_batch_norm_momentum_default(self):
        """Test default value for batch_norm_momentum."""
        config = ConvDiscriminatorConfig(
            name="defaults_test",
            input_shape=(3, 32, 32),
            hidden_dims=(64, 128),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert config.batch_norm_momentum == 0.9

    def test_output_dim_default(self):
        """Test default value for output_dim (discriminator output dimension)."""
        config = ConvDiscriminatorConfig(
            name="defaults_test",
            input_shape=(3, 32, 32),
            hidden_dims=(64, 128),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )
        assert config.output_dim == 1


class TestConvDiscriminatorConfigValidation:
    """Test validation for ConvDiscriminatorConfig."""

    def test_kernel_size_must_be_positive(self):
        """Test that kernel_size must have positive dimensions."""
        with pytest.raises(ValueError, match="kernel_size"):
            ConvDiscriminatorConfig(
                name="invalid_kernel",
                input_shape=(3, 32, 32),
                hidden_dims=(64, 128),
                activation="leaky_relu",
                kernel_size=(-1, 4),
                stride=(2, 2),
                padding="SAME",
            )

    def test_stride_must_be_positive(self):
        """Test that stride must have positive dimensions."""
        with pytest.raises(ValueError, match="stride"):
            ConvDiscriminatorConfig(
                name="invalid_stride",
                input_shape=(3, 32, 32),
                hidden_dims=(64, 128),
                activation="leaky_relu",
                kernel_size=(4, 4),
                stride=(-1, 2),
                padding="SAME",
            )

    def test_output_dim_must_be_positive(self):
        """Test that output_dim must be positive."""
        with pytest.raises(ValueError, match="output_dim"):
            ConvDiscriminatorConfig(
                name="invalid_output_dim",
                input_shape=(3, 32, 32),
                hidden_dims=(64, 128),
                activation="leaky_relu",
                kernel_size=(4, 4),
                stride=(2, 2),
                padding="SAME",
                output_dim=0,
            )


class TestConvDiscriminatorConfigSerialization:
    """Test serialization for ConvDiscriminatorConfig."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ConvDiscriminatorConfig(
            name="serialization_test",
            input_shape=(3, 64, 64),
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            output_dim=1,
        )
        data = config.to_dict()
        assert data["name"] == "serialization_test"
        assert data["input_shape"] == (3, 64, 64)
        assert data["kernel_size"] == (4, 4)
        assert data["output_dim"] == 1

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "from_dict_test",
            "input_shape": (3, 64, 64),
            "hidden_dims": (64, 128, 256),
            "activation": "leaky_relu",
            "kernel_size": (4, 4),
            "stride": (2, 2),
            "padding": "SAME",
        }
        config = ConvDiscriminatorConfig.from_dict(data)
        assert config.name == "from_dict_test"
        assert config.kernel_size == (4, 4)

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves all values."""
        original = ConvDiscriminatorConfig(
            name="roundtrip_test",
            input_shape=(3, 128, 128),
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            kernel_size=(5, 5),
            stride=(2, 2),
            padding="VALID",
            batch_norm=False,
            batch_norm_momentum=0.95,
            output_dim=1,
        )
        data = original.to_dict()
        restored = ConvDiscriminatorConfig.from_dict(data)
        assert restored == original


class TestConvConfigsCompatibility:
    """Test that ConvGeneratorConfig and ConvDiscriminatorConfig work together."""

    def test_matching_shapes(self):
        """Test that generator output_shape matches discriminator input_shape."""
        output_shape = (3, 64, 64)

        gen_config = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=output_shape,
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        disc_config = ConvDiscriminatorConfig(
            name="disc",
            input_shape=output_shape,  # Same as generator output
            hidden_dims=(64, 128, 256, 512),
            activation="leaky_relu",
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
        )

        assert gen_config.output_shape == disc_config.input_shape

    def test_asymmetric_kernel_sizes(self):
        """Test that generator and discriminator can have different kernel sizes."""
        gen_config = ConvGeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(3, 64, 64),
            hidden_dims=(512, 256),
            activation="relu",
            kernel_size=(5, 5),  # Larger kernel
            stride=(2, 2),
            padding="SAME",
        )

        disc_config = ConvDiscriminatorConfig(
            name="disc",
            input_shape=(3, 64, 64),
            hidden_dims=(64, 128),
            activation="leaky_relu",
            kernel_size=(4, 4),  # Smaller kernel
            stride=(2, 2),
            padding="SAME",
        )

        assert gen_config.kernel_size != disc_config.kernel_size
