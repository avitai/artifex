"""Tests for create_energy_function factory.

These tests define the expected behavior for the energy function factory following TDD principles.
The factory creates the appropriate energy function based on EnergyNetworkConfig.network_type.

Following Principle #4: Methods Take Configs, NOT Individual Parameters
"""

import pytest
from flax import nnx

from artifex.generative_models.core.configuration.energy_config import (
    EnergyNetworkConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rngs():
    """Create nnx.Rngs for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def mlp_energy_config():
    """Create an EnergyNetworkConfig for MLP energy function."""
    return EnergyNetworkConfig(
        name="test_mlp_energy",
        hidden_dims=(32, 64),
        activation="gelu",
        network_type="mlp",
        use_bias=True,
    )


@pytest.fixture
def cnn_energy_config():
    """Create an EnergyNetworkConfig for CNN energy function."""
    return EnergyNetworkConfig(
        name="test_cnn_energy",
        hidden_dims=(16, 32, 64),
        activation="silu",
        network_type="cnn",
        use_bias=True,
    )


# =============================================================================
# Factory Function Existence Tests
# =============================================================================


class TestCreateEnergyFunctionExists:
    """Test that create_energy_function factory exists."""

    def test_factory_function_exists(self):
        """Test that create_energy_function function exists."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        assert create_energy_function is not None
        assert callable(create_energy_function)

    def test_factory_accepts_config_and_rngs(self, mlp_energy_config, rngs):
        """Test that factory accepts config and rngs parameters."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        # Should not raise - just test call signature
        result = create_energy_function(mlp_energy_config, input_dim=10, rngs=rngs)
        assert result is not None


# =============================================================================
# MLP Energy Function Tests
# =============================================================================


class TestCreateMLPEnergyFunction:
    """Test creating MLP energy functions from config."""

    def test_create_mlp_energy_function(self, mlp_energy_config, rngs):
        """Test creating MLPEnergyFunction from config."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )
        from artifex.generative_models.models.energy.base import MLPEnergyFunction

        energy_fn = create_energy_function(mlp_energy_config, input_dim=10, rngs=rngs)

        assert energy_fn is not None
        assert isinstance(energy_fn, MLPEnergyFunction)

    def test_mlp_hidden_dims_from_config(self, mlp_energy_config, rngs):
        """Test that hidden_dims are used from config."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        energy_fn = create_energy_function(mlp_energy_config, input_dim=10, rngs=rngs)

        # Check that hidden_dims from config were used
        assert energy_fn.hidden_dims == list(mlp_energy_config.hidden_dims)

    def test_mlp_input_dim_required(self, mlp_energy_config, rngs):
        """Test that input_dim is required for MLP energy function."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        with pytest.raises((ValueError, TypeError)):
            # input_dim not provided
            create_energy_function(mlp_energy_config, rngs=rngs)

    def test_mlp_use_bias_from_config(self, rngs):
        """Test that use_bias is used from config."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        config_with_bias = EnergyNetworkConfig(
            name="test",
            hidden_dims=(32,),
            activation="gelu",
            network_type="mlp",
            use_bias=True,
        )
        config_no_bias = EnergyNetworkConfig(
            name="test",
            hidden_dims=(32,),
            activation="gelu",
            network_type="mlp",
            use_bias=False,
        )

        energy_with_bias = create_energy_function(config_with_bias, input_dim=10, rngs=rngs)
        energy_no_bias = create_energy_function(config_no_bias, input_dim=10, rngs=rngs)

        assert energy_with_bias.use_bias is True
        assert energy_no_bias.use_bias is False


# =============================================================================
# CNN Energy Function Tests
# =============================================================================


class TestCreateCNNEnergyFunction:
    """Test creating CNN energy functions from config."""

    def test_create_cnn_energy_function(self, cnn_energy_config, rngs):
        """Test creating CNNEnergyFunction from config."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )
        from artifex.generative_models.models.energy.base import CNNEnergyFunction

        energy_fn = create_energy_function(cnn_energy_config, input_channels=3, rngs=rngs)

        assert energy_fn is not None
        assert isinstance(energy_fn, CNNEnergyFunction)

    def test_cnn_hidden_dims_from_config(self, cnn_energy_config, rngs):
        """Test that hidden_dims are used from config."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        energy_fn = create_energy_function(cnn_energy_config, input_channels=3, rngs=rngs)

        # Check that hidden_dims from config were used
        assert energy_fn.hidden_dims == list(cnn_energy_config.hidden_dims)

    def test_cnn_input_channels_required(self, cnn_energy_config, rngs):
        """Test that input_channels is required for CNN energy function."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        with pytest.raises((ValueError, TypeError)):
            # input_channels not provided
            create_energy_function(cnn_energy_config, rngs=rngs)


# =============================================================================
# Network Type Dispatch Tests
# =============================================================================


class TestNetworkTypeDispatch:
    """Test that factory dispatches on network_type correctly."""

    def test_mlp_type_creates_mlp(self, rngs):
        """Test network_type='mlp' creates MLPEnergyFunction."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )
        from artifex.generative_models.models.energy.base import MLPEnergyFunction

        config = EnergyNetworkConfig(
            name="test",
            hidden_dims=(32, 64),
            activation="gelu",
            network_type="mlp",
        )

        energy_fn = create_energy_function(config, input_dim=10, rngs=rngs)

        assert isinstance(energy_fn, MLPEnergyFunction)

    def test_cnn_type_creates_cnn(self, rngs):
        """Test network_type='cnn' creates CNNEnergyFunction."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )
        from artifex.generative_models.models.energy.base import CNNEnergyFunction

        config = EnergyNetworkConfig(
            name="test",
            hidden_dims=(16, 32),
            activation="silu",
            network_type="cnn",
        )

        energy_fn = create_energy_function(config, input_channels=1, rngs=rngs)

        assert isinstance(energy_fn, CNNEnergyFunction)

    def test_invalid_network_type_raises_error(self, rngs):
        """Test that invalid network_type raises error."""
        # The validation should happen at config creation time
        with pytest.raises(ValueError):
            EnergyNetworkConfig(
                name="test",
                hidden_dims=(32,),
                network_type="invalid",
            )


# =============================================================================
# Activation Function Tests
# =============================================================================


class TestActivationFunction:
    """Test activation function mapping."""

    def test_gelu_activation(self, rngs):
        """Test gelu activation is mapped correctly."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        config = EnergyNetworkConfig(
            name="test",
            hidden_dims=(32,),
            network_type="mlp",
            activation="gelu",
        )

        energy_fn = create_energy_function(config, input_dim=10, rngs=rngs)

        # Check activation is set (exact comparison depends on implementation)
        assert energy_fn.activation is not None

    def test_silu_activation(self, rngs):
        """Test silu activation is mapped correctly."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        config = EnergyNetworkConfig(
            name="test",
            hidden_dims=(32,),
            network_type="cnn",
            activation="silu",
        )

        energy_fn = create_energy_function(config, input_channels=1, rngs=rngs)

        assert energy_fn.activation is not None


# =============================================================================
# Dropout Rate Tests
# =============================================================================


class TestDropoutRate:
    """Test dropout rate handling."""

    def test_mlp_dropout_rate_from_config(self, rngs):
        """Test that dropout_rate is used from config for MLP."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        config = EnergyNetworkConfig(
            name="test",
            hidden_dims=(32,),
            activation="gelu",
            network_type="mlp",
            dropout_rate=0.5,
        )

        energy_fn = create_energy_function(config, input_dim=10, rngs=rngs)

        assert energy_fn.dropout_rate == 0.5

    def test_mlp_zero_dropout_no_layer(self, rngs):
        """Test that zero dropout doesn't create dropout layer."""
        from artifex.generative_models.core.configuration.energy_config import (
            create_energy_function,
        )

        config = EnergyNetworkConfig(
            name="test",
            hidden_dims=(32,),
            activation="gelu",
            network_type="mlp",
            dropout_rate=0.0,
        )

        energy_fn = create_energy_function(config, input_dim=10, rngs=rngs)

        assert energy_fn.dropout is None
