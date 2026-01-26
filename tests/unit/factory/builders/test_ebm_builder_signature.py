"""Tests for EBMBuilder Principle #4 compliance.

These tests verify that EBMBuilder follows Principle #4:
"Methods Take Configs, NOT Individual Parameters"

The builder should:
- Accept dataclass configs (EBMConfig, DeepEBMConfig)
- NOT accept Pydantic ModelConfig
- Determine model class from config type, not model_class string

Following TDD - these tests define the expected behavior.
"""

import pytest
from flax import nnx

from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def rngs():
    """Create nnx.Rngs for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def energy_network_config():
    """Create an EnergyNetworkConfig for testing."""
    return EnergyNetworkConfig(
        name="test_energy_network",
        hidden_dims=(32, 64),
        activation="gelu",
        network_type="mlp",
        use_bias=True,
    )


@pytest.fixture
def mcmc_config():
    """Create an MCMCConfig for testing."""
    return MCMCConfig(
        name="test_mcmc",
        n_steps=10,  # Small for fast testing
        step_size=0.01,
        noise_scale=0.005,
    )


@pytest.fixture
def sample_buffer_config():
    """Create a SampleBufferConfig for testing."""
    return SampleBufferConfig(
        name="test_buffer",
        capacity=100,  # Small for testing
        reinit_prob=0.05,
    )


@pytest.fixture
def ebm_config(energy_network_config, mcmc_config, sample_buffer_config):
    """Create an EBMConfig for testing."""
    return EBMConfig(
        name="test_ebm",
        input_dim=10,
        energy_network=energy_network_config,
        mcmc=mcmc_config,
        sample_buffer=sample_buffer_config,
        alpha=0.01,
    )


@pytest.fixture
def deep_ebm_config(mcmc_config, sample_buffer_config):
    """Create a DeepEBMConfig for testing."""
    # Use CNN for DeepEBM
    cnn_energy_config = EnergyNetworkConfig(
        name="test_cnn_energy",
        hidden_dims=(16, 32),
        activation="silu",
        network_type="cnn",
        use_bias=True,
    )
    return DeepEBMConfig(
        name="test_deep_ebm",
        input_shape=(8, 8, 3),  # Small for testing
        energy_network=cnn_energy_config,
        mcmc=mcmc_config,
        sample_buffer=sample_buffer_config,
    )


# =============================================================================
# EBMBuilder Existence Tests
# =============================================================================


class TestEBMBuilderExists:
    """Test that EBMBuilder exists and has correct signature."""

    def test_builder_exists(self):
        """Test that EBMBuilder class exists."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        assert EBMBuilder is not None

    def test_builder_has_build_method(self):
        """Test that EBMBuilder has build method."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        builder = EBMBuilder()
        assert hasattr(builder, "build")
        assert callable(builder.build)


# =============================================================================
# EBMBuilder with EBMConfig Tests
# =============================================================================


class TestEBMBuilderWithEBMConfig:
    """Test building EBM with EBMConfig."""

    def test_build_ebm_with_ebm_config(self, ebm_config, rngs):
        """Test that builder can create EBM from EBMConfig."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder
        from artifex.generative_models.models.energy.ebm import EBM

        builder = EBMBuilder()
        model = builder.build(ebm_config, rngs=rngs)

        assert model is not None
        assert isinstance(model, EBM)

    def test_build_ebm_has_energy_fn(self, ebm_config, rngs):
        """Test that built EBM has energy_fn attribute."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        builder = EBMBuilder()
        model = builder.build(ebm_config, rngs=rngs)

        assert hasattr(model, "energy_fn")
        assert model.energy_fn is not None

    def test_build_ebm_preserves_config(self, ebm_config, rngs):
        """Test that built EBM preserves config values."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        builder = EBMBuilder()
        model = builder.build(ebm_config, rngs=rngs)

        assert model.mcmc_steps == ebm_config.mcmc.n_steps
        assert model.alpha == ebm_config.alpha


# =============================================================================
# EBMBuilder with DeepEBMConfig Tests
# =============================================================================


class TestEBMBuilderWithDeepEBMConfig:
    """Test building DeepEBM with DeepEBMConfig."""

    def test_build_deep_ebm_with_config(self, deep_ebm_config, rngs):
        """Test that builder can create DeepEBM from DeepEBMConfig."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder
        from artifex.generative_models.models.energy.ebm import DeepEBM

        builder = EBMBuilder()
        model = builder.build(deep_ebm_config, rngs=rngs)

        assert model is not None
        assert isinstance(model, DeepEBM)

    def test_deep_ebm_uses_cnn_energy(self, deep_ebm_config, rngs):
        """Test that DeepEBM uses CNN energy function from config."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder
        from artifex.generative_models.models.energy.ebm import DeepCNNEnergyFunction

        builder = EBMBuilder()
        model = builder.build(deep_ebm_config, rngs=rngs)

        # DeepEBM should use DeepCNNEnergyFunction
        assert isinstance(model.energy_fn, DeepCNNEnergyFunction)


# =============================================================================
# EBMBuilder Error Handling Tests
# =============================================================================


class TestEBMBuilderErrorHandling:
    """Test error handling in EBMBuilder."""

    def test_reject_model_configuration(self, rngs):
        """Test that Pydantic ModelConfig is rejected."""
        from artifex.generative_models.core.configuration import ModelConfig
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        old_config = ModelConfig(
            name="test",
            model_class="EBM",
            input_dim=10,
            hidden_dims=[32, 64],
            output_dim=1,
        )

        builder = EBMBuilder()
        with pytest.raises(TypeError):
            builder.build(old_config, rngs=rngs)

    def test_reject_dict_config(self, rngs):
        """Test that dict config is rejected."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        dict_config = {"input_dim": 10}

        builder = EBMBuilder()
        with pytest.raises(TypeError):
            builder.build(dict_config, rngs=rngs)

    def test_reject_none_config(self, rngs):
        """Test that None config is rejected."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        builder = EBMBuilder()
        with pytest.raises(TypeError):
            builder.build(None, rngs=rngs)

    def test_reject_unsupported_config_type(self, rngs):
        """Test that unsupported config types are rejected."""
        from artifex.generative_models.core.configuration.base_dataclass import (
            BaseConfig,
        )
        from artifex.generative_models.factory.builders.ebm import EBMBuilder

        # Create a different config type
        class UnsupportedConfig(BaseConfig):
            pass

        builder = EBMBuilder()
        with pytest.raises(TypeError):
            builder.build(UnsupportedConfig(name="unsupported"), rngs=rngs)


# =============================================================================
# Config Type Dispatch Tests
# =============================================================================


class TestConfigTypeDispatch:
    """Test that builder dispatches based on config type."""

    def test_ebm_config_creates_ebm(self, ebm_config, rngs):
        """Test that EBMConfig creates EBM model."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder
        from artifex.generative_models.models.energy.ebm import EBM

        builder = EBMBuilder()
        model = builder.build(ebm_config, rngs=rngs)

        assert type(model).__name__ == "EBM"
        assert isinstance(model, EBM)

    def test_deep_ebm_config_creates_deep_ebm(self, deep_ebm_config, rngs):
        """Test that DeepEBMConfig creates DeepEBM model."""
        from artifex.generative_models.factory.builders.ebm import EBMBuilder
        from artifex.generative_models.models.energy.ebm import DeepEBM

        builder = EBMBuilder()
        model = builder.build(deep_ebm_config, rngs=rngs)

        assert type(model).__name__ == "DeepEBM"
        assert isinstance(model, DeepEBM)
