"""Tests for EBM builder using dataclass configs.

These tests verify the EBMBuilder functionality with the new dataclass-based
configuration system (EBMConfig, DeepEBMConfig) following Principle #4.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.factory.builders.ebm import EBMBuilder


class TestEBMBuilder:
    """Test EBM builder functionality with dataclass configs."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing."""
        return nnx.Rngs(42)

    @pytest.fixture
    def energy_network_config(self):
        """Create EnergyNetworkConfig for testing."""
        return EnergyNetworkConfig(
            name="test_energy",
            hidden_dims=(256, 512, 256),
            activation="swish",
            network_type="mlp",
            use_bias=True,
        )

    @pytest.fixture
    def cnn_energy_config(self):
        """Create CNN EnergyNetworkConfig for DeepEBM."""
        return EnergyNetworkConfig(
            name="test_cnn_energy",
            hidden_dims=(64, 128, 256, 512),
            activation="swish",
            network_type="cnn",
            use_bias=True,
        )

    @pytest.fixture
    def mcmc_config(self):
        """Create MCMCConfig for testing."""
        return MCMCConfig(
            name="test_mcmc",
            n_steps=100,
            step_size=0.01,
            noise_scale=0.01,
        )

    @pytest.fixture
    def sample_buffer_config(self):
        """Create SampleBufferConfig for testing."""
        return SampleBufferConfig(
            name="test_buffer",
            capacity=10000,
            reinit_prob=0.05,
        )

    def test_build_standard_ebm(
        self, rngs, energy_network_config, mcmc_config, sample_buffer_config
    ):
        """Test building a standard EBM."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=32 * 32 * 3,  # Flattened image
            energy_network=energy_network_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )

        builder = EBMBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "energy")
        assert hasattr(model, "score")

    def test_build_deep_ebm(self, rngs, cnn_energy_config, mcmc_config, sample_buffer_config):
        """Test building a DeepEBM."""
        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=cnn_energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )

        builder = EBMBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None

    def test_ebm_with_custom_params(self, rngs, sample_buffer_config):
        """Test building EBM with custom parameters."""
        # Custom energy network with spectral norm
        energy_network = EnergyNetworkConfig(
            name="custom_energy",
            hidden_dims=(512, 256),
            activation="swish",
            network_type="mlp",
            use_bias=True,
            use_spectral_norm=True,
            dropout_rate=0.1,
        )

        # Custom MCMC config
        mcmc = MCMCConfig(
            name="custom_mcmc",
            n_steps=200,
            step_size=0.001,
            noise_scale=0.005,
        )

        config = EBMConfig(
            name="test_custom_ebm",
            input_dim=28 * 28 * 1,  # Flattened MNIST
            energy_network=energy_network,
            mcmc=mcmc,
            sample_buffer=sample_buffer_config,
        )

        builder = EBMBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        # Check if parameters were properly set
        assert model.mcmc_steps == 200

    def test_ebm_energy_computation(
        self, rngs, energy_network_config, mcmc_config, sample_buffer_config
    ):
        """Test EBM energy computation."""
        config = EBMConfig(
            name="test_ebm",
            input_dim=784,  # Flattened MNIST
            energy_network=EnergyNetworkConfig(
                name="small_energy",
                hidden_dims=(256, 128),
                activation="swish",
                network_type="mlp",
            ),
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )

        builder = EBMBuilder()
        model = builder.build(config, rngs=rngs)

        # Create test input
        batch_size = 4
        x = jnp.ones((batch_size, 784))

        # Compute energy
        energy = model.energy(x)
        assert energy.shape == (batch_size,)

    def test_ebm_sampling_params(self, rngs, energy_network_config, sample_buffer_config):
        """Test EBM with sampling parameters."""
        mcmc = MCMCConfig(
            name="sampling_mcmc",
            n_steps=100,
            step_size=0.01,
            noise_scale=0.01,
        )

        config = EBMConfig(
            name="test_ebm_sampling",
            input_dim=32 * 32 * 3,
            energy_network=energy_network_config,
            mcmc=mcmc,
            sample_buffer=sample_buffer_config,
        )

        builder = EBMBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        # Verify parameters were set from config
        assert model.mcmc_steps == 100

    def test_deep_ebm_config_preparation(
        self, rngs, cnn_energy_config, mcmc_config, sample_buffer_config
    ):
        """Test configuration for DeepEBM models."""
        config = DeepEBMConfig(
            name="test_deep_ebm",
            input_shape=(32, 32, 3),
            energy_network=cnn_energy_config,
            mcmc=mcmc_config,
            sample_buffer=sample_buffer_config,
        )

        builder = EBMBuilder()
        model = builder.build(config, rngs=rngs)

        assert model is not None
        # DeepEBM should derive input_dim from input_shape
        assert config.input_dim == 32 * 32 * 3
