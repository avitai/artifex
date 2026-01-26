"""Tests for the Glow normalizing flow model."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    CouplingNetworkConfig,
    GlowConfig,
)
from artifex.generative_models.models.flow.glow import (
    ActNormLayer,
    AffineCouplingLayer,
    Glow,
    GlowBlock,
    InvertibleConv1x1,
)


def create_glow_config():
    """Create GlowConfig for testing Glow."""
    coupling_network = CouplingNetworkConfig(
        name="glow_coupling",
        hidden_dims=(32, 32),
        activation="relu",
    )
    return GlowConfig(
        name="test_glow",
        coupling_network=coupling_network,
        input_dim=4,
        latent_dim=4,
        base_distribution="normal",
        base_distribution_params={"loc": 0.0, "scale": 1.0},
        image_shape=(8, 8, 4),  # (height, width, channels)
        num_scales=2,
        blocks_per_scale=2,
    )


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def rngs(key):
    """Fixture for nnx random number generators."""
    return nnx.Rngs(
        params=jax.random.key(0),
        dropout=jax.random.key(1),
        sample=jax.random.key(2),
    )


@pytest.fixture
def input_data():
    """Fixture for input data."""
    # Input shape: (batch_size, height, width, channels)
    return jnp.ones((4, 8, 8, 4))


@pytest.fixture
def config():
    """Fixture for simple configuration."""
    return create_glow_config()


class TestActNormLayer:
    """Test cases for ActNormLayer."""

    def test_init(self, rngs):
        """Test initialization of ActNormLayer."""
        num_channels = 4
        layer = ActNormLayer(num_channels, rngs=rngs)

        # Check attributes
        assert layer.num_channels == num_channels
        assert not layer.initialized

        # Check parameters
        assert layer.logs.shape == (1, 1, num_channels)
        assert layer.bias.shape == (1, 1, num_channels)

        # Initial values should be zeros
        assert jnp.allclose(layer.logs, jnp.zeros_like(layer.logs))
        assert jnp.allclose(layer.bias, jnp.zeros_like(layer.bias))

    def test_initialize_from_data(self, rngs, input_data):
        """Test _initialize_from_data method."""
        num_channels = input_data.shape[-1]
        layer = ActNormLayer(num_channels, rngs=rngs)

        # Initialize from data
        layer._initialize_from_data(input_data)

        # Check initialized flag
        assert layer.initialized

        # bias should be negative mean
        mean = jnp.mean(input_data, axis=(0, 1), keepdims=True)
        assert jnp.allclose(layer.bias, -mean)

        # logs should be log(1/std)
        std = jnp.std(input_data, axis=(0, 1), keepdims=True)
        assert jnp.allclose(layer.logs, jnp.log(1.0 / (std + 1e-6)))

    def test_forward(self, rngs, input_data):
        """Test forward transformation."""
        num_channels = input_data.shape[-1]
        layer = ActNormLayer(num_channels, rngs=rngs)

        # Forward pass (should initialize parameters)
        y, log_det = layer.forward(input_data, rngs=rngs)

        # Check output shape
        assert y.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check initialization
        assert layer.initialized

        # Check log determinant
        batch_size, height, width = input_data.shape[0:3]
        expected_log_det = height * width * jnp.sum(layer.logs)
        expected_log_det = jnp.repeat(expected_log_det, batch_size)
        assert jnp.allclose(log_det, expected_log_det)

    def test_inverse(self, rngs, input_data):
        """Test inverse transformation."""
        num_channels = input_data.shape[-1]
        layer = ActNormLayer(num_channels, rngs=rngs)

        # Forward pass
        y, _ = layer.forward(input_data, rngs=rngs)

        # Inverse pass
        x, log_det = layer.inverse(y, rngs=rngs)

        # Check output shape
        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check reconstruction
        assert jnp.allclose(x, input_data, rtol=1e-5, atol=1e-5)

        # Check log determinant
        batch_size, height, width = input_data.shape[0:3]
        expected_log_det = -height * width * jnp.sum(layer.logs)
        expected_log_det = jnp.repeat(expected_log_det, batch_size)
        assert jnp.allclose(log_det, expected_log_det)


class TestInvertibleConv1x1:
    """Test cases for InvertibleConv1x1."""

    def test_init(self, rngs):
        """Test initialization of InvertibleConv1x1."""
        num_channels = 4
        layer = InvertibleConv1x1(num_channels, rngs=rngs)

        # Check attributes
        assert layer.num_channels == num_channels

        # Check weight parameter
        assert layer.weight.shape == (num_channels, num_channels)

        # Weight should be orthogonal (W^T W = I)
        wTw = jnp.matmul(layer.weight.T, layer.weight)
        assert jnp.allclose(wTw, jnp.eye(num_channels), rtol=1e-5, atol=1e-5)

    def test_forward(self, rngs, input_data):
        """Test forward transformation."""
        num_channels = input_data.shape[-1]
        layer = InvertibleConv1x1(num_channels, rngs=rngs)

        # Forward pass
        y, log_det = layer.forward(input_data, rngs=rngs)

        # Check output shape
        assert y.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check log determinant
        batch_size, height, width = input_data.shape[0:3]
        expected_log_det = height * width * jnp.linalg.slogdet(layer.weight)[1]
        expected_log_det = jnp.repeat(expected_log_det, batch_size)
        assert jnp.allclose(log_det, expected_log_det, atol=1e-4)

    def test_inverse(self, rngs, input_data):
        """Test inverse transformation."""
        num_channels = input_data.shape[-1]
        layer = InvertibleConv1x1(num_channels, rngs=rngs)

        # Forward pass
        y, _ = layer.forward(input_data, rngs=rngs)

        # Inverse pass
        x, log_det = layer.inverse(y, rngs=rngs)

        # Check output shape
        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check reconstruction
        assert jnp.allclose(x, input_data, rtol=1e-3, atol=1e-3)

        # Check log determinant
        batch_size, height, width = input_data.shape[0:3]
        expected_log_det = -height * width * jnp.linalg.slogdet(layer.weight)[1]
        expected_log_det = jnp.repeat(expected_log_det, batch_size)
        assert jnp.allclose(log_det, expected_log_det)


class TestAffineCouplingLayer:
    """Test cases for AffineCouplingLayer."""

    def test_init(self, rngs):
        """Test initialization of AffineCouplingLayer."""
        num_channels = 4
        hidden_dims = [32, 32]
        layer = AffineCouplingLayer(num_channels=num_channels, hidden_dims=hidden_dims, rngs=rngs)

        # Check attributes
        assert layer.num_channels == num_channels
        assert layer.hidden_dims == hidden_dims
        assert layer.split_idx == num_channels // 2

        # Initially network is not built (lazy initialization)
        assert not layer.is_built
        assert not hasattr(layer, "nn_layers")
        assert not hasattr(layer, "nn_output")

        # Build network with dummy input to test network creation
        dummy_input = jnp.ones((2, 8, 8, num_channels))
        layer._build_network(dummy_input.shape, rngs=rngs)

        # After building, check neural network layers
        assert layer.is_built
        assert len(layer.nn_layers) == len(hidden_dims)
        assert layer.nn_output is not None

    def test_scale_and_translate(self, rngs, input_data):
        """Test _scale_and_translate method."""
        num_channels = input_data.shape[-1]
        layer = AffineCouplingLayer(num_channels=num_channels, rngs=rngs)

        # Call _scale_and_translate
        s, t = layer._scale_and_translate(input_data)

        # Check shapes
        batch_size, height, width = input_data.shape[0:3]
        half_channels = num_channels // 2
        assert s.shape == (batch_size, height, width, half_channels)
        assert t.shape == (batch_size, height, width, half_channels)

        # Scale should be bounded by tanh
        assert jnp.all((s >= -1.0) & (s <= 1.0))

    def test_forward(self, rngs, input_data):
        """Test forward transformation."""
        num_channels = input_data.shape[-1]
        layer = AffineCouplingLayer(num_channels=num_channels, rngs=rngs)

        # Forward pass
        y, log_det = layer.forward(input_data, rngs=rngs)

        # Check output shape
        assert y.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # First half should remain unchanged
        split_idx = num_channels // 2
        assert jnp.allclose(y[:, :, :, :split_idx], input_data[:, :, :, :split_idx])

        # Log determinant should be sum of scale
        s, _ = layer._scale_and_translate(input_data)
        expected_log_det = jnp.sum(s, axis=(1, 2, 3))
        assert jnp.allclose(log_det, expected_log_det)

    def test_inverse(self, rngs, input_data):
        """Test inverse transformation."""
        num_channels = input_data.shape[-1]
        layer = AffineCouplingLayer(num_channels=num_channels, rngs=rngs)

        # Forward pass
        y, _ = layer.forward(input_data, rngs=rngs)

        # Inverse pass
        x, log_det = layer.inverse(y, rngs=rngs)

        # Check output shape
        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check reconstruction
        assert jnp.allclose(x, input_data, rtol=1e-5, atol=1e-5)

        # Log determinant should be negative sum of scale
        s, _ = layer._scale_and_translate(input_data)
        expected_log_det = -jnp.sum(s, axis=(1, 2, 3))
        assert jnp.allclose(log_det, expected_log_det)


class TestGlowBlock:
    """Test cases for GlowBlock."""

    def test_init(self, rngs):
        """Test initialization of GlowBlock."""
        num_channels = 4
        hidden_dims = [32, 32]
        block = GlowBlock(num_channels=num_channels, hidden_dims=hidden_dims, rngs=rngs)

        # Check component layers
        assert isinstance(block.actnorm, ActNormLayer)
        assert isinstance(block.conv1x1, InvertibleConv1x1)
        assert isinstance(block.coupling, AffineCouplingLayer)

        # Check layer configurations
        assert block.actnorm.num_channels == num_channels
        assert block.conv1x1.num_channels == num_channels
        assert block.coupling.num_channels == num_channels
        assert block.coupling.hidden_dims == hidden_dims

    def test_forward(self, rngs, input_data):
        """Test forward transformation."""
        num_channels = input_data.shape[-1]
        block = GlowBlock(num_channels=num_channels, rngs=rngs)

        # Forward pass
        y, log_det = block.forward(input_data, rngs=rngs)

        # Check output shape
        assert y.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Individual components should be initialized
        assert block.actnorm.initialized

    def test_inverse(self, rngs, input_data):
        """Test inverse transformation."""
        num_channels = input_data.shape[-1]
        block = GlowBlock(num_channels=num_channels, rngs=rngs)

        # Forward pass
        y, _ = block.forward(input_data, rngs=rngs)

        # Inverse pass
        x, log_det = block.inverse(y, rngs=rngs)

        # Check output shape
        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check reconstruction
        assert jnp.allclose(x, input_data, rtol=1e-5, atol=1e-5)


class TestGlow:
    """Test cases for Glow model."""

    def test_init(self, config, rngs):
        """Test initialization of Glow."""
        model = Glow(config, rngs=rngs)

        # Check attributes
        assert model.image_shape == config.image_shape
        assert model.num_scales == config.num_scales
        assert model.blocks_per_scale == config.blocks_per_scale
        assert model.hidden_dims == list(config.coupling_network.hidden_dims)

        # Check flow layers
        expected_layers = config.num_scales * config.blocks_per_scale
        assert len(model.flow_layers) == expected_layers

        # Layers should be GlowBlocks
        for layer in model.flow_layers:
            assert isinstance(layer, GlowBlock)

    def test_init_flow_layers(self, config, rngs):
        """Test _init_flow_layers method."""
        model = Glow(config, rngs=rngs)

        # Check correct number of layers
        expected_layers = config.num_scales * config.blocks_per_scale
        assert len(model.flow_layers) == expected_layers

        # Check each block
        for i in range(expected_layers):
            scale_idx = i // config.blocks_per_scale

            # For first scale, channel count should match original channels
            if scale_idx == 0:
                expected_channels = config.image_shape[2]
            # For each subsequent scale, channels are multiplied by 4
            else:
                expected_channels = config.image_shape[2] * (4**scale_idx)

            block = model.flow_layers[i]
            assert isinstance(block, GlowBlock)
            assert block.coupling.num_channels == expected_channels

    def test_forward(self, config, rngs, input_data):
        """Test forward transformation through Glow."""
        model = Glow(config, rngs=rngs)

        # Forward transformation
        z, log_det = model.forward(input_data, rngs=rngs)

        # Check shapes
        assert z.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # z should be different from input
        assert not jnp.allclose(z, input_data)

    def test_inverse(self, config, rngs, input_data):
        """Test inverse transformation through Glow."""
        model = Glow(config, rngs=rngs)

        # Forward transformation
        z, _ = model.forward(input_data, rngs=rngs)

        # Inverse transformation
        x, log_det = model.inverse(z, rngs=rngs)

        # Check shapes
        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check we recover original input (with tolerance for numerical precision)
        assert jnp.allclose(x, input_data, rtol=1e-4, atol=1e-4)

    def test_log_prob(self, config, rngs, input_data):
        """Test log probability calculation."""
        model = Glow(config, rngs=rngs)

        # Calculate log probability
        log_prob = model.log_prob(input_data, rngs=rngs)

        # Check shape
        assert log_prob.shape == (input_data.shape[0],)

        # Log probability should be finite
        assert jnp.all(jnp.isfinite(log_prob))

    def test_call(self, config, rngs, input_data):
        """Test forward pass of Glow."""
        model = Glow(config, rngs=rngs)

        # Call model
        outputs = model(input_data)

        # Check outputs
        assert "z" in outputs
        assert "logdet" in outputs
        assert "log_prob" in outputs

        # Check shapes
        assert outputs["z"].shape == input_data.shape
        assert outputs["logdet"].shape == (input_data.shape[0],)
        assert outputs["log_prob"].shape == (input_data.shape[0],)

    def test_generate(self, config, rngs):
        """Test sample generation."""
        model = Glow(config, rngs=rngs)

        # Generate samples
        batch_size = 2
        samples = model.generate(n_samples=batch_size, rngs=rngs)

        # Check shape
        expected_shape = (batch_size, *config.image_shape)
        assert samples.shape == expected_shape

        # Samples should be finite
        assert jnp.all(jnp.isfinite(samples))

    def test_single_scale(self, rngs, input_data):
        """Test model with a single scale."""
        coupling_network = CouplingNetworkConfig(
            name="glow_coupling",
            hidden_dims=(32, 32),
            activation="relu",
        )
        config = GlowConfig(
            name="test_glow_single_scale",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            base_distribution="normal",
            image_shape=(8, 8, 4),
            num_scales=1,
            blocks_per_scale=2,
        )
        model = Glow(config, rngs=rngs)

        # Forward and inverse should work
        z, _ = model.forward(input_data, rngs=rngs)
        x, _ = model.inverse(z, rngs=rngs)

        # Check reconstruction
        assert jnp.allclose(x, input_data, rtol=1e-4, atol=1e-4)

    def test_no_blocks(self, rngs, input_data):
        """Test that config validation rejects zero blocks per scale."""
        coupling_network = CouplingNetworkConfig(
            name="glow_coupling",
            hidden_dims=(32, 32),
            activation="relu",
        )
        # GlowConfig validates that blocks_per_scale must be positive
        with pytest.raises(ValueError, match="blocks_per_scale must be positive"):
            GlowConfig(
                name="test_glow_no_blocks",
                coupling_network=coupling_network,
                input_dim=4,
                latent_dim=4,
                base_distribution="normal",
                image_shape=(8, 8, 4),
                num_scales=2,
                blocks_per_scale=0,
            )
