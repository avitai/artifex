"""Tests for the RealNVP model."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    CouplingNetworkConfig,
    RealNVPConfig,
)
from artifex.generative_models.models.flow.real_nvp import CouplingLayer, RealNVP


def create_real_nvp_config():
    """Create RealNVPConfig for testing RealNVP."""
    coupling_network = CouplingNetworkConfig(
        name="realnvp_coupling",
        hidden_dims=(32, 32),
        activation="relu",
    )
    return RealNVPConfig(
        name="test_real_nvp",
        coupling_network=coupling_network,
        input_dim=4,  # Need even number for coupling
        latent_dim=4,
        base_distribution="normal",
        base_distribution_params={"loc": 0.0, "scale": 1.0},
        num_coupling_layers=3,
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
    return jnp.ones((4, 4))


@pytest.fixture
def config():
    """Fixture for simple configuration."""
    return create_real_nvp_config()


@pytest.fixture
def mask():
    """Fixture for coupling mask."""
    mask = jnp.zeros(4)
    mask = mask.at[:2].set(1.0)  # Mask first half (2 dimensions)
    return mask


class TestCouplingLayer:
    """Test cases for CouplingLayer."""

    def test_init(self, mask, rngs):
        """Test initialization of CouplingLayer."""
        hidden_dims = [32, 32]
        layer = CouplingLayer(mask=mask, hidden_dims=hidden_dims, rngs=rngs)

        # Check attributes
        assert jnp.array_equal(layer.mask, mask)
        assert layer.scale_activation == jax.nn.tanh

        # Check neural network layers
        assert len(layer.scale_layers) == len(hidden_dims)
        assert len(layer.translation_layers) == len(hidden_dims)
        assert layer.scale_out is not None
        assert layer.translation_out is not None

    def test_scale_and_translate(self, mask, rngs, input_data):
        """Test _scale_and_translate method."""
        hidden_dims = [32, 32]
        layer = CouplingLayer(mask=mask, hidden_dims=hidden_dims, rngs=rngs)

        # Call _scale_and_translate
        s, t = layer._scale_and_translate(input_data, rngs=rngs)

        # Check shapes (s and t should have shape batch_size x masked_dims)
        # mask has sum=2, so 1-mask has sum=2 (4 dimensions total)
        assert s.shape == (input_data.shape[0], jnp.sum(1 - mask))
        assert t.shape == (input_data.shape[0], jnp.sum(1 - mask))

        # Scale should be bounded by tanh
        assert jnp.all((s >= -1.0) & (s <= 1.0))

    def test_forward(self, mask, rngs, input_data):
        """Test forward transformation."""
        layer = CouplingLayer(mask=mask, hidden_dims=[32, 32], rngs=rngs)

        # Forward pass
        y, log_det = layer.forward(input_data, rngs=rngs)

        # Check output shape
        assert y.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check identity part (masked)
        # First half should be unchanged (mask = 1)
        assert jnp.allclose(y[:, :2], input_data[:, :2])

        # Second half should be transformed
        assert not jnp.allclose(y[:, 2:], input_data[:, 2:])

    def test_inverse(self, mask, rngs, input_data):
        """Test inverse transformation."""
        layer = CouplingLayer(mask=mask, hidden_dims=[32, 32], rngs=rngs)

        # Forward transformation
        y, _ = layer.forward(input_data, rngs=rngs)

        # Inverse transformation
        x, log_det = layer.inverse(y, rngs=rngs)

        # Check shape
        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check we recover original input
        assert jnp.allclose(x, input_data, rtol=1e-4, atol=1e-4)

        # Log determinant should be additive inverse of forward log determinant
        _, forward_log_det = layer.forward(input_data, rngs=rngs)
        assert jnp.allclose(log_det, -forward_log_det)

    def test_custom_scale_activation(self, mask, rngs, input_data):
        """Test with custom scale activation function."""
        # Use sigmoid instead of tanh
        layer = CouplingLayer(
            mask=mask, hidden_dims=[32, 32], scale_activation=jax.nn.sigmoid, rngs=rngs
        )

        # Forward pass
        y, log_det = layer.forward(input_data, rngs=rngs)

        # Check scale is bounded by sigmoid, not tanh
        s, _ = layer._scale_and_translate(input_data, rngs=rngs)
        assert jnp.all((s >= 0.0) & (s <= 1.0))

        # Inverse should still work
        x, _ = layer.inverse(y, rngs=rngs)
        assert jnp.allclose(x, input_data, rtol=1e-4, atol=1e-4)


class TestRealNVP:
    """Test cases for RealNVP model."""

    def test_init(self, config, rngs):
        """Test initialization of RealNVP."""
        model = RealNVP(config, rngs=rngs)

        # Check attributes
        assert model.num_coupling_layers == config.num_coupling_layers
        assert model.hidden_dims == list(config.coupling_network.hidden_dims)
        assert model.input_dim == config.input_dim
        assert model.latent_dim == config.latent_dim

        # Check flow layers
        assert len(model.flow_layers) == config.num_coupling_layers
        for layer in model.flow_layers:
            assert isinstance(layer, CouplingLayer)

    def test_init_coupling_layers(self, config, rngs):
        """Test _init_coupling_layers method."""
        model = RealNVP(config, rngs=rngs)

        # Check correct number of layers
        assert len(model.flow_layers) == config.num_coupling_layers

        # Just verify that masks are created and have the correct shape
        # The actual mask patterns might vary based on implementation
        for layer in model.flow_layers:
            assert layer.mask.shape == (model.total_dim,)
            # Ensure each mask contains both 0s and 1s (not all same value)
            assert not jnp.all(layer.mask == 0)
            assert not jnp.all(layer.mask == 1)

    def test_forward(self, config, rngs, input_data):
        """Test forward transformation through RealNVP."""
        model = RealNVP(config, rngs=rngs)

        # Forward transformation
        z, log_det = model.forward(input_data, rngs=rngs)

        # Check shapes
        assert z.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # z should be different from input after multiple coupling layers
        assert not jnp.allclose(z, input_data)

    def test_inverse(self, config, rngs, input_data):
        """Test inverse transformation through RealNVP."""
        model = RealNVP(config, rngs=rngs)

        # Forward transformation
        z, _ = model.forward(input_data, rngs=rngs)

        # Inverse transformation
        x, log_det = model.inverse(z, rngs=rngs)

        # Check shapes
        assert x.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check we recover original input
        assert jnp.allclose(x, input_data, rtol=1e-4, atol=1e-4)

    def test_log_prob(self, config, rngs, input_data):
        """Test log probability calculation."""
        model = RealNVP(config, rngs=rngs)

        # Calculate log probability
        log_prob = model.log_prob(input_data, rngs=rngs)

        # Check shape
        assert log_prob.shape == (input_data.shape[0],)

        # Log probability should be finite
        assert jnp.all(jnp.isfinite(log_prob))

    def test_call(self, config, rngs, input_data):
        """Test forward pass of RealNVP."""
        model = RealNVP(config, rngs=rngs)

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
        model = RealNVP(config, rngs=rngs)

        # Generate samples
        batch_size = 4
        samples = model.generate(n_samples=batch_size, rngs=rngs)

        # Check shape
        assert samples.shape == (batch_size, config.input_dim)

        # Samples should be finite
        assert jnp.all(jnp.isfinite(samples))

    def test_loss_fn(self, config, rngs, input_data):
        """Test loss function calculation."""
        model = RealNVP(config, rngs=rngs)

        # Calculate loss - pass input_data as batch and an empty dict as model_outputs
        metrics = model.loss_fn(input_data, {}, rngs=rngs)

        # Check metrics contain expected keys
        assert "loss" in metrics
        assert "log_prob" in metrics

        # Extract loss
        loss = metrics["loss"]

        # Check scalar loss
        assert jnp.isscalar(loss) or loss.shape == ()

        # Loss should be negative log likelihood
        assert jnp.allclose(loss, -jnp.mean(model.log_prob(input_data, rngs=rngs)))

        # All values should be finite
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(jnp.array(list(metrics.values()))))

    def test_different_hidden_dims(self, rngs, input_data):
        """Test model with different hidden dimensions."""
        coupling_network = CouplingNetworkConfig(
            name="realnvp_coupling",
            hidden_dims=(64, 32, 16),
            activation="relu",
        )
        config = RealNVPConfig(
            name="test_real_nvp_diff_hidden",
            coupling_network=coupling_network,
            input_dim=4,
            latent_dim=4,
            num_coupling_layers=3,
        )
        model = RealNVP(config, rngs=rngs)

        # Check model works with different hidden dimensions
        z, log_det = model.forward(input_data, rngs=rngs)

        # Forward and inverse should still be consistent
        x, _ = model.inverse(z, rngs=rngs)
        assert jnp.allclose(x, input_data, rtol=1e-4, atol=1e-4)

    def test_no_coupling_layers(self, rngs, input_data):
        """Test model with no coupling layers - should raise ValueError."""
        coupling_network = CouplingNetworkConfig(
            name="realnvp_coupling",
            hidden_dims=(32, 32),
            activation="relu",
        )
        # RealNVPConfig validates that num_coupling_layers must be positive
        with pytest.raises(ValueError):
            RealNVPConfig(
                name="test_real_nvp_no_layers",
                coupling_network=coupling_network,
                input_dim=4,
                latent_dim=4,
                num_coupling_layers=0,
            )
