"""Tests for the base Normalizing Flow model."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    CouplingNetworkConfig,
    FlowConfig,
)
from artifex.generative_models.models.flow.base import FlowLayer, NormalizingFlow


class SimpleFlowLayer(FlowLayer):
    """Simple flow layer for testing.

    Implements a simple linear transformation: y = ax + b, log_det = log|a|.
    """

    def __init__(self, scale=2.0, bias=1.0, *, rngs=None):
        """Initialize simple flow layer.

        Args:
            scale: Scale factor for linear transformation
            bias: Bias term for linear transformation
            rngs: Random number generators
        """
        super().__init__(rngs=rngs)
        self.scale = scale
        self.bias = bias

    def forward(self, x, *, rngs=None):
        """Forward transformation: y = ax + b.

        Args:
            x: Input tensor of shape [batch_size, dim]
            rngs: Random number generators

        Returns:
            Tuple of (transformed_x, log_det_jacobian)
        """
        y = self.scale * x + self.bias
        # Log determinant is log|a| for each input dimension, summed over dimensions
        log_det = jnp.full(x.shape[0], jnp.sum(jnp.log(jnp.abs(self.scale))))
        return y, log_det

    def inverse(self, y, *, rngs=None):
        """Inverse transformation: x = (y - b) / a.

        Args:
            y: Input tensor of shape [batch_size, dim]
            rngs: Random number generators

        Returns:
            Tuple of (transformed_y, log_det_jacobian)
        """
        x = (y - self.bias) / self.scale
        # Log determinant is -log|a| for inverse transformation
        log_det = jnp.full(y.shape[0], -jnp.sum(jnp.log(jnp.abs(self.scale))))
        return x, log_det


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
    return jnp.ones((4, 2))


@pytest.fixture
def coupling_network():
    """Fixture for coupling network configuration."""
    return CouplingNetworkConfig(
        name="test_coupling",
        hidden_dims=(64, 64),
        activation="relu",
    )


@pytest.fixture
def config(coupling_network):
    """Fixture for model configuration."""
    return FlowConfig(
        name="test_flow",
        coupling_network=coupling_network,
        input_dim=2,
        latent_dim=2,
        base_distribution="normal",
        base_distribution_params={"loc": 0.0, "scale": 1.0},
    )


@pytest.fixture
def flow_model(config, rngs):
    """Fixture for normalizing flow model with a simple layer."""
    model = NormalizingFlow(config, rngs=rngs)
    model.flow_layers.append(SimpleFlowLayer(rngs=rngs))
    return model


class TestFlowLayer:
    """Test cases for FlowLayer base class."""

    def test_init(self, rngs):
        """Test initialization of FlowLayer."""
        layer = SimpleFlowLayer(rngs=rngs)
        assert layer.scale == 2.0
        assert layer.bias == 1.0

    def test_forward(self, rngs, input_data):
        """Test forward transformation of FlowLayer."""
        layer = SimpleFlowLayer(scale=2.0, bias=1.0, rngs=rngs)
        y, log_det = layer.forward(input_data, rngs=rngs)

        # Check output shape
        assert y.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check values
        expected_y = 2.0 * input_data + 1.0
        expected_log_det = jnp.full(input_data.shape[0], jnp.sum(jnp.log(jnp.abs(2.0))))
        assert jnp.allclose(y, expected_y)
        assert jnp.allclose(log_det, expected_log_det)

    def test_inverse(self, rngs, input_data):
        """Test inverse transformation of FlowLayer."""
        layer = SimpleFlowLayer(scale=2.0, bias=1.0, rngs=rngs)

        # First apply forward transformation
        y, _ = layer.forward(input_data, rngs=rngs)

        # Then apply inverse transformation
        x, log_det = layer.inverse(y, rngs=rngs)

        # Check that we recover the original input
        assert jnp.allclose(x, input_data)

        # Check log determinant
        expected_log_det = jnp.full(input_data.shape[0], -jnp.sum(jnp.log(jnp.abs(2.0))))
        assert jnp.allclose(log_det, expected_log_det)


class TestNormalizingFlow:
    """Test cases for NormalizingFlow base class."""

    def test_init(self, config, rngs):
        """Test initialization of NormalizingFlow."""
        model = NormalizingFlow(config, rngs=rngs)

        # Check that attributes are set correctly
        assert model.input_dim == config.input_dim
        assert model.latent_dim == config.latent_dim
        assert model.base_distribution_type == config.base_distribution
        assert len(model.flow_layers) == 0

    def test_setup_base_distribution_normal(self, config, rngs):
        """Test setup of normal base distribution."""
        model = NormalizingFlow(config, rngs=rngs)

        # Generate random input for log_prob
        x = jnp.zeros((4, config.latent_dim))
        log_prob = model.log_prob_fn(x)

        # Check shape of log probability
        assert log_prob.shape == (4,)

        # Generate samples
        key = jax.random.key(0)
        samples = model.sample_fn(key, 4)

        # Check shape of samples
        assert samples.shape == (4, config.latent_dim)

    def test_setup_base_distribution_uniform(self, coupling_network, rngs):
        """Test setup of uniform base distribution."""
        # Create config with uniform distribution
        config = FlowConfig(
            name="test_flow_uniform",
            coupling_network=coupling_network,
            input_dim=2,
            latent_dim=2,
            base_distribution="uniform",
            base_distribution_params={"low": -1.0, "high": 1.0},
        )

        model = NormalizingFlow(config, rngs=rngs)

        # Generate random input for log_prob
        x = jnp.zeros((4, config.latent_dim))
        log_prob = model.log_prob_fn(x)

        # Check shape of log probability
        assert log_prob.shape == (4,)

        # Generate samples
        key = jax.random.key(0)
        samples = model.sample_fn(key, 4)

        # Check shape of samples
        assert samples.shape == (4, config.latent_dim)

        # Check samples are within range
        assert jnp.all((samples >= -1.0) & (samples <= 1.0))

    def test_setup_base_distribution_invalid(self, coupling_network, rngs):
        """Test setup of invalid base distribution."""
        # Create config with invalid distribution - should fail at FlowConfig creation
        with pytest.raises(ValueError):
            FlowConfig(
                name="test_flow_invalid",
                coupling_network=coupling_network,
                input_dim=2,
                latent_dim=2,
                base_distribution="invalid",
            )

    def test_forward_empty(self, config, rngs, input_data):
        """Test forward transformation with empty flow."""
        model = NormalizingFlow(config, rngs=rngs)

        # Forward transformation
        z, log_det = model.forward(input_data, rngs=rngs)

        # With no layers, should return input unchanged
        assert jnp.allclose(z, input_data)
        assert jnp.allclose(log_det, jnp.zeros(input_data.shape[0]))

    def test_forward(self, flow_model, input_data, rngs):
        """Test forward transformation."""
        # Forward transformation
        z, log_det = flow_model.forward(input_data, rngs=rngs)

        # Check output shape
        assert z.shape == input_data.shape
        assert log_det.shape == (input_data.shape[0],)

        # Check values (should be transformed by SimpleFlowLayer)
        expected_z = 2.0 * input_data + 1.0
        expected_log_det = jnp.full(input_data.shape[0], jnp.sum(jnp.log(jnp.abs(2.0))))
        assert jnp.allclose(z, expected_z)
        assert jnp.allclose(log_det, expected_log_det)

    def test_inverse_empty(self, config, rngs, input_data):
        """Test inverse transformation with empty flow."""
        model = NormalizingFlow(config, rngs=rngs)

        # Inverse transformation
        x, log_det = model.inverse(input_data, rngs=rngs)

        # With no layers, should return input unchanged
        assert jnp.allclose(x, input_data)
        assert jnp.allclose(log_det, jnp.zeros(input_data.shape[0]))

    def test_inverse(self, flow_model, input_data, rngs):
        """Test inverse transformation."""
        # First apply forward transformation
        z, _ = flow_model.forward(input_data, rngs=rngs)

        # Then apply inverse transformation
        x, log_det = flow_model.inverse(z, rngs=rngs)

        # Check that we recover the original input
        assert jnp.allclose(x, input_data)

        # Check log determinant
        expected_log_det = jnp.full(input_data.shape[0], -jnp.sum(jnp.log(jnp.abs(2.0))))
        assert jnp.allclose(log_det, expected_log_det)

    def test_log_prob(self, flow_model, input_data, rngs):
        """Test log probability calculation."""
        # Calculate log probability
        log_prob = flow_model.log_prob(input_data, rngs=rngs)

        # Check shape
        assert log_prob.shape == (input_data.shape[0],)

        # Transform input to latent space
        z, log_det = flow_model.forward(input_data, rngs=rngs)

        # Calculate log probability of z under base distribution
        log_prob_z = flow_model.log_prob_fn(z)

        # Check that log_prob = log_prob_z + log_det
        assert jnp.allclose(log_prob, log_prob_z + log_det)

    def test_call(self, flow_model, input_data, rngs):
        """Test forward pass through model."""
        # Call model
        outputs = flow_model(input_data)

        # Check outputs
        assert "z" in outputs
        assert "logdet" in outputs
        assert "log_prob" in outputs

        # Check shapes
        assert outputs["z"].shape == input_data.shape
        assert outputs["logdet"].shape == (input_data.shape[0],)
        assert outputs["log_prob"].shape == (input_data.shape[0],)

        # Transform input to latent space
        z, log_det = flow_model.forward(input_data, rngs=rngs)

        # Check that outputs are consistent
        assert jnp.allclose(outputs["z"], z)
        assert jnp.allclose(outputs["logdet"], log_det)
        assert jnp.allclose(outputs["log_prob"], flow_model.log_prob_fn(z) + log_det)

    def test_generate(self, flow_model, rngs):
        """Test sample generation."""
        # Generate samples
        samples = flow_model.generate(n_samples=4, rngs=rngs)

        # Check shape
        assert samples.shape == (4, 2)

    def test_loss_fn(self, flow_model, input_data, rngs):
        """Test loss function calculation."""
        # Calculate loss - pass input_data as batch and empty dict as model_outputs
        metrics = flow_model.loss_fn(input_data, {}, rngs=rngs)

        # Extract loss
        loss = metrics["loss"]

        # Check scalar loss
        assert loss.ndim == 0

        # Check metrics
        assert "loss" in metrics
        assert "log_prob" in metrics

    def test_multiple_layers(self, config, rngs, input_data):
        """Test model with multiple flow layers."""
        model = NormalizingFlow(config, rngs=rngs)
        model.flow_layers.append(SimpleFlowLayer(scale=2.0, bias=1.0, rngs=rngs))
        model.flow_layers.append(SimpleFlowLayer(scale=0.5, bias=-0.5, rngs=rngs))

        # Forward transformation
        z, log_det = model.forward(input_data, rngs=rngs)

        # Check output shape
        assert z.shape == input_data.shape

        # Compose the transformations
        # Layer 1: y = 2x + 1
        # Layer 2: z = 0.5y - 0.5 = 0.5(2x + 1) - 0.5 = x
        # So the expected output is z = x
        assert jnp.allclose(z, input_data)

        # Log determinant: log|2| + log|0.5| = log(2) + log(0.5) = log(1) = 0
        expected_log_det = jnp.zeros(input_data.shape[0])
        assert jnp.allclose(log_det, expected_log_det)

        # Inverse transformation
        x, inv_log_det = model.inverse(z, rngs=rngs)

        # Check inverse
        assert jnp.allclose(x, input_data)
        assert jnp.allclose(inv_log_det, expected_log_det)

        # Log probability
        log_prob = model.log_prob(input_data, rngs=rngs)

        # Since the transformation is identity, log_prob should equal log_prob_fn(x)
        expected_log_prob = model.log_prob_fn(input_data)
        assert jnp.allclose(log_prob, expected_log_prob)
