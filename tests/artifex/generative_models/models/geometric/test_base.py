"""Tests for the geometric model base class."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ModelConfig
from artifex.generative_models.models.geometric.base import GeometricModel


class ConcreteGeometricModel(GeometricModel):
    """A minimal concrete implementation of GeometricModel for testing."""

    def __init__(self, config, *, rngs):
        super().__init__(config, rngs=rngs)
        # With unified config, access output_dim directly or from parameters
        self.output_dim = getattr(config, "output_dim", 3)
        self.model = nnx.Linear(in_features=4, out_features=self.output_dim, rngs=rngs)

    def __call__(self, x, auxiliary=None, *, deterministic=False):
        """Forward pass of the model."""
        return self.model(x)


@pytest.fixture
def geometric_model():
    """Create a concrete GeometricModel instance for testing."""
    rngs = nnx.Rngs(params=jax.random.key(0))
    config = ModelConfig(
        name="test_geometric",
        model_class="artifex.generative_models.models.geometric.base.GeometricModel",
        input_dim=3,
        hidden_dims=[64, 128, 64],
        output_dim=3,
    )
    return ConcreteGeometricModel(config, rngs=rngs)


class TestGeometricModel:
    """Tests for the GeometricModel base class."""

    def test_init(self, geometric_model):
        """Test that a concrete GeometricModel can be initialized."""
        assert isinstance(geometric_model, GeometricModel)
        assert geometric_model.config.output_dim == 3

    def test_call(self, geometric_model):
        """Test the forward pass of a GeometricModel."""
        batch_size = 2
        num_points = 5
        x = jnp.ones((batch_size, num_points, 4))  # [B, N, 4]

        output = geometric_model(x)

        # Check output shape is correct: [B, N, output_dim]
        expected_shape = (batch_size, num_points, geometric_model.output_dim)
        assert output.shape == expected_shape

    def test_with_rng(self, geometric_model):
        """Test the model with no RNGs (removed in refactoring)."""
        batch_size = 2
        num_points = 5
        x = jnp.ones((batch_size, num_points, 4))

        # Call without rngs parameter (removed in refactoring)
        output = geometric_model(x)

        expected_shape = (batch_size, num_points, geometric_model.output_dim)
        assert output.shape == expected_shape

    def test_with_deterministic(self, geometric_model):
        """Test that the model can be called with deterministic flag."""
        batch_size = 2
        num_points = 5
        x = jnp.ones((batch_size, num_points, 4))

        output_det = geometric_model(x, deterministic=True)

        expected_shape = (batch_size, num_points, geometric_model.output_dim)
        assert output_det.shape == expected_shape

    def test_with_auxiliary(self, geometric_model):
        """Test that the model can be called with auxiliary inputs."""
        batch_size = 2
        num_points = 5
        x = jnp.ones((batch_size, num_points, 4))
        auxiliary = {"condition": jnp.ones((batch_size, 10))}

        output = geometric_model(x, auxiliary=auxiliary)

        expected_shape = (batch_size, num_points, geometric_model.output_dim)
        assert output.shape == expected_shape
