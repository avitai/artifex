"""Tests for regularization loss functions."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.losses.regularization import (
    gradient_penalty,
    l1_regularization,
    l2_regularization,
    orthogonal_regularization,
    spectral_norm_regularization,
)


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def params():
    """Fixture for model parameters."""
    key = jax.random.key(0)
    return {
        "layer1": {
            "kernel": jax.random.normal(key, (10, 5)),
            "bias": jax.random.normal(jax.random.fold_in(key, 1), (5,)),
        },
        "layer2": {
            "kernel": jax.random.normal(jax.random.fold_in(key, 2), (5, 3)),
            "bias": jax.random.normal(jax.random.fold_in(key, 3), (3,)),
        },
    }


@pytest.fixture
def linear_model_fn():
    """Fixture for a simple linear model function."""

    def model(x, params):
        h = x @ params["layer1"]["kernel"] + params["layer1"]["bias"]
        h = nnx.relu(h)
        y = h @ params["layer2"]["kernel"] + params["layer2"]["bias"]
        return y

    return model


class TestL1Regularization:
    """Test cases for L1 regularization."""

    def test_l1_scalar(self, params):
        """Test L1 regularization with scalar factor."""
        # Compute L1 regularization loss
        loss = l1_regularization(params, 0.1)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test with different weight
        loss2 = l1_regularization(params, 0.2)
        # Double the weight, double the loss
        assert jnp.isclose(loss2, 2 * loss)

    def test_l1_predicate(self, params):
        """Test L1 regularization with a predicate function."""

        # Only regularize kernel weights, not biases
        def kernel_only(name, param):
            return "kernel" in name

        # Compute L1 regularization loss
        loss = l1_regularization(params, 0.1, predicate=kernel_only)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Calculate manually to verify
        manual_sum = 0.0
        for layer in params.values():
            manual_sum += jnp.sum(jnp.abs(layer["kernel"]))
        expected_loss = 0.1 * manual_sum

        assert jnp.isclose(loss, expected_loss)


class TestL2Regularization:
    """Test cases for L2 regularization (weight decay)."""

    def test_l2_scalar(self, params):
        """Test L2 regularization with scalar factor."""
        # Compute L2 regularization loss
        loss = l2_regularization(params, 0.1)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Manual calculation for verification
        manual_sum = 0.0
        for layer in params.values():
            for param in layer.values():
                manual_sum += jnp.sum(param**2)
        expected_loss = 0.1 * manual_sum

        assert jnp.isclose(loss, expected_loss)

        # Test with different weight
        loss2 = l2_regularization(params, 0.2)
        # Double the weight, double the loss
        assert jnp.isclose(loss2, 2 * loss)

    def test_l2_predicate(self, params):
        """Test L2 regularization with predicate functions."""

        # Set different weights for different parameter groups
        def kernel_only(name, param):
            return "kernel" in name

        # Compute weight decay loss
        loss = l2_regularization(params, 0.1, predicate=kernel_only)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Manual calculation for verification
        manual_sum = 0.0
        for layer_name, layer in params.items():
            manual_sum += jnp.sum(layer["kernel"] ** 2)
        expected_loss = 0.1 * manual_sum

        assert jnp.isclose(loss, expected_loss)


class TestGradientPenalty:
    """Test cases for gradient penalty regularization."""

    def test_gradient_penalty_basic(self, key, linear_model_fn, params):
        """Test basic gradient penalty calculation."""
        # Generate random input
        x = jax.random.normal(key, (4, 10))

        # Create fake real and generated samples
        x_real = x
        x_fake = x + 0.1

        def discriminator(x):
            return linear_model_fn(x, params)

        # Compute gradient penalty
        gp = gradient_penalty(x_real, x_fake, discriminator, key=key)

        # Loss should be scalar and finite
        assert gp.shape == ()
        assert jnp.isfinite(gp)

        # Gradient penalty should be non-negative
        assert gp >= 0.0

    def test_gradient_penalty_lambda(self, key, linear_model_fn, params):
        """Test gradient penalty with different lambda values."""
        # Generate random input
        x = jax.random.normal(key, (4, 10))

        # Create fake real and generated samples
        x_real = x
        x_fake = x + 0.1

        def discriminator(x):
            return linear_model_fn(x, params)

        # Compute gradient penalty with default lambda
        gp1 = gradient_penalty(x_real, x_fake, discriminator, key=key)

        # Compute with different lambda
        gp2 = gradient_penalty(x_real, x_fake, discriminator, lambda_gp=20.0, key=key)

        # Second penalty should be twice the first
        assert jnp.isclose(gp2, 2.0 * gp1)


class TestOrthogonalRegularization:
    """Test cases for orthogonal regularization."""

    def test_orthogonal_regularization_basic(self, params):
        """Test basic orthogonal regularization."""
        # Compute orthogonal regularization for a single weight matrix
        loss = orthogonal_regularization(params["layer1"]["kernel"])

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Orthogonal regularization should be non-negative
        assert loss >= 0.0

        # Create almost orthogonal parameters (orthogonal matrix)
        q, _ = jnp.linalg.qr(params["layer1"]["kernel"])

        # Regularization for orthogonal matrices should be close to zero
        ortho_loss = orthogonal_regularization(q)
        assert ortho_loss < loss

    def test_orthogonal_regularization_scale(self, params):
        """Test orthogonal regularization with different scale values."""
        # Compute regularization with default scale
        loss1 = orthogonal_regularization(params["layer1"]["kernel"])

        # Compute with different scale
        loss2 = orthogonal_regularization(params["layer1"]["kernel"], scale=2.0)

        # Second loss should be twice the first
        assert jnp.isclose(loss2, 2.0 * loss1)


class TestSpectralRegularization:
    """Test cases for spectral regularization."""

    def test_spectral_regularization_basic(self, params):
        """Test basic spectral regularization."""
        # Compute spectral regularization for a single weight matrix
        loss, u_vector = spectral_norm_regularization(params["layer1"]["kernel"])

        # Loss should be scalar and finite
        assert loss.shape == () or loss.shape == (1,)
        assert jnp.isfinite(loss)

        # Create parameters with small spectral norm (scaled down)
        small_params = params["layer1"]["kernel"] * 0.1

        # Regularization for scaled-down matrices should be smaller
        small_loss, small_u = spectral_norm_regularization(small_params)
        assert small_loss < loss

    def test_spectral_regularization_scale(self, params):
        """Test spectral regularization with different scale values."""
        # Compute regularization with default scale
        loss1, u1 = spectral_norm_regularization(params["layer1"]["kernel"])

        # Compute with different scale
        loss2, u2 = spectral_norm_regularization(params["layer1"]["kernel"], scale=2.0)

        # Second loss should be twice the first
        assert jnp.isclose(loss2, 2.0 * loss1)
