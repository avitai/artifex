"""Tests for the reconstruction losses module."""

import jax
import jax.numpy as jnp
import numpy as np

from artifex.generative_models.core.losses.reconstruction import (
    charbonnier_loss,
    huber_loss,
    mae_loss,
    mse_loss,
    psnr_loss,
)


class TestMSELoss:
    """Tests for the MSE loss function."""

    def test_identical_inputs(self):
        """Test MSE loss with identical inputs."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        result = mse_loss(predictions, targets)
        # Identical inputs should result in zero loss
        np.testing.assert_allclose(result, 0.0)

    def test_different_inputs(self):
        """Test MSE loss with different inputs."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        result = mse_loss(predictions, targets)
        # Expected: mean(1^2 + 2^2 + 3^2) = 14/3
        expected = jnp.mean(jnp.square(predictions - targets))
        np.testing.assert_allclose(result, expected)

    def test_with_weights(self):
        """Test MSE loss with weights."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        weights = jnp.array([2.0, 1.0, 0.5])
        result = mse_loss(predictions, targets, weights=weights)
        # Expected: mean((1^2 * 2.0) + (2^2 * 1.0) + (3^2 * 0.5))
        # = mean(2 + 4 + 4.5) = 10.5/3
        expected = jnp.mean(jnp.square(predictions - targets) * weights)
        np.testing.assert_allclose(result, expected)

    def test_sum_reduction(self):
        """Test MSE loss with sum reduction."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        result = mse_loss(predictions, targets, reduction="sum")
        # Expected: sum(1^2 + 2^2 + 3^2) = 14
        expected = jnp.sum(jnp.square(predictions - targets))
        np.testing.assert_allclose(result, expected)

    def test_no_reduction(self):
        """Test MSE loss with no reduction."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        result = mse_loss(predictions, targets, reduction="none")
        # Expected: [1^2, 2^2, 3^2] = [1, 4, 9]
        expected = jnp.square(predictions - targets)
        np.testing.assert_allclose(result, expected)

    def test_batched_input(self):
        """Test MSE loss with batched input."""
        predictions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        result = mse_loss(predictions, targets)
        # Expected: mean(1^2 + 2^2 + 3^2 + 4^2) = 30/4 = 7.5
        expected = jnp.mean(jnp.square(predictions - targets))
        np.testing.assert_allclose(result, expected)

    def test_axis_reduction(self):
        """Test MSE loss with axis reduction."""
        predictions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        result = mse_loss(predictions, targets, axis=0)
        # Expected: [mean(1^2 + 3^2), mean(2^2 + 4^2)] = [5, 10]
        expected = jnp.mean(jnp.square(predictions - targets), axis=0)
        np.testing.assert_allclose(result, expected)

    def test_gradient(self):
        """Test that gradients of MSE loss are computed correctly."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])

        # Define function to compute loss
        def loss_fn(preds):
            return mse_loss(preds, targets)

        # Compute gradients using JAX
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(predictions)

        # Expected gradients: d(MSE)/dx = 2(x-y)/n
        # For x=[1,2,3], y=[0,0,0], n=3, we get 2*[1,2,3]/3 = [2/3, 4/3, 2]
        expected = 2 * (predictions - targets) / len(predictions)
        np.testing.assert_allclose(gradients, expected)


class TestMAELoss:
    """Tests for the MAE loss function."""

    def test_identical_inputs(self):
        """Test MAE loss with identical inputs."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        result = mae_loss(predictions, targets)
        # Identical inputs should result in zero loss
        np.testing.assert_allclose(result, 0.0)

    def test_different_inputs(self):
        """Test MAE loss with different inputs."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        result = mae_loss(predictions, targets)
        # Expected: mean(|1| + |2| + |3|) = 6/3 = 2
        expected = jnp.mean(jnp.abs(predictions - targets))
        np.testing.assert_allclose(result, expected)

    def test_with_weights(self):
        """Test MAE loss with weights."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        weights = jnp.array([2.0, 1.0, 0.5])
        result = mae_loss(predictions, targets, weights=weights)
        # Expected: mean(|1| * 2.0 + |2| * 1.0 + |3| * 0.5)
        # = mean(2 + 2 + 1.5) = 5.5/3
        expected = jnp.mean(jnp.abs(predictions - targets) * weights)
        np.testing.assert_allclose(result, expected)


class TestHuberLoss:
    """Tests for the Huber loss function."""

    def test_quadratic_region(self):
        """Test Huber loss in the quadratic region."""
        predictions = jnp.array([0.5, -0.5])
        targets = jnp.array([0.0, 0.0])
        delta = 1.0
        result = huber_loss(predictions, targets, delta=delta)
        # All errors are <= delta, so we use the quadratic formula:
        # 0.5 * error^2
        expected = jnp.mean(0.5 * jnp.square(predictions - targets))
        np.testing.assert_allclose(result, expected)

    def test_linear_region(self):
        """Test Huber loss in the linear region."""
        predictions = jnp.array([2.0, -2.0])
        targets = jnp.array([0.0, 0.0])
        delta = 1.0
        result = huber_loss(predictions, targets, delta=delta)
        # All errors are > delta, so we use the linear formula:
        # delta * (|error| - 0.5 * delta)
        expected = jnp.mean(delta * (jnp.abs(predictions - targets) - 0.5 * delta))
        np.testing.assert_allclose(result, expected)

    def test_mixed_regions(self):
        """Test Huber loss with mixed quadratic and linear regions."""
        predictions = jnp.array([0.5, 2.0])
        targets = jnp.array([0.0, 0.0])
        delta = 1.0
        result = huber_loss(predictions, targets, delta=delta)

        # Manually compute expected loss
        errors = jnp.abs(predictions - targets)
        quadratic = 0.5 * jnp.square(errors)
        linear = delta * (errors - 0.5 * delta)
        expected = jnp.mean(jnp.where(errors <= delta, quadratic, linear))

        np.testing.assert_allclose(result, expected)


class TestCharbonnierLoss:
    """Tests for the Charbonnier loss function."""

    def test_basic(self):
        """Test basic Charbonnier loss."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])
        epsilon = 1e-3
        result = charbonnier_loss(predictions, targets, epsilon=epsilon)

        # Expected: mean(sqrt(1^2 + eps^2) + sqrt(2^2 + eps^2) + sqrt(3^2 + eps^2))
        errors = predictions - targets
        expected = jnp.mean(jnp.power(jnp.sqrt(jnp.square(errors) + epsilon**2), 1.0))
        np.testing.assert_allclose(result, expected)

    def test_approaches_l1(self):
        """Test that Charbonnier approaches L1 as epsilon approaches 0."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])

        # Compute L1 (MAE) loss
        l1_result = mae_loss(predictions, targets)

        # Compute Charbonnier with very small epsilon
        eps_small = 1e-8
        charb_result = charbonnier_loss(predictions, targets, epsilon=eps_small)

        # They should be very close
        np.testing.assert_allclose(l1_result, charb_result, rtol=1e-5)


class TestPSNRLoss:
    """Tests for the PSNR loss function."""

    def test_basic(self):
        """Test basic PSNR loss calculation."""
        predictions = jnp.array([[0.5, 0.6], [0.7, 0.8]])
        targets = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        max_value = 1.0

        result = psnr_loss(predictions, targets, max_value=max_value)

        # Compute MSE first
        mse = jnp.mean(jnp.square(predictions - targets))
        # Expected PSNR: -20 * log10(max_value / sqrt(mse))
        expected = -20 * jnp.log10(max_value / jnp.sqrt(mse + 1e-8))

        np.testing.assert_allclose(result, expected)

    def test_identical_inputs(self):
        """Test PSNR loss with identical inputs."""
        predictions = jnp.array([[0.5, 0.6], [0.7, 0.8]])
        targets = jnp.array([[0.5, 0.6], [0.7, 0.8]])
        max_value = 1.0

        # For identical inputs, MSE is 0, and PSNR approaches infinity
        # Due to the epsilon in the implementation,
        # it won't be exactly infinity
        result = psnr_loss(predictions, targets, max_value=max_value)

        # Should be a large negative number
        # (negative since we're expressing as a loss)
        assert result < -50.0
