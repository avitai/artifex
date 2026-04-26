"""Tests for the base loss module."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from artifex.generative_models.core.losses.base import reduce_loss


class TestReduceLoss:
    """Tests for the reduce_loss function."""

    def test_reduce_mean(self):
        """Test mean reduction."""
        loss = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = reduce_loss(loss, reduction="mean")
        expected = jnp.mean(loss)
        np.testing.assert_allclose(result, expected)

    def test_reduce_sum(self):
        """Test sum reduction."""
        loss = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = reduce_loss(loss, reduction="sum")
        expected = jnp.sum(loss)
        np.testing.assert_allclose(result, expected)

    def test_reduce_none(self):
        """Test no reduction."""
        loss = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = reduce_loss(loss, reduction="none")
        np.testing.assert_allclose(result, loss)

    def test_reduce_with_weights(self):
        """Test reduction with weights."""
        loss = jnp.array([1.0, 2.0, 3.0, 4.0])
        weights = jnp.array([2.0, 0.5, 1.0, 0.0])
        result = reduce_loss(loss, reduction="mean", weights=weights)
        expected = jnp.mean(loss * weights)
        np.testing.assert_allclose(result, expected)

    def test_reduce_with_axis(self):
        """Test reduction along specific axis."""
        loss = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = reduce_loss(loss, reduction="mean", axis=0)
        expected = jnp.mean(loss, axis=0)
        np.testing.assert_allclose(result, expected)

    def test_reduce_batch_sum_2d(self):
        """Test batch_sum reduction on 2D tensor (batch, features)."""
        # Shape: (2, 3) - 2 samples, 3 features each
        loss = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = reduce_loss(loss, reduction="batch_sum")
        # Sum over features: [6.0, 15.0], then mean over batch: 10.5
        expected = jnp.mean(jnp.array([6.0, 15.0]))
        np.testing.assert_allclose(result, expected)

    def test_reduce_batch_sum_4d(self):
        """Test batch_sum reduction on 4D tensor (batch, H, W, C) - typical for images."""
        # Shape: (2, 2, 2, 1) - 2 images, 2x2 pixels, 1 channel
        loss = jnp.array([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[5.0], [6.0]], [[7.0], [8.0]]]])
        result = reduce_loss(loss, reduction="batch_sum")
        # Sum over spatial: [10.0, 26.0], then mean over batch: 18.0
        expected = jnp.mean(jnp.array([10.0, 26.0]))
        np.testing.assert_allclose(result, expected)

    def test_reduce_batch_sum_1d(self):
        """Test batch_sum reduction on 1D tensor (falls back to mean)."""
        loss = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = reduce_loss(loss, reduction="batch_sum")
        # 1D tensor: just mean
        expected = jnp.mean(loss)
        np.testing.assert_allclose(result, expected)

    def test_invalid_reduction(self):
        """Test that invalid reduction raises ValueError."""
        loss = jnp.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError):
            reduce_loss(loss, reduction="invalid")

    @pytest.mark.parametrize("reduction", ["mean", "sum", "batch_sum"])
    def test_reduce_loss_is_jittable_and_differentiable(self, reduction):
        """Jittable training paths should be able to transform shared reduction logic."""
        loss = jnp.array([[0.5, 1.5], [2.0, 3.0]], dtype=jnp.float32)
        weights = jnp.array([[1.0, 0.5], [0.75, 1.25]], dtype=jnp.float32)

        def scalar_loss(values: jax.Array) -> jax.Array:
            return reduce_loss(values, reduction=reduction, weights=weights)

        compiled_value = jax.jit(scalar_loss)(loss)
        gradients = jax.grad(scalar_loss)(loss)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert gradients.shape == loss.shape
        assert jnp.all(jnp.isfinite(gradients))

    def test_reduce_loss_none_is_jittable(self):
        """The no-reduction path should preserve shape under JIT."""
        loss = jnp.array([[0.5, 1.5], [2.0, 3.0]], dtype=jnp.float32)

        compiled_value = jax.jit(lambda values: reduce_loss(values, reduction="none"))(loss)

        assert compiled_value.shape == loss.shape
        assert jnp.allclose(compiled_value, loss)


def test_base_module_only_exposes_reduce_loss_contract() -> None:
    """The base loss module should not keep dead management helpers around."""
    base_module = importlib.import_module("artifex.generative_models.core.losses.base")

    assert hasattr(base_module, "reduce_loss")
    assert not hasattr(base_module, "LossCollection")
    assert not hasattr(base_module, "LossMetrics")
    assert not hasattr(base_module, "LossScheduler")
    assert not hasattr(base_module, "LossRegistry")
    assert not hasattr(base_module, "validate_loss_inputs")
    assert not hasattr(base_module, "safe_log")
    assert not hasattr(base_module, "safe_divide")
