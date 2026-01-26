"""Tests for the base loss module."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from artifex.generative_models.core.losses.base import (
    LossCollection,
    reduce_loss,
)


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


class TestLossCollection:
    """Tests for the LossCollection class."""

    def test_empty_collection(self):
        """Test an empty loss collection."""
        collection = LossCollection()
        assert len(collection.losses) == 0

    def test_add_loss(self):
        """Test adding losses to the collection."""
        collection = LossCollection()

        def loss_fn(p, t):
            return jnp.mean(jnp.square(p - t))

        # Test method chaining
        result = collection.add(loss_fn, weight=2.0, name="loss1")
        assert result is collection
        assert len(collection.losses) == 1
        assert collection.losses[0][0] is loss_fn
        assert collection.losses[0][1] == 2.0
        assert collection.losses[0][2] == "loss1"

        # Add another loss
        collection.add(loss_fn, weight=0.5)
        assert len(collection.losses) == 2
        assert collection.losses[1][0] is loss_fn
        assert collection.losses[1][1] == 0.5
        assert collection.losses[1][2] == "loss_1"  # Auto-generated name

    def test_call(self):
        """Test calling the loss collection."""
        collection = LossCollection()

        # Define simple loss functions
        def loss1(p, t):
            return jnp.sum(jnp.square(p - t))

        def loss2(p, t):
            return jnp.sum(jnp.abs(p - t))

        # Add losses to collection
        collection.add(loss1, weight=2.0, name="l2")
        collection.add(loss2, weight=0.5, name="l1")

        # Test calling the collection
        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([0.0, 0.0])

        total_loss, loss_dict = collection(predictions, targets)

        # Expected values
        expected_l2 = jnp.sum(jnp.square(predictions - targets))  # 5.0
        expected_l1 = jnp.sum(jnp.abs(predictions - targets))  # 3.0
        # Final calculation: 2*5.0 + 0.5*3.0 = 10.0 + 1.5 = 11.5
        expected_total = 2.0 * expected_l2 + 0.5 * expected_l1

        # Verify results
        np.testing.assert_allclose(loss_dict["l2"], expected_l2)
        np.testing.assert_allclose(loss_dict["l1"], expected_l1)
        np.testing.assert_allclose(total_loss, expected_total)

    def test_call_with_kwargs(self):
        """Test calling collection with additional keyword arguments."""
        collection = LossCollection()

        # Define a loss function that takes additional kwargs
        def loss_with_kwargs(p, t, scale=1.0):
            return scale * jnp.sum(jnp.square(p - t))

        collection.add(loss_with_kwargs, weight=1.0, name="loss")

        # Test calling the collection with kwargs
        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([0.0, 0.0])

        total_loss, loss_dict = collection(predictions, targets, scale=2.0)

        # Expected value: 2.0 * (1^2 + 2^2) = 2.0 * 5.0 = 10.0
        expected = 2.0 * jnp.sum(jnp.square(predictions - targets))

        np.testing.assert_allclose(loss_dict["loss"], expected)
        np.testing.assert_allclose(total_loss, expected)

    def test_jit_compatibility(self):
        """Test that collection is compatible with JAX transformations."""
        collection = LossCollection()

        # Define a simple loss function
        def mse_loss(p, t):
            return jnp.mean(jnp.square(p - t))

        collection.add(mse_loss, weight=1.0)

        # Create a jitted version
        @nnx.jit
        def jitted_loss(p, t):
            return collection(p, t)[0]  # Return only total_loss

        # Test with some data
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 0.0, 0.0])

        # Run the jitted function (should compile without errors)
        result = jitted_loss(predictions, targets)

        # Expected: mean((1^2 + 2^2 + 3^2) / 3) = 14/3
        expected = jnp.mean(jnp.square(predictions - targets))

        np.testing.assert_allclose(result, expected)
