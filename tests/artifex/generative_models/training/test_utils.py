"""Tests for training utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.training.utils import (
    expand_dims_to_match,
    extract_batch_data,
    extract_model_prediction,
    reshape_for_broadcast,
    sample_logit_normal,
    sample_u_shaped,
)


class TestExtractModelPrediction:
    """Tests for extract_model_prediction utility."""

    def test_passthrough_array(self):
        """Test that array inputs are passed through unchanged."""
        arr = jnp.ones((2, 4))
        result = extract_model_prediction(arr)
        assert result.shape == (2, 4)
        assert jnp.allclose(result, 1.0)

    def test_extract_predicted_noise_key(self):
        """Test extraction with predicted_noise key (default priority)."""
        output = {"predicted_noise": jnp.ones((2, 4))}
        result = extract_model_prediction(output)
        assert result.shape == (2, 4)
        assert jnp.allclose(result, 1.0)

    def test_extract_prediction_key(self):
        """Test extraction with prediction key."""
        output = {"prediction": jnp.zeros((3, 5))}
        result = extract_model_prediction(output)
        assert result.shape == (3, 5)
        assert jnp.allclose(result, 0.0)

    def test_extract_output_key(self):
        """Test extraction with output key."""
        output = {"output": jnp.full((4, 6), 2.0)}
        result = extract_model_prediction(output)
        assert result.shape == (4, 6)
        assert jnp.allclose(result, 2.0)

    def test_extract_noise_key(self):
        """Test extraction with noise key."""
        output = {"noise": jnp.full((2, 8), -1.0)}
        result = extract_model_prediction(output)
        assert result.shape == (2, 8)
        assert jnp.allclose(result, -1.0)

    def test_key_priority_order(self):
        """Test that keys are checked in priority order."""
        # predicted_noise should be selected over prediction
        output = {
            "prediction": jnp.zeros((2, 4)),
            "predicted_noise": jnp.ones((2, 4)),
        }
        result = extract_model_prediction(output)
        assert jnp.allclose(result, 1.0)

    def test_custom_key_priority(self):
        """Test with custom key priority."""
        output = {
            "custom_output": jnp.zeros((2, 4)),
            "predicted_noise": jnp.ones((2, 4)),
        }
        # Custom key takes priority
        result = extract_model_prediction(output, keys=("custom_output",))
        assert jnp.allclose(result, 0.0)

    def test_fallback_to_first_value(self):
        """Test fallback to first value when no standard keys found."""
        output = {"unknown_key": jnp.full((2, 4), 5.0)}
        result = extract_model_prediction(output)
        assert result.shape == (2, 4)
        assert jnp.allclose(result, 5.0)

    def test_empty_keys_uses_fallback(self):
        """Test that empty keys tuple falls back to first value."""
        output = {"any_key": jnp.ones((2, 4))}
        result = extract_model_prediction(output, keys=())
        assert jnp.allclose(result, 1.0)

    def test_preserves_dtype(self):
        """Test that dtype is preserved."""
        output = {"predicted_noise": jnp.ones((2, 4), dtype=jnp.float16)}
        result = extract_model_prediction(output)
        assert result.dtype == jnp.float16


class TestExtractBatchData:
    """Tests for extract_batch_data utility."""

    def test_extract_image_key(self):
        """Test extraction with image key."""
        batch = {"image": jnp.ones((32, 28, 28, 1))}
        data = extract_batch_data(batch)
        assert data.shape == (32, 28, 28, 1)

    def test_extract_data_key(self):
        """Test extraction with data key."""
        batch = {"data": jnp.ones((32, 28, 28, 1))}
        data = extract_batch_data(batch)
        assert data.shape == (32, 28, 28, 1)

    def test_image_priority_over_data(self):
        """Test that image key has priority over data key."""
        batch = {
            "image": jnp.ones((32, 28, 28, 1)),
            "data": jnp.zeros((32, 28, 28, 1)),
        }
        data = extract_batch_data(batch)
        assert jnp.allclose(data, 1.0)

    def test_custom_keys(self):
        """Test with custom key priority."""
        batch = {"features": jnp.ones((32, 64))}
        data = extract_batch_data(batch, keys=("features",))
        assert data.shape == (32, 64)

    def test_raises_on_missing_key(self):
        """Test that KeyError is raised when no keys found."""
        batch = {"unknown": jnp.ones((32, 28, 28, 1))}
        with pytest.raises(KeyError):
            extract_batch_data(batch)


class TestExpandDimsToMatch:
    """Tests for expand_dims_to_match utility."""

    def test_expand_1d_to_4d(self):
        """Test expanding 1D array to 4D."""
        arr = jnp.array([0.5, 0.3])
        result = expand_dims_to_match(arr, 4)
        assert result.shape == (2, 1, 1, 1)

    def test_no_expansion_needed(self):
        """Test when array already has target dims."""
        arr = jnp.ones((2, 3, 4))
        result = expand_dims_to_match(arr, 3)
        assert result.shape == (2, 3, 4)


class TestReshapeForBroadcast:
    """Tests for reshape_for_broadcast utility."""

    def test_reshape_to_broadcast(self):
        """Test reshaping for broadcast."""
        arr = jnp.array([[0.5], [0.3]])
        result = reshape_for_broadcast(arr, 2, 4)
        assert result.shape == (2, 1, 1, 1)


class TestSampleLogitNormal:
    """Tests for sample_logit_normal utility."""

    def test_output_in_range(self):
        """Test that output is in (0, 1) range."""
        import jax

        key = jax.random.key(0)
        samples = sample_logit_normal(key, (1000,))
        assert jnp.all(samples > 0)
        assert jnp.all(samples < 1)


class TestSampleUShaped:
    """Tests for sample_u_shaped utility."""

    def test_output_in_range(self):
        """Test that output is in [0, 1] range."""
        import jax

        key = jax.random.key(0)
        samples = sample_u_shaped(key, (1000,))
        assert jnp.all(samples >= 0)
        assert jnp.all(samples <= 1)


class TestTrainingUtilsJAXTransformCompatibility:
    """Tensor utilities used by trainer steps should remain transform safe."""

    def test_batch_and_prediction_extractors_are_jittable_and_differentiable(self):
        data = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

        def batch_loss(values):
            return jnp.sum(extract_batch_data({"image": values}) ** 2)

        def prediction_loss(values):
            return jnp.sum(extract_model_prediction({"predicted_noise": values}) ** 2)

        batch_value = jax.jit(batch_loss)(data)
        prediction_value = jax.jit(prediction_loss)(data)
        batch_grad = jax.grad(batch_loss)(data)
        prediction_grad = jax.grad(prediction_loss)(data)

        assert batch_value.shape == ()
        assert prediction_value.shape == ()
        assert jnp.isfinite(batch_value)
        assert jnp.isfinite(prediction_value)
        assert batch_grad.shape == data.shape
        assert prediction_grad.shape == data.shape
        assert jnp.all(jnp.isfinite(batch_grad))
        assert jnp.all(jnp.isfinite(prediction_grad))

    def test_broadcast_helpers_are_jittable_and_differentiable(self):
        values = jnp.array([0.2, 0.4, 0.6], dtype=jnp.float32)

        def expand_loss(inputs):
            expanded = expand_dims_to_match(inputs, 4)
            return jnp.sum(expanded * jnp.ones((3, 1, 1, 1), dtype=jnp.float32))

        def reshape_loss(inputs):
            reshaped = reshape_for_broadcast(inputs, batch_size=3, target_ndim=4)
            return jnp.sum(reshaped * jnp.ones((3, 1, 1, 1), dtype=jnp.float32))

        expand_value = jax.jit(expand_loss)(values)
        reshape_value = jax.jit(reshape_loss)(values)
        expand_grad = jax.grad(expand_loss)(values)
        reshape_grad = jax.grad(reshape_loss)(values)

        assert expand_value.shape == ()
        assert reshape_value.shape == ()
        assert jnp.isfinite(expand_value)
        assert jnp.isfinite(reshape_value)
        assert expand_grad.shape == values.shape
        assert reshape_grad.shape == values.shape
        assert jnp.all(jnp.isfinite(expand_grad))
        assert jnp.all(jnp.isfinite(reshape_grad))

    def test_sampling_helpers_are_jittable_with_differentiable_parameters(self):
        key = jax.random.key(42)

        def logit_normal_loss(loc, scale):
            return jnp.sum(sample_logit_normal(key, (8,), loc=loc, scale=scale))

        logit_value = jax.jit(logit_normal_loss)(0.1, 0.7)
        loc_grad, scale_grad = jax.grad(logit_normal_loss, argnums=(0, 1))(0.1, 0.7)
        u_samples = jax.jit(lambda: sample_u_shaped(key, (8,)))()

        assert logit_value.shape == ()
        assert jnp.isfinite(logit_value)
        assert jnp.isfinite(loc_grad)
        assert jnp.isfinite(scale_grad)
        assert u_samples.shape == (8,)
        assert jnp.all(jnp.isfinite(u_samples))
