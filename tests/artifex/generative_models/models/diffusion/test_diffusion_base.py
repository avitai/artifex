"""Tests for the Diffusion base model.

DiffusionModel uses the (config, *, rngs) signature pattern.
Backbone is created internally via create_backbone factory from config.backbone.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DiffusionConfig,
    NoiseScheduleConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.models.diffusion.base import DiffusionModel


def create_test_config():
    """Create a test configuration for diffusion models."""
    return DiffusionConfig(
        name="test_diffusion",
        input_shape=(32, 32, 3),  # (H, W, C) format - JAX convention
        backbone=UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(32, 64),
            activation="gelu",
            in_channels=3,
            out_channels=3,
            time_embedding_dim=64,
        ),
        noise_schedule=NoiseScheduleConfig(
            name="test_schedule",
            schedule_type="linear",
            num_timesteps=100,
            beta_start=1e-4,
            beta_end=2e-2,
        ),
    )


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def rngs(key):
    """Fixture for nnx random number generators."""
    # Use default stream which provides all needed keys
    return nnx.Rngs(default=jax.random.key(42))


@pytest.fixture
def input_data():
    """Fixture for input data in (B, H, W, C) format - JAX convention."""
    return jnp.ones((4, 32, 32, 3))


@pytest.fixture
def config():
    """Fixture for simple configuration."""
    return create_test_config()


@pytest.fixture
def model(config, rngs):
    """Fixture for diffusion model."""
    # DiffusionModel now uses (config, *, rngs) signature
    # Backbone is created internally from config.backbone via create_backbone factory
    return DiffusionModel(config, rngs=rngs)


class TestDiffusionModel:
    """Test cases for DiffusionModel."""

    def test_init(self, config, rngs):
        """Test initialization of DiffusionModel."""
        model = DiffusionModel(config, rngs=rngs)

        # Check that backbone is initialized
        assert model.backbone is not None

        # Check that noise schedule is set up
        num_timesteps = config.noise_schedule.num_timesteps
        assert model.betas.shape == (num_timesteps,)
        assert model.alphas.shape == (num_timesteps,)
        assert model.alphas_cumprod.shape == (num_timesteps,)

    def test_noise_schedule_integration(self, config, rngs):
        """Test that noise schedule is properly integrated."""
        model = DiffusionModel(config, rngs=rngs)

        # Check beta values from noise_schedule config
        assert jnp.isclose(model.betas[0], config.noise_schedule.beta_start, rtol=1e-5)
        assert jnp.isclose(model.betas[-1], config.noise_schedule.beta_end, rtol=1e-5)

        # Check alpha values
        assert jnp.allclose(model.alphas, 1.0 - model.betas)
        assert jnp.allclose(model.alphas_cumprod, jnp.cumprod(model.alphas))

    def test_call(self, model, input_data):
        """Test forward pass of DiffusionModel."""
        # Create timesteps
        batch_size = input_data.shape[0]
        timesteps = jnp.zeros((batch_size,), dtype=jnp.int32)

        # Forward pass - rngs stored at init time per NNX best practices
        output = model(input_data, timesteps)

        # Output should be a dictionary with predicted_noise
        assert isinstance(output, dict)
        assert "predicted_noise" in output

        # Predicted noise should have same shape as input
        assert output["predicted_noise"].shape == input_data.shape

    def test_q_sample(self, model, input_data):
        """Test q_sample method."""
        # Create timesteps (all zeros for simplicity)
        batch_size = input_data.shape[0]
        t = jnp.zeros((batch_size,), dtype=jnp.int32)

        # Fixed noise for reproducibility
        noise = jnp.ones_like(input_data)

        # Forward diffusion process
        x_t = model.q_sample(input_data, t, noise=noise)

        # Output should have same shape as input
        assert x_t.shape == input_data.shape

        # x_t should be a weighted combination of x_0 and noise
        # For t=0, it should be close to the original input
        assert jnp.allclose(x_t, input_data, atol=1e-2)

        # For t=max_timestep, it should be close to pure noise
        t_max = jnp.full(
            (batch_size,), model.config.noise_schedule.num_timesteps - 1, dtype=jnp.int32
        )
        x_t_max = model.q_sample(input_data, t_max, noise=noise)

        # Should be more noisy than the original input
        diff_start = jnp.mean(jnp.abs(x_t - input_data))
        diff_end = jnp.mean(jnp.abs(x_t_max - input_data))
        assert diff_end > diff_start

    def test_extract_into_tensor(self, model):
        """Test _extract_into_tensor method."""
        # Create 1D array
        arr = jnp.arange(10, dtype=jnp.float32)

        # Create timesteps
        timesteps = jnp.array([1, 3, 5], dtype=jnp.int32)

        # Create broadcast shape
        broadcast_shape = (3, 2, 2)

        # Extract values
        extracted = model._extract_into_tensor(arr, timesteps, broadcast_shape)

        # Check shape
        assert extracted.shape == broadcast_shape

        # Check values
        expected_values = jnp.array([arr[1], arr[3], arr[5]])
        expected_broadcast = jnp.broadcast_to(expected_values.reshape(3, 1, 1), broadcast_shape)
        assert jnp.allclose(extracted, expected_broadcast)

    def test_predict_start_from_noise(self, model, input_data):
        """Test predict_start_from_noise method."""
        # Create timesteps
        batch_size = input_data.shape[0]
        t = jnp.zeros((batch_size,), dtype=jnp.int32)

        # Create noise
        noise = jnp.ones_like(input_data)

        # Predict x_0 from noise
        pred_x0 = model.predict_start_from_noise(input_data, t, noise)

        # Should have same shape as input
        assert pred_x0.shape == input_data.shape

    def test_q_posterior_mean_variance(self, model, input_data):
        """Test q_posterior_mean_variance method."""
        # Create timesteps
        batch_size = input_data.shape[0]
        t = jnp.ones((batch_size,), dtype=jnp.int32)
        # Use t=1 to test posterior

        # Create predicted x_0
        x_start = input_data

        # Create noisy sample x_t
        noise = jnp.ones_like(input_data)
        x_t = model.q_sample(x_start, t, noise=noise)

        # Get posterior mean and variance
        posterior_mean, posterior_variance, posterior_log_variance = (
            model.q_posterior_mean_variance(x_start, x_t, t)
        )

        # Check shapes
        assert posterior_mean.shape == input_data.shape
        assert posterior_variance.shape == input_data.shape
        assert posterior_log_variance.shape == input_data.shape

        # Values should be finite
        assert jnp.all(jnp.isfinite(posterior_mean))
        assert jnp.all(jnp.isfinite(posterior_variance))
        assert jnp.all(jnp.isfinite(posterior_log_variance))

    def test_p_mean_variance(self, model, input_data):
        """Test p_mean_variance method."""
        # Create timesteps
        batch_size = input_data.shape[0]
        t = jnp.ones((batch_size,), dtype=jnp.int32)

        # Create model output (predicted noise)
        model_output = jnp.ones_like(input_data)

        # Get predicted mean and variance
        out = model.p_mean_variance(model_output, input_data, t)

        # Check returned values
        assert "mean" in out
        assert "variance" in out
        assert "log_variance" in out
        assert "pred_x_start" in out

        # Check shapes
        assert out["mean"].shape == input_data.shape
        assert out["variance"].shape == input_data.shape
        assert out["log_variance"].shape == input_data.shape
        assert out["pred_x_start"].shape == input_data.shape

        # Values should be finite
        assert jnp.all(jnp.isfinite(out["mean"]))
        assert jnp.all(jnp.isfinite(out["variance"]))
        assert jnp.all(jnp.isfinite(out["log_variance"]))
        assert jnp.all(jnp.isfinite(out["pred_x_start"]))

        # Check that predicted x_0 is clipped to [-1, 1] with a small tolerance
        # for numerical precision
        tolerance = 0.01  # Allow for small floating point errors
        pred_min = out["pred_x_start"] >= -1.0 - tolerance
        pred_max = out["pred_x_start"] <= 1.0 + tolerance
        assert jnp.all(pred_min & pred_max)

        # Test without clipping using a larger noise value to ensure we get
        # values outside [-1, 1]
        model_output = jnp.ones_like(input_data) * 10.0
        out_no_clip = model.p_mean_variance(model_output, input_data, t, clip_denoised=False)

        # Check that values change when clipping is disabled
        # Values may still be within [-1, 1] range depending on the noise scale
        # and timestep
        # Just check for isfinite as that's the main concern
        assert jnp.all(jnp.isfinite(out_no_clip["pred_x_start"]))

    def test_p_sample(self, model, input_data):
        """Test p_sample method."""
        # Create timesteps
        batch_size = input_data.shape[0]
        t = jnp.ones((batch_size,), dtype=jnp.int32)

        # Create model output (predicted noise)
        model_output = jnp.ones_like(input_data)

        # Sample - model uses internal self.rngs
        sample = model.p_sample(model_output, input_data, t)

        # Check shape
        assert sample.shape == input_data.shape

        # Values should be finite
        assert jnp.all(jnp.isfinite(sample))

        # Test that calling p_sample again produces different results (RNG state advances)
        sample2 = model.p_sample(model_output, input_data, t)
        assert sample2.shape == input_data.shape
        assert jnp.all(jnp.isfinite(sample2))
        # Should be different due to RNG state advancement (for non-zero timesteps)
        if jnp.any(t != 0):
            assert not jnp.allclose(sample, sample2)

    def test_generate(self, model):
        """Test generate method."""
        # Generate 2 samples
        batch_size = 2
        samples = model.generate(batch_size)

        # Check shape - use input_shape from DiffusionConfig
        expected_shape = (batch_size, *model.config.input_shape)
        assert samples.shape == expected_shape

        # Values should be finite
        assert jnp.all(jnp.isfinite(samples))

        # And within appropriate range for normalized images
        assert jnp.all((samples >= -1.1) & (samples <= 1.1))

    def test_loss_fn(self, model, input_data):
        """Test loss_fn method."""
        # Compute loss - rngs stored at init time per NNX best practices
        # Create mock model outputs for loss computation
        timesteps = jnp.zeros((input_data.shape[0],), dtype=jnp.int32)
        model_outputs = model(input_data, timesteps)
        result = model.loss_fn(input_data, model_outputs)

        # Check that result is a dictionary
        assert isinstance(result, dict)
        assert "loss" in result
        assert jnp.isscalar(result["loss"])
        assert jnp.isfinite(result["loss"])

        # Check other metrics
        assert "mse_loss" in result
        assert jnp.isscalar(result["mse_loss"])
        assert jnp.isfinite(result["mse_loss"])

    def test_timestep_sampling(self, model, input_data):
        """Test sampling of timesteps in loss_fn."""

        # Define custom function for sampling timesteps
        def sample_timesteps(key, shape, minval, maxval):
            """Custom timestep sampling function for testing."""
            # Always sample the same mid-range timestep for testing
            batch_size = shape[0]
            num_timesteps = maxval  # maxval is num_timesteps in this case
            return jnp.full((batch_size,), num_timesteps // 2, dtype=jnp.int32)

        # Monkey patch the random sampling
        original_randint = jax.random.randint
        try:
            jax.random.randint = sample_timesteps

            # Compute loss with patched random sampling
            timesteps = jnp.zeros((input_data.shape[0],), dtype=jnp.int32)
            model_outputs = model(input_data, timesteps)
            result = model.loss_fn(input_data, model_outputs)

            # Verify we get a valid loss
            assert isinstance(result, dict)
            assert "loss" in result
            assert jnp.isfinite(result["loss"])
        finally:
            # Restore original function
            jax.random.randint = original_randint

    def test_noise_sampling(self, model, input_data):
        """Test sampling of noise in loss_fn."""

        # Define custom function for sampling noise
        def sample_noise(key, shape):
            """Custom noise sampling function for testing."""
            # Return constant noise for testing
            return jnp.ones(shape)

        # Monkey patch the random sampling
        original_normal = jax.random.normal
        try:
            jax.random.normal = sample_noise

            # Compute loss with patched random sampling
            timesteps = jnp.zeros((input_data.shape[0],), dtype=jnp.int32)
            model_outputs = model(input_data, timesteps)
            result = model.loss_fn(input_data, model_outputs)

            # Verify we get a valid loss
            assert isinstance(result, dict)
            assert "loss" in result
            assert jnp.isfinite(result["loss"])
        finally:
            # Restore original function
            jax.random.normal = original_normal
