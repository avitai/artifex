"""Integration tests for diffusion models."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.sampling.diffusion import DiffusionSampler
from artifex.generative_models.models.diffusion import (
    DDPMModel,
    LDMModel,
    ScoreDiffusionModel,
)
from tests.utils.test_helpers import should_run_diffusion_tests


DIFFUSION_SKIP_REASON = (
    "Diffusion integration test skipped due to GroupNorm reshape issues in flax.nnx. "
    "The error occurs when GroupNorm tries to reshape tensors with incompatible dimensions. "
    "For example, reshaping (batch, height, width, 64) into (batch, height, width, 32, 1). "
    "Set RUN_DIFFUSION_TESTS=1 to force execution of these tests."
)


@pytest.fixture
def rng():
    """Random number generator fixture."""
    return jax.random.key(0)


@pytest.fixture
def diffusion_config():
    """Create diffusion model configuration for testing."""
    from artifex.generative_models.core.configuration import ModelConfig

    return ModelConfig(
        name="test_ddpm",
        model_class="artifex.generative_models.models.diffusion.DDPMModel",
        input_dim=(16, 16, 3),  # Small image size for testing
        output_dim=(16, 16, 3),  # Same as input for diffusion
        hidden_dims=[32, 64],  # Small network for testing
        parameters={
            "noise_steps": 20,  # Small number of steps for testing
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "time_embedding_dim": 16,
        },
    )


@pytest.fixture
def ldm_config():
    """Create latent diffusion model configuration for testing."""
    from artifex.generative_models.core.configuration import ModelConfig

    return ModelConfig(
        name="test_ldm",
        model_class="artifex.generative_models.models.diffusion.LDMModel",
        input_dim=(16, 16, 3),  # Small image size for testing
        output_dim=(16, 16, 3),  # Same as input for diffusion
        hidden_dims=[32, 64],  # Small network for testing
        parameters={
            "noise_steps": 20,  # Small number of steps for testing
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "time_embedding_dim": 16,
            "latent_dim": 8,  # Small latent dimension for testing
            "encoder_hidden_dims": [16, 32],  # Small encoder for testing
            "decoder_hidden_dims": [32, 16],  # Small decoder for testing
        },
    )


class TestDiffusionIntegration:
    """Integration tests for diffusion models."""

    def test_ddpm_forward_process_reverse_process(self, rng, diffusion_config):
        """Test DDPM forward and reverse processes."""
        if not should_run_diffusion_tests():
            pytest.skip(DIFFUSION_SKIP_REASON)

        # Initialize model
        model = DDPMModel(diffusion_config, rngs=nnx.Rngs(params=rng))

        # Generate clean sample
        batch_size = 2
        x_0 = jnp.ones((batch_size, 16, 16, 3))  # Simple image for testing

        # Select timestep
        t = jnp.array([4, 6])  # Different timestep for each sample

        # Apply forward process to add noise
        noise = jnp.ones_like(x_0) * 0.5  # Fixed noise for testing
        noisy_x = model.q_sample(x_0, t, noise)

        # Model should predict the noise
        model_output = model(noisy_x, t)

        # Extract predicted noise from model output dictionary
        pred_noise = model_output["predicted_noise"]

        # Check outputs
        assert pred_noise.shape == noise.shape
        # Ideally we'd check accuracy, but just ensure output is finite
        assert jnp.all(jnp.isfinite(pred_noise))

    def test_ddpm_sampling_process(self, rng, diffusion_config):
        """Test DDPM sampling process."""
        if not should_run_diffusion_tests():
            pytest.skip(DIFFUSION_SKIP_REASON)

        # Initialize model
        model = DDPMModel(diffusion_config, rngs=nnx.Rngs(params=rng))

        # Sample from model
        n_samples = 2
        sample_key = jax.random.key(1)
        samples = model.sample(n_samples, rngs=nnx.Rngs(params=sample_key))

        # Check output shape
        assert samples.shape == (n_samples, 16, 16, 3)
        # Ensure outputs are finite
        assert jnp.all(jnp.isfinite(samples))

    def test_ldm_latent_diffusion(self, rng, ldm_config):
        """Test latent diffusion model with encoder/decoder."""
        if not should_run_diffusion_tests():
            pytest.skip(DIFFUSION_SKIP_REASON)

        # Create model
        model = LDMModel(ldm_config, rngs=nnx.Rngs(params=rng))

        # Create dummy input
        batch_size = 2
        input_shape = (batch_size,) + ldm_config["input_dim"]
        x = jnp.ones(input_shape)

        # Encode to latent space
        latent = model.encode(x)

        # Verify latent shape
        # Assuming 4x downsampling for spatial dimensions
        expected_height = ldm_config["input_dim"][0] // 4
        expected_width = ldm_config["input_dim"][1] // 4
        expected_latent_shape = (
            batch_size,
            expected_height,
            expected_width,
            ldm_config["latent_dim"],
        )
        assert latent.shape == expected_latent_shape

        # Test forward diffusion in latent space
        t = jnp.array([5, 10])  # Different timesteps for each batch item

        # Add noise through forward process
        sample_key = jax.random.key(4)
        noisy_latent, noise = model.forward_diffusion(latent, t, rngs=nnx.Rngs(params=sample_key))

        # Verify shapes
        assert noisy_latent.shape == latent.shape
        assert noise.shape == latent.shape

        # Test noise prediction in latent space
        pred_noise = model(noisy_latent, t)
        assert pred_noise.shape == noise.shape

        # Test decoding from latent space
        decoded = model.decode(latent)
        assert decoded.shape == input_shape

        # Test end-to-end sampling
        n_samples = 2
        key5 = jax.random.key(5)
        samples = model.sample(n_samples, rngs=nnx.Rngs(params=key5))
        assert samples.shape == (n_samples,) + ldm_config["input_dim"]

    def test_score_diffusion_integration(self, rng, diffusion_config):
        """Test score-based diffusion model integration."""
        if not should_run_diffusion_tests():
            pytest.skip(DIFFUSION_SKIP_REASON)

        # Create config
        config = diffusion_config.copy()
        # Use different schedule for variety
        config["beta_schedule"] = "cosine"
        config["num_timesteps"] = 500

        # Create model
        model = ScoreDiffusionModel(config, rngs=nnx.Rngs(params=rng))

        # Generate clean sample
        batch_size = 2
        x_0 = jnp.ones((batch_size, 16, 16, 3))  # Simple image

        # Sample timestep
        t = jnp.array([250, 400])

        # Add noise
        noise = jnp.ones_like(x_0) * 0.1
        noisy_x = model.q_sample(x_0, t, noise)

        # Get score
        score = model(noisy_x, t)

        # Check outputs
        assert score.shape == x_0.shape
        assert jnp.all(jnp.isfinite(score))

        # Test sampling
        samples = model.sample(
            n_samples=1,
            steps=10,  # Fewer steps for testing
            rngs=nnx.Rngs(params=jax.random.key(3)),
        )
        assert samples.shape[0] == 1
        assert samples.shape[1:] == x_0.shape[1:]

    def test_diffusion_sampler_integration(self, rng, diffusion_config):
        """Test integration with diffusion samplers."""
        if not should_run_diffusion_tests():
            pytest.skip(DIFFUSION_SKIP_REASON)

        # Create model
        model = DDPMModel(diffusion_config, rngs=nnx.Rngs(params=rng))

        # Create sampler
        sampler = DiffusionSampler(model)

        # Sample using the sampler with default parameters
        samples = sampler.sample(
            n_samples=2,
            steps=10,  # Reduced for testing
            rngs=nnx.Rngs(params=jax.random.key(4)),
        )

        # Check outputs
        assert samples.shape == (2, 16, 16, 3)
        assert jnp.all(jnp.isfinite(samples))

    def test_conditional_diffusion(self, rng, diffusion_config):
        """Test conditional diffusion."""
        if not should_run_diffusion_tests():
            pytest.skip(DIFFUSION_SKIP_REASON)

        # Create config with conditioning
        config = diffusion_config.copy()
        config["num_classes"] = 10
        config["class_embed_dim"] = 8

        # Create model
        model = DDPMModel(config, rngs=nnx.Rngs(params=rng))

        # Create dummy input and conditions
        batch_size = 2
        input_shape = (batch_size,) + config["input_dim"]
        x = jnp.ones(input_shape)
        # Two different classes
        conditions = jnp.array([0, 1], dtype=jnp.int32)

        # Test forward process with conditions
        t = jnp.array([5, 10])
        noisy_x, noise = model.forward_diffusion(x, t, rngs=nnx.Rngs(params=jax.random.key(20)))

        # Test conditional denoising
        pred_noise = model(noisy_x, t, conditions)
        assert pred_noise.shape == noise.shape

        # Test conditional sampling
        n_samples = 3
        class_labels = jnp.array([2, 3, 4], dtype=jnp.int32)

        samples = model.sample(
            n_samples,
            conditions=class_labels,
            rngs=nnx.Rngs(params=jax.random.key(21)),
        )

        # Verify conditional sample shape
        assert samples.shape == (n_samples,) + config["input_dim"]

    def test_noise_prediction_loss(self, rng, diffusion_config):
        """Test noise prediction loss computation."""
        if not should_run_diffusion_tests():
            pytest.skip(DIFFUSION_SKIP_REASON)

        # Create model
        model = DDPMModel(diffusion_config, rngs=nnx.Rngs(params=rng))

        # Create dummy input
        batch_size = 2
        input_shape = (batch_size,) + diffusion_config["input_dim"]
        x = jnp.ones(input_shape)

        # Compute loss
        key30 = jax.random.key(30)
        loss, metrics = model.loss_fn(x, rngs=nnx.Rngs(params=key30))

        # Check loss is finite and reasonable
        assert jnp.isfinite(loss)
        assert loss > 0.0  # Loss should be positive

        # Check metrics
        assert "loss" in metrics
        assert jnp.isfinite(metrics["loss"])
        assert jnp.allclose(loss, metrics["loss"])
