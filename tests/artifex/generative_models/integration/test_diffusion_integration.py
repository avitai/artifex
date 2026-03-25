"""Integration tests for diffusion models."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DDPMConfig,
    LatentDiffusionConfig,
    NoiseScheduleConfig,
    ScoreDiffusionConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.sampling.diffusion import DiffusionSampler
from artifex.generative_models.models.diffusion import (
    DDPMModel,
    LDMModel,
    ScoreDiffusionModel,
)


@pytest.fixture
def rngs():
    """Random number generators fixture with all needed streams."""
    return nnx.Rngs(
        params=jax.random.key(0),
        dropout=jax.random.key(1),
        sample=jax.random.key(2),
        noise=jax.random.key(3),
        timestep=jax.random.key(4),
    )


@pytest.fixture
def ddpm_config():
    """Create DDPM configuration for testing."""
    backbone = UNetBackboneConfig(
        name="test_unet",
        hidden_dims=(32, 64),
        activation="relu",
        in_channels=3,
        out_channels=3,
        channel_mult=(1, 2),
        num_res_blocks=1,
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        num_timesteps=20,
        schedule_type="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )
    return DDPMConfig(
        name="test_ddpm",
        backbone=backbone,
        noise_schedule=noise_schedule,
        input_shape=(16, 16, 3),
        loss_type="mse",
        clip_denoised=True,
    )


@pytest.fixture
def ldm_config():
    """Create latent diffusion model configuration for testing."""
    backbone = UNetBackboneConfig(
        name="test_unet_ldm",
        hidden_dims=(32, 64),
        activation="relu",
        in_channels=8,
        out_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        num_timesteps=20,
        schedule_type="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )
    encoder = EncoderConfig(
        name="test_encoder",
        input_shape=(16, 16, 3),
        hidden_dims=(16, 32),
        latent_dim=8,
        activation="relu",
    )
    decoder = DecoderConfig(
        name="test_decoder",
        latent_dim=8,
        hidden_dims=(32, 16),
        output_shape=(16, 16, 3),
        activation="relu",
    )
    return LatentDiffusionConfig(
        name="test_ldm",
        backbone=backbone,
        noise_schedule=noise_schedule,
        input_shape=(16, 16, 3),
        encoder=encoder,
        decoder=decoder,
        latent_scale_factor=1.0,
    )


@pytest.fixture
def score_config():
    """Create score diffusion configuration for testing."""
    backbone = UNetBackboneConfig(
        name="test_unet_score",
        hidden_dims=(32, 64),
        activation="relu",
        in_channels=3,
        out_channels=3,
        channel_mult=(1, 2),
        num_res_blocks=1,
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        num_timesteps=20,
        schedule_type="linear",
        beta_start=1e-4,
        beta_end=0.02,
    )
    return ScoreDiffusionConfig(
        name="test_score",
        backbone=backbone,
        noise_schedule=noise_schedule,
        input_shape=(16, 16, 3),
        sigma_min=0.01,
        sigma_max=50.0,
    )


class TestDiffusionIntegration:
    """Integration tests for diffusion models."""

    def test_ddpm_forward_process_reverse_process(self, rngs, ddpm_config):
        """Test DDPM forward and reverse processes."""
        model = DDPMModel(ddpm_config, rngs=rngs)

        batch_size = 2
        x_0 = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([4, 6])

        # Apply forward process to add noise
        noise = jnp.ones_like(x_0) * 0.5
        noisy_x = model.q_sample(x_0, t, noise)

        # Model should predict the noise
        model_output = model(noisy_x, t)
        pred_noise = model_output["predicted_noise"]

        assert pred_noise.shape == noise.shape
        assert jnp.all(jnp.isfinite(pred_noise))

    def test_ddpm_sampling_process(self, rngs, ddpm_config):
        """Test DDPM sampling process."""
        model = DDPMModel(ddpm_config, rngs=rngs)

        n_samples = 2
        samples = model.sample(n_samples)

        assert samples.shape == (n_samples, 16, 16, 3)
        assert jnp.all(jnp.isfinite(samples))

    def test_ldm_latent_diffusion(self, rngs, ldm_config):
        """Test latent diffusion model with encoder/decoder."""
        model = LDMModel(ldm_config, rngs=rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 3))

        # Encode to latent space (returns mean, log_var)
        mean, log_var = model.encode(x)
        assert mean.ndim >= 2
        assert mean.shape[0] == batch_size
        assert mean.shape[1] == ldm_config.encoder.latent_dim

        # Test forward diffusion in latent space
        t = jnp.array([5, 10])
        noisy_latent, noise = model.forward_diffusion(mean, t)
        assert noisy_latent.shape == mean.shape
        assert noise.shape == mean.shape

        # Test full forward pass (encode → diffuse → denoise → decode)
        output = model(x)
        assert "predicted_noise" in output
        assert "reconstructed" in output
        assert "mean" in output
        assert "log_var" in output

        # Test decoding from latent space
        decoded = model.decode(mean)
        assert decoded.shape == x.shape

        # Test end-to-end sampling
        n_samples = 2
        samples = model.sample(n_samples)
        assert samples.shape[0] == n_samples

    def test_score_diffusion_integration(self, rngs, score_config):
        """Test score-based diffusion model integration."""
        model = ScoreDiffusionModel(score_config, rngs=rngs)

        batch_size = 2
        x_0 = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([5, 10])

        # Add noise
        noise = jnp.ones_like(x_0) * 0.1
        noisy_x = model.q_sample(x_0, t, noise)

        # Get score
        output = model(noisy_x, t)
        score = output["predicted_noise"]

        assert score.shape == x_0.shape
        assert jnp.all(jnp.isfinite(score))

        # Test sampling
        samples = model.sample(num_samples=1, num_steps=10)
        assert samples.shape[0] == 1
        assert samples.shape[1:] == x_0.shape[1:]

    def test_diffusion_sampler_integration(self, rngs, ddpm_config):
        """Test integration with diffusion samplers."""
        model = DDPMModel(ddpm_config, rngs=rngs)
        sampler = DiffusionSampler(model=model)

        samples = sampler.sample(
            n_samples=2,
            steps=10,
            rngs=rngs,
        )

        assert samples.shape == (2, 16, 16, 3)
        assert jnp.all(jnp.isfinite(samples))

    def test_ddpm_with_cosine_schedule(self, rngs, ddpm_config):
        """Test DDPM with cosine noise schedule."""
        cosine_schedule = replace(ddpm_config.noise_schedule, schedule_type="cosine")
        config = replace(ddpm_config, noise_schedule=cosine_schedule)

        model = DDPMModel(config, rngs=rngs)

        batch_size = 2
        x_0 = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([4, 6])

        model_output = model(x_0, t)
        pred_noise = model_output["predicted_noise"]

        assert pred_noise.shape == x_0.shape
        assert jnp.all(jnp.isfinite(pred_noise))

    def test_noise_prediction_loss(self, rngs, ddpm_config):
        """Test noise prediction loss computation."""
        model = DDPMModel(ddpm_config, rngs=rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([5, 10])

        # Forward pass to get model outputs
        model_output = model(x, t)

        # Compute loss using model outputs
        loss_dict = model.loss_fn({"x": x}, model_output)

        assert "total_loss" in loss_dict
        assert jnp.isfinite(loss_dict["total_loss"])
        assert loss_dict["total_loss"] > 0.0
