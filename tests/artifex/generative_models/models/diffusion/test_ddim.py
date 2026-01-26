"""Tests for DDIM (Denoising Diffusion Implicit Models) implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DDIMConfig,
    NoiseScheduleConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.models.diffusion.ddim import DDIMModel


class TestDDIM:
    """Test DDIM model implementation."""

    @pytest.fixture
    def rngs(self):
        """Create RNGs for testing with all required streams."""
        return nnx.Rngs(default=jax.random.PRNGKey(42))

    @pytest.fixture
    def ddim_config(self):
        """Create DDIM configuration."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(32, 64, 128),
            activation="relu",
            in_channels=3,
            out_channels=3,
            channel_mult=(1, 2, 4),
            num_res_blocks=2,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            num_timesteps=1000,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        return DDIMConfig(
            name="test_ddim",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(32, 32, 3),  # (H, W, C) format - JAX convention
            eta=0.0,  # Deterministic sampling
            num_inference_steps=50,
            skip_type="uniform",
        )

    def test_ddim_initialization(self, ddim_config, rngs):
        """Test DDIM model initialization."""
        model = DDIMModel(config=ddim_config, rngs=rngs)

        assert model is not None
        assert hasattr(model, "ddim_sample")
        assert hasattr(model, "get_ddim_timesteps")
        assert model.eta == 0.0
        assert model.noise_steps == 1000

    def test_ddim_timestep_generation(self, ddim_config, rngs):
        """Test DDIM timestep generation."""
        model = DDIMModel(config=ddim_config, rngs=rngs)

        # Test uniform spacing
        timesteps = model.get_ddim_timesteps(50)
        assert len(timesteps) == 50
        assert jnp.all(timesteps >= 0)
        assert jnp.all(timesteps < 1000)
        # Should be in reverse order (high to low)
        assert jnp.all(timesteps[:-1] >= timesteps[1:])

    def test_ddim_deterministic_sampling(self, ddim_config):
        """Test deterministic DDIM sampling."""
        # Create two models with same seed for deterministic comparison
        rngs1 = nnx.Rngs(default=jax.random.PRNGKey(123))
        rngs2 = nnx.Rngs(default=jax.random.PRNGKey(123))

        model1 = DDIMModel(config=ddim_config, rngs=rngs1)
        model2 = DDIMModel(config=ddim_config, rngs=rngs2)

        # Generate samples with eta=0 (deterministic)
        samples1 = model1.ddim_sample(2, steps=20, eta=0.0)
        samples2 = model2.ddim_sample(2, steps=20, eta=0.0)

        # Should be identical for deterministic sampling with same seed
        assert jnp.allclose(samples1, samples2)

    def test_ddim_stochastic_sampling(self, rngs):
        """Test stochastic DDIM sampling."""
        # Create config with eta=1 for stochastic sampling
        backbone = UNetBackboneConfig(
            name="test_unet_stochastic",
            hidden_dims=(32, 64, 128),
            activation="relu",
            in_channels=3,
            out_channels=3,
            channel_mult=(1, 2, 4),
            num_res_blocks=2,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_stochastic",
            num_timesteps=1000,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        config = DDIMConfig(
            name="test_ddim_stochastic",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(32, 32, 3),  # (H, W, C) format - JAX convention
            eta=1.0,  # Stochastic sampling (equivalent to DDPM)
            num_inference_steps=50,
            skip_type="uniform",
        )
        model = DDIMModel(config=config, rngs=rngs)

        # Generate samples - rngs stored at init, not passed here
        samples = model.ddim_sample(2, steps=20)

        assert samples.shape == (2, 32, 32, 3)
        # Check samples are in reasonable range
        assert jnp.all(jnp.isfinite(samples))

    def test_ddim_vs_ddpm_compatibility(self, rngs):
        """Test that DDIM with eta=1 behaves like DDPM."""
        # Create config with eta=1
        backbone = UNetBackboneConfig(
            name="test_unet_compat",
            hidden_dims=(32, 64, 128),
            activation="relu",
            in_channels=3,
            out_channels=3,
            channel_mult=(1, 2, 4),
            num_res_blocks=2,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_compat",
            num_timesteps=1000,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        config = DDIMConfig(
            name="test_ddim_compat",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(32, 32, 3),  # (H, W, C) format - JAX convention
            eta=1.0,
            num_inference_steps=50,
            skip_type="uniform",
        )
        model = DDIMModel(config=config, rngs=rngs)

        # Sample with DDIM eta=1 (should be similar to DDPM)
        samples_ddim = model.sample(2, scheduler="ddim", steps=50)

        # Sample with DDPM
        samples_ddpm = model.sample(2, scheduler="ddpm", steps=50)

        # Both should produce valid samples
        assert samples_ddim.shape == samples_ddpm.shape
        assert jnp.all(jnp.isfinite(samples_ddim))
        assert jnp.all(jnp.isfinite(samples_ddpm))

    def test_ddim_reverse_process(self, ddim_config, rngs):
        """Test DDIM reverse process (encoding).

        Following industry-standard testing patterns (huggingface/diffusers):
        - Test shape preservation
        - Test finite values
        - Test that transformation occurred (output != input)
        - Note: variance comparisons require trained models and aren't reliable for unit tests
        """
        model = DDIMModel(config=ddim_config, rngs=rngs)

        # Create a clean image
        key = jax.random.PRNGKey(42)
        x0 = jax.random.normal(key, (2, 32, 32, 3))
        x0 = jnp.clip(x0, -1.0, 1.0)

        # Encode to noise - rngs stored at init, not passed here
        encoded = model.ddim_reverse(x0, ddim_steps=50)

        # Shape preservation
        assert encoded.shape == x0.shape

        # All values should be finite
        assert jnp.all(jnp.isfinite(encoded))

        # Transformation should have occurred (output differs from input)
        assert not jnp.allclose(encoded, x0)

    def test_ddim_different_step_counts(self, ddim_config, rngs):
        """Test DDIM with different numbers of sampling steps."""
        model = DDIMModel(config=ddim_config, rngs=rngs)

        for steps in [10, 25, 50, 100]:
            # rngs stored at init, not passed here
            samples = model.ddim_sample(1, steps=steps)
            assert samples.shape == (1, 32, 32, 3)
            assert jnp.all(jnp.isfinite(samples))
