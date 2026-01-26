"""Tests for DDPM (Denoising Diffusion Probabilistic Models) implementation.

This module provides comprehensive tests for the DDPM model, covering
initialization, forward and reverse diffusion processes, noise scheduling,
and sampling consistency.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.configuration import (
    DDPMConfig,
    NoiseScheduleConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.models.diffusion.ddpm import DDPMModel
from tests.utils.model_testing import DiffusionTestUtils, ModelTestRunner


def update_config(config: DDPMConfig, **updates) -> DDPMConfig:
    """Helper to create a new DDPMConfig with updated parameters."""
    # Handle nested noise_schedule updates
    if "beta_schedule" in updates:
        schedule_type = updates.pop("beta_schedule")
        new_schedule = replace(config.noise_schedule, schedule_type=schedule_type)
        return replace(config, noise_schedule=new_schedule)
    if "noise_steps" in updates:
        num_timesteps = updates.pop("noise_steps")
        new_schedule = replace(config.noise_schedule, num_timesteps=num_timesteps)
        return replace(config, noise_schedule=new_schedule)
    return replace(config, **updates)


class TestDDPMModel:
    """Comprehensive DDPM model tests following standardized patterns."""

    @pytest.fixture
    def ddpm_config(self):
        """Basic DDPM configuration for testing."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(64, 128),
            activation="relu",
            in_channels=1,
            out_channels=1,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            num_timesteps=100,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
        return DDPMConfig(
            name="test_ddpm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(28, 28, 1),  # (H, W, C) format - JAX convention
            loss_type="mse",
            clip_denoised=True,
        )

    @pytest.fixture
    def advanced_ddpm_config(self):
        """Advanced DDPM configuration with attention."""
        backbone = UNetBackboneConfig(
            name="test_unet_advanced",
            hidden_dims=(64, 128, 256),
            activation="relu",
            in_channels=3,
            out_channels=3,
            channel_mult=(1, 2, 4),
            num_res_blocks=2,
            attention_resolutions=(16,),
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_cosine",
            num_timesteps=1000,
            schedule_type="cosine",
            beta_start=1e-4,
            beta_end=0.02,
        )
        return DDPMConfig(
            name="test_ddpm_advanced",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(32, 32, 3),  # (H, W, C) format - JAX convention
            loss_type="mse",
            clip_denoised=True,
        )

    def test_ddpm_initialization(self, ddpm_config, base_rngs):
        """Test DDPM model initialization with different configurations."""
        # Test basic initialization
        model = ModelTestRunner.test_model_initialization(DDPMModel, ddpm_config, base_rngs)

        # Debug: print actual vs expected
        print(f"Expected noise_steps: {ddpm_config.noise_schedule.num_timesteps}")
        print(f"Actual noise_steps: {model.noise_steps}")
        print(f"Betas shape: {model.betas.shape}")

        # Check DDPM-specific attributes
        assert hasattr(model, "noise_steps"), "DDPM must have noise_steps"
        assert hasattr(model, "betas"), "DDPM must have betas schedule"
        assert hasattr(model, "alphas"), "DDPM must have alphas"
        assert hasattr(model, "alphas_cumprod"), "DDPM must have cumulative alphas"

        # Check shapes
        num_timesteps = ddpm_config.noise_schedule.num_timesteps
        assert model.betas.shape == (num_timesteps,)
        assert model.alphas.shape == (num_timesteps,)
        assert model.alphas_cumprod.shape == (num_timesteps,)

    def test_ddpm_advanced_initialization(self, advanced_ddpm_config, base_rngs):
        """Test DDPM with advanced configuration."""
        model = ModelTestRunner.test_model_initialization(
            DDPMModel, advanced_ddpm_config, base_rngs
        )

        # Should handle more complex configurations
        assert model.noise_steps == 1000

    def test_noise_schedule_properties(self, ddpm_config, base_rngs):
        """Test noise schedule generation and properties."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        # Test noise schedule
        DiffusionTestUtils.test_noise_schedule(model, ddpm_config.noise_schedule.num_timesteps)

        # Test different schedules
        for schedule in ["linear", "cosine"]:
            # Create new config with updated metadata
            config = update_config(ddpm_config, beta_schedule=schedule)
            model = DDPMModel(config=config, rngs=base_rngs)
            DiffusionTestUtils.test_noise_schedule(model, config.noise_schedule.num_timesteps)

    def test_forward_process(self, ddpm_config, base_rngs, diffusion_test_data):
        """Test forward diffusion process."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        # Test with different timesteps
        timesteps = jnp.array([0, 10, 50, 99])  # Various timesteps

        xt = DiffusionTestUtils.test_forward_process(model, diffusion_test_data, timesteps)

        # Check that noise level increases with timestep
        if len(timesteps) > 1:
            # Should be more noisy at later timesteps
            noise_levels = jnp.var(xt, axis=(1, 2, 3))  # Variance per timestep
            # Generally increasing (allowing some tolerance)
            assert jnp.mean(noise_levels[1:] - noise_levels[:-1]) > -0.01

    def test_reverse_process_single_step(self, ddpm_config, base_rngs, diffusion_test_data):
        """Test single-step reverse process."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        # Add noise to data
        t = jnp.array([50])  # Middle timestep
        noisy_data = model.q_sample(diffusion_test_data, t)

        # Test reverse step
        if hasattr(model, "p_sample_loop"):
            # Test single reverse step
            denoised = DiffusionTestUtils.test_reverse_process(model, noisy_data, t)

            # Should be less noisy than input
            input_noise = jnp.var(noisy_data)
            output_noise = jnp.var(denoised)
            assert output_noise <= input_noise or abs(output_noise - input_noise) < 0.1

    def test_sampling_consistency(self, ddpm_config, base_rngs):
        """Test sampling produces diverse but valid outputs."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        if hasattr(model, "sample"):
            sample_shape = (4, *ddpm_config.input_shape)
            samples = DiffusionTestUtils.test_sampling_consistency(
                model, sample_shape, num_samples=2, tolerance=1e-2
            )

            # Check sample properties
            assert samples.shape == (2, *sample_shape)
            assert jnp.all(jnp.isfinite(samples))

    def test_forward_pass_with_timesteps(
        self, ddpm_config, base_rngs, diffusion_test_data, test_timesteps
    ):
        """Test model forward pass with timestep conditioning."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        # Test with noise prediction
        if hasattr(model, "forward") or hasattr(model, "__call__"):
            # Add noise to create noisy input
            noisy_data = model.q_sample(diffusion_test_data, test_timesteps)

            # Forward pass should predict noise
            predicted_noise = ModelTestRunner.test_forward_pass(
                model,
                noisy_data,
                expected_output_shape=diffusion_test_data.shape,
                timesteps=test_timesteps,
            )

            # Predicted noise should have reasonable statistics
            # Handle both dictionary and array outputs
            if isinstance(predicted_noise, dict):
                # For dictionary outputs, use the predicted_noise field
                noise_array = predicted_noise.get(
                    "predicted_noise", next(iter(predicted_noise.values()))
                )
            else:
                noise_array = predicted_noise

            noise_mean = jnp.mean(noise_array)
            noise_std = jnp.std(noise_array)

            assert abs(noise_mean) < 1.0, f"Predicted noise mean too extreme: {noise_mean}"
            assert 0.1 < noise_std < 5.0, f"Predicted noise std unreasonable: {noise_std}"

    def test_loss_computation(self, ddpm_config, base_rngs, diffusion_test_data, test_timesteps):
        """Test loss computation for DDPM training."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        if hasattr(model, "compute_loss") or hasattr(model, "loss_fn"):
            # Test loss computation
            def simple_loss(pred, target):
                # Handle both dictionary and array outputs
                if isinstance(pred, dict):
                    # For dictionary outputs, use the predicted_noise field
                    pred_array = pred.get("predicted_noise", next(iter(pred.values())))
                else:
                    pred_array = pred
                return jnp.mean((pred_array - target) ** 2)

            # Test gradient flow through loss
            grads = ModelTestRunner.test_gradient_flow(model, diffusion_test_data, simple_loss)

            # Check that gradients exist and are reasonable
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
            assert 1e-6 < grad_norm < 1e3, f"Gradient norm unreasonable: {grad_norm}"

    def test_batch_consistency(self, ddpm_config, base_rngs, diffusion_test_data):
        """Test batch processing consistency."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        # Test single vs batch processing
        single_data = diffusion_test_data[:1]  # First sample
        batch_data = diffusion_test_data  # Full batch

        ModelTestRunner.test_batch_consistency(model, single_data, batch_data)

    def test_parameter_shapes(self, ddpm_config, base_rngs):
        """Test that parameters have reasonable shapes."""
        model = DDPMModel(config=ddpm_config, rngs=base_rngs)

        # Check total parameter count is reasonable
        # Get model parameters using NNX approach
        if hasattr(model, "params"):
            model_params = model.params
        else:
            from flax import nnx

            model_params = nnx.state(model, nnx.Param)

        total_params = sum(p.size for p in jax.tree_util.tree_leaves(model_params))

        # Should have reasonable number of parameters for the configuration
        expected_min_params = 1000  # At least 1K parameters
        expected_max_params = 50_000_000  # At most 50M parameters

        assert expected_min_params <= total_params <= expected_max_params, (
            f"Parameter count {total_params:,} outside expected range"
        )

    @pytest.mark.slow
    def test_full_sampling_process(self, ddpm_config, base_rngs):
        """Test complete sampling from noise to data (slow test)."""
        # Use much smaller timesteps and batch for faster testing on CPU
        config = update_config(
            ddpm_config,
            noise_steps=10,  # Reduced from 20 to 10 for CPU testing
        )
        model = DDPMModel(config=config, rngs=base_rngs)

        if hasattr(model, "sample"):
            with ModelTestRunner.timing_test(
                "Full sampling", max_time=120.0
            ):  # Increased to 120s for slower CPU execution
                sample_shape = (1, *config.input_shape)  # Reduced batch size from 2 to 1
                samples = model.sample(sample_shape)

                # Check output properties
                assert samples.shape == sample_shape
                assert jnp.all(jnp.isfinite(samples))

                # Should be different from pure noise
                noise = jax.random.normal(jax.random.PRNGKey(42), sample_shape)
                assert not jnp.allclose(samples, noise, atol=0.1)

    @pytest.mark.parametrize("schedule", ["linear", "cosine"])
    def test_different_schedules(self, ddpm_config, base_rngs, diffusion_test_data, schedule):
        """Test DDPM with different noise schedules."""
        config = update_config(ddpm_config, beta_schedule=schedule)
        model = DDPMModel(config=config, rngs=base_rngs)

        # Basic functionality should work with any schedule
        timesteps = jnp.array([10, 50])
        noisy_data = model.q_sample(diffusion_test_data, timesteps)

        assert noisy_data.shape == diffusion_test_data.shape
        assert jnp.all(jnp.isfinite(noisy_data))

    def test_model_with_attention(self, advanced_ddpm_config, base_rngs):
        """Test DDPM model with attention layers."""
        model = DDPMModel(config=advanced_ddpm_config, rngs=base_rngs)

        # Create test data matching the advanced config (32, 32, 3)
        import jax.numpy as jnp

        test_data = jnp.ones((8, 32, 32, 3))  # batch_size=8, RGB images

        # Should handle RGB data with attention
        output = ModelTestRunner.test_forward_pass(
            model,
            test_data,
            expected_output_shape=test_data.shape,
        )

        # With attention, might be slower but should work
        # Handle both dictionary and array outputs
        if isinstance(output, dict):
            # For dictionary outputs, check each value
            for key, value in output.items():
                assert jnp.all(jnp.isfinite(value)), f"Output[{key}] must be finite"
        else:
            assert jnp.all(jnp.isfinite(output))

    @pytest.mark.benchmark
    @pytest.mark.parametrize("variant", ["fast", "realistic"])
    def test_ddpm_performance(self, ddpm_config, base_rngs, diffusion_test_data, variant):
        """Benchmark DDPM performance."""
        # Configure based on variant for CPU-optimized testing
        if variant == "fast":
            # Create minimal config for fast testing
            backbone = UNetBackboneConfig(
                name="test_unet_fast",
                hidden_dims=(8, 16),
                activation="relu",
                in_channels=1,
                out_channels=1,
                channel_mult=(1, 2),
                num_res_blocks=1,
            )
            noise_schedule = NoiseScheduleConfig(
                name="test_schedule_fast",
                num_timesteps=10,  # Very minimal for CPU
                schedule_type="linear",
                beta_start=1e-4,
                beta_end=0.02,
            )
            config = DDPMConfig(
                name="test_ddpm_fast",
                backbone=backbone,
                noise_schedule=noise_schedule,
                input_shape=(28, 28, 1),  # (H, W, C) format - JAX convention
                loss_type="mse",
                clip_denoised=True,
            )
            init_time_limit = 1.0
            forward_time_limit = 5.5  # Increased for CPU reality
            batch_size = 1
        else:  # realistic
            # Create moderate config for realistic testing
            backbone = UNetBackboneConfig(
                name="test_unet_realistic",
                hidden_dims=(16, 32),
                activation="relu",
                in_channels=1,
                out_channels=1,
                channel_mult=(1, 2),
                num_res_blocks=1,
            )
            noise_schedule = NoiseScheduleConfig(
                name="test_schedule_realistic",
                num_timesteps=30,  # Moderate for CPU
                schedule_type="linear",
                beta_start=1e-4,
                beta_end=0.02,
            )
            config = DDPMConfig(
                name="test_ddpm_realistic",
                backbone=backbone,
                noise_schedule=noise_schedule,
                input_shape=(28, 28, 1),  # (H, W, C) format - JAX convention
                loss_type="mse",
                clip_denoised=True,
            )
            init_time_limit = 2.0  # Reduced from 3.0s
            forward_time_limit = 6.0  # Increased for CPU reality
            batch_size = 1  # Reduced from 2

        model = DDPMModel(config=config, rngs=base_rngs)

        # Test initialization time
        with ModelTestRunner.timing_test("DDPM initialization", max_time=init_time_limit):
            DDPMModel(config=config, rngs=base_rngs)

        # Test forward pass time
        test_data = diffusion_test_data[:batch_size]
        with ModelTestRunner.timing_test("DDPM forward pass", max_time=forward_time_limit):
            ModelTestRunner.test_forward_pass(model, test_data)


class TestDDPMForwardDiffusion:
    """Tests for DDPM forward diffusion process."""

    @pytest.fixture
    def ddpm_model(self, base_rngs):
        """Create a DDPM model for testing."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(32, 64),
            activation="relu",
            in_channels=1,
            out_channels=1,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            num_timesteps=50,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
        config = DDPMConfig(
            name="test_ddpm_forward",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(16, 16, 1),
            loss_type="mse",
            clip_denoised=True,
        )
        return DDPMModel(config=config, rngs=base_rngs)

    def test_forward_diffusion_returns_noisy_and_noise(self, ddpm_model):
        """Test forward_diffusion returns both noisy sample and noise."""
        x = jnp.ones((2, 16, 16, 1))
        t = jnp.array([10, 20])

        noisy_x, noise = ddpm_model.forward_diffusion(x, t)

        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape
        assert jnp.all(jnp.isfinite(noisy_x))
        assert jnp.all(jnp.isfinite(noise))

    def test_forward_diffusion_noise_is_gaussian(self, ddpm_model):
        """Test that the noise component is approximately Gaussian."""
        x = jnp.zeros((100, 16, 16, 1))
        t = jnp.full((100,), 25)

        _, noise = ddpm_model.forward_diffusion(x, t)

        # Check noise statistics (should be approximately standard normal)
        noise_mean = jnp.mean(noise)
        noise_std = jnp.std(noise)

        assert abs(noise_mean) < 0.5, f"Noise mean {noise_mean} too far from 0"
        assert 0.5 < noise_std < 1.5, f"Noise std {noise_std} too far from 1"

    def test_forward_diffusion_noisier_at_higher_timesteps(self, ddpm_model):
        """Test that samples are noisier at higher timesteps."""
        x = jnp.ones((1, 16, 16, 1))

        t_low = jnp.array([5])
        t_high = jnp.array([45])

        noisy_low, _ = ddpm_model.forward_diffusion(x, t_low)
        noisy_high, _ = ddpm_model.forward_diffusion(x, t_high)

        # Higher timestep should have more variance from original
        diff_low = jnp.mean((noisy_low - x) ** 2)
        diff_high = jnp.mean((noisy_high - x) ** 2)

        # Allow for some randomness, but generally should hold
        # This is a statistical test, so we're lenient
        assert diff_high > diff_low * 0.5 or diff_high > 0.01


class TestDDPMDenoiseStep:
    """Tests for DDPM denoising step."""

    @pytest.fixture
    def ddpm_model(self, base_rngs):
        """Create a DDPM model for testing."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(32, 64),
            activation="relu",
            in_channels=1,
            out_channels=1,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            num_timesteps=50,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
        config = DDPMConfig(
            name="test_ddpm_denoise",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(16, 16, 1),
            loss_type="mse",
            clip_denoised=True,
        )
        return DDPMModel(config=config, rngs=base_rngs)

    def test_denoise_step_output_shape(self, ddpm_model):
        """Test denoise_step returns correct output shape."""
        x_t = jnp.ones((2, 16, 16, 1))
        t = jnp.array([25, 30])
        predicted_noise = jax.random.normal(jax.random.key(0), x_t.shape)

        denoised = ddpm_model.denoise_step(x_t, t, predicted_noise)

        assert denoised.shape == x_t.shape
        assert jnp.all(jnp.isfinite(denoised))

    def test_denoise_step_with_clipping(self, ddpm_model):
        """Test denoise_step with clip_denoised=True.

        Note: clip_denoised clips the predicted x0, but the returned posterior_mean
        is a weighted combination of pred_x0 and x_t, so it's not guaranteed to be
        in [-1, 1]. The test verifies the method executes correctly with clipping enabled.
        """
        x_t = jnp.ones((2, 16, 16, 1))
        t = jnp.array([25, 30])
        predicted_noise = jax.random.normal(jax.random.key(0), x_t.shape)

        denoised = ddpm_model.denoise_step(x_t, t, predicted_noise, clip_denoised=True)

        # Method should execute and return finite values
        assert denoised.shape == x_t.shape
        assert jnp.all(jnp.isfinite(denoised))

    def test_denoise_step_without_clipping(self, ddpm_model):
        """Test denoise_step with clip_denoised=False."""
        x_t = jnp.ones((2, 16, 16, 1))
        t = jnp.array([25, 30])
        predicted_noise = jax.random.normal(jax.random.key(0), x_t.shape)

        denoised = ddpm_model.denoise_step(x_t, t, predicted_noise, clip_denoised=False)

        # Without clipping, values can be outside [-1, 1]
        assert jnp.all(jnp.isfinite(denoised))


class TestDDPMSampling:
    """Tests for DDPM sampling methods."""

    @pytest.fixture
    def ddpm_model(self, base_rngs):
        """Create a DDPM model for testing."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(16, 32),
            activation="relu",
            in_channels=1,
            out_channels=1,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            num_timesteps=20,  # Small for testing
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )
        config = DDPMConfig(
            name="test_ddpm_sample",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(8, 8, 1),
            loss_type="mse",
            clip_denoised=True,
        )
        return DDPMModel(config=config, rngs=base_rngs)

    def test_sample_with_int_n_samples(self, ddpm_model):
        """Test sample with integer n_samples argument."""
        n_samples = 2
        samples = ddpm_model.sample(n_samples, scheduler="ddpm")

        assert samples.shape[0] == n_samples
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_with_tuple_shape(self, ddpm_model):
        """Test sample with tuple shape argument."""
        shape = (2, 8, 8, 1)
        samples = ddpm_model.sample(shape, scheduler="ddpm")

        assert samples.shape == shape
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_with_ddim_scheduler(self, ddpm_model):
        """Test sample with DDIM scheduler."""
        n_samples = 1
        samples = ddpm_model.sample(n_samples, scheduler="ddim", steps=10)

        assert samples.shape[0] == n_samples
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_with_unknown_scheduler_raises_error(self, ddpm_model):
        """Test sample with unknown scheduler raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            ddpm_model.sample(1, scheduler="unknown_scheduler")

    def test_sample_with_default_steps(self, ddpm_model):
        """Test sample with default steps uses noise_steps."""
        samples = ddpm_model.sample(1, scheduler="ddpm", steps=None)

        assert samples.shape[0] == 1
        assert jnp.all(jnp.isfinite(samples))


class TestDDPMDDIMSampling:
    """Tests for DDPM DDIM sampling implementation."""

    @pytest.fixture
    def ddpm_model(self, base_rngs):
        """Create a DDPM model for testing."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(16, 32),
            activation="relu",
            in_channels=1,
            out_channels=1,
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
        config = DDPMConfig(
            name="test_ddpm_ddim",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(8, 8, 1),
            loss_type="mse",
            clip_denoised=True,
        )
        return DDPMModel(config=config, rngs=base_rngs)

    def test_sample_ddim_output_shape(self, ddpm_model):
        """Test _sample_ddim returns correct output shape."""
        n_samples = 2
        shape = (8, 8, 1)
        steps = 10

        samples = ddpm_model._sample_ddim(n_samples, shape, steps)

        assert samples.shape == (n_samples, *shape)
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_ddim_with_eta_zero(self, ddpm_model):
        """Test DDIM with eta=0 (deterministic)."""
        n_samples = 1
        shape = (8, 8, 1)
        steps = 5

        samples = ddpm_model._sample_ddim(n_samples, shape, steps, eta=0.0)

        assert samples.shape == (n_samples, *shape)
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_ddim_with_eta_nonzero(self, ddpm_model):
        """Test DDIM with eta > 0 (stochastic)."""
        n_samples = 1
        shape = (8, 8, 1)
        steps = 5

        samples = ddpm_model._sample_ddim(n_samples, shape, steps, eta=0.5)

        assert samples.shape == (n_samples, *shape)
        assert jnp.all(jnp.isfinite(samples))

    def test_ddim_step_output_shape(self, ddpm_model):
        """Test _ddim_step returns correct output shape."""
        batch_size = 2
        x_t = jax.random.normal(jax.random.key(0), (batch_size, 8, 8, 1))
        t = jnp.full((batch_size,), 15, dtype=jnp.int32)
        t_prev = jnp.full((batch_size,), 10, dtype=jnp.int32)
        predicted_noise = jax.random.normal(jax.random.key(1), x_t.shape)

        x_prev = ddpm_model._ddim_step(x_t, t, t_prev, predicted_noise, eta=0.0)

        assert x_prev.shape == x_t.shape
        assert jnp.all(jnp.isfinite(x_prev))


class TestDDPMGetSampleShape:
    """Tests for DDPM _get_sample_shape method."""

    @pytest.fixture
    def base_rngs(self):
        """Base RNGs fixture."""
        from flax import nnx

        return nnx.Rngs(
            params=jax.random.key(42),
            noise=jax.random.key(123),
            sample=jax.random.key(456),
        )

    def test_get_sample_shape_with_tuple_input_dim(self, base_rngs):
        """Test _get_sample_shape with tuple input_dim."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(16, 32),
            activation="relu",
            in_channels=1,
            out_channels=1,
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
        config = DDPMConfig(
            name="test_ddpm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(16, 16, 3),
            loss_type="mse",
            clip_denoised=True,
        )
        model = DDPMModel(config=config, rngs=base_rngs)

        shape = model._get_sample_shape()

        assert shape == (16, 16, 3)

    def test_get_sample_shape_with_list_input_dim(self, base_rngs):
        """Test _get_sample_shape with list input_dim."""
        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(16, 32),
            activation="relu",
            in_channels=1,
            out_channels=1,
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
        config = DDPMConfig(
            name="test_ddpm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(8, 8, 1),
            loss_type="mse",
            clip_denoised=True,
        )
        model = DDPMModel(config=config, rngs=base_rngs)

        # Manually set input_dim to a list for testing
        model.input_dim = [8, 8, 1]

        shape = model._get_sample_shape()

        assert shape == (8, 8, 1)
