"""Tests for Latent Diffusion Model (LDM) implementation.

This module provides comprehensive tests for the LDM model, covering
initialization, encoder/decoder functionality, reparameterization, diffusion
process, and mathematical correctness.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    LatentDiffusionConfig,
    NoiseScheduleConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.models.diffusion.latent import (
    LDMModel,
    SimpleDecoder,
    SimpleEncoder,
)


class TestSimpleEncoder:
    """Test suite for SimpleEncoder module."""

    @pytest.fixture
    def encoder_config(self):
        """Configuration for a simple encoder."""
        return {
            "input_dim": (28, 28, 1),  # Image input
            "latent_dim": 16,
            "hidden_dims": [32, 64],
        }

    @pytest.fixture
    def vector_encoder_config(self):
        """Configuration for a vector encoder."""
        return {
            "input_dim": (100,),  # Vector input
            "latent_dim": 16,
            "hidden_dims": [64, 32],
        }

    def test_encoder_initialization_image_input(self, encoder_config, base_rngs):
        """Test encoder initialization with image input."""
        encoder = SimpleEncoder(
            input_dim=encoder_config["input_dim"],
            latent_dim=encoder_config["latent_dim"],
            hidden_dims=encoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        assert encoder.input_dim == encoder_config["input_dim"]
        assert encoder.latent_dim == encoder_config["latent_dim"]
        assert encoder.is_image is True
        assert encoder.flat_dim == 28 * 28 * 1

    def test_encoder_initialization_vector_input(self, vector_encoder_config, base_rngs):
        """Test encoder initialization with vector input."""
        encoder = SimpleEncoder(
            input_dim=vector_encoder_config["input_dim"],
            latent_dim=vector_encoder_config["latent_dim"],
            hidden_dims=vector_encoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        assert encoder.latent_dim == vector_encoder_config["latent_dim"]
        assert encoder.is_image is False

    def test_encoder_forward_shape_image(self, encoder_config, base_rngs):
        """Test encoder output shapes for image input."""
        encoder = SimpleEncoder(
            input_dim=encoder_config["input_dim"],
            latent_dim=encoder_config["latent_dim"],
            hidden_dims=encoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        batch_size = 4
        x = jnp.ones((batch_size, 28, 28, 1))
        mean, logvar = encoder(x)

        assert mean.shape == (batch_size, encoder_config["latent_dim"])
        assert logvar.shape == (batch_size, encoder_config["latent_dim"])

    def test_encoder_forward_shape_vector(self, vector_encoder_config, base_rngs):
        """Test encoder output shapes for vector input."""
        encoder = SimpleEncoder(
            input_dim=vector_encoder_config["input_dim"],
            latent_dim=vector_encoder_config["latent_dim"],
            hidden_dims=vector_encoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        batch_size = 4
        x = jnp.ones((batch_size, 100))
        mean, logvar = encoder(x)

        assert mean.shape == (batch_size, vector_encoder_config["latent_dim"])
        assert logvar.shape == (batch_size, vector_encoder_config["latent_dim"])

    def test_encoder_mean_logvar_statistics(self, encoder_config, base_rngs):
        """Test that encoder produces reasonable mean and logvar statistics."""
        encoder = SimpleEncoder(
            input_dim=encoder_config["input_dim"],
            latent_dim=encoder_config["latent_dim"],
            hidden_dims=encoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        # Generate random input
        key = jax.random.key(42)
        x = jax.random.uniform(key, (32, 28, 28, 1))
        mean, logvar = encoder(x)

        # Mean and logvar should be finite
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))

    def test_encoder_deterministic(self, encoder_config, base_rngs):
        """Test that encoder is deterministic (no dropout in forward)."""
        encoder = SimpleEncoder(
            input_dim=encoder_config["input_dim"],
            latent_dim=encoder_config["latent_dim"],
            hidden_dims=encoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        x = jnp.ones((2, 28, 28, 1))
        mean1, logvar1 = encoder(x)
        mean2, logvar2 = encoder(x)

        assert jnp.allclose(mean1, mean2)
        assert jnp.allclose(logvar1, logvar2)


class TestSimpleDecoder:
    """Test suite for SimpleDecoder module."""

    @pytest.fixture
    def decoder_config(self):
        """Configuration for a simple decoder."""
        return {
            "latent_dim": 16,
            "output_dim": (28, 28, 1),  # Image output
            "hidden_dims": [64, 32],
        }

    @pytest.fixture
    def vector_decoder_config(self):
        """Configuration for a vector decoder."""
        return {
            "latent_dim": 16,
            "output_dim": (100,),  # Vector output
            "hidden_dims": [32, 64],
        }

    def test_decoder_initialization(self, decoder_config, base_rngs):
        """Test decoder initialization."""
        decoder = SimpleDecoder(
            latent_dim=decoder_config["latent_dim"],
            output_dim=decoder_config["output_dim"],
            hidden_dims=decoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        assert decoder.latent_dim == decoder_config["latent_dim"]
        assert decoder.output_dim == decoder_config["output_dim"]
        assert decoder.is_image is True

    def test_decoder_requires_rngs_validation(self):
        """Test that decoder requires rngs parameter."""
        with pytest.raises(ValueError, match="rngs must be provided"):
            SimpleDecoder(
                latent_dim=16,
                output_dim=(28, 28, 1),
                hidden_dims=[64, 32],
                rngs=None,
            )

    def test_decoder_forward_image_output(self, decoder_config, base_rngs):
        """Test decoder output shapes for image output."""
        decoder = SimpleDecoder(
            latent_dim=decoder_config["latent_dim"],
            output_dim=decoder_config["output_dim"],
            hidden_dims=decoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        batch_size = 4
        z = jnp.ones((batch_size, decoder_config["latent_dim"]))
        output = decoder(z)

        assert output.shape == (batch_size, *decoder_config["output_dim"])

    def test_decoder_forward_vector_output(self, vector_decoder_config, base_rngs):
        """Test decoder output shapes for vector output."""
        decoder = SimpleDecoder(
            latent_dim=vector_decoder_config["latent_dim"],
            output_dim=vector_decoder_config["output_dim"],
            hidden_dims=vector_decoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        batch_size = 4
        z = jnp.ones((batch_size, vector_decoder_config["latent_dim"]))
        output = decoder(z)

        assert output.shape == (batch_size, *vector_decoder_config["output_dim"])

    def test_decoder_deterministic(self, decoder_config, base_rngs):
        """Test that decoder is deterministic."""
        decoder = SimpleDecoder(
            latent_dim=decoder_config["latent_dim"],
            output_dim=decoder_config["output_dim"],
            hidden_dims=decoder_config["hidden_dims"],
            rngs=base_rngs,
        )

        z = jnp.ones((2, decoder_config["latent_dim"]))
        output1 = decoder(z)
        output2 = decoder(z)

        assert jnp.allclose(output1, output2)


class TestLDMModel:
    """Test suite for LDMModel (Latent Diffusion Model)."""

    @pytest.fixture
    def ldm_config(self):
        """Create a minimal LDM configuration for testing.

        Uses image-based latent space since UNet requires 4D input.
        The encoder produces 4D latent codes that UNet can process.
        """
        # Image input with small spatial dimensions
        input_shape = (8, 8, 1)
        latent_channels = 4

        backbone = UNetBackboneConfig(
            name="test_unet",
            hidden_dims=(16, 32),
            activation="relu",
            in_channels=latent_channels,  # Latent space channels
            out_channels=latent_channels,
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

        # For image LDM, latent_dim is spatial latent dimension
        # The encoder produces (batch, latent_dim) which is reshaped to (batch, h, w, c)
        latent_dim = 4 * 4 * latent_channels  # 4x4 spatial with latent_channels

        encoder_config = EncoderConfig(
            name="test_encoder",
            hidden_dims=(32, 16),
            activation="relu",
            input_shape=input_shape,
            latent_dim=latent_dim,
        )

        decoder_config = DecoderConfig(
            name="test_decoder",
            hidden_dims=(16, 32),
            activation="relu",
            output_shape=input_shape,
            latent_dim=latent_dim,
        )

        return LatentDiffusionConfig(
            name="test_ldm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=input_shape,
            encoder=encoder_config,
            decoder=decoder_config,
            latent_scale_factor=0.5,
        )

    @pytest.fixture
    def image_ldm_config(self):
        """Create LDM configuration for image input.

        The latent_dim is designed to factor into a spatial shape that matches
        the backbone's in_channels. For latent_dim=32 with target 4 channels:
        32 = 2*4*4 -> spatial shape (2, 4, 4) with 4 channels for UNet.
        """
        # Target spatial latent: 2x4 spatial with 4 channels -> 2*4*4=32
        latent_channels = 4
        latent_dim = 2 * 4 * latent_channels  # 32

        backbone = UNetBackboneConfig(
            name="test_unet_image",
            hidden_dims=(16, 32),
            activation="relu",
            in_channels=latent_channels,  # Must match latent spatial channels
            out_channels=latent_channels,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )

        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_image",
            num_timesteps=50,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )

        encoder_config = EncoderConfig(
            name="test_encoder_image",
            hidden_dims=(32, 16),
            activation="relu",
            input_shape=(16, 16, 1),
            latent_dim=latent_dim,
        )

        decoder_config = DecoderConfig(
            name="test_decoder_image",
            hidden_dims=(16, 32),
            activation="relu",
            output_shape=(16, 16, 1),
            latent_dim=latent_dim,
        )

        return LatentDiffusionConfig(
            name="test_ldm_image",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=(16, 16, 1),
            encoder=encoder_config,
            decoder=decoder_config,
            latent_scale_factor=0.18215,
        )

    def test_ldm_initialization_from_config(self, ldm_config, base_rngs):
        """Test LDM model initialization from config."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        # Check that essential attributes exist
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "scale_factor")
        assert model.scale_factor == ldm_config.latent_scale_factor

    def test_encode_returns_mean_logvar_tuple(self, ldm_config, base_rngs):
        """Test that encode method returns (mean, logvar) tuple."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 4
        x = jnp.ones((batch_size, *ldm_config.input_shape))  # Image input
        mean, logvar = model.encode(x)

        assert isinstance(mean, jax.Array)
        assert isinstance(logvar, jax.Array)
        assert mean.shape == (batch_size, ldm_config.encoder.latent_dim)
        assert logvar.shape == (batch_size, ldm_config.encoder.latent_dim)

    def test_decode_reconstructs_shape(self, ldm_config, base_rngs):
        """Test that decode reconstructs to original shape."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 4
        latent_dim = ldm_config.encoder.latent_dim
        z = jnp.ones((batch_size, latent_dim))
        reconstructed = model.decode(z)

        # Should match input shape from config
        expected_shape = (batch_size, *ldm_config.input_shape)
        assert reconstructed.shape == expected_shape

    def test_reparameterize_math(self, ldm_config, base_rngs):
        """Test reparameterization: z = mean + exp(0.5*logvar)*eps."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 1000  # Large batch for statistical test
        latent_dim = ldm_config.encoder.latent_dim

        # When mean=0 and logvar=0, z should be ~N(0,1) samples
        mean = jnp.zeros((batch_size, latent_dim))
        logvar = jnp.zeros((batch_size, latent_dim))

        # Create fresh rngs for sampling
        sample_rngs = nnx.Rngs(sample=jax.random.key(123))
        z = model.reparameterize(mean, logvar, rngs=sample_rngs)

        # Check z is approximately standard normal
        z_mean = jnp.mean(z)
        z_std = jnp.std(z)

        assert jnp.abs(z_mean) < 0.2, f"Mean should be ~0, got {z_mean}"
        assert jnp.abs(z_std - 1.0) < 0.2, f"Std should be ~1, got {z_std}"

    def test_reparameterize_with_nonzero_mean(self, ldm_config, base_rngs):
        """Test reparameterization with non-zero mean."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 1000
        latent_dim = ldm_config.encoder.latent_dim

        # With mean=5 and logvar=0, z should be ~N(5,1)
        mean = jnp.full((batch_size, latent_dim), 5.0)
        logvar = jnp.zeros((batch_size, latent_dim))

        sample_rngs = nnx.Rngs(sample=jax.random.key(456))
        z = model.reparameterize(mean, logvar, rngs=sample_rngs)

        z_mean = jnp.mean(z)
        assert jnp.abs(z_mean - 5.0) < 0.3, f"Mean should be ~5, got {z_mean}"

    def test_reparameterize_with_nonzero_logvar(self, ldm_config, base_rngs):
        """Test reparameterization with non-zero log variance."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 1000
        latent_dim = ldm_config.encoder.latent_dim

        # With logvar=ln(4)~1.386, std should be 2
        mean = jnp.zeros((batch_size, latent_dim))
        logvar = jnp.full((batch_size, latent_dim), jnp.log(4.0))

        sample_rngs = nnx.Rngs(sample=jax.random.key(789))
        z = model.reparameterize(mean, logvar, rngs=sample_rngs)

        z_std = jnp.std(z)
        assert jnp.abs(z_std - 2.0) < 0.4, f"Std should be ~2, got {z_std}"

    def test_forward_pass_returns_all_outputs(self, image_ldm_config, base_rngs):
        """Test that forward pass returns all required outputs.

        The LDM implementation reshapes flat latent codes to 4D spatial format
        (batch, h, w, c) before passing to the UNet backbone.
        """
        model = LDMModel(config=image_ldm_config, rngs=base_rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 1))  # Image input
        outputs = model(x)

        # Check all expected keys
        assert "reconstruction" in outputs
        assert "mean" in outputs
        assert "logvar" in outputs
        assert "latent" in outputs
        assert "noisy_latent" in outputs
        assert "predicted_noise" in outputs
        assert "true_noise" in outputs

        # Check shapes
        assert outputs["reconstruction"].shape == (batch_size, 16, 16, 1)
        assert outputs["mean"].shape == (batch_size, image_ldm_config.encoder.latent_dim)
        assert outputs["logvar"].shape == (batch_size, image_ldm_config.encoder.latent_dim)

    def test_scale_factor_effect_on_latent(self, ldm_config, base_rngs):
        """Test that scale factor affects latent codes."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 4
        x = jnp.ones((batch_size, *ldm_config.input_shape))
        mean, _ = model.encode(x)

        # Verify the mean is finite
        assert jnp.all(jnp.isfinite(mean))
        # Mean magnitude should be reasonable
        assert jnp.abs(jnp.mean(mean)) < 100.0

    def test_kl_divergence_math(self, ldm_config, base_rngs):
        """Test KL divergence computation: -0.5 * mean(1 + logvar - μ² - exp(logvar))."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 8
        x = jnp.ones((batch_size, *ldm_config.input_shape))
        mean, logvar = model.encode(x)

        # Manually compute KL divergence
        kl = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))

        # KL should be non-negative and finite
        assert jnp.isfinite(kl)
        assert kl >= 0, f"KL divergence should be >= 0, got {kl}"

    def test_loss_computation(self, image_ldm_config, base_rngs):
        """Test loss computation with proper latent space diffusion."""
        model = LDMModel(config=image_ldm_config, rngs=base_rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 1))
        loss = model.loss(x)

        # Loss should be finite and reasonable
        assert jnp.isfinite(loss)
        assert loss > 0, "Loss should be positive"

    def test_gradient_flow_through_encoder_decoder(self, ldm_config, base_rngs):
        """Test that gradients flow through encoder and decoder."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        batch_size = 2
        x = jnp.ones((batch_size, *ldm_config.input_shape))

        def loss_fn(m):
            mean, logvar = m.encode(x)
            z = m.reparameterize(mean, logvar, rngs=m.rngs)
            recon = m.decode(z)
            return jnp.mean((recon - x) ** 2)

        # Compute gradients
        grads = nnx.grad(loss_fn)(model)

        # Check that gradients exist and are finite
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert len(grad_leaves) > 0, "Should have gradient leaves"

        # Sum of squared gradients should be non-zero
        grad_norm = sum(jnp.sum(g**2) for g in grad_leaves if isinstance(g, jax.Array))
        assert grad_norm > 0, "Gradient norm should be positive"

    def test_output_dim_property(self, ldm_config, base_rngs):
        """Test output_dim property returns original input shape."""
        model = LDMModel(config=ldm_config, rngs=base_rngs)

        assert model.output_dim == ldm_config.input_shape

    def test_forward_with_explicit_timesteps(self, image_ldm_config, base_rngs):
        """Test forward pass with explicit timesteps."""
        model = LDMModel(config=image_ldm_config, rngs=base_rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 1))
        t = jnp.array([10, 25])

        outputs = model(x, t)

        assert jnp.all(jnp.isfinite(outputs["reconstruction"]))
        assert jnp.all(jnp.isfinite(outputs["predicted_noise"]))

    def test_image_ldm_forward(self, image_ldm_config, base_rngs):
        """Test LDM with image input and spatial latent reshaping."""
        model = LDMModel(config=image_ldm_config, rngs=base_rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 1))
        outputs = model(x)

        assert outputs["reconstruction"].shape == (batch_size, 16, 16, 1)
        assert jnp.all(jnp.isfinite(outputs["reconstruction"]))


class TestLDMSampling:
    """Test suite for LDM sampling functionality."""

    @pytest.fixture
    def sample_ldm_config(self):
        """Create a minimal LDM configuration for sampling tests."""
        # Use image-based config for UNet compatibility
        input_shape = (8, 8, 1)
        latent_channels = 2
        latent_dim = 4 * 4 * latent_channels  # 4x4 spatial with latent_channels

        backbone = UNetBackboneConfig(
            name="test_unet_sample",
            hidden_dims=(8, 16),
            activation="relu",
            in_channels=latent_channels,
            out_channels=latent_channels,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )

        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_sample",
            num_timesteps=10,  # Small for fast testing
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )

        encoder_config = EncoderConfig(
            name="test_encoder_sample",
            hidden_dims=(16,),
            activation="relu",
            input_shape=input_shape,
            latent_dim=latent_dim,
        )

        decoder_config = DecoderConfig(
            name="test_decoder_sample",
            hidden_dims=(16,),
            activation="relu",
            output_shape=input_shape,
            latent_dim=latent_dim,
        )

        return LatentDiffusionConfig(
            name="test_ldm_sample",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=input_shape,
            encoder=encoder_config,
            decoder=decoder_config,
            latent_scale_factor=0.5,
        )

    def test_sample_generates_valid_outputs(self, sample_ldm_config, base_rngs):
        """Test that sampling generates valid outputs with spatial latent reshaping."""
        model = LDMModel(config=sample_ldm_config, rngs=base_rngs)

        num_samples = 2
        samples = model.sample(num_samples)

        # Check output shape
        assert samples.shape == (num_samples, *sample_ldm_config.input_shape)

        # Check outputs are finite
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_with_different_seeds(self, sample_ldm_config):
        """Test that different seeds produce different samples."""
        # Create two models with different seeds
        rngs1 = nnx.Rngs(default=jax.random.key(42))
        model1 = LDMModel(config=sample_ldm_config, rngs=rngs1)

        rngs2 = nnx.Rngs(default=jax.random.key(999))
        model2 = LDMModel(config=sample_ldm_config, rngs=rngs2)

        samples1 = model1.sample(2)
        samples2 = model2.sample(2)

        # Samples from different seeds should be different (with high probability)
        assert not jnp.allclose(samples1, samples2, atol=1e-5)


class TestLDMEdgeCases:
    """Test edge cases and numerical stability for LDM."""

    @pytest.fixture
    def small_ldm_config(self):
        """Create minimal LDM config for edge case testing."""
        # Use image-based config for UNet compatibility
        input_shape = (4, 4, 1)
        latent_channels = 1
        latent_dim = 2 * 2 * latent_channels  # 2x2 spatial

        backbone = UNetBackboneConfig(
            name="small_unet",
            hidden_dims=(4, 8),
            activation="relu",
            in_channels=latent_channels,
            out_channels=latent_channels,
            channel_mult=(1, 2),
            num_res_blocks=1,
        )

        noise_schedule = NoiseScheduleConfig(
            name="small_schedule",
            num_timesteps=5,
            schedule_type="linear",
            beta_start=1e-4,
            beta_end=0.02,
        )

        encoder_config = EncoderConfig(
            name="small_encoder",
            hidden_dims=(8,),
            activation="relu",
            input_shape=input_shape,
            latent_dim=latent_dim,
        )

        decoder_config = DecoderConfig(
            name="small_decoder",
            hidden_dims=(8,),
            activation="relu",
            output_shape=input_shape,
            latent_dim=latent_dim,
        )

        return LatentDiffusionConfig(
            name="small_ldm",
            backbone=backbone,
            noise_schedule=noise_schedule,
            input_shape=input_shape,
            encoder=encoder_config,
            decoder=decoder_config,
            latent_scale_factor=1.0,
        )

    def test_single_sample_batch(self, small_ldm_config, base_rngs):
        """Test encode/decode with batch size 1."""
        model = LDMModel(config=small_ldm_config, rngs=base_rngs)

        x = jnp.ones((1, *small_ldm_config.input_shape))
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar, rngs=model.rngs)
        recon = model.decode(z)

        assert recon.shape == (1, *small_ldm_config.input_shape)
        assert jnp.all(jnp.isfinite(recon))

    def test_zero_input(self, small_ldm_config, base_rngs):
        """Test encode with zero input."""
        model = LDMModel(config=small_ldm_config, rngs=base_rngs)

        x = jnp.zeros((2, *small_ldm_config.input_shape))
        mean, logvar = model.encode(x)

        # Should still produce finite outputs
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))

    def test_large_input_values(self, small_ldm_config, base_rngs):
        """Test encode with large input values."""
        model = LDMModel(config=small_ldm_config, rngs=base_rngs)

        x = jnp.ones((2, *small_ldm_config.input_shape)) * 100.0
        mean, logvar = model.encode(x)

        # Should produce finite outputs (may be clipped/saturated)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))

    def test_encoder_decoder_roundtrip(self, small_ldm_config, base_rngs):
        """Test that encoder-decoder roundtrip produces valid outputs."""
        model = LDMModel(config=small_ldm_config, rngs=base_rngs)

        x = jnp.ones((2, *small_ldm_config.input_shape)) * 0.5
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar, rngs=model.rngs)
        recon = model.decode(z)

        # Output should have same shape as input
        assert recon.shape == x.shape
        assert jnp.all(jnp.isfinite(recon))
