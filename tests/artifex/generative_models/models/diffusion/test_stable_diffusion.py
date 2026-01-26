"""Tests for Stable Diffusion implementation.

This module provides comprehensive tests for the StableDiffusionModel,
which uses proper components: UNet2DCondition, CLIPTextEncoder, SpatialEncoder/Decoder.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DecoderConfig,
    EncoderConfig,
    NoiseScheduleConfig,
    StableDiffusionConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.models.diffusion.stable_diffusion import StableDiffusionModel


def get_latent_shape(input_shape: tuple[int, ...], latent_channels: int) -> tuple[int, ...]:
    """Calculate latent shape from input shape.

    With 3 encoder layers (hidden_dims of length 3), we get 8x spatial downsampling.
    """
    h, w, _ = input_shape
    return (h // 8, w // 8, latent_channels)


def create_sd_config(
    input_shape: tuple[int, ...] = (32, 32, 3),
    latent_channels: int = 4,
    hidden_dims: tuple[int, ...] = (32, 64),
    noise_steps: int = 50,
    text_embedding_dim: int = 64,
    text_max_length: int = 16,
    vocab_size: int = 500,
    guidance_scale: float = 7.5,
    use_guidance: bool = True,
) -> StableDiffusionConfig:
    """Helper to create StableDiffusionConfig with all nested configs.

    Uses small dimensions for fast testing.
    """
    backbone = UNetBackboneConfig(
        name="test_unet",
        hidden_dims=hidden_dims,
        in_channels=input_shape[-1],
        out_channels=input_shape[-1],
        activation="gelu",
    )
    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        num_timesteps=noise_steps,
    )
    # Encoder/decoder need at least 3 hidden dims for 8x spatial downsampling
    encoder = EncoderConfig(
        name="test_encoder",
        hidden_dims=(32, 64, 128),  # 3 layers for 8x downsampling
        input_shape=input_shape,
        latent_dim=latent_channels,
        activation="gelu",
    )
    decoder = DecoderConfig(
        name="test_decoder",
        hidden_dims=(128, 64, 32),  # Reverse order for upsampling
        latent_dim=latent_channels,
        output_shape=input_shape,
        activation="gelu",
    )
    return StableDiffusionConfig(
        name="test_stable_diffusion",
        backbone=backbone,
        noise_schedule=noise_schedule,
        encoder=encoder,
        decoder=decoder,
        input_shape=input_shape,
        text_embedding_dim=text_embedding_dim,
        text_max_length=text_max_length,
        vocab_size=vocab_size,
        guidance_scale=guidance_scale,
        use_guidance=use_guidance,
    )


class TestStableDiffusionModel:
    """Tests for Stable Diffusion model."""

    @pytest.fixture
    def sd_config(self):
        """Configuration for Stable Diffusion model."""
        return create_sd_config()

    def test_sd_initialization(self, sd_config, base_rngs):
        """Test StableDiffusionModel initialization."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Check attributes from config
        assert model.text_embedding_dim == 64
        assert model.text_max_length == 16
        assert model.vocab_size == 500

        # Check components exist
        assert hasattr(model, "text_encoder")
        assert hasattr(model, "unet")
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")

    def test_sd_initialization_with_guidance(self, sd_config, base_rngs):
        """Test StableDiffusionModel initialization with guidance."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Should have guidance enabled
        assert model.use_guidance is True
        assert model.guidance_scale == 7.5
        assert hasattr(model, "unconditional_token")

    def test_sd_initialization_without_guidance(self, base_rngs):
        """Test StableDiffusionModel initialization without guidance."""
        config = create_sd_config(use_guidance=False)
        model = StableDiffusionModel(config, rngs=base_rngs)

        assert model.use_guidance is False

    def test_encode_text(self, sd_config, base_rngs):
        """Test text encoding functionality."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create test text tokens
        batch_size = 2
        seq_len = 16
        key = jax.random.key(45)
        text_tokens = jax.random.randint(key, (batch_size, seq_len), 0, model.vocab_size)

        # Encode text
        embeddings = model.encode_text(text_tokens)

        # Check output shape: [batch, seq_len, text_embedding_dim]
        assert embeddings.shape == (batch_size, seq_len, model.text_embedding_dim)
        assert jnp.all(jnp.isfinite(embeddings))

    def test_encode_text_with_mask(self, sd_config, base_rngs):
        """Test text encoding with attention mask."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create test text tokens and mask
        batch_size = 2
        seq_len = 16
        key = jax.random.key(46)
        text_tokens = jax.random.randint(key, (batch_size, seq_len), 0, model.vocab_size)
        attention_mask = jnp.ones((batch_size, seq_len))
        attention_mask = attention_mask.at[:, 8:].set(0)  # Mask second half

        # Encode text with mask
        embeddings = model.encode_text(text_tokens, attention_mask)

        # Output should still be valid (masking affects attention, not output shape)
        assert embeddings.shape == (batch_size, seq_len, model.text_embedding_dim)
        assert jnp.all(jnp.isfinite(embeddings))

    def test_sd_forward_pass(self, sd_config, base_rngs):
        """Test forward pass through Stable Diffusion model in latent space."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create latent codes (UNet operates in latent space)
        batch_size = 2
        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((batch_size, *latent_shape))
        timesteps = jnp.array([25, 25])

        # Create text embeddings
        key = jax.random.key(47)
        text_tokens = jax.random.randint(key, (batch_size, 16), 0, model.vocab_size)
        text_embeddings = model.encode_text(text_tokens)

        # Forward pass with latent codes
        output = model(z, timesteps, text_embeddings=text_embeddings)

        # Check output shape matches latent shape
        assert isinstance(output, dict)
        assert "predicted_noise" in output
        assert output["predicted_noise"].shape == z.shape
        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_sd_forward_pass_without_text(self, sd_config, base_rngs):
        """Test forward pass without explicit text embeddings (uses unconditional)."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create latent codes (UNet operates in latent space)
        batch_size = 2
        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((batch_size, *latent_shape))
        timesteps = jnp.array([25, 25])

        # Forward pass without text (uses unconditional embedding)
        output = model(z, timesteps)

        # Should still work with unconditional embeddings
        assert isinstance(output, dict)
        assert "predicted_noise" in output
        assert output["predicted_noise"].shape == z.shape
        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_compute_text_similarity(self, sd_config, base_rngs):
        """Test text similarity computation."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create two sets of text tokens
        key1, key2 = jax.random.split(jax.random.key(48))
        text_tokens_1 = jax.random.randint(key1, (2, 16), 0, model.vocab_size)
        text_tokens_2 = jax.random.randint(key2, (2, 16), 0, model.vocab_size)

        # Compute similarity
        similarity = model.compute_text_similarity(text_tokens_1, text_tokens_2)

        # Check output
        assert similarity.shape == (2,)
        assert jnp.all(jnp.isfinite(similarity))
        # Cosine similarity should be in [-1, 1]
        assert jnp.all(similarity >= -1.0) and jnp.all(similarity <= 1.0)

    def test_text_similarity_identical_inputs(self, sd_config, base_rngs):
        """Test that identical text inputs have high similarity."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create text tokens
        key = jax.random.key(49)
        text_tokens = jax.random.randint(key, (2, 16), 0, model.vocab_size)

        # Compute similarity with itself
        similarity = model.compute_text_similarity(text_tokens, text_tokens)

        # Should be very close to 1.0 for identical inputs
        assert jnp.allclose(similarity, 1.0, atol=1e-5)


class TestStableDiffusionTraining:
    """Tests for Stable Diffusion training functionality."""

    @pytest.fixture
    def sd_config(self):
        """Configuration for Stable Diffusion model."""
        return create_sd_config()

    def test_train_eval_mode_switching(self, sd_config, base_rngs):
        """Test that train/eval mode switching works correctly."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create latent codes (UNet operates in latent space)
        batch_size = 2
        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((batch_size, *latent_shape))
        timesteps = jnp.array([25, 25])
        key = jax.random.key(50)
        text_tokens = jax.random.randint(key, (batch_size, 16), 0, model.vocab_size)
        text_embeddings = model.encode_text(text_tokens)

        # Test in training mode
        model.train()
        output_train = model(z, timesteps, text_embeddings=text_embeddings)
        assert isinstance(output_train, dict)
        assert jnp.all(jnp.isfinite(output_train["predicted_noise"]))

        # Test in eval mode
        model.eval()
        output_eval = model(z, timesteps, text_embeddings=text_embeddings)
        assert isinstance(output_eval, dict)
        assert jnp.all(jnp.isfinite(output_eval["predicted_noise"]))

    def test_batch_consistency(self, sd_config, base_rngs):
        """Test batch processing consistency."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)
        model.eval()

        # Create latent codes with different batch sizes
        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z_single = jnp.ones((1, *latent_shape))
        z_batch = jnp.ones((4, *latent_shape))
        t_single = jnp.array([25])
        t_batch = jnp.array([25, 25, 25, 25])

        # Process single and batch
        output_single = model(z_single, t_single)
        output_batch = model(z_batch, t_batch)

        # Both should work and have correct latent shapes
        assert output_single["predicted_noise"].shape == (1, *latent_shape)
        assert output_batch["predicted_noise"].shape == (4, *latent_shape)


class TestStableDiffusionIntegration:
    """Integration tests for Stable Diffusion."""

    @pytest.fixture
    def sd_config(self):
        """Minimal configuration for integration tests."""
        return create_sd_config(
            input_shape=(16, 16, 3),
            hidden_dims=(16, 32),
            latent_channels=4,
            noise_steps=10,
            text_embedding_dim=32,
            text_max_length=8,
            vocab_size=100,
            guidance_scale=5.0,
        )

    def test_end_to_end_forward(self, sd_config, base_rngs):
        """Test end-to-end forward pass with text conditioning in latent space."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)
        model.eval()

        # Create text prompt
        key = jax.random.key(52)
        text_tokens = jax.random.randint(key, (2, 8), 0, model.vocab_size)

        # Encode text
        text_embeddings = model.encode_text(text_tokens)

        # Create latent codes (UNet operates in latent space)
        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((2, *latent_shape))
        timesteps = jnp.array([5, 5])

        # Forward pass with text conditioning
        output = model(z, timesteps, text_embeddings=text_embeddings)

        # Should produce valid output
        assert isinstance(output, dict)
        assert "predicted_noise" in output
        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_parameter_count_reasonable(self, sd_config, base_rngs):
        """Test that total parameter count is reasonable."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Get all parameters
        params = nnx.state(model, nnx.Param)
        total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

        # Should have reasonable parameter count
        min_params = 10_000  # At least 10K
        max_params = 100_000_000  # At most 100M for this test config

        assert min_params <= total_params <= max_params, (
            f"Parameter count {total_params:,} outside expected range"
        )

    def test_encoder_decoder_spatial(self, sd_config, base_rngs):
        """Test that encoder/decoder preserve spatial structure."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create test image
        x = jax.random.normal(jax.random.key(53), (2, 16, 16, 3))

        # Encode to latent
        mean, log_var = model.encode_image(x)

        # Latent should have reduced spatial dimensions (8x downsampling)
        # For 16x16 input with 3 encoder layers (2^3=8), we get 2x2 latent
        assert mean.shape[0] == 2  # batch
        assert mean.shape[1] == 2  # height/8
        assert mean.shape[2] == 2  # width/8
        assert mean.shape[3] == sd_config.encoder.latent_dim  # latent channels

        # Decode back
        decoded = model.decode_latent(mean)

        # Decoded should be back to original spatial dimensions
        assert decoded.shape == (2, 16, 16, 3)
        assert jnp.all(jnp.isfinite(decoded))


class TestStableDiffusionNoiseSchedule:
    """Tests for noise schedule behavior in Stable Diffusion."""

    @pytest.fixture
    def sd_config(self):
        """Configuration for noise schedule tests."""
        return create_sd_config(noise_steps=100)

    def test_noise_schedule_initialization(self, sd_config, base_rngs):
        """Test that noise schedule is properly initialized."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Check schedule arrays exist and have correct length
        assert model.alphas_cumprod.shape == (100,)
        assert model.sqrt_alphas_cumprod.shape == (100,)
        assert model.sqrt_one_minus_alphas_cumprod.shape == (100,)

    def test_noise_schedule_values_valid(self, sd_config, base_rngs):
        """Test that noise schedule values are in valid ranges."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Alphas_cumprod should decrease monotonically from ~1 to ~0
        assert jnp.all(model.alphas_cumprod > 0)
        assert jnp.all(model.alphas_cumprod <= 1)
        assert model.alphas_cumprod[0] > model.alphas_cumprod[-1]

        # Sqrt values should be positive
        assert jnp.all(model.sqrt_alphas_cumprod > 0)
        assert jnp.all(model.sqrt_one_minus_alphas_cumprod >= 0)

    def test_noise_schedule_monotonicity(self, sd_config, base_rngs):
        """Test that alphas_cumprod is monotonically decreasing."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Each element should be less than the previous
        for i in range(1, len(model.alphas_cumprod)):
            assert model.alphas_cumprod[i] < model.alphas_cumprod[i - 1]


class TestStableDiffusionForwardDiffusion:
    """Tests for forward diffusion process."""

    @pytest.fixture
    def sd_config(self):
        """Configuration for forward diffusion tests."""
        return create_sd_config(noise_steps=50)

    def test_forward_diffusion_shape(self, sd_config, base_rngs):
        """Test forward diffusion preserves input shape."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        x = jnp.ones((2, 32, 32, 3))
        t = jnp.array([10, 20])

        noisy, noise = model.forward_diffusion(x, t)

        assert noisy.shape == x.shape
        assert noise.shape == x.shape

    def test_forward_diffusion_finite(self, sd_config, base_rngs):
        """Test forward diffusion produces finite values."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        x = jax.random.normal(jax.random.key(60), (2, 32, 32, 3))
        t = jnp.array([25, 25])

        noisy, noise = model.forward_diffusion(x, t)

        assert jnp.all(jnp.isfinite(noisy))
        assert jnp.all(jnp.isfinite(noise))

    def test_forward_diffusion_with_pregenerated_noise(self, sd_config, base_rngs):
        """Test forward diffusion with pre-generated noise."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        x = jnp.ones((2, 32, 32, 3))
        t = jnp.array([10, 10])
        custom_noise = jax.random.normal(jax.random.key(61), x.shape)

        noisy, returned_noise = model.forward_diffusion(x, t, noise=custom_noise)

        # Should use the provided noise
        assert jnp.allclose(returned_noise, custom_noise)
        assert jnp.all(jnp.isfinite(noisy))

    def test_forward_diffusion_t0_preserves_signal(self, sd_config, base_rngs):
        """Test that t=0 (first timestep) preserves most of the signal."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        x = jnp.ones((2, 32, 32, 3)) * 5.0  # Non-trivial signal
        t = jnp.array([0, 0])  # First timestep
        noise = jax.random.normal(jax.random.key(62), x.shape) * 0.1

        noisy, _ = model.forward_diffusion(x, t, noise=noise)

        # At t=0, noisy should be close to original (high alpha)
        # The sqrt_alphas_cumprod[0] should be close to 1
        assert jnp.mean(jnp.abs(noisy - x)) < jnp.mean(jnp.abs(x))

    def test_forward_diffusion_high_t_adds_noise(self, sd_config, base_rngs):
        """Test that high t adds significant noise."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        x = jnp.ones((2, 32, 32, 3))
        t_low = jnp.array([5, 5])
        t_high = jnp.array([45, 45])
        noise = jax.random.normal(jax.random.key(63), x.shape)

        noisy_low, _ = model.forward_diffusion(x, t_low, noise=noise)
        noisy_high, _ = model.forward_diffusion(x, t_high, noise=noise)

        # Higher t should have more noise influence (farther from original)
        diff_low = jnp.mean(jnp.abs(noisy_low - x))
        diff_high = jnp.mean(jnp.abs(noisy_high - x))
        assert diff_high > diff_low


class TestStableDiffusionTrainStep:
    """Tests for training step functionality."""

    @pytest.fixture
    def sd_config(self):
        """Configuration for train step tests."""
        return create_sd_config(
            input_shape=(16, 16, 3),
            hidden_dims=(16, 32),
            noise_steps=20,
            text_embedding_dim=32,
            text_max_length=8,
            vocab_size=100,
        )

    def test_train_step_returns_loss(self, sd_config, base_rngs):
        """Test that train_step returns a loss dictionary."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        # Create training data
        images = jax.random.normal(jax.random.key(70), (2, 16, 16, 3))
        text_tokens = jax.random.randint(jax.random.key(71), (2, 8), 0, 100)

        # Run train step
        result = model.train_step(images, text_tokens)

        assert isinstance(result, dict)
        assert "loss" in result
        assert result["loss"].shape == ()  # Scalar loss

    def test_train_step_loss_finite(self, sd_config, base_rngs):
        """Test that train_step produces finite loss."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        images = jax.random.normal(jax.random.key(72), (2, 16, 16, 3))
        text_tokens = jax.random.randint(jax.random.key(73), (2, 8), 0, 100)

        result = model.train_step(images, text_tokens)

        assert jnp.isfinite(result["loss"])

    def test_train_step_loss_positive(self, sd_config, base_rngs):
        """Test that MSE loss is non-negative."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        images = jax.random.normal(jax.random.key(74), (2, 16, 16, 3))
        text_tokens = jax.random.randint(jax.random.key(75), (2, 8), 0, 100)

        result = model.train_step(images, text_tokens)

        assert result["loss"] >= 0

    def test_train_step_different_batches(self, sd_config, base_rngs):
        """Test train_step with different batch sizes."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        for batch_size in [1, 2, 4]:
            images = jax.random.normal(jax.random.key(76 + batch_size), (batch_size, 16, 16, 3))
            text_tokens = jax.random.randint(
                jax.random.key(77 + batch_size), (batch_size, 8), 0, 100
            )

            result = model.train_step(images, text_tokens)

            assert jnp.isfinite(result["loss"])


class TestStableDiffusionGradients:
    """Tests for gradient computation through Stable Diffusion."""

    @pytest.fixture
    def sd_config(self):
        """Small config for gradient tests."""
        return create_sd_config(
            input_shape=(8, 8, 3),
            hidden_dims=(8, 16),
            noise_steps=10,
            text_embedding_dim=16,
            text_max_length=4,
            vocab_size=50,
        )

    def test_gradients_computable(self, sd_config, base_rngs):
        """Test that gradients can be computed through the model."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        def loss_fn(model):
            images = jnp.ones((1, 8, 8, 3))
            text_tokens = jnp.ones((1, 4), dtype=jnp.int32)
            result = model.train_step(images, text_tokens)
            return result["loss"]

        # Compute gradients
        grads = nnx.grad(loss_fn)(model)

        # Should have gradients (not None)
        assert grads is not None

    def test_gradients_finite(self, sd_config, base_rngs):
        """Test that computed gradients are finite."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        def loss_fn(model):
            images = jax.random.normal(jax.random.key(80), (1, 8, 8, 3))
            text_tokens = jax.random.randint(jax.random.key(81), (1, 4), 0, 50)
            result = model.train_step(images, text_tokens)
            return result["loss"]

        grads = nnx.grad(loss_fn)(model)

        # Check all gradient leaves are finite
        grad_leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
        for leaf in grad_leaves:
            if hasattr(leaf, "value"):
                assert jnp.all(jnp.isfinite(leaf.value)), "Found non-finite gradient"


class TestStableDiffusionConfigPropagation:
    """Tests for configuration propagation to model."""

    def test_custom_guidance_scale(self, base_rngs):
        """Test that custom guidance scale is propagated."""
        config = create_sd_config(guidance_scale=3.0)
        model = StableDiffusionModel(config, rngs=base_rngs)

        assert model.guidance_scale == 3.0

    def test_custom_vocab_size(self, base_rngs):
        """Test that custom vocab size is propagated."""
        config = create_sd_config(vocab_size=1000)
        model = StableDiffusionModel(config, rngs=base_rngs)

        assert model.vocab_size == 1000

    def test_custom_text_embedding_dim(self, base_rngs):
        """Test that custom text embedding dim is propagated."""
        config = create_sd_config(text_embedding_dim=128)
        model = StableDiffusionModel(config, rngs=base_rngs)

        assert model.text_embedding_dim == 128

    def test_custom_text_max_length(self, base_rngs):
        """Test that custom text max length is propagated."""
        config = create_sd_config(text_max_length=32)
        model = StableDiffusionModel(config, rngs=base_rngs)

        assert model.text_max_length == 32

    def test_noise_steps_propagated(self, base_rngs):
        """Test that noise steps are correctly propagated."""
        config = create_sd_config(noise_steps=200)
        model = StableDiffusionModel(config, rngs=base_rngs)

        assert model.noise_steps == 200
        assert model.alphas_cumprod.shape[0] == 200

    def test_latent_scale_factor(self, base_rngs):
        """Test latent scale factor is set from config."""
        config = create_sd_config()
        model = StableDiffusionModel(config, rngs=base_rngs)

        # Default latent scale factor from LatentDiffusionConfig
        assert hasattr(model, "latent_scale_factor")
        assert model.latent_scale_factor > 0


class TestStableDiffusionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def sd_config(self):
        """Configuration for edge case tests."""
        return create_sd_config(
            input_shape=(16, 16, 3),
            hidden_dims=(16, 32),
            noise_steps=20,
            text_embedding_dim=32,
            text_max_length=8,
            vocab_size=100,
        )

    def test_single_batch_processing(self, sd_config, base_rngs):
        """Test processing with batch size 1."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((1, *latent_shape))
        t = jnp.array([10])

        output = model(z, t)

        assert output["predicted_noise"].shape == (1, *latent_shape)

    def test_timestep_zero(self, sd_config, base_rngs):
        """Test forward pass at timestep 0."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((2, *latent_shape))
        t = jnp.array([0, 0])

        output = model(z, t)

        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_timestep_max(self, sd_config, base_rngs):
        """Test forward pass at maximum timestep."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((2, *latent_shape))
        t = jnp.array([19, 19])  # noise_steps - 1

        output = model(z, t)

        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_mixed_timesteps(self, sd_config, base_rngs):
        """Test forward pass with different timesteps per sample."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((4, *latent_shape))
        t = jnp.array([0, 5, 10, 19])  # Different timesteps

        output = model(z, t)

        assert output["predicted_noise"].shape == (4, *latent_shape)
        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_zeros_input(self, sd_config, base_rngs):
        """Test forward pass with zero latent input."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.zeros((2, *latent_shape))
        t = jnp.array([10, 10])

        output = model(z, t)

        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_negative_input(self, sd_config, base_rngs):
        """Test forward pass with negative latent values."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = -jnp.ones((2, *latent_shape))
        t = jnp.array([10, 10])

        output = model(z, t)

        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_large_input_values(self, sd_config, base_rngs):
        """Test forward pass with large latent values."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((2, *latent_shape)) * 10.0
        t = jnp.array([10, 10])

        output = model(z, t)

        assert jnp.all(jnp.isfinite(output["predicted_noise"]))

    def test_short_text_sequence(self, base_rngs):
        """Test with shorter text sequence than max length."""
        config = create_sd_config(text_max_length=16)
        model = StableDiffusionModel(config, rngs=base_rngs)

        # Create text with only 8 tokens (shorter than max 16)
        text_tokens = jax.random.randint(jax.random.key(90), (2, 8), 0, 500)

        # Should work with shorter sequence
        embeddings = model.encode_text(text_tokens)
        assert embeddings.shape == (2, 8, model.text_embedding_dim)


class TestStableDiffusionDeterminism:
    """Tests for deterministic behavior."""

    @pytest.fixture
    def sd_config(self):
        """Configuration for determinism tests."""
        return create_sd_config(
            input_shape=(16, 16, 3),
            hidden_dims=(16, 32),
            noise_steps=20,
        )

    def test_eval_mode_deterministic(self, sd_config, base_rngs):
        """Test that eval mode produces consistent outputs."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)
        model.eval()

        latent_shape = get_latent_shape(sd_config.input_shape, model.latent_channels)
        z = jnp.ones((2, *latent_shape))
        t = jnp.array([10, 10])

        # Run twice
        output1 = model(z, t)
        output2 = model(z, t)

        # Should be identical in eval mode
        assert jnp.allclose(output1["predicted_noise"], output2["predicted_noise"], atol=1e-5)

    def test_text_encoding_deterministic(self, sd_config, base_rngs):
        """Test that text encoding is deterministic."""
        model = StableDiffusionModel(sd_config, rngs=base_rngs)
        model.eval()

        text_tokens = jax.random.randint(jax.random.key(100), (2, 16), 0, 500)

        emb1 = model.encode_text(text_tokens)
        emb2 = model.encode_text(text_tokens)

        assert jnp.allclose(emb1, emb2, atol=1e-5)
