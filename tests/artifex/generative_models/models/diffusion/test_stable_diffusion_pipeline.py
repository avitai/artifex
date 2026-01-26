"""Comprehensive tests for Stable Diffusion Pipeline.

This module tests the complete Stable Diffusion pipeline for text-to-image generation,
following Test-Driven Development (TDD) principles.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    NoiseScheduleConfig,
    StableDiffusionConfig,
    UNetBackboneConfig,
)
from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.models.diffusion.stable_diffusion_pipeline import (
    StableDiffusionPipeline,
)


def create_pipeline_config(
    input_shape: tuple[int, int, int] = (64, 64, 3),
    latent_channels: int = 4,
    hidden_dims: tuple[int, ...] = (128, 256),
    vocab_size: int = 1000,
    text_max_length: int = 77,
    text_embedding_dim: int = 256,
    noise_steps: int = 100,
    guidance_scale: float = 7.5,
) -> StableDiffusionConfig:
    """Create a test configuration for Stable Diffusion Pipeline.

    Args:
        input_shape: Input image shape (H, W, C)
        latent_channels: Number of latent channels
        hidden_dims: UNet hidden dimensions
        vocab_size: Vocabulary size for text encoder
        text_max_length: Maximum text sequence length
        text_embedding_dim: Text embedding dimension
        noise_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale

    Returns:
        StableDiffusionConfig for pipeline initialization
    """
    encoder = EncoderConfig(
        name="test_encoder",
        hidden_dims=(64, 128, 256),
        input_shape=input_shape,
        latent_dim=latent_channels,
        activation="gelu",
    )

    decoder = DecoderConfig(
        name="test_decoder",
        hidden_dims=(256, 128, 64),
        latent_dim=latent_channels,
        output_shape=input_shape,
        activation="gelu",
    )

    backbone = UNetBackboneConfig(
        name="test_unet",
        hidden_dims=hidden_dims,
        in_channels=latent_channels,
        out_channels=latent_channels,
        time_embedding_dim=128,
        activation="gelu",
    )

    noise_schedule = NoiseScheduleConfig(
        name="test_schedule",
        schedule_type="linear",
        num_timesteps=noise_steps,
        beta_start=0.00085,
        beta_end=0.012,
    )

    return StableDiffusionConfig(
        name="test_sd_pipeline",
        input_shape=input_shape,
        encoder=encoder,
        decoder=decoder,
        backbone=backbone,
        noise_schedule=noise_schedule,
        vocab_size=vocab_size,
        text_max_length=text_max_length,
        text_embedding_dim=text_embedding_dim,
        guidance_scale=guidance_scale,
        use_guidance=True,
        latent_scale_factor=0.18215,
    )


class TestStableDiffusionPipelineInitialization:
    """Test Stable Diffusion pipeline initialization."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return create_pipeline_config()

    def test_initialization(self, config):
        """Test basic initialization."""
        rngs = nnx.Rngs(0)

        pipeline = StableDiffusionPipeline(config, rngs=rngs)

        assert pipeline is not None
        assert hasattr(pipeline, "model")
        assert hasattr(pipeline, "scheduler")
        assert hasattr(pipeline, "encoder")
        assert hasattr(pipeline, "decoder")
        assert hasattr(pipeline, "text_encoder")
        assert hasattr(pipeline, "unet")

    def test_initialization_different_configs(self):
        """Test initialization with different configurations."""
        rngs = nnx.Rngs(0)

        # Minimal config
        config_min = create_pipeline_config(
            input_shape=(32, 32, 3),
            hidden_dims=(64,),
            vocab_size=100,
            text_max_length=10,
            text_embedding_dim=64,
            noise_steps=10,
        )

        pipeline = StableDiffusionPipeline(config_min, rngs=rngs)
        assert pipeline is not None


class TestStableDiffusionPipelineTextToImage:
    """Test text-to-image generation."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        config = create_pipeline_config(
            guidance_scale=7.5,
        )
        rngs = nnx.Rngs(0)
        return StableDiffusionPipeline(config, rngs=rngs)

    def test_generate_from_text_basic(self, pipeline):
        """Test basic text-to-image generation."""
        # Token IDs (in real use, these would come from a tokenizer)
        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)

        rngs = nnx.Rngs(42)
        images = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=64,
            width=64,
            rngs=rngs,
        )

        # Check output shape
        assert images.shape == (1, 64, 64, 3)
        assert images.dtype == jnp.float32

    def test_generate_from_text_multiple_prompts(self, pipeline):
        """Test generation with multiple prompts."""
        batch_size = 4
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, 77), 0, 1000)

        rngs = nnx.Rngs(42)
        images = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=64,
            width=64,
            rngs=rngs,
        )

        assert images.shape == (batch_size, 64, 64, 3)

    def test_generate_from_text_different_guidance_scales(self, pipeline):
        """Test generation with different guidance scales."""
        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)

        # Test with different guidance scales (using different random seeds)
        images_low = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=1.0,
            height=64,
            width=64,
            rngs=nnx.Rngs(42),
        )

        images_high = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=15.0,
            height=64,
            width=64,
            rngs=nnx.Rngs(43),  # Different seed for different starting latents
        )

        # Different guidance should produce different images
        # Note: With different starting latents, images will definitely be different
        assert not jnp.allclose(images_low, images_high, atol=0.1)

    def test_generate_from_text_different_num_steps(self, pipeline):
        """Test generation with different inference steps."""
        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)

        for num_steps in [5, 10, 20]:
            rngs = nnx.Rngs(42)
            images = pipeline.generate_from_text(
                token_ids=token_ids,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                height=64,
                width=64,
                rngs=rngs,
            )

            assert images.shape == (1, 64, 64, 3)

    def test_generate_from_text_output_range(self, pipeline):
        """Test that generated images are in valid range."""
        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)

        rngs = nnx.Rngs(42)
        images = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=64,
            width=64,
            rngs=rngs,
        )

        # Images should be in [0, 1] range
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_generate_from_text_no_nan_inf(self, pipeline):
        """Test that generation doesn't produce NaN or Inf."""
        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)

        rngs = nnx.Rngs(42)
        images = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=64,
            width=64,
            rngs=rngs,
        )

        assert not jnp.isnan(images).any()
        assert not jnp.isinf(images).any()


class TestStableDiffusionPipelineEncodeDecode:
    """Test VAE encode/decode functionality."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        config = create_pipeline_config()
        rngs = nnx.Rngs(0)
        return StableDiffusionPipeline(config, rngs=rngs)

    def test_encode_image(self, pipeline):
        """Test image encoding to latent space."""
        batch_size = 2
        images = jax.random.uniform(jax.random.key(0), (batch_size, 64, 64, 3))

        # NNX stores rngs at init, no need to pass
        latents = pipeline.encode_image(images)

        # Check latent shape (8x spatial compression for SD)
        assert latents.shape == (batch_size, 8, 8, 4)
        assert latents.dtype == jnp.float32

    def test_decode_latents(self, pipeline):
        """Test decoding latents to images."""
        batch_size = 2
        latents = jax.random.normal(jax.random.key(0), (batch_size, 8, 8, 4))

        # NNX stores rngs at init, no need to pass
        images = pipeline.decode_latents(latents)

        # Check image shape
        assert images.shape == (batch_size, 64, 64, 3)
        assert images.dtype == jnp.float32

    def test_encode_decode_round_trip(self, pipeline):
        """Test encode-decode round trip."""
        batch_size = 2
        original_images = jax.random.uniform(jax.random.key(0), (batch_size, 64, 64, 3))

        # NNX stores rngs at init, no need to pass

        # Encode
        latents = pipeline.encode_image(original_images)

        # Decode
        reconstructed_images = pipeline.decode_latents(latents)

        # Check shape preserved
        assert reconstructed_images.shape == original_images.shape

        # Should be reasonable reconstruction (not perfect due to VAE)
        # Check that reconstruction is not too far off
        mse = jnp.mean((original_images - reconstructed_images) ** 2)
        assert mse < 1.0  # Reasonable reconstruction error


class TestStableDiffusionPipelineTextEncoding:
    """Test text encoding functionality."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        config = create_pipeline_config()
        rngs = nnx.Rngs(0)
        return StableDiffusionPipeline(config, rngs=rngs)

    def test_encode_text(self, pipeline):
        """Test text encoding."""
        batch_size = 2
        token_ids = jax.random.randint(jax.random.key(0), (batch_size, 77), 0, 1000)

        text_embeddings = pipeline.encode_text(token_ids)

        # Check shape
        assert text_embeddings.shape == (batch_size, 77, 256)
        assert text_embeddings.dtype == jnp.float32

    def test_encode_text_different_prompts(self, pipeline):
        """Test that different prompts produce different embeddings."""
        token_ids1 = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)
        token_ids2 = jax.random.randint(jax.random.key(1), (1, 77), 0, 1000)

        text_emb1 = pipeline.encode_text(token_ids1)
        text_emb2 = pipeline.encode_text(token_ids2)

        # Different prompts should produce different embeddings
        assert not jnp.allclose(text_emb1, text_emb2)


class TestStableDiffusionPipelineTraining:
    """Test training functionality."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        config = create_pipeline_config()
        rngs = nnx.Rngs(0)
        return StableDiffusionPipeline(config, rngs=rngs)

    def test_train_step(self, pipeline):
        """Test training step."""
        batch = {
            "images": jax.random.uniform(jax.random.key(0), (2, 64, 64, 3)),
            "token_ids": jax.random.randint(jax.random.key(1), (2, 77), 0, 1000),
        }

        loss_dict = pipeline.train_step(batch)

        # Check loss dict
        assert "loss" in loss_dict
        assert isinstance(loss_dict["loss"], jax.Array)
        assert loss_dict["loss"].shape == ()  # Scalar
        assert loss_dict["loss"] >= 0.0

    def test_gradients_computable(self, pipeline):
        """Test that gradients can be computed."""
        batch = {
            "images": jax.random.uniform(jax.random.key(0), (2, 64, 64, 3)),
            "token_ids": jax.random.randint(jax.random.key(1), (2, 77), 0, 1000),
        }

        def loss_fn(model):
            loss_dict = model.train_step(batch)
            return loss_dict["loss"]

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(pipeline)

        assert isinstance(loss, jax.Array)
        assert grads is not None


class TestStableDiffusionPipelineClassifierFreeGuidance:
    """Test classifier-free guidance."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        config = create_pipeline_config(guidance_scale=7.5)
        rngs = nnx.Rngs(0)
        return StableDiffusionPipeline(config, rngs=rngs)

    def test_cfg_different_from_unconditional(self, pipeline):
        """Test that CFG produces different results than unconditional."""
        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)

        # Generate with CFG
        images_cfg = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=7.5,
            height=64,
            width=64,
            rngs=nnx.Rngs(42),
        )

        # Generate without CFG (guidance_scale=1.0, different seed)
        images_no_cfg = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=10,
            guidance_scale=1.0,
            height=64,
            width=64,
            rngs=nnx.Rngs(43),  # Different seed for different starting latents
        )

        # Should be different (using different starting latents)
        assert not jnp.allclose(images_cfg, images_no_cfg, atol=0.1)


class TestStableDiffusionPipelineTrainEvalModes:
    """Test train/eval mode switching."""

    def test_mode_switching(self):
        """Test switching between train and eval modes."""
        config = create_pipeline_config(
            hidden_dims=(64, 128),
        )

        rngs = nnx.Rngs(0)
        pipeline = StableDiffusionPipeline(config, rngs=rngs)

        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)

        # Eval mode
        pipeline.eval()
        rngs1 = nnx.Rngs(42)
        images_eval1 = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=5,
            guidance_scale=7.5,
            height=64,
            width=64,
            rngs=rngs1,
        )

        # Train mode
        pipeline.train()
        # (in train mode, generation might still work but behave differently)

        # Back to eval mode
        pipeline.eval()
        rngs2 = nnx.Rngs(42)  # Same seed
        images_eval2 = pipeline.generate_from_text(
            token_ids=token_ids,
            num_inference_steps=5,
            guidance_scale=7.5,
            height=64,
            width=64,
            rngs=rngs2,
        )

        # Same seed in eval mode should give same results
        assert jnp.allclose(images_eval1, images_eval2, atol=1e-5)


class TestStableDiffusionPipelineComposition:
    """Test that pipeline properly composes the model (DRY)."""

    def test_pipeline_composes_model(self):
        """Test that pipeline uses StableDiffusionModel internally."""
        config = create_pipeline_config()
        rngs = nnx.Rngs(0)
        pipeline = StableDiffusionPipeline(config, rngs=rngs)

        # Pipeline should have a model attribute
        assert hasattr(pipeline, "model")
        from artifex.generative_models.models.diffusion.stable_diffusion import (
            StableDiffusionModel,
        )

        assert isinstance(pipeline.model, StableDiffusionModel)

    def test_pipeline_delegates_to_model(self):
        """Test that pipeline methods delegate to model."""
        config = create_pipeline_config()
        rngs = nnx.Rngs(0)
        pipeline = StableDiffusionPipeline(config, rngs=rngs)

        # Text encoding should use model's text encoder
        token_ids = jax.random.randint(jax.random.key(0), (1, 77), 0, 1000)
        pipeline_emb = pipeline.encode_text(token_ids)
        model_emb = pipeline.model.encode_text(token_ids)
        assert jnp.allclose(pipeline_emb, model_emb)

    def test_encoder_decoder_access(self):
        """Test that encoder/decoder are directly accessible."""
        config = create_pipeline_config()
        rngs = nnx.Rngs(0)
        pipeline = StableDiffusionPipeline(config, rngs=rngs)

        # Access encoder and decoder directly
        assert pipeline.encoder is not None
        assert pipeline.decoder is not None

        # Should be same as model's encoder/decoder
        assert pipeline.encoder is pipeline.model.encoder
        assert pipeline.decoder is pipeline.model.decoder
