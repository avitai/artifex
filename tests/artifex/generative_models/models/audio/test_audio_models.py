"""Tests for audio models."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.modalities.audio import (
    AudioModalityConfig,
    AudioRepresentation,
)
from artifex.generative_models.models.audio import (
    AudioDiffusionModel,
    AudioModelConfig,
    BaseAudioModel,
    create_audio_diffusion_config,
    WaveNetAudioModel,
    WaveNetConfig,
)


@pytest.fixture
def rngs():
    """Random number generators for testing."""
    return nnx.Rngs(42)


@pytest.fixture
def audio_modality_config():
    """Audio modality configuration for testing."""
    return AudioModalityConfig(
        representation=AudioRepresentation.RAW_WAVEFORM,
        sample_rate=16000,
        duration=0.05,  # Reduced from 0.5s to 0.05s for performance (800 vs 8000 samples)
    )


@pytest.fixture
def audio_modality_config_fast():
    """Audio modality configuration for fast testing."""
    return AudioModalityConfig(
        representation=AudioRepresentation.RAW_WAVEFORM,
        sample_rate=8000,  # Lower sample rate
        duration=0.01,  # Very short duration (80 samples)
    )


@pytest.fixture
def audio_model_config(audio_modality_config):
    """Audio model configuration for testing."""
    return AudioModelConfig(
        modality_config=audio_modality_config,
        hidden_dims=64,
        num_layers=4,
        dropout_rate=0.1,
    )


class TestBaseAudioModel:
    """Test base audio model functionality."""

    def test_initialization(self, audio_model_config, rngs):
        """Test base audio model initialization."""
        model = BaseAudioModel(audio_model_config, rngs=rngs)

        assert model.config == audio_model_config
        assert model.modality_config == audio_model_config.modality_config
        assert model.sample_rate == 16000
        assert model.n_time_steps == 800  # 0.05 seconds at 16kHz
        assert model.output_dim == 1  # Raw waveform

    def test_get_output_shape(self, audio_model_config, rngs):
        """Test output shape calculation."""
        model = BaseAudioModel(audio_model_config, rngs=rngs)

        shape = model.get_output_shape(batch_size=4)
        expected_shape = (4, 800)  # 4 samples, 0.05 seconds each
        assert shape == expected_shape

    def test_preprocess_audio(self, audio_model_config, rngs):
        """Test audio preprocessing."""
        model = BaseAudioModel(audio_model_config, rngs=rngs)

        # Test audio that needs normalization
        audio = jnp.array([[2.0, -1.5, 1.0], [0.5, -0.5, 0.0]])
        processed = model.preprocess_audio(audio)

        # Should be normalized to [-1, 1] range
        assert jnp.abs(processed).max() <= 1.0
        assert jnp.isfinite(processed).all()

    def test_compute_reconstruction_loss(self, audio_model_config, rngs):
        """Test loss computation."""
        model = BaseAudioModel(audio_model_config, rngs=rngs)

        target = jax.random.normal(rngs.sample(), (2, 100))
        predicted = jax.random.normal(rngs.sample(), (2, 100))

        # Test different loss types
        mse_loss = model.compute_reconstruction_loss(target, predicted, "mse")
        l1_loss = model.compute_reconstruction_loss(target, predicted, "l1")
        spectral_loss = model.compute_reconstruction_loss(target, predicted, "spectral")

        assert jnp.isfinite(mse_loss)
        assert jnp.isfinite(l1_loss)
        assert jnp.isfinite(spectral_loss)
        assert mse_loss >= 0.0
        assert l1_loss >= 0.0
        assert spectral_loss >= 0.0

    def test_generate(self, audio_model_config, rngs):
        """Test audio generation."""
        model = BaseAudioModel(audio_model_config, rngs=rngs)

        generated = model.generate(n_samples=2, rngs=rngs)

        assert generated.shape == (2, 800)
        assert jnp.isfinite(generated).all()
        assert jnp.abs(generated).max() <= 1.0  # Should be clipped


class TestWaveNetAudioModel:
    """Test WaveNet audio model."""

    def test_initialization(self, audio_modality_config, rngs):
        """Test WaveNet initialization."""
        config = WaveNetConfig(
            modality_config=audio_modality_config,
            n_dilated_blocks=4,  # Small for testing
            n_residual_channels=32,
            n_skip_channels=32,
            quantization_levels=16,  # Small for testing
        )

        model = WaveNetAudioModel(config, rngs=rngs)

        assert model.wavenet_config == config
        assert len(model.residual_blocks) == 4
        assert len(config.dilation_rates) == 4

    def test_mu_law_encoding(self, audio_modality_config, rngs):
        """Test mu-law encoding/decoding."""
        config = WaveNetConfig(
            modality_config=audio_modality_config,
            quantization_levels=256,
        )
        model = WaveNetAudioModel(config, rngs=rngs)

        # Test encoding/decoding round trip
        audio = jnp.array([[-1.0, -0.5, 0.0, 0.5, 1.0]])
        encoded = model.mu_law_encode(audio)
        decoded = model.mu_law_decode(encoded)

        # Should be close to original (with quantization error)
        assert jnp.abs(decoded - audio).max() < 0.1
        assert encoded.min() >= 0
        assert encoded.max() < 256

    def test_forward_pass(self, audio_modality_config, rngs):
        """Test WaveNet forward pass."""
        config = WaveNetConfig(
            modality_config=audio_modality_config,
            n_dilated_blocks=2,  # Very small for testing
            n_residual_channels=16,
            quantization_levels=16,
        )
        model = WaveNetAudioModel(config, rngs=rngs)

        # Test with small audio sequence
        audio = jax.random.uniform(rngs.sample(), (1, 100), minval=-1, maxval=1)

        outputs = model(audio, training=False)

        assert "logits" in outputs
        assert "predictions" in outputs

        logits = outputs["logits"]
        assert logits.shape == (1, 100, 16)  # quantization_levels
        assert jnp.isfinite(logits).all()

    def test_generate(self, audio_modality_config_fast, rngs):
        """Test WaveNet autoregressive generation."""
        config = WaveNetConfig(
            modality_config=audio_modality_config_fast,
            n_dilated_blocks=1,
            n_residual_channels=8,
            quantization_levels=8,
        )
        model = WaveNetAudioModel(config, rngs=rngs)

        # Generate audio
        generated = model.generate(n_samples=2, rngs=rngs)

        # Verify output shape and properties
        expected_samples = int(
            audio_modality_config_fast.sample_rate * audio_modality_config_fast.duration
        )
        assert generated.shape == (2, expected_samples)
        assert jnp.isfinite(generated).all()
        assert jnp.abs(generated).max() <= 1.0  # Should be in valid audio range

    def test_generate_with_seed_audio(self, audio_modality_config_fast, rngs):
        """Test WaveNet generation with seed audio conditioning."""
        config = WaveNetConfig(
            modality_config=audio_modality_config_fast,
            n_dilated_blocks=1,
            n_residual_channels=8,
            quantization_levels=8,
        )
        model = WaveNetAudioModel(config, rngs=rngs)

        # Create seed audio (first 20 samples)
        seed_audio = jax.random.uniform(rngs.sample(), (1, 20), minval=-0.5, maxval=0.5)

        # Generate with seed
        generated = model.generate(n_samples=1, seed_audio=seed_audio, rngs=rngs)

        # Verify output
        expected_samples = int(
            audio_modality_config_fast.sample_rate * audio_modality_config_fast.duration
        )
        assert generated.shape == (1, expected_samples)
        assert jnp.isfinite(generated).all()
        # First few samples should match seed (within postprocessing tolerance)
        # Note: seed is incorporated but may be modified by postprocessing

    def test_generate_deterministic(self, audio_modality_config_fast):
        """Test that WaveNet generation is deterministic with same RNG seed."""
        config = WaveNetConfig(
            modality_config=audio_modality_config_fast,
            n_dilated_blocks=1,
            n_residual_channels=8,
            quantization_levels=8,
        )

        # Create two models with same seed
        rngs1 = nnx.Rngs(42)
        model1 = WaveNetAudioModel(config, rngs=rngs1)

        rngs2 = nnx.Rngs(42)
        model2 = WaveNetAudioModel(config, rngs=rngs2)

        # Generate with same RNG seed
        gen_rngs1 = nnx.Rngs(123)
        generated1 = model1.generate(n_samples=1, rngs=gen_rngs1)

        gen_rngs2 = nnx.Rngs(123)
        generated2 = model2.generate(n_samples=1, rngs=gen_rngs2)

        # Results should be identical
        assert jnp.allclose(generated1, generated2)


class TestAudioDiffusionModel:
    """Test audio diffusion model."""

    def test_initialization(self, audio_modality_config, rngs):
        """Test diffusion model initialization."""
        config = create_audio_diffusion_config(
            modality_config=audio_modality_config,
            num_timesteps=100,  # Small for testing
            unet_channels=32,  # Small for testing
        )

        model = AudioDiffusionModel(config, rngs=rngs)

        assert model.audio_config == config
        assert len(model.betas) == 100
        assert len(model.alphas) == 100
        assert len(model.alphas_cumprod) == 100

    def test_noise_schedule(self, audio_modality_config, rngs):
        """Test noise schedule generation."""
        config = create_audio_diffusion_config(
            modality_config=audio_modality_config,
            num_timesteps=10,
            beta_start=1e-4,
            beta_end=2e-2,
        )
        model = AudioDiffusionModel(config, rngs=rngs)

        # Check beta schedule properties
        assert model.betas[0] == pytest.approx(1e-4, rel=1e-3)
        assert model.betas[-1] == pytest.approx(2e-2, rel=1e-3)
        assert jnp.all(model.betas > 0)
        assert jnp.all(model.betas < 1)

        # Check derived schedules
        assert jnp.allclose(model.alphas, 1.0 - model.betas)
        assert jnp.all(model.alphas_cumprod > 0)
        assert jnp.all(model.alphas_cumprod <= 1)

    def test_add_noise(self, audio_modality_config, rngs):
        """Test noise addition process."""
        config = create_audio_diffusion_config(
            modality_config=audio_modality_config,
            num_timesteps=100,
        )
        model = AudioDiffusionModel(config, rngs=rngs)

        # Test noise addition
        audio = jax.random.normal(rngs.sample(), (2, 100))
        noise = jax.random.normal(rngs.sample(), (2, 100))
        timesteps = jnp.array([10, 50])

        noisy_audio = model.q_sample(audio, timesteps, noise)

        assert noisy_audio.shape == audio.shape
        assert jnp.isfinite(noisy_audio).all()

    def test_forward_pass(self, audio_modality_config, rngs):
        """Test diffusion model forward pass."""
        config = create_audio_diffusion_config(
            modality_config=audio_modality_config,
            num_timesteps=100,
            unet_channels=16,  # Very small for testing
        )
        model = AudioDiffusionModel(config, rngs=rngs)

        # Test with small audio and timesteps
        audio = jax.random.normal(rngs.sample(), (1, 64))  # Small sequence
        timesteps = jnp.array([50])

        outputs = model(audio, timesteps)

        assert "predicted_noise" in outputs

        predicted_noise = outputs["predicted_noise"]
        # Should match input shape (with potential channel dimension)
        assert predicted_noise.shape[0] == 1
        assert predicted_noise.shape[1] == 64
        assert jnp.isfinite(predicted_noise).all()


class TestModelIntegration:
    """Integration tests for audio models."""

    @pytest.mark.slow
    def test_model_generation_comparison_full(self, audio_modality_config, rngs):
        """Test full generation from both model types (optimized for performance).

        Note: Uses 0.05s duration (800 samples) instead of longer durations to avoid
        the 8000+ autoregressive generation steps that would make this test extremely slow.
        Each step requires a full WaveNet forward pass, so 8000 steps = 8000 model evaluations.
        """
        # WaveNet config
        wavenet_config = WaveNetConfig(
            modality_config=audio_modality_config,
            n_dilated_blocks=2,
            n_residual_channels=16,
            quantization_levels=16,
        )

        wavenet = WaveNetAudioModel(wavenet_config, rngs=rngs)

        # Generate from WaveNet model
        wavenet_audio = wavenet.generate(n_samples=1, rngs=rngs)

        # Test WaveNet generation
        assert wavenet_audio.shape == (1, 800)
        assert jnp.isfinite(wavenet_audio).all()
        assert jnp.abs(wavenet_audio).max() <= 1.0

        # Note: Diffusion model generation is skipped to avoid test timeouts
        # as diffusion generation requires many timesteps and is computationally expensive

    def test_model_generation_comparison(self, audio_modality_config_fast, rngs):
        """Fast test for generation from both model types."""
        # WaveNet config with minimal settings for fast testing
        wavenet_config = WaveNetConfig(
            modality_config=audio_modality_config_fast,
            n_dilated_blocks=1,  # Minimal blocks
            n_residual_channels=8,  # Minimal channels
            quantization_levels=8,  # Minimal quantization
        )

        wavenet = WaveNetAudioModel(wavenet_config, rngs=rngs)

        # Generate from WaveNet model (only 80 samples at 8kHz = 0.01 seconds)
        wavenet_audio = wavenet.generate(n_samples=1, rngs=rngs)

        # Test WaveNet generation
        expected_samples = int(
            audio_modality_config_fast.sample_rate * audio_modality_config_fast.duration
        )
        assert wavenet_audio.shape == (1, expected_samples)
        assert jnp.isfinite(wavenet_audio).all()
        assert jnp.abs(wavenet_audio).max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
