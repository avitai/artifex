"""Tests for spatial autoencoder components."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.models.vae.spatial_autoencoder import (
    SpatialDecoder,
    SpatialEncoder,
)


@pytest.fixture
def rngs():
    """Fixture providing random number generators."""
    return nnx.Rngs(0)


class TestSpatialEncoder:
    """Tests for SpatialEncoder."""

    def test_initialization(self, rngs):
        """Test that SpatialEncoder can be initialized."""
        config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(64, 128, 256),
            activation="relu",
        )
        encoder = SpatialEncoder(config=config, rngs=rngs)

        assert encoder is not None
        assert len(encoder.conv_layers) == 3

    def test_output_shape(self, rngs):
        """Test that encoder outputs correct spatial shape."""
        config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(64, 128, 256),
            activation="relu",
        )
        encoder = SpatialEncoder(config=config, rngs=rngs)

        # Input: (batch=2, height=64, width=64, channels=3)
        x = jnp.ones((2, 64, 64, 3))

        # Encode
        mean, log_var = encoder(x)

        # With 3 layers of stride-2 convolutions, spatial dims should be reduced by 8x
        # 64 -> 32 -> 16 -> 8
        assert mean.shape == (2, 8, 8, 4), f"Expected (2, 8, 8, 4), got {mean.shape}"
        assert log_var.shape == (2, 8, 8, 4), f"Expected (2, 8, 8, 4), got {log_var.shape}"

    def test_returns_tuple(self, rngs):
        """Test that encoder returns (mean, log_var) tuple."""
        config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(64, 128),
            activation="relu",
        )
        encoder = SpatialEncoder(config=config, rngs=rngs)

        x = jnp.ones((2, 64, 64, 3))
        output = encoder(x)

        assert isinstance(output, tuple)
        assert len(output) == 2

    def test_different_activations(self, rngs):
        """Test encoder with different activation functions."""
        for activation in ["relu", "silu", "gelu"]:
            config = EncoderConfig(
                name="spatial_encoder",
                input_shape=(64, 64, 3),
                latent_dim=4,
                hidden_dims=(64, 128),
                activation=activation,
            )
            encoder = SpatialEncoder(config=config, rngs=rngs)

            x = jnp.ones((2, 64, 64, 3))
            mean, log_var = encoder(x)

            assert mean.shape[0] == 2
            assert log_var.shape[0] == 2


class TestSpatialDecoder:
    """Tests for SpatialDecoder."""

    def test_initialization(self, rngs):
        """Test that SpatialDecoder can be initialized."""
        config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(64, 64, 3),
            hidden_dims=(256, 128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=config, rngs=rngs)

        assert decoder is not None
        assert len(decoder.conv_transpose_layers) == 2  # len(hidden_dims) - 1

    def test_output_shape(self, rngs):
        """Test that decoder outputs correct spatial shape."""
        config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(64, 64, 3),
            hidden_dims=(256, 128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=config, rngs=rngs)

        # Input: (batch=2, height=8, width=8, latent_channels=4)
        z = jnp.ones((2, 8, 8, 4))

        # Decode
        output = decoder(z)

        # With 3 transpose conv layers (len(hidden_dims) - 1 + final), spatial dims should be 8x
        # 8 -> 16 -> 32 -> 64
        assert output.shape == (2, 64, 64, 3), f"Expected (2, 64, 64, 3), got {output.shape}"

    def test_output_range(self, rngs):
        """Test that decoder output is in [0, 1] range due to sigmoid."""
        config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(64, 64, 3),
            hidden_dims=(256, 128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=config, rngs=rngs)

        z = jnp.ones((2, 8, 8, 4))
        output = decoder(z)

        assert jnp.all(output >= 0.0)
        assert jnp.all(output <= 1.0)

    def test_different_activations(self, rngs):
        """Test decoder with different activation functions."""
        for activation in ["relu", "silu", "gelu"]:
            config = DecoderConfig(
                name="spatial_decoder",
                latent_dim=4,
                output_shape=(32, 32, 3),
                hidden_dims=(256, 128),
                activation=activation,
            )
            decoder = SpatialDecoder(config=config, rngs=rngs)

            z = jnp.ones((2, 8, 8, 4))
            output = decoder(z)

            assert output.shape[0] == 2


class TestSpatialAutoencoder:
    """Integration tests for encoder-decoder pipeline."""

    def test_encode_decode_shapes_match(self, rngs):
        """Test that encoder and decoder have compatible shapes."""
        # Create encoder with 3 downsampling layers (64x64 -> 8x8)
        encoder_config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(64, 128, 256),
            activation="relu",
        )
        encoder = SpatialEncoder(config=encoder_config, rngs=rngs)

        # Create decoder with 3 upsampling layers (8x8 -> 64x64)
        decoder_config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(64, 64, 3),
            hidden_dims=(256, 128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=decoder_config, rngs=rngs)

        # Test encode-decode pipeline
        x = jnp.ones((2, 64, 64, 3))
        mean, _log_var = encoder(x)

        # Latent should be 8x smaller in each spatial dimension
        assert mean.shape == (2, 8, 8, 4)

        # Decode from mean
        reconstructed = decoder(mean)

        # Should reconstruct to original spatial size
        assert reconstructed.shape == (2, 64, 64, 3)

    def test_stable_diffusion_configuration(self, rngs):
        """Test configuration matching Stable Diffusion architecture."""
        # Stable Diffusion typically uses these configurations
        encoder_config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(64, 64, 3),
            latent_dim=4,
            hidden_dims=(64, 128, 256, 512),
            activation="relu",
        )
        encoder = SpatialEncoder(config=encoder_config, rngs=rngs)

        decoder_config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(64, 64, 3),
            hidden_dims=(512, 256, 128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=decoder_config, rngs=rngs)

        # Test with 64x64 input (common for Stable Diffusion)
        x = jnp.ones((2, 64, 64, 3))
        mean, _log_var = encoder(x)

        # 4 downsampling layers -> 64 / (2^4) = 64/16 = 4
        assert mean.shape == (2, 4, 4, 4)

        reconstructed = decoder(mean)

        # 4 upsampling layers -> 4 * (2^4) = 4*16 = 64
        # Note: decoder has len(hidden_dims)-1 + final = 4 transpose conv layers
        assert reconstructed.shape == (2, 64, 64, 3)


class TestSpatialAutoencoderJITCompatibility:
    """Comprehensive JIT compatibility tests for SpatialEncoder and SpatialDecoder."""

    def test_spatial_encoder_jit_forward_pass(self, rngs):
        """Test that SpatialEncoder forward pass can be JIT compiled."""
        config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(32, 32, 3),
            latent_dim=4,
            hidden_dims=(64, 128),
            activation="relu",
        )
        encoder = SpatialEncoder(config=config, rngs=rngs)

        x = jnp.ones((2, 32, 32, 3))

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test JIT compilation
        mean, log_var = forward(encoder, x)

        # Check shapes (2 downsampling layers -> 32 / 4 = 8)
        assert mean.shape == (2, 8, 8, 4)
        assert log_var.shape == (2, 8, 8, 4)
        assert jnp.isfinite(mean).all()
        assert jnp.isfinite(log_var).all()

    def test_spatial_decoder_jit_forward_pass(self, rngs):
        """Test that SpatialDecoder forward pass can be JIT compiled."""
        config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(32, 32, 3),
            hidden_dims=(128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=config, rngs=rngs)

        # Latent input (8x8 spatial size)
        z = jnp.ones((2, 8, 8, 4))

        @nnx.jit
        def forward(model, z):
            return model(z)

        # Test JIT compilation
        reconstructed = forward(decoder, z)

        # Check shape (2 upsampling layers -> 8 * 4 = 32)
        assert reconstructed.shape == (2, 32, 32, 3)
        assert jnp.isfinite(reconstructed).all()

    def test_spatial_encoder_jit_with_different_batch_sizes(self, rngs):
        """Test SpatialEncoder JIT compilation with different batch sizes."""
        config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(32, 32, 3),
            latent_dim=4,
            hidden_dims=(64, 128),
            activation="relu",
        )
        encoder = SpatialEncoder(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            x = jnp.ones((batch_size, 32, 32, 3))
            mean, log_var = forward(encoder, x)

            assert mean.shape == (batch_size, 8, 8, 4)
            assert log_var.shape == (batch_size, 8, 8, 4)
            assert jnp.isfinite(mean).all()

    def test_spatial_decoder_jit_with_different_batch_sizes(self, rngs):
        """Test SpatialDecoder JIT compilation with different batch sizes."""
        config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(32, 32, 3),
            hidden_dims=(128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, z):
            return model(z)

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            z = jnp.ones((batch_size, 8, 8, 4))
            reconstructed = forward(decoder, z)

            assert reconstructed.shape == (batch_size, 32, 32, 3)
            assert jnp.isfinite(reconstructed).all()

    def test_spatial_encoder_jit_with_different_spatial_sizes(self, rngs):
        """Test SpatialEncoder JIT compilation with different spatial sizes."""
        config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(64, 64, 3),  # Max expected size
            latent_dim=4,
            hidden_dims=(64, 128),
            activation="relu",
        )
        encoder = SpatialEncoder(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test with different spatial sizes
        for size in [16, 32, 64]:
            x = jnp.ones((2, size, size, 3))
            mean, log_var = forward(encoder, x)

            expected_size = size // 4  # 2 downsampling layers
            assert mean.shape == (2, expected_size, expected_size, 4)
            assert log_var.shape == (2, expected_size, expected_size, 4)

    def test_spatial_encoder_jit_gradient_computation(self, rngs):
        """Test that SpatialEncoder gradient computation can be JIT compiled."""
        config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(32, 32, 3),
            latent_dim=4,
            hidden_dims=(64, 128),
            activation="relu",
        )
        encoder = SpatialEncoder(config=config, rngs=rngs)

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def loss_fn(model, x):
            mean, log_var = model(x)
            return jnp.mean(mean**2 + log_var**2)

        # Compute gradients using nnx.grad
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(encoder, x)

        # Check that gradients were computed
        assert grads is not None

    def test_spatial_decoder_jit_gradient_computation(self, rngs):
        """Test that SpatialDecoder gradient computation can be JIT compiled."""
        config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(32, 32, 3),
            hidden_dims=(128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=config, rngs=rngs)

        z = jnp.ones((2, 8, 8, 4))

        @jax.jit
        def loss_fn(model, z):
            reconstructed = model(z)
            return jnp.mean(reconstructed**2)

        # Compute gradients using nnx.grad
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(decoder, z)

        # Check that gradients were computed
        assert grads is not None

    def test_spatial_autoencoder_jit_end_to_end(self, rngs):
        """Test end-to-end SpatialEncoder + SpatialDecoder pipeline with JIT compilation."""
        encoder_config = EncoderConfig(
            name="spatial_encoder",
            input_shape=(32, 32, 3),
            latent_dim=4,
            hidden_dims=(64, 128),
            activation="relu",
        )
        encoder = SpatialEncoder(config=encoder_config, rngs=rngs)

        decoder_config = DecoderConfig(
            name="spatial_decoder",
            latent_dim=4,
            output_shape=(32, 32, 3),
            hidden_dims=(128, 64),
            activation="relu",
        )
        decoder = SpatialDecoder(config=decoder_config, rngs=rngs)

        x = jnp.ones((2, 32, 32, 3))

        @jax.jit
        def autoencoder_forward(encoder, decoder, x):
            mean, log_var = encoder(x)
            # Reparameterization
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(jax.random.key(0), mean.shape)
            z = mean + eps * std
            # Decode
            reconstructed = decoder(z)
            return reconstructed, mean, log_var

        reconstructed, mean, log_var = autoencoder_forward(encoder, decoder, x)

        # Check shapes
        assert reconstructed.shape == (2, 32, 32, 3)
        assert mean.shape == (2, 8, 8, 4)
        assert log_var.shape == (2, 8, 8, 4)
        assert jnp.isfinite(reconstructed).all()

    def test_spatial_encoder_jit_with_different_activations(self, rngs):
        """Test SpatialEncoder JIT compilation with different activation functions."""
        for activation in ["relu", "silu", "gelu"]:
            config = EncoderConfig(
                name="spatial_encoder",
                input_shape=(32, 32, 3),
                latent_dim=4,
                hidden_dims=(64, 128),
                activation=activation,
            )
            encoder = SpatialEncoder(config=config, rngs=rngs)

            x = jnp.ones((2, 32, 32, 3))

            @nnx.jit
            def forward(model, x):
                return model(x)

            mean, log_var = forward(encoder, x)

            assert mean.shape == (2, 8, 8, 4)
            assert log_var.shape == (2, 8, 8, 4)

    def test_spatial_decoder_jit_with_different_activations(self, rngs):
        """Test SpatialDecoder JIT compilation with different activation functions."""
        for activation in ["relu", "silu", "gelu"]:
            config = DecoderConfig(
                name="spatial_decoder",
                latent_dim=4,
                output_shape=(32, 32, 3),
                hidden_dims=(128, 64),
                activation=activation,
            )
            decoder = SpatialDecoder(config=config, rngs=rngs)

            z = jnp.ones((2, 8, 8, 4))

            @nnx.jit
            def forward(model, z):
                return model(z)

            reconstructed = forward(decoder, z)

            assert reconstructed.shape == (2, 32, 32, 3)
