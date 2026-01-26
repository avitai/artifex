"""Unit tests for VQ-VAE implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VQVAEConfig
from artifex.generative_models.models.vae.vq_vae import VQVAE


@pytest.fixture
def rng_key():
    """Fixture for random number generator."""
    return jax.random.key(42)


@pytest.fixture
def rngs(rng_key):
    """Fixture for nnx random number generators."""
    params_key, dropout_key, sample_key = jax.random.split(rng_key, 3)
    return nnx.Rngs(params=params_key, dropout=dropout_key, sample=sample_key)


@pytest.fixture
def vqvae_config():
    """Fixture for VQVAEConfig."""
    input_dim = 100
    embedding_dim = 64
    num_embeddings = 512

    encoder_config = EncoderConfig(
        name="test_encoder",
        hidden_dims=(64, 32),
        activation="relu",
        input_shape=(input_dim,),
        latent_dim=embedding_dim,  # VQ-VAE encoder outputs embedding_dim
    )

    decoder_config = DecoderConfig(
        name="test_decoder",
        hidden_dims=(32, 64),
        activation="relu",
        output_shape=(input_dim,),
        latent_dim=embedding_dim,
    )

    return VQVAEConfig(
        name="test_vqvae",
        encoder=encoder_config,
        decoder=decoder_config,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=0.25,
    )


@pytest.fixture
def vqvae_components(vqvae_config):
    """Fixture for VQ-VAE config and test data."""
    input_dim = 100
    latent_dim = 10
    embedding_dim = 64
    batch_size = 2

    # Create sample input
    x = jnp.ones((batch_size, input_dim))

    return {
        "config": vqvae_config,
        "x": x,
        "latent_dim": latent_dim,
        "embedding_dim": embedding_dim,
        "input_dim": input_dim,
        "batch_size": batch_size,
    }


class TestVQVAE:
    """Test suite for the VQ-VAE class."""

    def test_initialization(self, rngs, vqvae_components):
        """Test VQ-VAE initialization."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Check attributes
        assert vqvae.encoder is not None
        assert vqvae.decoder is not None
        assert vqvae.num_embeddings == config.num_embeddings
        assert vqvae.embedding_dim == embedding_dim
        assert vqvae.commitment_cost == config.commitment_cost

        # Check embedding table initialization
        assert hasattr(vqvae, "embeddings")
        assert isinstance(vqvae.embeddings, nnx.Embed)

        # Check auxiliary data storage
        assert hasattr(vqvae, "_last_quantize_aux")
        assert isinstance(vqvae._last_quantize_aux, nnx.Dict)

    def test_quantize(self, rngs, vqvae_components):
        """Test VQ-VAE quantization function."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Generate an encoding for testing quantization
        batch_size = x.shape[0]
        encoding = jnp.ones((batch_size, embedding_dim))

        # Test quantize function
        quantized, aux = vqvae.quantize(encoding)

        # Check shapes
        assert quantized.shape == encoding.shape

        # Check auxiliary data
        assert "commitment_loss" in aux
        assert "codebook_loss" in aux
        assert "encoding_indices" in aux

        # Check encoding indices shape
        # Flattened batch dimension * 1 (single vector per sample)
        assert aux["encoding_indices"].shape == (batch_size * 1,)

        # Check loss values are reasonable
        assert not jnp.isnan(aux["commitment_loss"])
        assert not jnp.isnan(aux["codebook_loss"])

    def test_encode(self, rngs, vqvae_components):
        """Test VQ-VAE encode method."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Test encode method
        mean, log_var = vqvae.encode(x)

        # Check shapes - VQ-VAE uses a dummy log_var for VAE compatibility
        assert mean.shape[0] == x.shape[0]  # Batch dimension matches
        assert mean.shape[1] == embedding_dim  # Embedding dimension

        # Dummy log_var should match mean shape
        assert log_var.shape == mean.shape

        # Log variance should be zeros (dummy values for VAE interface)
        assert jnp.allclose(log_var, jnp.zeros_like(log_var))

    def test_decode(self, rngs, vqvae_components):
        """Test VQ-VAE decode method."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Create sample quantized encoding
        batch_size = x.shape[0]
        z = jnp.ones((batch_size, embedding_dim))

        # Test decode method
        reconstructed = vqvae.decode(z)

        # Check shape
        assert reconstructed.shape == x.shape

    def test_forward_pass(self, rngs, vqvae_components):
        """Test VQ-VAE forward pass."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Test forward pass
        outputs = vqvae(x)

        # Check outputs
        assert "reconstructed" in outputs
        assert "z" in outputs  # Quantized encoding
        assert "z_e" in outputs  # Pre-quantization encoding
        assert "commitment_loss" in outputs
        assert "codebook_loss" in outputs

        # Check shapes
        assert outputs["reconstructed"].shape == x.shape
        assert outputs["z"].shape == (x.shape[0], embedding_dim)
        assert outputs["z_e"].shape == (x.shape[0], embedding_dim)

    def test_loss_function(self, rngs, vqvae_components):
        """Test VQ-VAE loss function."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Get model outputs for loss computation
        outputs = vqvae(x)

        # Test loss function
        losses = vqvae.loss_fn(x, outputs)

        # Check loss components
        assert "reconstruction_loss" in losses
        assert "commitment_loss" in losses
        assert "codebook_loss" in losses
        assert "loss" in losses

        # Check values are reasonable
        assert not jnp.isnan(losses["reconstruction_loss"])
        assert not jnp.isnan(losses["commitment_loss"])
        assert not jnp.isnan(losses["codebook_loss"])
        assert not jnp.isnan(losses["loss"])

        # Verify commitment loss weight is applied
        commitment_cost = config.commitment_cost
        expected_commitment_term = commitment_cost * losses["commitment_loss"]
        expected_total = (
            losses["reconstruction_loss"] + expected_commitment_term + losses["codebook_loss"]
        )
        assert jnp.isclose(losses["loss"], expected_total)

    def test_sample_and_generate(self, rngs, vqvae_components):
        """Test VQ-VAE sample and generate methods."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Test sample method
        n_samples = 3
        samples = vqvae.sample(n_samples)

        # Check shape
        assert samples.shape == (n_samples, x.shape[1])

        # Test generate method
        generated = vqvae.generate(n_samples)
        assert generated.shape == (n_samples, x.shape[1])

    def test_reconstruct(self, rngs, vqvae_components):
        """Test VQ-VAE reconstruct method."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Test reconstruct method
        reconstructed = vqvae.reconstruct(x)

        # Check shape
        assert reconstructed.shape == x.shape


class TestVQVAEJITCompatibility:
    """Comprehensive JIT compatibility tests for VQ-VAE."""

    def test_vqvae_jit_forward_pass(self, rngs, vqvae_components):
        """Test that VQ-VAE forward pass can be JIT compiled."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test JIT compilation
        output = forward(vqvae, x)

        # Check outputs
        assert "reconstructed" in output
        assert "quantized" in output
        assert "encoding_indices" in output
        assert output["reconstructed"].shape == x.shape

    def test_vqvae_jit_encode(self, rngs, vqvae_components):
        """Test that VQ-VAE encode method can be JIT compiled."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        @nnx.jit
        def encode_fn(model, x):
            return model.encode(x)

        # encode() returns (mean, log_var) for VAE interface compatibility
        embeddings, log_var = encode_fn(vqvae, x)

        # Check shape
        assert embeddings.shape == (x.shape[0], embedding_dim)
        assert jnp.isfinite(embeddings).all()
        # For VQ-VAE, log_var should be zeros
        assert jnp.allclose(log_var, jnp.zeros_like(log_var))

    def test_vqvae_jit_quantize(self, rngs, vqvae_components):
        """Test that VQ-VAE quantize method can be JIT compiled."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Create sample embeddings
        embeddings = jnp.ones((2, embedding_dim))

        @nnx.jit
        def quantize_fn(model, embeddings):
            return model.quantize(embeddings)

        # quantize() returns (quantized, aux_dict)
        quantized, aux = quantize_fn(vqvae, embeddings)
        indices = aux["encoding_indices"]

        # Check shapes
        assert quantized.shape == embeddings.shape
        assert indices.shape == (embeddings.shape[0],)
        assert jnp.isfinite(quantized).all()
        # Check auxiliary data
        assert "commitment_loss" in aux
        assert "codebook_loss" in aux

    def test_vqvae_jit_decode(self, rngs, vqvae_components):
        """Test that VQ-VAE decode method can be JIT compiled."""
        config = vqvae_components["config"]
        embedding_dim = vqvae_components["embedding_dim"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Create sample quantized embeddings
        z = jnp.ones((x.shape[0], embedding_dim))

        @nnx.jit
        def decode_fn(model, z):
            return model.decode(z)

        reconstructed = decode_fn(vqvae, z)

        # Check shape
        assert reconstructed.shape == x.shape
        assert jnp.isfinite(reconstructed).all()

    def test_vqvae_jit_loss_function(self, rngs, vqvae_components):
        """Test that VQ-VAE loss function can be JIT compiled."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        # Get outputs first
        outputs = vqvae(x)

        @jax.jit
        def compute_loss(model, x, outputs):
            return model.loss_fn(x=x, outputs=outputs)

        losses = compute_loss(vqvae, x, outputs)

        # Check loss components
        assert "reconstruction_loss" in losses
        assert "vq_loss" in losses
        assert "commitment_loss" in losses
        assert "loss" in losses
        assert jnp.isfinite(losses["loss"])

    def test_vqvae_jit_with_different_batch_sizes(self, rngs, vqvae_components):
        """Test VQ-VAE JIT compilation with different batch sizes."""
        config = vqvae_components["config"]
        input_dim = vqvae_components["input_dim"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = jnp.ones((batch_size, input_dim))
            output = forward(vqvae, x)

            assert output["reconstructed"].shape == (batch_size, input_dim)
            assert not jnp.isnan(output["reconstructed"]).any()

    def test_vqvae_jit_gradient_computation(self, rngs, vqvae_components):
        """Test that VQ-VAE gradient computation can be JIT compiled."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        @jax.jit
        def loss_fn(model, x):
            outputs = model(x)
            losses = model.loss_fn(x=x, outputs=outputs)
            return losses["loss"]

        # Compute gradients using nnx.grad
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(vqvae, x)

        # Check that gradients were computed
        assert grads is not None

    def test_vqvae_jit_reconstruct(self, rngs, vqvae_components):
        """Test that VQ-VAE reconstruct method can be JIT compiled."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        @nnx.jit
        def reconstruct_fn(model, x):
            return model.reconstruct(x)

        reconstructed = reconstruct_fn(vqvae, x)

        # Check shape
        assert reconstructed.shape == x.shape
        assert jnp.isfinite(reconstructed).all()

    def test_vqvae_jit_end_to_end(self, rngs, vqvae_components):
        """Test end-to-end VQ-VAE pipeline with JIT compilation."""
        config = vqvae_components["config"]
        x = vqvae_components["x"]

        # Initialize VQ-VAE
        vqvae = VQVAE(config=config, rngs=rngs)

        @jax.jit
        def train_step(model, x):
            # Forward pass
            outputs = model(x)
            # Compute loss
            losses = model.loss_fn(x=x, outputs=outputs)
            return losses["loss"], outputs

        # Run training step
        loss, outputs = train_step(vqvae, x)

        # Check outputs
        assert jnp.isfinite(loss)
        assert outputs["reconstructed"].shape == x.shape
        assert jnp.isfinite(outputs["reconstructed"]).all()
        assert "quantized" in outputs
        assert "encoding_indices" in outputs
