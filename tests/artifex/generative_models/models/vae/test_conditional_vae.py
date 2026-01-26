"""Unit tests for Conditional VAE implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import ConditionalVAEConfig
from artifex.generative_models.models.vae.conditional import ConditionalVAE


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
def cvae_config():
    """Fixture for ConditionalVAEConfig."""
    input_dim = 100
    latent_dim = 10
    condition_dim = 5

    encoder_config = EncoderConfig(
        name="test_encoder",
        hidden_dims=(64, 32),
        activation="relu",
        input_shape=(input_dim,),
        latent_dim=latent_dim,
    )

    decoder_config = DecoderConfig(
        name="test_decoder",
        hidden_dims=(32, 64),
        activation="relu",
        output_shape=(input_dim,),
        latent_dim=latent_dim,
    )

    return ConditionalVAEConfig(
        name="test_cvae",
        encoder=encoder_config,
        decoder=decoder_config,
        num_classes=condition_dim,  # num_classes is the required field
        condition_type="concat",
    )


@pytest.fixture
def cvae_components(cvae_config):
    """Fixture for CVAE config and test data."""
    input_dim = 100
    latent_dim = 10
    condition_dim = 5
    batch_size = 2

    # Create sample input
    x = jnp.ones((batch_size, input_dim))

    # Create sample condition
    y_onehot = jnp.zeros((batch_size, condition_dim))
    y_onehot = y_onehot.at[:, 0].set(1)  # One-hot condition

    y_int = jnp.zeros((batch_size,), dtype=jnp.int32)  # Integer condition

    return {
        "config": cvae_config,
        "x": x,
        "y_onehot": y_onehot,
        "y_int": y_int,
        "latent_dim": latent_dim,
        "condition_dim": condition_dim,
        "input_dim": input_dim,
        "batch_size": batch_size,
    }


class TestConditionalVAE:
    """Test suite for the Conditional VAE class."""

    def test_initialization(self, rngs, cvae_components):
        """Test Conditional VAE initialization."""
        config = cvae_components["config"]
        latent_dim = cvae_components["latent_dim"]
        condition_dim = cvae_components["condition_dim"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Check attributes
        assert cvae.encoder is not None
        assert cvae.decoder is not None
        assert cvae.latent_dim == latent_dim
        assert cvae.condition_dim == condition_dim
        assert cvae.condition_type == "concat"

    def test_reshape_condition(self, rngs, cvae_components):
        """Test condition reshaping for different input shapes."""
        config = cvae_components["config"]
        condition_dim = cvae_components["condition_dim"]
        y_int = cvae_components["y_int"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Test reshaping for 2D inputs
        batch_size = 2
        target_shape = (batch_size, 100)  # 1D input (batch_size, features)
        reshaped = cvae._reshape_condition(y_int, target_shape)

        # Should become one-hot encoded without further reshaping
        assert reshaped.shape == (batch_size, condition_dim)

        # Test reshaping for 3D inputs
        target_shape = (batch_size, 10, 10)  # 2D input (batch_size, h, w)
        reshaped = cvae._reshape_condition(y_int, target_shape)

        # Should be expanded to (batch_size, h, w, condition_dim)
        assert reshaped.shape == (batch_size, 10, 10, condition_dim)

        # Test reshaping for 4D inputs
        target_shape = (batch_size, 8, 8, 3)  # Image (batch_size, h, w, c)
        reshaped = cvae._reshape_condition(y_int, target_shape)

        # Should be expanded to (batch_size, h, w, condition_dim)
        assert reshaped.shape == (batch_size, 8, 8, condition_dim)

    def test_encode_with_condition(self, rngs, cvae_components):
        """Test conditional encoding."""
        config = cvae_components["config"]
        latent_dim = cvae_components["latent_dim"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]
        y_int = cvae_components["y_int"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Test encode with one-hot condition
        mean1, log_var1 = cvae.encode(x, y=y_onehot)

        # Check shapes
        assert mean1.shape == (x.shape[0], latent_dim)
        assert log_var1.shape == (x.shape[0], latent_dim)

        # Test encode with integer condition
        mean2, log_var2 = cvae.encode(x, y=y_int)

        # Check shapes
        assert mean2.shape == (x.shape[0], latent_dim)
        assert log_var2.shape == (x.shape[0], latent_dim)

        # Test with default condition (None)
        mean3, log_var3 = cvae.encode(x)

        # Check shapes
        assert mean3.shape == (x.shape[0], latent_dim)
        assert log_var3.shape == (x.shape[0], latent_dim)

    def test_decode_with_condition(self, rngs, cvae_components):
        """Test conditional decoding."""
        config = cvae_components["config"]
        latent_dim = cvae_components["latent_dim"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]
        y_int = cvae_components["y_int"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Create latent vectors for testing
        z = jnp.ones((x.shape[0], latent_dim))

        # Test decode with one-hot condition
        reconstructed1 = cvae.decode(z, y=y_onehot)
        assert reconstructed1.shape == x.shape

        # Test decode with integer condition
        reconstructed2 = cvae.decode(z, y=y_int)
        assert reconstructed2.shape == x.shape

        # Test with default condition (None)
        reconstructed3 = cvae.decode(z)
        assert reconstructed3.shape == x.shape

    def test_forward_pass_with_condition(self, rngs, cvae_components):
        """Test forward pass with conditioning."""
        config = cvae_components["config"]
        latent_dim = cvae_components["latent_dim"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]
        y_int = cvae_components["y_int"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Test with one-hot condition
        outputs1 = cvae(x, y=y_onehot)
        assert outputs1["reconstructed"].shape == x.shape
        assert outputs1["mean"].shape == (x.shape[0], latent_dim)
        assert outputs1["log_var"].shape == (x.shape[0], latent_dim)
        assert outputs1["z"].shape == (x.shape[0], latent_dim)

        # Test with integer condition
        outputs2 = cvae(x, y=y_int)
        assert outputs2["reconstructed"].shape == x.shape

        # Test with no condition
        outputs3 = cvae(x)
        assert outputs3["reconstructed"].shape == x.shape

    def test_sample_with_condition(self, rngs, cvae_components):
        """Test conditional sampling."""
        config = cvae_components["config"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]
        y_int = cvae_components["y_int"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Number of samples
        n_samples = 3

        # Test sample with one-hot condition (uses self.rngs internally)
        samples1 = cvae.sample(n_samples, y=y_onehot[:1])
        assert samples1.shape == (n_samples, x.shape[1])

        # Test sample with integer condition
        samples2 = cvae.sample(n_samples, y=y_int[:1])
        assert samples2.shape == (n_samples, x.shape[1])

        # Test with no condition
        samples3 = cvae.sample(n_samples)
        assert samples3.shape == (n_samples, x.shape[1])

    def test_different_condition_types(self, rngs, cvae_components):
        """Test different condition types in Conditional VAE."""
        input_dim = cvae_components["input_dim"]
        latent_dim = cvae_components["latent_dim"]
        condition_dim = cvae_components["condition_dim"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]

        # Create config with "concat" type
        encoder_config = EncoderConfig(
            name="test_encoder",
            hidden_dims=(64, 32),
            activation="relu",
            input_shape=(input_dim,),
            latent_dim=latent_dim,
        )
        decoder_config = DecoderConfig(
            name="test_decoder",
            hidden_dims=(32, 64),
            activation="relu",
            output_shape=(input_dim,),
            latent_dim=latent_dim,
        )
        config = ConditionalVAEConfig(
            name="test_cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            num_classes=condition_dim,  # num_classes is the required field
            condition_type="concat",
        )

        cvae_concat = ConditionalVAE(config=config, rngs=rngs)
        outputs_concat = cvae_concat(x, y=y_onehot)
        assert outputs_concat["reconstructed"].shape == x.shape


class TestConditionalVAEImageInput:
    """Test ConditionalVAE with image-like inputs (multi-dimensional input_shape)."""

    @pytest.fixture
    def image_cvae_config(self):
        """Fixture for ConditionalVAEConfig with image input shape."""
        # MNIST-like image dimensions
        input_shape = (28, 28, 1)
        latent_dim = 20
        num_classes = 10

        encoder_config = EncoderConfig(
            name="image_encoder",
            hidden_dims=(512, 256),
            activation="relu",
            input_shape=input_shape,
            latent_dim=latent_dim,
        )

        decoder_config = DecoderConfig(
            name="image_decoder",
            hidden_dims=(256, 512),
            activation="relu",
            output_shape=input_shape,
            latent_dim=latent_dim,
        )

        return ConditionalVAEConfig(
            name="image_cvae",
            encoder=encoder_config,
            decoder=decoder_config,
            encoder_type="dense",
            num_classes=num_classes,
            condition_type="concat",
        )

    def test_image_cvae_initialization(self, rngs, image_cvae_config):
        """Test ConditionalVAE initializes correctly with image input shape."""
        cvae = ConditionalVAE(config=image_cvae_config, rngs=rngs)

        assert cvae.encoder is not None
        assert cvae.decoder is not None
        assert cvae.latent_dim == 20
        assert cvae.condition_dim == 10

    def test_image_cvae_forward_pass(self, rngs, image_cvae_config):
        """Test ConditionalVAE forward pass with image inputs.

        This test specifically catches the bug where create_encoder() only
        handles 1D input_shape for conditional encoders, causing shape mismatch
        when using image-like input shapes (e.g., (28, 28, 1)).

        Bug: When input_shape has len > 1, the encoder is not adjusted for
        the additional condition dimensions, leading to:
        'dot_general requires contracting dimensions to have the same shape'
        """
        cvae = ConditionalVAE(config=image_cvae_config, rngs=rngs)

        # Create image batch
        batch_size = 4
        x = jnp.ones((batch_size, 28, 28, 1))

        # Create integer labels
        y = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

        # Forward pass should work without shape mismatch errors
        outputs = cvae(x, y=y)

        # Check output shapes
        assert outputs["reconstructed"].shape == x.shape
        assert outputs["mean"].shape == (batch_size, 20)
        assert outputs["log_var"].shape == (batch_size, 20)
        assert outputs["z"].shape == (batch_size, 20)

    def test_image_cvae_encode(self, rngs, image_cvae_config):
        """Test ConditionalVAE encoding with image inputs."""
        cvae = ConditionalVAE(config=image_cvae_config, rngs=rngs)

        batch_size = 4
        x = jnp.ones((batch_size, 28, 28, 1))
        x_flat = x.reshape(batch_size, -1)  # (4, 784)
        y = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

        # Encode should handle the flattened input + condition correctly
        mean, log_var = cvae.encode(x_flat, y=y)

        assert mean.shape == (batch_size, 20)
        assert log_var.shape == (batch_size, 20)

    def test_image_cvae_decode(self, rngs, image_cvae_config):
        """Test ConditionalVAE decoding with image inputs."""
        cvae = ConditionalVAE(config=image_cvae_config, rngs=rngs)

        batch_size = 4
        z = jnp.ones((batch_size, 20))
        y = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

        # Decode should work with latent + condition
        reconstructed = cvae.decode(z, y=y)

        # Output should be flattened (MLPDecoder with dense encoder_type)
        assert reconstructed.shape[0] == batch_size

    def test_image_cvae_sample(self, rngs, image_cvae_config):
        """Test ConditionalVAE sampling with image inputs."""
        cvae = ConditionalVAE(config=image_cvae_config, rngs=rngs)

        n_samples = 4
        # Single condition to broadcast
        y = jnp.array([5], dtype=jnp.int32)

        samples = cvae.sample(n_samples, y=y)

        # Should generate n_samples images
        assert samples.shape[0] == n_samples


class TestConditionalVAEJITCompatibility:
    """Comprehensive JIT compatibility tests for ConditionalVAE."""

    def test_cvae_jit_forward_pass(self, rngs, cvae_components):
        """Test that ConditionalVAE forward pass can be JIT compiled."""
        config = cvae_components["config"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x, y):
            return model(x, y=y)

        # Test JIT compilation with condition
        output = forward(cvae, x, y_onehot)

        # Check outputs
        assert "reconstructed" in output
        assert "mean" in output
        assert output["reconstructed"].shape == x.shape

    def test_cvae_jit_encode_with_condition(self, rngs, cvae_components):
        """Test that ConditionalVAE encode with condition can be JIT compiled."""
        config = cvae_components["config"]
        latent_dim = cvae_components["latent_dim"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]
        y_int = cvae_components["y_int"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        @nnx.jit
        def encode_fn(model, x, y):
            return model.encode(x, y=y)

        # Test with one-hot condition
        mean1, log_var1 = encode_fn(cvae, x, y_onehot)
        assert mean1.shape == (x.shape[0], latent_dim)
        assert log_var1.shape == (x.shape[0], latent_dim)

        # Test with integer condition
        mean2, log_var2 = encode_fn(cvae, x, y_int)
        assert mean2.shape == (x.shape[0], latent_dim)
        assert log_var2.shape == (x.shape[0], latent_dim)

    def test_cvae_jit_decode_with_condition(self, rngs, cvae_components):
        """Test that ConditionalVAE decode with condition can be JIT compiled."""
        config = cvae_components["config"]
        latent_dim = cvae_components["latent_dim"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        z = jnp.ones((x.shape[0], latent_dim))

        @nnx.jit
        def decode_fn(model, z, y):
            return model.decode(z, y=y)

        reconstructed = decode_fn(cvae, z, y_onehot)

        # Check shape
        assert reconstructed.shape == x.shape
        assert jnp.isfinite(reconstructed).all()

    def test_cvae_jit_sample_with_condition(self, rngs, cvae_components):
        """Test that ConditionalVAE sample with condition can be JIT compiled."""
        config = cvae_components["config"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Mark n_samples as static for JIT compatibility
        @nnx.jit(static_argnums=(1,))
        def sample_fn(model, n_samples, y):
            return model.sample(n_samples, y=y)

        n_samples = 3
        samples = sample_fn(cvae, n_samples, y_onehot[:1])

        # Check shape
        assert samples.shape == (n_samples, x.shape[1])
        assert jnp.isfinite(samples).all()

    def test_cvae_jit_reshape_condition(self, rngs, cvae_components):
        """Test that ConditionalVAE _reshape_condition can be JIT compiled."""
        config = cvae_components["config"]
        condition_dim = cvae_components["condition_dim"]
        y_int = cvae_components["y_int"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        # Mark target_shape as static for JIT compatibility (as documented in _reshape_condition)
        def reshape_fn(model, y, target_shape):
            return model._reshape_condition(y, target_shape)

        reshape_fn_jit = jax.jit(reshape_fn, static_argnums=(2,))

        # Test reshaping for different target shapes
        batch_size = y_int.shape[0]
        target_shape = (batch_size, 10, 10)
        reshaped = reshape_fn_jit(cvae, y_int, target_shape)

        # Check shape
        assert reshaped.shape == (batch_size, 10, 10, condition_dim)
        assert jnp.isfinite(reshaped).all()

    def test_cvae_jit_with_different_batch_sizes(self, rngs, cvae_components):
        """Test ConditionalVAE JIT compilation with different batch sizes."""
        config = cvae_components["config"]
        condition_dim = cvae_components["condition_dim"]
        input_dim = cvae_components["input_dim"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x, y):
            return model(x, y=y)

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            x = jnp.ones((batch_size, input_dim))
            y = jnp.zeros((batch_size, condition_dim))
            y = y.at[:, 0].set(1)  # One-hot encoding

            output = forward(cvae, x, y)

            assert output["reconstructed"].shape == (batch_size, input_dim)
            assert not jnp.isnan(output["reconstructed"]).any()

    def test_cvae_jit_without_condition(self, rngs, cvae_components):
        """Test ConditionalVAE JIT compilation without providing condition."""
        config = cvae_components["config"]
        x = cvae_components["x"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test JIT compilation without condition
        output = forward(cvae, x)

        # Check outputs
        assert output["reconstructed"].shape == x.shape
        assert jnp.isfinite(output["reconstructed"]).all()

    def test_cvae_jit_end_to_end(self, rngs, cvae_components):
        """Test end-to-end ConditionalVAE pipeline with JIT compilation."""
        config = cvae_components["config"]
        x = cvae_components["x"]
        y_onehot = cvae_components["y_onehot"]

        # Initialize CVAE
        cvae = ConditionalVAE(config=config, rngs=rngs)

        @jax.jit
        def train_step(model, x, y):
            # Forward pass
            outputs = model(x, y=y)
            # Compute reconstruction loss
            recon_loss = jnp.mean((x - outputs["reconstructed"]) ** 2)
            return recon_loss, outputs

        # Run training step
        loss, outputs = train_step(cvae, x, y_onehot)

        # Check outputs
        assert jnp.isfinite(loss)
        assert outputs["reconstructed"].shape == x.shape
        assert jnp.isfinite(outputs["reconstructed"]).all()
