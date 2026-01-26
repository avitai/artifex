"""Unit tests for base VAE implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import VAEConfig
from artifex.generative_models.models.vae.base import VAE


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
def vae_config():
    """Fixture for VAEConfig."""
    input_dim = 100
    latent_dim = 10

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

    return VAEConfig(
        name="test_vae",
        encoder=encoder_config,
        decoder=decoder_config,
        kl_weight=1.0,
    )


@pytest.fixture
def vae_components(rngs, vae_config):
    """Fixture for VAE and test data."""
    input_dim = 100
    latent_dim = 10
    batch_size = 2

    # Create sample input
    x = jnp.ones((batch_size, input_dim))

    return {
        "config": vae_config,
        "x": x,
        "latent_dim": latent_dim,
        "input_dim": input_dim,
        "batch_size": batch_size,
    }


class TestVAE:
    """Test suite for the base VAE class."""

    def test_initialization(self, rngs, vae_components):
        """Test VAE initialization."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Check attributes are correctly set
        assert vae.latent_dim == latent_dim
        assert vae.encoder is not None
        assert vae.decoder is not None

    def test_encode(self, rngs, vae_components):
        """Test VAE encode method."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Test encode method
        mean, log_var = vae.encode(x)

        # Check shapes
        assert mean.shape == (x.shape[0], latent_dim)
        assert log_var.shape == (x.shape[0], latent_dim)

    def test_decode(self, rngs, vae_components):
        """Test VAE decode method."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Create latent vectors
        z = jnp.ones((x.shape[0], latent_dim))

        # Test decode method
        reconstructed = vae.decode(z)

        # Check shape
        assert reconstructed.shape == x.shape

    def test_reparameterize(self, rngs, vae_components):
        """Test VAE reparameterize method."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Create mean and log_var
        mean = jnp.zeros((x.shape[0], latent_dim))
        log_var = jnp.zeros((x.shape[0], latent_dim))

        # Test reparameterize method
        z = vae.reparameterize(mean, log_var)

        # Check shape
        assert z.shape == (x.shape[0], latent_dim)

    def test_numerical_stability(self, rngs, vae_components):
        """Test VAE handles extreme values without numerical issues."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Test with very large log_var values
        mean = jnp.zeros((2, latent_dim))
        large_log_var = jnp.ones((2, latent_dim)) * 100.0

        # This should not produce NaN or Inf due to clipping
        z = vae.reparameterize(mean, large_log_var)
        assert not jnp.any(jnp.isnan(z))
        assert not jnp.any(jnp.isinf(z))

        # Test with very small log_var values
        small_log_var = jnp.ones((2, latent_dim)) * -100.0
        z = vae.reparameterize(mean, small_log_var)
        assert not jnp.any(jnp.isnan(z))
        assert not jnp.any(jnp.isinf(z))

    def test_forward_pass(self, rngs, vae_components):
        """Test VAE forward pass."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        outputs = vae(x)

        # Check outputs
        assert "reconstructed" in outputs
        assert "mean" in outputs
        assert "log_var" in outputs
        assert "z" in outputs

        # Check shapes
        assert outputs["reconstructed"].shape == x.shape
        assert outputs["mean"].shape == (x.shape[0], latent_dim)
        assert outputs["log_var"].shape == (x.shape[0], latent_dim)
        assert outputs["z"].shape == (x.shape[0], latent_dim)

    def test_loss_function(self, rngs, vae_components):
        """Test VAE loss function."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Forward pass to get outputs
        outputs = vae(x)

        # Test loss function
        losses = vae.loss_fn(x=x, outputs=outputs)

        # Check loss components
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "loss" in losses

        # Check values are reasonable
        assert not jnp.isnan(losses["reconstruction_loss"])
        assert not jnp.isnan(losses["kl_loss"])
        assert not jnp.isnan(losses["loss"])

        # Test with custom beta
        beta = 0.5
        custom_losses = vae.loss_fn(x=x, outputs=outputs, beta=beta)

        # Check beta affects the total loss
        expected_loss = custom_losses["reconstruction_loss"] + beta * custom_losses["kl_loss"]
        assert jnp.isclose(custom_losses["loss"], expected_loss)

    def test_custom_reconstruction_loss(self, rngs, vae_components):
        """Test VAE with custom reconstruction loss function."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Run forward pass
        outputs = vae(x)

        # Define custom loss function (JAX/Optax convention: predictions, targets)
        def custom_loss_fn(predictions, targets):
            return jnp.mean(jnp.abs(predictions - targets))

        # Calculate losses with custom function
        losses = vae.loss_fn(x=x, outputs=outputs, reconstruction_loss_fn=custom_loss_fn)

        # Verify loss is reasonable
        assert not jnp.isnan(losses["loss"])
        assert losses["reconstruction_loss"] >= 0.0

        # Verify using a different loss function changes the reconstruction loss value
        default_losses = vae.loss_fn(x=x, outputs=outputs)
        # The values should be different unless data is exactly 0 or 1
        if not jnp.all((x == 0) | (x == 1)):
            assert not jnp.isclose(
                losses["reconstruction_loss"], default_losses["reconstruction_loss"]
            )

    def test_sample(self, rngs, vae_components):
        """Test VAE sample method."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Test sample method
        n_samples = 5
        samples = vae.sample(n_samples)

        # Check shape
        assert samples.shape == (n_samples, x.shape[1])

        # Test with temperature parameter
        hot_samples = vae.sample(n_samples, temperature=2.0)
        cold_samples = vae.sample(n_samples, temperature=0.1)

        # Higher temperature should generally result in more varied samples
        assert jnp.std(hot_samples) > jnp.std(cold_samples)

    def test_reconstruct(self, rngs, vae_components):
        """Test VAE reconstruct method."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Test regular reconstruct
        reconstructed = vae.reconstruct(x)
        assert reconstructed.shape == x.shape

        # Test deterministic reconstruct
        deterministic = vae.reconstruct(x, deterministic=True)
        assert deterministic.shape == x.shape

        # Multiple deterministic reconstructions should be identical
        det1 = vae.reconstruct(x, deterministic=True)
        det2 = vae.reconstruct(x, deterministic=True)
        assert jnp.allclose(det1, det2)

        # Non-deterministic reconstructions should generally be different
        nondet1 = vae.reconstruct(x, deterministic=False)
        nondet2 = vae.reconstruct(x, deterministic=False)
        assert not jnp.allclose(nondet1, nondet2)

    def test_generate(self, rngs, vae_components):
        """Test VAE generate method from GenerativeModel base class."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Create sample RNG
        params_key = rngs["params"].key.value
        sample_rngs = nnx.Rngs(sample=params_key)

        n_samples = 3
        samples = vae.generate(n_samples, rngs=sample_rngs)

        # Check shape
        assert samples.shape == (n_samples, x.shape[1])

        # Test with temperature
        samples_hot = vae.generate(n_samples, temperature=2.0, rngs=sample_rngs)
        assert samples_hot.shape == (n_samples, x.shape[1])

    def test_interpolate(self, rngs, vae_components):
        """Test VAE interpolate method."""
        config = vae_components["config"]
        input_dim = vae_components["input_dim"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Create two test inputs
        x1 = jnp.ones(input_dim)
        x2 = jnp.zeros(input_dim)

        # Test interpolate method
        steps = 5
        interp = vae.interpolate(x1, x2, steps=steps)

        # Check shape
        assert interp.shape == (steps, input_dim)

        # Check first and last steps are close to reconstructions of original inputs
        recon1 = vae.reconstruct(x1[None, ...], deterministic=True)
        recon2 = vae.reconstruct(x2[None, ...], deterministic=True)

        assert jnp.allclose(interp[0], recon1[0], atol=1e-4)
        assert jnp.allclose(interp[-1], recon2[0], atol=1e-4)

    def test_latent_traversal(self, rngs, vae_components):
        """Test VAE latent traversal method."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        input_dim = vae_components["input_dim"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Create test input
        x = jnp.ones(input_dim)

        # Test traversal method
        steps = 7
        dim = 2
        traversal = vae.latent_traversal(x, dim, range_vals=(-2.0, 2.0), steps=steps)

        # Check shape
        assert traversal.shape == (steps, input_dim)

        # Test invalid dimension
        with pytest.raises(ValueError, match="Dimension .* out of range"):
            vae.latent_traversal(x, latent_dim + 5, steps=steps)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    @pytest.mark.parametrize("input_dim", [50, 100, 200])
    @pytest.mark.parametrize("latent_dim", [5, 10, 20])
    def test_vae_shapes(self, rngs, batch_size, input_dim, latent_dim):
        """Test VAE handles different shapes correctly."""
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
        config = VAEConfig(
            name="test_vae",
            encoder=encoder_config,
            decoder=decoder_config,
        )

        # Create sample input
        x = jnp.ones((batch_size, input_dim))

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Forward pass
        outputs = vae(x)

        # Check shapes
        assert outputs["reconstructed"].shape == (batch_size, input_dim)
        assert outputs["mean"].shape == (batch_size, latent_dim)
        assert outputs["log_var"].shape == (batch_size, latent_dim)
        assert outputs["z"].shape == (batch_size, latent_dim)


class TestCNNVAE:
    """Test suite for CNN-based VAE architecture."""

    @pytest.fixture
    def cnn_vae_config_2layer(self):
        """Fixture for 2-layer CNN VAE config (standard for 28x28)."""
        return VAEConfig(
            name="cnn_vae_2layer",
            encoder=EncoderConfig(
                name="cnn_encoder",
                input_shape=(28, 28, 1),
                latent_dim=16,
                hidden_dims=(32, 64),
                activation="relu",
                use_batch_norm=False,
            ),
            decoder=DecoderConfig(
                name="cnn_decoder",
                latent_dim=16,
                output_shape=(28, 28, 1),
                hidden_dims=(32, 64),
                activation="relu",
                batch_norm=False,
            ),
            encoder_type="cnn",
            kl_weight=1.0,
        )

    @pytest.fixture
    def cnn_vae_config_3layer(self):
        """Fixture for 3-layer CNN VAE config (tests dimension handling)."""
        return VAEConfig(
            name="cnn_vae_3layer",
            encoder=EncoderConfig(
                name="cnn_encoder",
                input_shape=(28, 28, 1),
                latent_dim=20,
                hidden_dims=(32, 64, 128),
                activation="relu",
                use_batch_norm=False,
            ),
            decoder=DecoderConfig(
                name="cnn_decoder",
                latent_dim=20,
                output_shape=(28, 28, 1),
                hidden_dims=(32, 64, 128),
                activation="relu",
                batch_norm=False,
            ),
            encoder_type="cnn",
            kl_weight=1.0,
        )

    def test_cnn_vae_2layer_forward(self, rngs, cnn_vae_config_2layer):
        """Test 2-layer CNN VAE forward pass with 28x28 input."""
        vae = VAE(config=cnn_vae_config_2layer, rngs=rngs)
        x = jnp.ones((2, 28, 28, 1))

        outputs = vae(x)

        assert outputs["reconstructed"].shape == (2, 28, 28, 1)
        assert outputs["mean"].shape == (2, 16)
        assert outputs["log_var"].shape == (2, 16)
        assert jnp.isfinite(outputs["reconstructed"]).all()

    def test_cnn_vae_3layer_forward(self, rngs, cnn_vae_config_3layer):
        """Test 3-layer CNN VAE forward pass with 28x28 input.

        This tests the dimension handling fix for non-power-of-2 sizes.
        With 3 stride-2 layers: 28->14->7->4 (encoder), 4->8->16->32 (decoder)
        The decoder must resize output to match target 28x28.
        """
        vae = VAE(config=cnn_vae_config_3layer, rngs=rngs)
        x = jnp.ones((2, 28, 28, 1))

        outputs = vae(x)

        # Output should match input shape exactly
        assert outputs["reconstructed"].shape == (2, 28, 28, 1)
        assert outputs["mean"].shape == (2, 20)
        assert outputs["log_var"].shape == (2, 20)
        assert jnp.isfinite(outputs["reconstructed"]).all()

    def test_cnn_vae_3layer_jit(self, rngs, cnn_vae_config_3layer):
        """Test 3-layer CNN VAE is JIT compatible."""
        vae = VAE(config=cnn_vae_config_3layer, rngs=rngs)
        x = jnp.ones((2, 28, 28, 1))

        @nnx.jit
        def forward(model, x):
            return model(x)

        outputs = forward(vae, x)

        assert outputs["reconstructed"].shape == (2, 28, 28, 1)
        assert jnp.isfinite(outputs["reconstructed"]).all()

    @pytest.mark.parametrize(
        "input_shape,hidden_dims",
        [
            ((28, 28, 1), (32, 64)),  # MNIST with 2 layers
            ((28, 28, 1), (32, 64, 128)),  # MNIST with 3 layers
            ((32, 32, 1), (32, 64)),  # Power-of-2 with 2 layers
            ((32, 32, 3), (32, 64, 128)),  # RGB with 3 layers
        ],
    )
    def test_cnn_vae_various_sizes(self, rngs, input_shape, hidden_dims):
        """Test CNN VAE with various input sizes and depths."""
        config = VAEConfig(
            name="cnn_vae_test",
            encoder=EncoderConfig(
                name="encoder",
                input_shape=input_shape,
                latent_dim=16,
                hidden_dims=hidden_dims,
                activation="relu",
            ),
            decoder=DecoderConfig(
                name="decoder",
                latent_dim=16,
                output_shape=input_shape,
                hidden_dims=hidden_dims,
                activation="relu",
            ),
            encoder_type="cnn",
        )

        vae = VAE(config=config, rngs=rngs)
        x = jnp.ones((2, *input_shape))

        outputs = vae(x)

        assert outputs["reconstructed"].shape == x.shape
        assert jnp.isfinite(outputs["reconstructed"]).all()

    def test_cnn_vae_sample(self, rngs, cnn_vae_config_3layer):
        """Test sampling from 3-layer CNN VAE."""
        vae = VAE(config=cnn_vae_config_3layer, rngs=rngs)

        samples = vae.sample(n_samples=4)

        assert samples.shape == (4, 28, 28, 1)
        assert jnp.isfinite(samples).all()

    def test_cnn_vae_reconstruct(self, rngs, cnn_vae_config_3layer):
        """Test reconstruction with 3-layer CNN VAE."""
        vae = VAE(config=cnn_vae_config_3layer, rngs=rngs)
        x = jnp.ones((2, 28, 28, 1))

        reconstructed = vae.reconstruct(x, deterministic=True)

        assert reconstructed.shape == x.shape
        assert jnp.isfinite(reconstructed).all()


class TestVAEJITCompatibility:
    """Comprehensive JIT compatibility tests for VAE."""

    def test_vae_jit_forward_pass(self, rngs, vae_components):
        """Test that VAE forward pass can be JIT compiled."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # First call (should trigger compilation)
        output1 = forward(vae, x)

        # Second call (should use cached compilation)
        forward(vae, x)

        # Both outputs should have correct structure
        assert "reconstructed" in output1
        assert "mean" in output1
        assert "log_var" in output1
        assert "z" in output1

        # Check shapes
        assert output1["reconstructed"].shape == x.shape
        assert output1["mean"].shape == (x.shape[0], latent_dim)

    def test_vae_jit_encode(self, rngs, vae_components):
        """Test that VAE encode method can be JIT compiled."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        @nnx.jit
        def encode_fn(model, x):
            return model.encode(x)

        mean, log_var = encode_fn(vae, x)

        # Check shapes
        assert mean.shape == (x.shape[0], latent_dim)
        assert log_var.shape == (x.shape[0], latent_dim)
        assert jnp.isfinite(mean).all()
        assert jnp.isfinite(log_var).all()

    def test_vae_jit_decode(self, rngs, vae_components):
        """Test that VAE decode method can be JIT compiled."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        z = jnp.ones((x.shape[0], latent_dim))

        @nnx.jit
        def decode_fn(model, z):
            return model.decode(z)

        reconstructed = decode_fn(vae, z)

        # Check shape
        assert reconstructed.shape == x.shape
        assert jnp.isfinite(reconstructed).all()

    def test_vae_jit_reparameterize(self, rngs, vae_components):
        """Test that VAE reparameterize method can be JIT compiled."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        mean = jnp.zeros((x.shape[0], latent_dim))
        log_var = jnp.zeros((x.shape[0], latent_dim))

        @nnx.jit
        def reparameterize_fn(model, mean, log_var):
            return model.reparameterize(mean, log_var)

        z = reparameterize_fn(vae, mean, log_var)

        # Check shape
        assert z.shape == (x.shape[0], latent_dim)
        assert jnp.isfinite(z).all()

    def test_vae_jit_sample(self, rngs, vae_components):
        """Test that VAE sample method can be JIT compiled."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        @nnx.jit(static_argnums=(1,))
        def sample_fn(model, n_samples):
            return model.sample(n_samples)

        n_samples = 5
        samples = sample_fn(vae, n_samples)

        # Check shape
        assert samples.shape == (n_samples, x.shape[1])
        assert jnp.isfinite(samples).all()

    def test_vae_jit_with_different_batch_sizes(self, rngs, vae_components):
        """Test VAE JIT compilation with different batch sizes."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        input_dim = vae_components["input_dim"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test with different batch sizes (each triggers recompilation)
        for batch_size in [1, 2, 4, 8]:
            x = jnp.ones((batch_size, input_dim))
            output = forward(vae, x)

            assert output["reconstructed"].shape == (batch_size, input_dim)
            assert output["mean"].shape == (batch_size, latent_dim)
            assert not jnp.isnan(output["reconstructed"]).any()

    def test_vae_jit_gradient_computation(self, rngs, vae_components):
        """Test that VAE gradient computation can be JIT compiled."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        @jax.jit
        def loss_fn(model, x):
            outputs = model(x)
            losses = model.loss_fn(x=x, outputs=outputs)
            return losses["loss"]

        # Compute gradients using nnx.grad
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(vae, x)

        # Check that gradients were computed
        assert grads is not None

    def test_vae_jit_loss_function(self, rngs, vae_components):
        """Test that VAE loss function can be JIT compiled."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        # Get outputs first
        outputs = vae(x)

        @jax.jit
        def compute_loss(model, x, outputs):
            return model.loss_fn(x=x, outputs=outputs)

        losses = compute_loss(vae, x, outputs)

        # Check loss components
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "loss" in losses
        assert jnp.isfinite(losses["loss"])

    def test_vae_jit_reconstruct(self, rngs, vae_components):
        """Test that VAE reconstruct method can be JIT compiled."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        @nnx.jit
        def reconstruct_fn(model, x):
            return model.reconstruct(x)

        reconstructed = reconstruct_fn(vae, x)

        # Check shape
        assert reconstructed.shape == x.shape
        assert jnp.isfinite(reconstructed).all()

    def test_vae_jit_interpolate(self, rngs, vae_components):
        """Test that VAE interpolate method can be JIT compiled."""
        config = vae_components["config"]
        input_dim = vae_components["input_dim"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        x1 = jnp.ones(input_dim)
        x2 = jnp.zeros(input_dim)
        steps = 5

        # Mark steps as static for JIT compatibility (as documented in VAE.interpolate)
        @nnx.jit(static_argnums=(3,))
        def interpolate_fn(model, x1, x2, steps):
            return model.interpolate(x1, x2, steps=steps)

        interp = interpolate_fn(vae, x1, x2, steps)

        # Check shape
        assert interp.shape == (steps, input_dim)
        assert jnp.isfinite(interp).all()

    def test_vae_jit_end_to_end(self, rngs, vae_components):
        """Test end-to-end VAE pipeline with JIT compilation."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize VAE
        vae = VAE(config=config, rngs=rngs)

        @jax.jit
        def train_step(model, x):
            # Forward pass
            outputs = model(x)
            # Compute loss
            losses = model.loss_fn(x=x, outputs=outputs)
            return losses["loss"], outputs

        # Run training step
        loss, outputs = train_step(vae, x)

        # Check outputs
        assert jnp.isfinite(loss)
        assert outputs["reconstructed"].shape == x.shape
        assert jnp.isfinite(outputs["reconstructed"]).all()
