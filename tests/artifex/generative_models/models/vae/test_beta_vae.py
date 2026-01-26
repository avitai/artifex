"""Unit tests for Beta-VAE implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.network_configs import (
    DecoderConfig,
    EncoderConfig,
)
from artifex.generative_models.core.configuration.vae_config import (
    BetaVAEConfig,
    BetaVAEWithCapacityConfig,
)
from artifex.generative_models.models.vae.beta_vae import BetaVAE, BetaVAEWithCapacity


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
def beta_vae_config():
    """Fixture for BetaVAEConfig."""
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

    return BetaVAEConfig(
        name="test_beta_vae",
        encoder=encoder_config,
        decoder=decoder_config,
        beta_default=2.0,
        beta_warmup_steps=0,
        reconstruction_loss_type="mse",
    )


@pytest.fixture
def vae_components(beta_vae_config):
    """Fixture for BetaVAE config and test data."""
    input_dim = 100
    latent_dim = 10
    batch_size = 2

    # Create sample input
    x = jnp.ones((batch_size, input_dim))

    return {
        "config": beta_vae_config,
        "x": x,
        "latent_dim": latent_dim,
        "input_dim": input_dim,
        "batch_size": batch_size,
    }


class TestBetaVAE:
    """Test suite for the Beta-VAE class."""

    def test_initialization(self, rngs, vae_components):
        """Test Beta-VAE initialization."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]

        # Create config with specific beta settings
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
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=2.0,
            beta_warmup_steps=100,
            reconstruction_loss_type="mse",
        )

        # Standard initialization
        beta_vae = BetaVAE(config=config, rngs=rngs)

        # Check attributes
        assert beta_vae.encoder is not None
        assert beta_vae.decoder is not None
        assert beta_vae.latent_dim == latent_dim
        assert beta_vae.beta_default == 2.0
        assert beta_vae.beta_warmup_steps == 100
        assert beta_vae.reconstruction_loss_type == "mse"

        # Test with different reconstruction loss type
        config_bce = BetaVAEConfig(
            name="test_beta_vae_bce",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=1.5,
            reconstruction_loss_type="bce",
        )
        beta_vae_bce = BetaVAE(config=config_bce, rngs=rngs)
        assert beta_vae_bce.reconstruction_loss_type == "bce"

    def test_loss_function_default_beta(self, rngs, vae_components):
        """Test Beta-VAE loss function with default beta."""
        config = vae_components["config"]
        x = vae_components["x"]

        # Initialize BetaVAE with default beta of 2.0
        beta_value = config.beta_default
        beta_vae = BetaVAE(config=config, rngs=rngs)

        outputs = beta_vae(x)

        # Test loss function with default beta (should use beta_default)
        losses = beta_vae.loss_fn(x=x, outputs=outputs)

        # Check loss components
        assert "loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "beta" in losses

        # Check beta is correctly applied
        assert jnp.isclose(losses["beta"], beta_value)

        # Verify total loss calculation
        expected_total_loss = losses["reconstruction_loss"] + beta_value * losses["kl_loss"]
        assert jnp.isclose(losses["loss"], expected_total_loss)

    def test_loss_function_custom_beta(self, rngs, vae_components):
        """Test Beta-VAE loss function with custom beta."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Create config with beta_default=1.0
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
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=1.0,
        )

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Test loss function with custom beta
        custom_beta = 2.5
        losses = beta_vae.loss_fn(x=x, outputs=outputs, beta=custom_beta)

        # Check beta is correctly applied
        assert jnp.isclose(losses["beta"], custom_beta)

        # Verify total loss calculation
        expected_total_loss = losses["reconstruction_loss"] + custom_beta * losses["kl_loss"]
        assert jnp.isclose(losses["loss"], expected_total_loss)

    def test_beta_warmup(self, rngs, vae_components):
        """Test Beta-VAE beta annealing during warmup."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Create config with warmup
        beta_default = 4.0
        warmup_steps = 1000
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
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=beta_default,
            beta_warmup_steps=warmup_steps,
        )

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Test at different steps during warmup
        steps = [0, 250, 500, 750, 1000, 1500]

        for step in steps:
            losses = beta_vae.loss_fn(x=x, outputs=outputs, step=step)

            # Expected beta value based on linear annealing
            expected_beta = min(beta_default, beta_default * step / warmup_steps)

            # Check beta is correctly annealed
            assert jnp.isclose(losses["beta"], expected_beta)

            # Verify total loss calculation
            expected_total_loss = losses["reconstruction_loss"] + expected_beta * losses["kl_loss"]
            assert jnp.isclose(losses["loss"], expected_total_loss)

    def test_bce_loss(self, rngs, vae_components):
        """Test Beta-VAE with BCE reconstruction loss."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Create config with BCE loss
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
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=1.0,
            reconstruction_loss_type="bce",
        )

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Test loss function
        losses = beta_vae.loss_fn(x=x, outputs=outputs)

        # Check loss components
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "loss" in losses

        # Check values are reasonable
        assert not jnp.isnan(losses["reconstruction_loss"])
        assert not jnp.isnan(losses["kl_loss"])
        assert not jnp.isnan(losses["loss"])

    def test_mse_loss(self, rngs, vae_components):
        """Test Beta-VAE with MSE reconstruction loss."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Test loss function
        losses = beta_vae.loss_fn(x=x, outputs=outputs)

        # Check loss components and values
        assert not jnp.isnan(losses["reconstruction_loss"])
        assert not jnp.isnan(losses["kl_loss"])
        assert not jnp.isnan(losses["loss"])

    def test_inherited_methods(self, rngs, vae_components):
        """Test methods inherited from VAE base class."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize BetaVAE
        beta_vae = BetaVAE(config=config, rngs=rngs)

        # Test encode method
        mean, log_var = beta_vae.encode(x)
        assert mean.shape == (x.shape[0], latent_dim)
        assert log_var.shape == (x.shape[0], latent_dim)

        # Test reparameterize method
        z = beta_vae.reparameterize(mean, log_var)
        assert z.shape == (x.shape[0], latent_dim)

        # Test decode method
        reconstructed = beta_vae.decode(z)
        assert reconstructed.shape == x.shape

        # Test sample method (uses self.rngs internally)
        n_samples = 5
        samples = beta_vae.sample(n_samples)
        assert samples.shape == (n_samples, x.shape[1])


class TestBetaVAEJITCompatibility:
    """Comprehensive JIT compatibility tests for BetaVAE."""

    def test_beta_vae_jit_forward_pass(self, rngs, vae_components):
        """Test that BetaVAE forward pass can be JIT compiled."""
        config = vae_components["config"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Initialize BetaVAE
        beta_vae = BetaVAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test JIT compilation
        output = forward(beta_vae, x)

        # Check outputs
        assert "reconstructed" in output
        assert "mean" in output
        assert "log_var" in output
        assert output["reconstructed"].shape == x.shape
        assert output["mean"].shape == (x.shape[0], latent_dim)

    def test_beta_vae_jit_loss_with_beta_warmup(self, rngs, vae_components):
        """Test BetaVAE loss function with beta warmup under JIT."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Create config with warmup
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
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
            beta_warmup_steps=1000,
        )

        beta_vae = BetaVAE(config=config, rngs=rngs)

        @jax.jit
        def compute_loss_at_step(model, x, step):
            outputs = model(x)
            return model.loss_fn(x=x, outputs=outputs, step=step)

        # Test at different steps
        for step in [0, 250, 500, 1000]:
            losses = compute_loss_at_step(beta_vae, x, step)
            assert "beta" in losses
            assert jnp.isfinite(losses["loss"])

    def test_beta_vae_jit_loss_with_custom_beta(self, rngs, vae_components):
        """Test BetaVAE loss function with custom beta under JIT."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)

        @jax.jit
        def compute_loss_with_beta(model, x, beta):
            outputs = model(x)
            return model.loss_fn(x=x, outputs=outputs, beta=beta)

        # Test with different beta values
        for beta_val in [0.5, 1.0, 2.0, 4.0]:
            losses = compute_loss_with_beta(beta_vae, x, beta_val)
            assert jnp.isclose(losses["beta"], beta_val)
            assert jnp.isfinite(losses["loss"])

    def test_beta_vae_jit_gradient_computation(self, rngs, vae_components):
        """Test that BetaVAE gradient computation can be JIT compiled."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)

        @jax.jit
        def loss_fn(model, x):
            outputs = model(x)
            losses = model.loss_fn(x=x, outputs=outputs)
            return losses["loss"]

        # Compute gradients using nnx.grad
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(beta_vae, x)

        # Check that gradients were computed
        assert grads is not None

    def test_beta_vae_jit_with_different_batch_sizes(self, rngs, vae_components):
        """Test BetaVAE JIT compilation with different batch sizes."""
        config = vae_components["config"]
        input_dim = vae_components["input_dim"]

        beta_vae = BetaVAE(config=config, rngs=rngs)

        @nnx.jit
        def forward(model, x):
            return model(x)

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            x = jnp.ones((batch_size, input_dim))
            output = forward(beta_vae, x)

            assert output["reconstructed"].shape == (batch_size, input_dim)
            assert not jnp.isnan(output["reconstructed"]).any()

    def test_beta_vae_jit_bce_loss(self, rngs, vae_components):
        """Test BetaVAE with BCE loss under JIT compilation."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Create config with BCE loss
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
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=1.0,
            reconstruction_loss_type="bce",
        )

        beta_vae = BetaVAE(config=config, rngs=rngs)

        @jax.jit
        def compute_loss(model, x):
            outputs = model(x)
            return model.loss_fn(x=x, outputs=outputs)

        losses = compute_loss(beta_vae, x)

        assert "reconstruction_loss" in losses
        assert jnp.isfinite(losses["loss"])

    def test_beta_vae_jit_end_to_end(self, rngs, vae_components):
        """Test end-to-end BetaVAE training pipeline with JIT compilation."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

        # Create config with warmup
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
        config = BetaVAEConfig(
            name="test_beta_vae",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=4.0,
            beta_warmup_steps=1000,
        )

        beta_vae = BetaVAE(config=config, rngs=rngs)

        @jax.jit
        def train_step(model, x, step):
            # Forward pass
            outputs = model(x)
            # Compute loss with warmup
            losses = model.loss_fn(x=x, outputs=outputs, step=step)
            return losses["loss"], losses["beta"], outputs

        # Run training steps with warmup
        for step in [0, 250, 500, 1000]:
            loss, beta, outputs = train_step(beta_vae, x, step)

            # Check outputs
            assert jnp.isfinite(loss)
            assert jnp.isfinite(beta)
            assert outputs["reconstructed"].shape == x.shape


class TestBetaVAEInputHandling:
    """Tests for BetaVAE input handling edge cases."""

    def test_loss_fn_with_batch_dict_inputs_key(self, rngs, vae_components):
        """Test loss_fn with batch dict containing 'inputs' key."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Pass x through batch dict with 'inputs' key
        batch = {"inputs": x}
        losses = beta_vae.loss_fn(batch=batch, outputs=outputs)

        assert "loss" in losses
        assert jnp.isfinite(losses["loss"])

    def test_loss_fn_with_batch_dict_no_inputs_key(self, rngs, vae_components):
        """Test loss_fn with batch dict without 'inputs' key (x treated as batch)."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Pass x directly as batch (without 'inputs' key)
        losses = beta_vae.loss_fn(batch=x, outputs=outputs)

        assert "loss" in losses
        assert jnp.isfinite(losses["loss"])

    def test_loss_fn_with_x_dict_inputs_key(self, rngs, vae_components):
        """Test loss_fn with x as dict containing 'inputs' key."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Pass x as dict with 'inputs' key
        x_dict = {"inputs": x}
        losses = beta_vae.loss_fn(x=x_dict, outputs=outputs)

        assert "loss" in losses
        assert jnp.isfinite(losses["loss"])

    def test_loss_fn_with_x_dict_input_key(self, rngs, vae_components):
        """Test loss_fn with x as dict containing 'input' key."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Pass x as dict with 'input' key (singular)
        x_dict = {"input": x}
        losses = beta_vae.loss_fn(x=x_dict, outputs=outputs)

        assert "loss" in losses
        assert jnp.isfinite(losses["loss"])

    def test_loss_fn_with_x_dict_invalid_keys_raises_error(self, rngs, vae_components):
        """Test loss_fn raises error when x is dict without valid keys."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)
        outputs = beta_vae(x)

        # Pass x as dict without recognized keys
        x_dict = {"data": x, "other": jnp.zeros_like(x)}

        with pytest.raises(ValueError, match="Input 'x' is a dictionary"):
            beta_vae.loss_fn(x=x_dict, outputs=outputs)

    def test_loss_fn_missing_reconstructed_key_raises_error(self, rngs, vae_components):
        """Test loss_fn raises KeyError when outputs missing 'reconstructed' key."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)

        # Create invalid outputs without 'reconstructed' key
        outputs = {
            "mean": jnp.zeros((x.shape[0], 10)),
            "log_var": jnp.zeros((x.shape[0], 10)),
        }

        with pytest.raises(KeyError, match="reconstructed"):
            beta_vae.loss_fn(x=x, outputs=outputs)

    def test_loss_fn_missing_log_var_key_raises_error(self, rngs, vae_components):
        """Test loss_fn raises KeyError when outputs missing 'log_var' key."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)

        # Create invalid outputs without 'log_var' key
        outputs = {
            "reconstructed": x,
            "mean": jnp.zeros((x.shape[0], 10)),
        }

        with pytest.raises(KeyError, match="log_var"):
            beta_vae.loss_fn(x=x, outputs=outputs)

    def test_loss_fn_outputs_none_calls_model(self, rngs, vae_components):
        """Test loss_fn with outputs=None calls model directly."""
        config = vae_components["config"]
        x = vae_components["x"]

        beta_vae = BetaVAE(config=config, rngs=rngs)

        # Call loss_fn without outputs - should call model internally
        losses = beta_vae.loss_fn(x=x, outputs=None)

        assert "loss" in losses
        assert jnp.isfinite(losses["loss"])


class TestBetaVAEWithCapacity:
    """Tests for BetaVAEWithCapacity model."""

    @pytest.fixture
    def capacity_config(self, vae_components):
        """Fixture for BetaVAEWithCapacityConfig."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]

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

        return BetaVAEWithCapacityConfig(
            name="test_beta_vae_capacity",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=1.0,
            beta_warmup_steps=0,
            reconstruction_loss_type="mse",
            use_capacity_control=True,
            capacity_max=25.0,
            capacity_num_iter=10000,
            gamma=1000.0,
        )

    def test_initialization(self, rngs, capacity_config):
        """Test BetaVAEWithCapacity initialization."""
        model = BetaVAEWithCapacity(config=capacity_config, rngs=rngs)

        assert model.use_capacity_control is True
        assert model.capacity_max == 25.0
        assert model.capacity_num_iter == 10000
        assert model.gamma == 1000.0

    def test_loss_with_capacity_control_enabled(self, rngs, capacity_config, vae_components):
        """Test loss function with capacity control enabled."""
        x = vae_components["x"]

        model = BetaVAEWithCapacity(config=capacity_config, rngs=rngs)
        outputs = model(x)

        # Compute loss at different steps
        for step in [0, 2500, 5000, 10000]:
            losses = model.loss_fn(x=x, outputs=outputs, step=step)

            assert "loss" in losses
            assert "reconstruction_loss" in losses
            assert "kl_loss" in losses
            assert "capacity_loss" in losses
            assert "current_capacity" in losses

            # Check values are finite
            assert jnp.isfinite(losses["loss"])
            assert jnp.isfinite(losses["capacity_loss"])
            assert jnp.isfinite(losses["current_capacity"])

    def test_capacity_increases_with_steps(self, rngs, capacity_config, vae_components):
        """Test that current_capacity increases with training steps."""
        x = vae_components["x"]

        model = BetaVAEWithCapacity(config=capacity_config, rngs=rngs)
        outputs = model(x)

        capacities = []
        steps = [0, 2500, 5000, 7500, 10000, 15000]

        for step in steps:
            losses = model.loss_fn(x=x, outputs=outputs, step=step)
            capacities.append(float(losses["current_capacity"]))

        # Capacity should increase until reaching max
        for i in range(len(capacities) - 1):
            if steps[i] < capacity_config.capacity_num_iter:
                assert capacities[i + 1] >= capacities[i]

        # After capacity_num_iter, capacity should be at max
        assert jnp.isclose(capacities[-1], capacity_config.capacity_max)

    def test_loss_without_capacity_control(self, rngs, vae_components):
        """Test loss function with capacity control disabled."""
        input_dim = vae_components["input_dim"]
        latent_dim = vae_components["latent_dim"]
        x = vae_components["x"]

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

        config = BetaVAEWithCapacityConfig(
            name="test_beta_vae_no_capacity",
            encoder=encoder_config,
            decoder=decoder_config,
            beta_default=2.0,
            use_capacity_control=False,  # Disabled
            capacity_max=25.0,
            capacity_num_iter=10000,
            gamma=1000.0,
        )

        model = BetaVAEWithCapacity(config=config, rngs=rngs)
        outputs = model(x)
        losses = model.loss_fn(x=x, outputs=outputs, step=5000)

        # Without capacity control, should return base BetaVAE losses
        assert "loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "beta" in losses

        # Should NOT have capacity-specific keys
        assert "capacity_loss" not in losses
        assert "current_capacity" not in losses

    def test_capacity_loss_calculation(self, rngs, capacity_config, vae_components):
        """Test capacity loss is calculated correctly: γ * |KL - C|."""
        x = vae_components["x"]

        model = BetaVAEWithCapacity(config=capacity_config, rngs=rngs)
        outputs = model(x)

        step = 5000
        losses = model.loss_fn(x=x, outputs=outputs, step=step)

        # Calculate expected capacity
        expected_capacity = min(
            capacity_config.capacity_max,
            capacity_config.capacity_max * step / capacity_config.capacity_num_iter,
        )

        # Check current_capacity matches expected
        assert jnp.isclose(losses["current_capacity"], expected_capacity)

        # Verify capacity loss structure: γ * |KL - C|
        expected_capacity_loss = capacity_config.gamma * jnp.abs(
            losses["kl_loss"] - expected_capacity
        )
        assert jnp.isclose(losses["capacity_loss"], expected_capacity_loss)

    def test_jit_compatibility(self, rngs, capacity_config, vae_components):
        """Test BetaVAEWithCapacity is JIT compatible."""
        x = vae_components["x"]

        model = BetaVAEWithCapacity(config=capacity_config, rngs=rngs)

        @jax.jit
        def compute_loss(model, x, step):
            outputs = model(x)
            return model.loss_fn(x=x, outputs=outputs, step=step)

        # Should compile and run without errors
        for step in [0, 5000, 10000]:
            losses = compute_loss(model, x, step)
            assert jnp.isfinite(losses["loss"])
            assert jnp.isfinite(losses["capacity_loss"])
