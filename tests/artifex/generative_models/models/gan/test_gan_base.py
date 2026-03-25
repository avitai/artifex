"""Tests for the GAN base model."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.gan_config import GANConfig
from artifex.generative_models.core.configuration.network_configs import (
    DiscriminatorConfig,
    GeneratorConfig,
)
from artifex.generative_models.models.gan.base import Discriminator, GAN, Generator


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def rngs(key):
    """Fixture for nnx random number generators."""
    # Initialize with explicit keys for each RNG stream needed by the models
    params_key = jax.random.key(0)
    dropout_key = jax.random.key(1)
    sample_key = jax.random.key(2)

    return nnx.Rngs(params=params_key, dropout=dropout_key, sample=sample_key)


@pytest.fixture
def input_data():
    """Fixture for input data."""
    return jnp.ones((4, 3, 32, 32))


@pytest.fixture
def generator_config():
    """Fixture for GeneratorConfig."""
    return GeneratorConfig(
        name="test_generator",
        latent_dim=32,
        hidden_dims=(64, 128),
        output_shape=(1, 3, 32, 32),
        activation="relu",
        batch_norm=True,
        dropout_rate=0.0,
    )


@pytest.fixture
def discriminator_config():
    """Fixture for DiscriminatorConfig."""
    return DiscriminatorConfig(
        name="test_discriminator",
        input_shape=(4, 3, 32, 32),
        hidden_dims=(128, 64),
        activation="leaky_relu",
        leaky_relu_slope=0.2,
        batch_norm=False,
        dropout_rate=0.1,
    )


@pytest.fixture
def gan_config(generator_config, discriminator_config):
    """Fixture for GANConfig."""
    return GANConfig(
        name="test_gan",
        generator=generator_config,
        discriminator=discriminator_config,
        loss_type="vanilla",
        gradient_penalty_weight=0.0,
    )


class TestGenerator:
    """Test cases for Generator module."""

    def test_init(self, rngs, generator_config):
        """Test initialization of Generator."""
        generator = Generator(config=generator_config, rngs=rngs)

        # Check that attributes are set correctly
        assert generator.hidden_dims == list(generator_config.hidden_dims)
        assert generator.output_shape == generator_config.output_shape
        assert generator.latent_dim == generator_config.latent_dim
        assert len(generator.layers) == len(generator_config.hidden_dims)

    def test_init_requires_rngs(self, generator_config):
        """Test that Generator requires rngs."""
        with pytest.raises(ValueError, match="rngs must be provided"):
            Generator(config=generator_config, rngs=None)

    def test_init_requires_generator_config(self, rngs):
        """Test that Generator requires GeneratorConfig."""
        with pytest.raises(TypeError, match="config must be GeneratorConfig"):
            Generator(config="not a config", rngs=rngs)

    def test_call(self, rngs, generator_config):
        """Test forward pass of Generator."""
        batch_size = 4
        generator = Generator(config=generator_config, rngs=rngs)

        # Create random latent vector
        z = jnp.ones((batch_size, generator_config.latent_dim))

        # Forward pass
        output = generator(z)

        # Check output shape
        assert output.shape == (batch_size, 3, 32, 32)

        # Check output range (should be between -1 and 1 due to tanh)
        assert jnp.all((output >= -1.0) & (output <= 1.0))

    def test_no_batch_norm(self, rngs):
        """Test Generator without batch normalization."""
        config = GeneratorConfig(
            name="test_gen_no_bn",
            latent_dim=32,
            hidden_dims=(64, 128),
            output_shape=(1, 3, 32, 32),
            activation="relu",
            batch_norm=False,
            dropout_rate=0.0,
        )
        generator = Generator(config=config, rngs=rngs)

        assert generator.bn_layers is None

        # Test forward pass works
        z = jnp.ones((2, 32))
        output = generator(z)
        assert output.shape == (2, 3, 32, 32)

    def test_with_dropout(self, rngs):
        """Test Generator with dropout."""
        config = GeneratorConfig(
            name="test_gen_dropout",
            latent_dim=32,
            hidden_dims=(64, 128),
            output_shape=(1, 3, 32, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.5,
        )
        generator = Generator(config=config, rngs=rngs)

        assert generator.dropout is not None

        # Test forward pass works
        z = jnp.ones((2, 32))
        output = generator(z)
        assert output.shape == (2, 3, 32, 32)


class TestDiscriminator:
    """Test cases for Discriminator module."""

    def test_init(self, rngs, discriminator_config):
        """Test initialization of Discriminator."""
        discriminator = Discriminator(config=discriminator_config, rngs=rngs)

        # Check that attributes are set correctly
        assert discriminator.hidden_dims == list(discriminator_config.hidden_dims)
        assert discriminator.input_shape == discriminator_config.input_shape
        assert discriminator.activation == discriminator_config.activation
        assert discriminator.leaky_relu_slope == discriminator_config.leaky_relu_slope
        assert discriminator.batch_norm == discriminator_config.batch_norm
        assert discriminator.dropout_rate == discriminator_config.dropout_rate
        assert not hasattr(discriminator, "use_spectral_norm")

        # Check that layers were pre-allocated (not lazy)
        assert len(discriminator.layers) == len(discriminator_config.hidden_dims)
        assert discriminator.output_layer is not None

    def test_init_requires_rngs(self, discriminator_config):
        """Test that Discriminator requires rngs."""
        with pytest.raises(ValueError, match="rngs must be provided"):
            Discriminator(config=discriminator_config, rngs=None)

    def test_init_requires_discriminator_config(self, rngs):
        """Test that Discriminator requires DiscriminatorConfig."""
        with pytest.raises(TypeError, match="config must be DiscriminatorConfig"):
            Discriminator(config="not a config", rngs=rngs)

    def test_call(self, rngs, discriminator_config, input_data):
        """Test forward pass of Discriminator."""
        discriminator = Discriminator(config=discriminator_config, rngs=rngs)

        # Forward pass
        output = discriminator(input_data)

        # Check output shape (should be batch_size x 1)
        assert output.shape == (input_data.shape[0], 1)

        # Check output range (should be between 0 and 1 due to sigmoid)
        assert jnp.all((output >= 0.0) & (output <= 1.0))

    def test_removed_spectral_norm_surface(self):
        """Test that the dead spectral-norm config surface is removed."""
        with pytest.raises(TypeError, match="use_spectral_norm"):
            DiscriminatorConfig(
                name="test_disc_spectral",
                input_shape=(4, 3, 32, 32),
                hidden_dims=(128, 64),
                activation="leaky_relu",
                leaky_relu_slope=0.2,
                batch_norm=False,
                dropout_rate=0.3,
                use_spectral_norm=True,
            )

    def test_with_batch_norm(self, rngs, input_data):
        """Test Discriminator with batch normalization."""
        config = DiscriminatorConfig(
            name="test_disc_bn",
            input_shape=input_data.shape,
            hidden_dims=(128, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            dropout_rate=0.0,
        )
        discriminator = Discriminator(config=config, rngs=rngs)

        assert discriminator.bn_layers is not None

        # Forward pass
        output = discriminator(input_data)
        assert output.shape == (input_data.shape[0], 1)


class TestGAN:
    """Test cases for GAN model."""

    def test_init(self, rngs, gan_config):
        """Test initialization of GAN."""
        gan = GAN(config=gan_config, rngs=rngs)

        # Check that generator and discriminator are initialized
        assert gan.generator is not None
        assert gan.discriminator is not None
        assert gan.latent_dim == gan_config.generator.latent_dim
        assert gan.loss_type == gan_config.loss_type
        assert gan.gradient_penalty_weight == gan_config.gradient_penalty_weight

    def test_init_requires_rngs(self, gan_config):
        """Test that GAN requires rngs."""
        with pytest.raises(ValueError, match="rngs must be provided"):
            GAN(config=gan_config, rngs=None)

    def test_init_requires_gan_config(self, rngs):
        """Test that GAN requires GANConfig."""
        with pytest.raises(TypeError, match="config must be GANConfig"):
            GAN(config="not a config", rngs=rngs)

    def test_init_requires_sample_stream(self, gan_config):
        """Test that GAN requires 'sample' stream in rngs."""
        rngs_no_sample = nnx.Rngs(params=jax.random.key(0))
        with pytest.raises(ValueError, match="'sample' stream"):
            GAN(config=gan_config, rngs=rngs_no_sample)

    def test_call(self, rngs, gan_config, input_data):
        """Test forward pass of GAN."""
        gan = GAN(config=gan_config, rngs=rngs)

        # Forward pass
        outputs = gan(input_data)

        # Check output dictionary
        assert "real_scores" in outputs

        # In the refactored implementation, fake scores and samples are None in __call__
        assert outputs["fake_scores"] is None
        assert outputs["fake_samples"] is None

        # Check shapes of real scores
        assert outputs["real_scores"].shape == (input_data.shape[0], 1)

    def test_generate(self, rngs, gan_config):
        """Test sample generation."""
        gan = GAN(config=gan_config, rngs=rngs)

        # Generate samples (no rngs parameter - uses stored self.rngs)
        batch_size = 4
        samples = gan.generate(batch_size=batch_size)

        # Check output shape matches the configuration
        expected_shape = (batch_size, *gan_config.generator.output_shape[1:])
        assert samples.shape == expected_shape

    def test_generator_objective_vanilla(self, rngs, input_data):
        """Test generator objective with vanilla GAN loss."""
        gen_config = GeneratorConfig(
            name="gen_vanilla",
            latent_dim=32,
            hidden_dims=(64, 128),
            output_shape=(1, 3, 32, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )
        disc_config = DiscriminatorConfig(
            name="disc_vanilla",
            input_shape=(4, 3, 32, 32),
            hidden_dims=(128, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
            dropout_rate=0.1,
        )
        config = GANConfig(
            name="gan_vanilla",
            generator=gen_config,
            discriminator=disc_config,
            loss_type="vanilla",
            gradient_penalty_weight=0.0,
        )
        gan = GAN(config=config, rngs=rngs)

        result = gan.generator_objective(input_data)

        loss = result["total_loss"]
        assert jnp.isscalar(loss) or loss.shape == ()
        assert jnp.isfinite(loss)

        assert "generator_loss" in result
        assert jnp.isfinite(result["generator_loss"])
        assert jnp.isclose(result["total_loss"], result["generator_loss"])

    def test_discriminator_objective_vanilla(self, rngs, input_data):
        """Test discriminator objective with vanilla GAN loss."""
        gen_config = GeneratorConfig(
            name="gen_ls",
            latent_dim=32,
            hidden_dims=(64, 128),
            output_shape=(1, 3, 32, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )
        disc_config = DiscriminatorConfig(
            name="disc_ls",
            input_shape=(4, 3, 32, 32),
            hidden_dims=(128, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
            dropout_rate=0.1,
        )
        config = GANConfig(
            name="gan_vanilla",
            generator=gen_config,
            discriminator=disc_config,
            loss_type="vanilla",
            gradient_penalty_weight=0.0,
        )
        gan = GAN(config=config, rngs=rngs)

        result = gan.discriminator_objective(input_data)

        loss = result["total_loss"]
        assert jnp.isfinite(loss)
        assert jnp.isfinite(result["discriminator_loss"])
        assert jnp.isclose(result["total_loss"], result["discriminator_loss"])

    @pytest.mark.parametrize("loss_type", ["least_squares", "wasserstein", "hinge"])
    def test_objectives_support_all_loss_types(self, rngs, input_data, loss_type):
        """Generator and discriminator objectives should support all configured GAN losses."""
        gen_config = GeneratorConfig(
            name="gen_wgan",
            latent_dim=32,
            hidden_dims=(64, 128),
            output_shape=(1, 3, 32, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )
        disc_config = DiscriminatorConfig(
            name="disc_wgan",
            input_shape=(4, 3, 32, 32),
            hidden_dims=(128, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
            dropout_rate=0.1,
        )
        config = GANConfig(
            name=f"gan_{loss_type}",
            generator=gen_config,
            discriminator=disc_config,
            loss_type=loss_type,
            gradient_penalty_weight=0.0,
        )
        gan = GAN(config=config, rngs=rngs)

        generator_result = gan.generator_objective(input_data)
        discriminator_result = gan.discriminator_objective(input_data)

        assert jnp.isfinite(generator_result["total_loss"])
        assert jnp.isfinite(discriminator_result["total_loss"])
        assert jnp.isfinite(generator_result["generator_loss"])
        assert jnp.isfinite(discriminator_result["discriminator_loss"])

    def test_loss_fn_not_supported_for_multi_objective_training(self, rngs, gan_config, input_data):
        """Combined GAN loss_fn should not exist as a fake single-objective API."""
        gan = GAN(config=gan_config, rngs=rngs)

        with pytest.raises(NotImplementedError, match="separate generator and discriminator"):
            gan.loss_fn(input_data, None)

    def test_invalid_loss_type(self, rngs, input_data):
        """Test that invalid loss type raises an error at config creation."""
        gen_config = GeneratorConfig(
            name="gen_invalid",
            latent_dim=32,
            hidden_dims=(64, 128),
            output_shape=(1, 3, 32, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )
        disc_config = DiscriminatorConfig(
            name="disc_invalid",
            input_shape=(4, 3, 32, 32),
            hidden_dims=(128, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
            dropout_rate=0.1,
        )

        # Check that error is raised at config creation (validation in __post_init__)
        with pytest.raises(ValueError, match="loss_type must be one of"):
            GANConfig(
                name="gan_invalid",
                generator=gen_config,
                discriminator=disc_config,
                loss_type="invalid",
                gradient_penalty_weight=0.0,
            )


class TestGeneratorJIT:
    """JIT compatibility tests for Generator."""

    def test_jit_call(self, rngs, generator_config):
        """Test that Generator __call__ is JIT compatible."""
        generator = Generator(config=generator_config, rngs=rngs)

        @nnx.jit
        def jit_generate(model, z):
            return model(z)

        z = jnp.ones((2, generator_config.latent_dim))
        output = jit_generate(generator, z)

        assert output.shape == (2, 3, 32, 32)
        assert jnp.all((output >= -1.0) & (output <= 1.0))


class TestDiscriminatorJIT:
    """JIT compatibility tests for Discriminator."""

    def test_jit_call(self, rngs, discriminator_config, input_data):
        """Test that Discriminator __call__ is JIT compatible."""
        discriminator = Discriminator(config=discriminator_config, rngs=rngs)

        @nnx.jit
        def jit_discriminate(model, x):
            return model(x)

        output = jit_discriminate(discriminator, input_data)

        assert output.shape == (input_data.shape[0], 1)
        assert jnp.all((output >= 0.0) & (output <= 1.0))


class TestGANJIT:
    """JIT compatibility tests for GAN."""

    def test_jit_call(self, rngs, gan_config, input_data):
        """Test that GAN __call__ is JIT compatible."""
        gan = GAN(config=gan_config, rngs=rngs)

        @nnx.jit
        def jit_call(model, x):
            return model(x)

        outputs = jit_call(gan, input_data)

        assert "real_scores" in outputs
        assert outputs["real_scores"].shape == (input_data.shape[0], 1)

    def test_jit_generate(self, rngs, gan_config):
        """Test that GAN generate is JIT compatible with static n_samples."""
        gan = GAN(config=gan_config, rngs=rngs)

        # n_samples must be static since it's used in array shapes
        @nnx.jit(static_argnums=(1,))
        def jit_generate(model, n_samples):
            return model.generate(n_samples)

        # Note: JIT with generate uses stored RNG which advances
        samples = jit_generate(gan, 2)

        expected_shape = (2, *gan_config.generator.output_shape[1:])
        assert samples.shape == expected_shape
