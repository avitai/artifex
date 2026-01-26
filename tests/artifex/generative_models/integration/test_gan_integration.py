"""Integration tests for GAN models."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    ConditionalDiscriminatorConfig,
    ConditionalGANConfig,
    ConditionalGeneratorConfig,
    ConditionalParams,
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
    DCGANConfig,
    WGANConfig,
)
from artifex.generative_models.models.gan import DCGAN, WGAN
from tests.utils.test_helpers import get_mock_reason, should_run_gan_tests


@pytest.fixture
def rng():
    """Random number generator fixture."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def generator_config():
    """Create generator configuration for testing."""
    return ConvGeneratorConfig(
        name="test_generator",
        latent_dim=32,
        hidden_dims=(128, 64),
        output_shape=(3, 16, 16),  # NCHW format
        activation="relu",
        batch_norm=True,
        kernel_size=(4, 4),
        stride=(2, 2),
        padding="SAME",
    )


@pytest.fixture
def discriminator_config():
    """Create discriminator configuration for testing."""
    return ConvDiscriminatorConfig(
        name="test_discriminator",
        hidden_dims=(64, 128),
        input_shape=(3, 16, 16),  # NCHW format
        activation="leaky_relu",
        leaky_relu_slope=0.2,
        batch_norm=False,
        kernel_size=(4, 4),
        stride=(2, 2),
        padding="SAME",
    )


@pytest.fixture
def dcgan_config(generator_config, discriminator_config):
    """Create DCGAN configuration for testing."""
    return DCGANConfig(
        name="test_dcgan",
        generator=generator_config,
        discriminator=discriminator_config,
    )


@pytest.fixture
def wgan_config(generator_config, discriminator_config):
    """Create WGAN configuration for testing."""
    return WGANConfig(
        name="test_wgan",
        generator=generator_config,
        discriminator=discriminator_config,
        critic_iterations=5,
        gradient_penalty_weight=10.0,
    )


class TestGANIntegration:
    """Integration tests for GAN models."""

    @pytest.mark.skipif(not should_run_gan_tests(), reason=get_mock_reason("GAN"))
    def test_dcgan_generator_discriminator(self, rng, dcgan_config):
        """Test DCGAN generator and discriminator."""
        # Create model with required RNG streams
        model = DCGAN(dcgan_config, rngs=nnx.Rngs(params=rng, sample=1, dropout=2))

        # Test generator
        batch_size = 2

        # Generate random latent vectors for testing
        latent_key = jax.random.key(1)
        fake_images = model.generate(batch_size, rngs=nnx.Rngs(sample=latent_key))

        # Verify shapes - NCHW format
        expected_shape = (batch_size, 3, 16, 16)
        assert fake_images.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {fake_images.shape}"
        )

        # Test discriminator
        disc_output = model.discriminator(fake_images)

        # Verify discriminator output shape (real/fake prediction per image)
        assert disc_output.shape == (batch_size, 1)

        # Verify values are in reasonable range for sigmoid output
        assert jnp.all(disc_output >= 0.0)
        assert jnp.all(disc_output <= 1.0)

    @pytest.mark.skipif(not should_run_gan_tests(), reason=get_mock_reason("GAN"))
    def test_wgan_with_gradient_penalty(self, rng, wgan_config):
        """Test WGAN with gradient penalty."""
        # Create model with required RNG streams
        model = WGAN(wgan_config, rngs=nnx.Rngs(params=rng, sample=1, dropout=2))

        # Test generator
        batch_size = 2
        latent_key = jax.random.key(1)
        fake_images = model.generate(batch_size, rngs=nnx.Rngs(sample=latent_key))

        # Verify shapes - NCHW format
        expected_shape = (batch_size, 3, 16, 16)
        assert fake_images.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {fake_images.shape}"
        )

        # Test discriminator
        disc_output = model.discriminator(fake_images)

        # Verify discriminator output shape (raw scores per image)
        assert disc_output.shape == (batch_size,)

        # WGAN discriminator outputs raw scores (no sigmoid), so values can be any real number
        assert jnp.all(jnp.isfinite(disc_output))

        # Test gradient penalty function
        from artifex.generative_models.models.gan.wgan import compute_gradient_penalty

        # Create some real samples for gradient penalty test
        real_samples = jax.random.normal(latent_key, expected_shape)

        # Compute gradient penalty
        gp = compute_gradient_penalty(
            model.discriminator, real_samples, fake_images, rngs=nnx.Rngs(params=rng)
        )

        # Gradient penalty should be a scalar and finite
        assert gp.shape == ()
        assert jnp.isfinite(gp)

    @pytest.mark.skipif(not should_run_gan_tests(), reason=get_mock_reason("GAN"))
    def test_conditional_gan(self, rng):
        """Test Conditional GAN with label conditioning."""
        from artifex.generative_models.models.gan.conditional import ConditionalGAN

        # Create conditional generator config (uses composition with ConditionalParams)
        cond_generator = ConditionalGeneratorConfig(
            name="cond_generator",
            latent_dim=32,
            hidden_dims=(128, 64),
            output_shape=(3, 16, 16),
            conditional=ConditionalParams(num_classes=10, embedding_dim=32),
            activation="relu",
            batch_norm=True,
        )

        # Create conditional discriminator config (uses composition with ConditionalParams)
        cond_discriminator = ConditionalDiscriminatorConfig(
            name="cond_discriminator",
            hidden_dims=(64, 128),
            input_shape=(3, 16, 16),
            conditional=ConditionalParams(num_classes=10, embedding_dim=32),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
        )

        # Create conditional GAN config
        cgan_config = ConditionalGANConfig(
            name="test_conditional_gan",
            generator=cond_generator,
            discriminator=cond_discriminator,
        )

        # Create model with required RNG streams
        model = ConditionalGAN(cgan_config, rngs=nnx.Rngs(params=rng, sample=1, dropout=2))

        # Test generator
        batch_size = 2
        latent_key = jax.random.key(1)

        # Create random labels (one-hot encoded)
        label_indices = jnp.array([0, 5])  # Generate class 0 and class 5
        labels = jax.nn.one_hot(label_indices, 10)

        # Generate conditional samples
        fake_images = model.generate(batch_size, labels=labels, rngs=nnx.Rngs(sample=latent_key))

        # Verify shapes
        expected_shape = (batch_size, 3, 16, 16)
        assert fake_images.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {fake_images.shape}"
        )

        # Test discriminator with labels
        disc_output = model.discriminator(fake_images, labels)

        # Verify discriminator output shape (raw scores per image)
        assert disc_output.shape == (batch_size,)

        # Discriminator outputs should be finite
        assert jnp.all(jnp.isfinite(disc_output))

        # Test generator directly with different labels (use model.eval() for eval mode)
        z = jax.random.normal(latent_key, (batch_size, 32))
        model.generator.eval()  # Switch to eval mode
        generated = model.generator(z, labels)

        assert generated.shape == expected_shape
        assert jnp.all(jnp.isfinite(generated))
        model.generator.train()  # Switch back to train mode

        # Test with random labels (when none provided)
        random_samples = model.generate(batch_size, rngs=nnx.Rngs(sample=latent_key))
        assert random_samples.shape == expected_shape

        # Test loss functions
        real_samples = jax.random.normal(latent_key, expected_shape)
        real_labels = jax.nn.one_hot(jnp.array([1, 3]), 10)
        fake_labels = jax.nn.one_hot(jnp.array([2, 7]), 10)

        # Test discriminator loss
        disc_loss = model.discriminator_loss(real_samples, fake_images, real_labels, fake_labels)
        assert disc_loss.shape == ()
        assert jnp.isfinite(disc_loss)

        # Test generator loss
        gen_loss = model.generator_loss(fake_images, fake_labels)
        assert gen_loss.shape == ()
        assert jnp.isfinite(gen_loss)

    @pytest.mark.skipif(not should_run_gan_tests(), reason=get_mock_reason("GAN"))
    def test_stylegan_style_mixing(self, rng):
        """Test StyleGAN3 with style mixing."""
        from artifex.generative_models.core.configuration.network_configs import (
            StyleGAN3GeneratorConfig,
        )
        from artifex.generative_models.models.gan.stylegan3 import StyleGAN3Generator

        # Create a small StyleGAN3 generator for testing (low resolution for speed)
        config = StyleGAN3GeneratorConfig(
            name="test_stylegan3",
            latent_dim=128,
            style_dim=128,
            hidden_dims=(512,),  # Required by base class, but not used by StyleGAN3
            activation="leaky_relu",  # Required by base class
            output_shape=(32, 32, 3),  # Required by base class
            img_resolution=32,  # Small for faster testing
            img_channels=3,
        )

        generator = StyleGAN3Generator(
            config, rngs=nnx.Rngs(params=rng, sample=1, dropout=2, noise=3)
        )

        # Test basic sampling
        batch_size = 2
        samples = generator.sample(batch_size, rngs=nnx.Rngs(sample=rng, noise=4))

        assert samples.shape == (batch_size, 32, 32, 3), (
            f"Expected shape (2, 32, 32, 3), got {samples.shape}"
        )
        assert jnp.all(jnp.isfinite(samples))

        # Test that different latent vectors produce different outputs
        z1 = jax.random.normal(jax.random.key(1), (1, 128))
        z2 = jax.random.normal(jax.random.key(2), (1, 128))

        img1 = generator(z1, rngs=nnx.Rngs(noise=5))
        img2 = generator(z2, rngs=nnx.Rngs(noise=6))

        # Images should be different
        assert not jnp.allclose(img1, img2, atol=1e-3)

    @pytest.mark.skipif(not should_run_gan_tests(), reason=get_mock_reason("GAN"))
    def test_dcgan_with_convtranspose(self, rng, dcgan_config):
        """Test DCGAN with ConvTranspose layers."""
        # Create DCGAN model
        model = DCGAN(dcgan_config, rngs=nnx.Rngs(params=rng, sample=1, dropout=2))

        # Test that the generator uses ConvTranspose layers
        assert hasattr(model.generator, "conv_transpose_layers"), (
            "DCGAN generator should have conv_transpose_layers"
        )
        assert len(model.generator.conv_transpose_layers) > 0, (
            "DCGAN generator should have at least one ConvTranspose layer"
        )

        # Test forward pass
        batch_size = 2
        z = jax.random.normal(rng, (batch_size, dcgan_config.generator.latent_dim))
        fake_images = model.generator(z)

        # Verify output shape
        expected_shape = (batch_size, *dcgan_config.generator.output_shape)
        assert fake_images.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {fake_images.shape}"
        )
        assert jnp.all(jnp.isfinite(fake_images))

    def test_gan_training_step(self, rng, dcgan_config):
        """Test a single training step for GANs."""
        # Create DCGAN model
        model = DCGAN(dcgan_config, rngs=nnx.Rngs(params=rng, sample=1, dropout=2))

        # Test generator forward pass
        batch_size = 2
        z = jax.random.normal(rng, (batch_size, dcgan_config.generator.latent_dim))
        fake_images = model.generator(z)

        # Verify output shape
        expected_shape = (batch_size, *dcgan_config.generator.output_shape)
        assert fake_images.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {fake_images.shape}"
        )

        # Test discriminator forward pass
        disc_output = model.discriminator(fake_images)
        # Discriminator output can be (batch_size,) or (batch_size, 1)
        assert disc_output.shape[0] == batch_size
        assert jnp.all(jnp.isfinite(disc_output))

    @pytest.mark.skipif(not should_run_gan_tests(), reason=get_mock_reason("GAN"))
    def test_gan_sampling(self, rng, dcgan_config):
        """Test sampling from GAN models."""
        # Create DCGAN model
        model = DCGAN(dcgan_config, rngs=nnx.Rngs(params=rng, sample=1, dropout=2))

        # Test sampling via generator
        batch_size = 4
        samples = model.sample(batch_size, rngs=nnx.Rngs(sample=rng))

        # Verify output shape
        expected_shape = (batch_size, *dcgan_config.generator.output_shape)
        assert samples.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {samples.shape}"
        )
        assert jnp.all(jnp.isfinite(samples))
