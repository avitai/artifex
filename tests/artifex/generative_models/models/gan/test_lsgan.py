"""Tests for LSGAN implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.gan_config import LSGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.models.gan.lsgan import (
    LSGAN,
    LSGANDiscriminator,
    LSGANGenerator,
)


class TestLSGANGenerator:
    """Test cases for LSGAN Generator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def output_shape(self):
        """Output shape fixture."""
        return (3, 32, 32)  # Small for testing

    @pytest.fixture
    def generator_config(self, output_shape):
        """Generator config fixture."""
        return ConvGeneratorConfig(
            name="test_lsgan_gen",
            latent_dim=32,  # Small for testing
            output_shape=output_shape,
            hidden_dims=(64, 32),  # Small for testing
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )

    @pytest.fixture
    def generator(self, rng, generator_config):
        """Generator fixture."""
        return LSGANGenerator(config=generator_config, rngs=nnx.Rngs(rng))

    def test_generator_initialization(self, generator, output_shape):
        """Test generator initialization."""
        assert generator.output_shape == output_shape
        assert generator.latent_dim == 32
        assert generator.hidden_dims == [64, 32]
        assert generator.batch_norm is True
        assert generator.dropout_rate == 0.0

    def test_generator_forward_pass(self, generator, rng):
        """Test generator forward pass."""
        batch_size = 4
        latent_dim = 32

        # Create latent input
        z = jax.random.normal(rng, (batch_size, latent_dim))

        # Forward pass
        output = generator(z)

        # Check output shape
        expected_shape = (batch_size, 3, 32, 32)
        assert output.shape == expected_shape

        # Check output range (tanh activation)
        assert jnp.all(output >= -1.0)
        assert jnp.all(output <= 1.0)

        # Check output is finite
        assert jnp.all(jnp.isfinite(output))

    def test_generator_deterministic_inference(self, generator, rng):
        """Test generator produces same output in inference mode."""
        batch_size = 2
        latent_dim = 32

        z = jax.random.normal(rng, (batch_size, latent_dim))

        # Two forward passes in inference mode
        generator.eval()
        output1 = generator(z)
        output2 = generator(z)

        # Should be identical in inference mode
        assert jnp.allclose(output1, output2, atol=1e-6)

    def test_generator_different_latent_inputs(self, generator, rng):
        """Test generator produces different outputs for different inputs."""
        batch_size = 2
        latent_dim = 32

        z1 = jax.random.normal(rng, (batch_size, latent_dim))
        z2 = jax.random.normal(jax.random.split(rng)[0], (batch_size, latent_dim))

        generator.eval()
        output1 = generator(z1)
        output2 = generator(z2)

        # Should be different for different inputs
        assert not jnp.allclose(output1, output2, atol=1e-3)

    def test_generator_batch_sizes(self, generator, rng):
        """Test generator works with different batch sizes."""
        latent_dim = 32

        generator.eval()
        for batch_size in [1, 2, 4, 8]:
            z = jax.random.normal(rng, (batch_size, latent_dim))
            output = generator(z)
            expected_shape = (batch_size, 3, 32, 32)
            assert output.shape == expected_shape

    def test_generator_requires_conv_config(self, rng):
        """Test that generator requires ConvGeneratorConfig."""
        from artifex.generative_models.core.configuration.network_configs import (
            GeneratorConfig,
        )

        base_config = GeneratorConfig(
            name="base_gen",
            latent_dim=32,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32),
            activation="relu",
        )

        with pytest.raises(TypeError, match="ConvGeneratorConfig"):
            LSGANGenerator(config=base_config, rngs=nnx.Rngs(rng))


class TestLSGANDiscriminator:
    """Test cases for LSGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture."""
        return (3, 32, 32)

    @pytest.fixture
    def discriminator_config(self, input_shape):
        """Discriminator config fixture."""
        return ConvDiscriminatorConfig(
            name="test_lsgan_disc",
            input_shape=input_shape,
            hidden_dims=(32, 64),  # Small for testing
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.1,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )

    @pytest.fixture
    def discriminator(self, rng, discriminator_config):
        """Discriminator fixture."""
        return LSGANDiscriminator(config=discriminator_config, rngs=nnx.Rngs(rng))

    def test_discriminator_initialization(self, discriminator, input_shape):
        """Test discriminator initialization."""
        assert discriminator.input_shape == input_shape
        assert discriminator.hidden_dims == [32, 64]
        assert discriminator.activation == "leaky_relu"
        assert discriminator.leaky_relu_slope == 0.2
        assert discriminator.batch_norm is False
        assert discriminator.dropout_rate == 0.1

    def test_discriminator_forward_pass(self, discriminator, rng):
        """Test discriminator forward pass."""
        batch_size = 4
        input_shape = (3, 32, 32)

        # Create input images
        images = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        discriminator.train()
        scores = discriminator(images)

        # Check output shape (should be scalar per sample)
        expected_shape = (batch_size, 1)
        assert scores.shape == expected_shape

        # Check output is finite
        assert jnp.all(jnp.isfinite(scores))

    def test_discriminator_no_sigmoid_activation(self, discriminator, rng):
        """Test discriminator outputs raw scores (no sigmoid for LSGAN)."""
        batch_size = 2
        input_shape = (3, 32, 32)

        # Create input images
        images = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        discriminator.eval()
        scores = discriminator(images)

        # LSGAN discriminator should output raw scores, not probabilities
        # So values can be outside [0, 1] range
        assert scores.shape == (batch_size, 1)
        assert jnp.all(jnp.isfinite(scores))

    def test_discriminator_different_inputs(self, discriminator, rng):
        """Test discriminator produces different outputs for different inputs."""
        batch_size = 2
        input_shape = (3, 32, 32)

        images1 = jax.random.normal(rng, (batch_size, *input_shape))
        images2 = jax.random.normal(jax.random.split(rng)[0], (batch_size, *input_shape))

        discriminator.eval()
        scores1 = discriminator(images1)
        scores2 = discriminator(images2)

        # Should be different for different inputs
        assert not jnp.allclose(scores1, scores2, atol=1e-3)

    def test_discriminator_batch_sizes(self, discriminator, rng):
        """Test discriminator works with different batch sizes."""
        input_shape = (3, 32, 32)

        discriminator.eval()
        for batch_size in [1, 2, 4, 8]:
            images = jax.random.normal(rng, (batch_size, *input_shape))
            scores = discriminator(images)
            expected_shape = (batch_size, 1)
            assert scores.shape == expected_shape

    def test_discriminator_requires_conv_config(self, rng):
        """Test that discriminator requires ConvDiscriminatorConfig."""
        from artifex.generative_models.core.configuration.network_configs import (
            DiscriminatorConfig,
        )

        base_config = DiscriminatorConfig(
            name="base_disc",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
        )

        with pytest.raises(TypeError, match="ConvDiscriminatorConfig"):
            LSGANDiscriminator(config=base_config, rngs=nnx.Rngs(rng))


class TestLSGAN:
    """Test cases for complete LSGAN model."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def image_shape(self):
        """Image shape fixture."""
        return (3, 32, 32)

    @pytest.fixture
    def lsgan_config(self, image_shape):
        """LSGAN configuration fixture."""
        generator = ConvGeneratorConfig(
            name="test_lsgan_gen",
            latent_dim=32,
            output_shape=image_shape,
            hidden_dims=(64, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )
        discriminator = ConvDiscriminatorConfig(
            name="test_lsgan_disc",
            input_shape=image_shape,
            hidden_dims=(32, 64),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.1,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )
        return LSGANConfig(
            name="test_lsgan",
            generator=generator,
            discriminator=discriminator,
            a=0.0,
            b=1.0,
            c=1.0,
        )

    @pytest.fixture
    def lsgan(self, rng, lsgan_config):
        """LSGAN fixture."""
        return LSGAN(
            config=lsgan_config,
            rngs=nnx.Rngs(params=rng, dropout=rng, sample=rng),
        )

    def test_lsgan_initialization(self, lsgan):
        """Test LSGAN initialization."""
        assert lsgan.loss_type == "least_squares"
        assert isinstance(lsgan.generator, LSGANGenerator)
        assert isinstance(lsgan.discriminator, LSGANDiscriminator)

    def test_lsgan_loss_type(self, lsgan):
        """Test LSGAN has correct loss type."""
        # LSGAN always uses least squares loss
        assert lsgan.loss_type == "least_squares"

    def test_generator_loss_computation(self, lsgan, rng):
        """Test generator loss computation."""
        batch_size = 4
        fake_scores = jax.random.normal(rng, (batch_size, 1))

        loss = lsgan.generator_loss(fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0.0  # Least squares loss is always non-negative

    def test_discriminator_loss_computation(self, lsgan, rng):
        """Test discriminator loss computation."""
        batch_size = 4
        real_scores = jax.random.normal(rng, (batch_size, 1))
        fake_scores = jax.random.normal(jax.random.split(rng)[0], (batch_size, 1))

        loss = lsgan.discriminator_loss(real_scores, fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0.0  # Least squares loss is always non-negative

    def test_training_step(self, lsgan, rng):
        """Test complete training step."""
        batch_size = 2
        image_shape = (3, 32, 32)
        latent_dim = 32

        # Create batch data
        real_images = jax.random.normal(rng, (batch_size, *image_shape))
        latent_vectors = jax.random.normal(jax.random.split(rng)[0], (batch_size, latent_dim))

        # Perform training step
        lsgan.train()
        results = lsgan.training_step(real_images, latent_vectors)

        # Check results structure
        assert "generator_loss" in results
        assert "discriminator_loss" in results
        assert "real_scores" in results
        assert "fake_scores" in results
        assert "fake_images" in results

        # Check loss values
        assert jnp.isfinite(results["generator_loss"])
        assert jnp.isfinite(results["discriminator_loss"])
        assert results["generator_loss"] >= 0.0
        assert results["discriminator_loss"] >= 0.0

        # Check score shapes
        assert results["real_scores"].shape == (batch_size, 1)
        assert results["fake_scores"].shape == (batch_size, 1)

        # Check generated images
        assert results["fake_images"].shape == (batch_size, *image_shape)
        assert jnp.all(results["fake_images"] >= -1.0)
        assert jnp.all(results["fake_images"] <= 1.0)

    def test_loss_functions_use_least_squares(self, lsgan):
        """Test that loss functions properly use least squares formulation."""
        # Test generator loss with perfect scores (should be 0)
        perfect_fake_scores = jnp.ones((4, 1))  # All 1.0 (target)
        perfect_gen_loss = lsgan.generator_loss(perfect_fake_scores)
        assert jnp.allclose(perfect_gen_loss, 0.0, atol=1e-6)

        # Test discriminator loss with perfect scores (should be 0)
        perfect_real_scores = jnp.ones((4, 1))  # All 1.0 (target for real)
        perfect_fake_scores = jnp.zeros((4, 1))  # All 0.0 (target for fake)
        perfect_disc_loss = lsgan.discriminator_loss(perfect_real_scores, perfect_fake_scores)
        assert jnp.allclose(perfect_disc_loss, 0.0, atol=1e-6)

    def test_different_target_values(self, lsgan, rng):
        """Test generator and discriminator losses with different target values."""
        batch_size = 4
        fake_scores = jax.random.normal(rng, (batch_size, 1))
        real_scores = jax.random.normal(jax.random.split(rng)[0], (batch_size, 1))

        # Test with different target values
        gen_loss_target_09 = lsgan.generator_loss(fake_scores, target_real=0.9)
        gen_loss_target_10 = lsgan.generator_loss(fake_scores, target_real=1.0)

        disc_loss_default = lsgan.discriminator_loss(real_scores, fake_scores)
        disc_loss_custom = lsgan.discriminator_loss(
            real_scores, fake_scores, target_real=0.9, target_fake=0.1
        )

        # All losses should be finite
        assert jnp.isfinite(gen_loss_target_09)
        assert jnp.isfinite(gen_loss_target_10)
        assert jnp.isfinite(disc_loss_default)
        assert jnp.isfinite(disc_loss_custom)

        # Different targets should give different losses
        assert not jnp.allclose(gen_loss_target_09, gen_loss_target_10, atol=1e-6)
        assert not jnp.allclose(disc_loss_default, disc_loss_custom, atol=1e-6)


class TestLSGANIntegration:
    """Integration tests for LSGAN."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(123)

    def test_lsgan_end_to_end_training_step(self, rng):
        """Test end-to-end LSGAN training step."""
        # Create LSGAN with small dimensions for testing
        image_shape = (3, 16, 16)  # Very small for speed
        latent_dim = 16
        batch_size = 2

        # Create nested configs
        generator = ConvGeneratorConfig(
            name="test_lsgan_gen_e2e",
            latent_dim=latent_dim,
            output_shape=image_shape,
            hidden_dims=(32, 16),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )
        discriminator = ConvDiscriminatorConfig(
            name="test_lsgan_disc_e2e",
            input_shape=image_shape,
            hidden_dims=(16, 32),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.1,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )
        config = LSGANConfig(
            name="test_lsgan_e2e",
            generator=generator,
            discriminator=discriminator,
            a=0.0,
            b=1.0,
            c=1.0,
        )

        lsgan = LSGAN(
            config=config,
            rngs=nnx.Rngs(params=rng, dropout=rng, sample=rng),
        )

        # Create training data
        real_images = jax.random.normal(rng, (batch_size, *image_shape))
        latent_vectors = jax.random.normal(jax.random.split(rng)[0], (batch_size, latent_dim))

        # Perform multiple training steps
        lsgan.train()
        for step in range(3):
            results = lsgan.training_step(real_images, latent_vectors)

            # Verify all outputs are valid
            assert jnp.all(jnp.isfinite(results["generator_loss"]))
            assert jnp.all(jnp.isfinite(results["discriminator_loss"]))
            assert jnp.all(jnp.isfinite(results["real_scores"]))
            assert jnp.all(jnp.isfinite(results["fake_scores"]))
            assert jnp.all(jnp.isfinite(results["fake_images"]))

            # Generated images should be in valid range
            assert jnp.all(results["fake_images"] >= -1.0)
            assert jnp.all(results["fake_images"] <= 1.0)


class TestLSGANGeneratorJIT:
    """JIT compatibility tests for LSGAN Generator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def generator(self, rng):
        """Generator fixture for JIT tests."""
        config = ConvGeneratorConfig(
            name="test_lsgan_gen_jit",
            latent_dim=32,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )
        return LSGANGenerator(config=config, rngs=nnx.Rngs(rng))

    def test_jit_forward_pass(self, generator, rng):
        """Test that generator forward pass is JIT compatible."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        z = jax.random.normal(rng, (4, 32))
        generator.eval()
        output_regular = generator(z)
        output_jit = generate_jit(generator, z)

        # GPU floating-point arithmetic with JIT can produce differences up to ~1e-5
        assert jnp.allclose(output_regular, output_jit, rtol=1e-4, atol=1e-5)

    def test_jit_compilation_without_errors(self, generator, rng):
        """Test that JIT compilation succeeds without errors."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        z = jax.random.normal(rng, (4, 32))
        generator.eval()

        output = generate_jit(generator, z)
        assert output.shape == (4, 3, 32, 32)

    def test_jit_multiple_calls_consistent(self, generator, rng):
        """Test that multiple JIT calls produce consistent results."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        z = jax.random.normal(rng, (4, 32))
        generator.eval()

        output1 = generate_jit(generator, z)
        output2 = generate_jit(generator, z)

        # JIT calls should be exactly consistent (same compilation)
        assert jnp.allclose(output1, output2, rtol=1e-7, atol=1e-9)

    def test_jit_with_different_batch_sizes(self, generator, rng):
        """Test JIT with different batch sizes."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        generator.eval()

        for batch_size in [1, 2, 4, 8]:
            z = jax.random.normal(rng, (batch_size, 32))
            output = generate_jit(generator, z)
            assert output.shape == (batch_size, 3, 32, 32)


class TestLSGANDiscriminatorJIT:
    """JIT compatibility tests for LSGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def discriminator(self, rng):
        """Discriminator fixture for JIT tests."""
        config = ConvDiscriminatorConfig(
            name="test_lsgan_disc_jit",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,  # No dropout for deterministic JIT tests
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )
        return LSGANDiscriminator(config=config, rngs=nnx.Rngs(rng))

    def test_jit_forward_pass(self, discriminator, rng):
        """Test that discriminator forward pass is JIT compatible."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        discriminator.eval()
        output_regular = discriminator(x)
        output_jit = discriminate_jit(discriminator, x)

        # GPU floating-point arithmetic with JIT can produce differences up to ~1e-5
        assert jnp.allclose(output_regular, output_jit, rtol=1e-4, atol=1e-5)

    def test_jit_compilation_without_errors(self, discriminator, rng):
        """Test that JIT compilation succeeds without errors."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        discriminator.eval()

        output = discriminate_jit(discriminator, x)
        assert output.shape == (4, 1)

    def test_jit_multiple_calls_consistent(self, discriminator, rng):
        """Test that multiple JIT calls produce consistent results."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        discriminator.eval()

        output1 = discriminate_jit(discriminator, x)
        output2 = discriminate_jit(discriminator, x)

        # JIT calls should be exactly consistent (same compilation)
        assert jnp.allclose(output1, output2, rtol=1e-7, atol=1e-9)

    def test_jit_with_different_batch_sizes(self, discriminator, rng):
        """Test JIT with different batch sizes."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        discriminator.eval()

        for batch_size in [1, 2, 4, 8]:
            x = jax.random.normal(rng, (batch_size, 3, 32, 32))
            output = discriminate_jit(discriminator, x)
            assert output.shape == (batch_size, 1)


class TestLSGANJIT:
    """JIT compatibility tests for full LSGAN model."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def lsgan(self, rng):
        """LSGAN fixture for JIT tests."""
        image_shape = (3, 32, 32)

        generator = ConvGeneratorConfig(
            name="test_lsgan_gen_jit",
            latent_dim=32,
            output_shape=image_shape,
            hidden_dims=(64, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )
        discriminator = ConvDiscriminatorConfig(
            name="test_lsgan_disc_jit",
            input_shape=image_shape,
            hidden_dims=(32, 64),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,  # No dropout for deterministic JIT tests
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )
        config = LSGANConfig(
            name="test_lsgan_jit",
            generator=generator,
            discriminator=discriminator,
            a=0.0,
            b=1.0,
            c=1.0,
        )

        return LSGAN(
            config=config,
            rngs=nnx.Rngs(params=rng, dropout=rng, sample=rng),
        )

    def test_jit_generator_loss(self, lsgan, rng):
        """Test that generator_loss() is JIT compatible."""

        @nnx.jit
        def generator_loss_jit(model, fake_scores):
            return model.generator_loss(fake_scores)

        fake_scores = jax.random.normal(rng, (4, 1))
        lsgan.eval()

        loss_regular = lsgan.generator_loss(fake_scores)
        loss_jit = generator_loss_jit(lsgan, fake_scores)

        # Loss computation should be exactly the same
        assert jnp.allclose(loss_regular, loss_jit, rtol=1e-7, atol=1e-9)

    def test_jit_discriminator_loss(self, lsgan, rng):
        """Test that discriminator_loss() is JIT compatible."""

        @nnx.jit
        def discriminator_loss_jit(model, real_scores, fake_scores):
            return model.discriminator_loss(real_scores, fake_scores)

        real_scores = jax.random.normal(rng, (4, 1))
        fake_scores = jax.random.normal(jax.random.split(rng)[0], (4, 1))
        lsgan.eval()

        loss_regular = lsgan.discriminator_loss(real_scores, fake_scores)
        loss_jit = discriminator_loss_jit(lsgan, real_scores, fake_scores)

        # Loss computation should be exactly the same
        assert jnp.allclose(loss_regular, loss_jit, rtol=1e-7, atol=1e-9)

    def test_jit_full_training_step(self, lsgan, rng):
        """Test that a full training step can be JIT compiled."""

        @nnx.jit
        def training_step_jit(model, real_images, latent_vectors):
            return model.training_step(real_images, latent_vectors)

        batch_size = 2
        real_images = jax.random.normal(rng, (batch_size, 3, 32, 32))
        latent_vectors = jax.random.normal(jax.random.split(rng)[0], (batch_size, 32))

        lsgan.train()

        results = training_step_jit(lsgan, real_images, latent_vectors)
        assert "generator_loss" in results
        assert "discriminator_loss" in results
        assert jnp.isfinite(results["generator_loss"])
        assert jnp.isfinite(results["discriminator_loss"])

    def test_jit_gradient_computation(self, lsgan, rng):
        """Test that gradient computation is JIT compatible."""

        @nnx.jit
        def generator_loss_fn(model, latent_vecs):
            fake_images = model.generator(latent_vecs)
            fake_scores = model.discriminator(fake_images)
            return model.generator_loss(fake_scores)

        batch_size = 2
        latent_vectors = jax.random.normal(rng, (batch_size, 32))

        lsgan.eval()

        loss = generator_loss_fn(lsgan, latent_vectors)
        assert jnp.isfinite(loss)
