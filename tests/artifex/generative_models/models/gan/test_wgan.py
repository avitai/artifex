"""Tests for WGAN-GP implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.gan_config import WGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.models.gan.wgan import (
    compute_gradient_penalty,
    WGAN,
    WGANDiscriminator,
    WGANGenerator,
)


class TestWGANGenerator:
    """Test cases for WGAN Generator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def output_shape(self):
        """Output shape fixture."""
        return (3, 64, 64)  # C, H, W format

    @pytest.fixture
    def generator_config(self, output_shape):
        """Generator config fixture."""
        return ConvGeneratorConfig(
            name="test_wgan_gen",
            latent_dim=100,
            output_shape=output_shape,
            hidden_dims=(256, 128, 64, 32),  # 4 dims for 64x64 output
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
        return WGANGenerator(config=generator_config, rngs=nnx.Rngs(rng))

    def test_generator_initialization(self, generator, output_shape):
        """Test generator initialization."""
        assert generator.output_shape == output_shape
        assert generator.latent_dim == 100
        assert generator.batch_norm is True
        assert generator.dropout_rate == 0.0

    def test_generator_forward_pass(self, generator, rng):
        """Test generator forward pass."""
        batch_size = 4
        z = jax.random.normal(rng, (batch_size, 100))

        # Forward pass
        output = generator(z)

        # Check output shape (NCHW format)
        assert output.shape == (batch_size, 3, 64, 64)
        assert output.dtype == jnp.float32

        # Check output is bounded (tanh activation)
        assert jnp.all(output >= -1.0)
        assert jnp.all(output <= 1.0)

    def test_generator_eval_mode_deterministic(self, generator, rng):
        """Test generator deterministic behavior in eval mode."""
        batch_size = 2
        z = jax.random.normal(rng, (batch_size, 100))

        # Two forward passes in eval mode should be identical
        generator.eval()
        output1 = generator(z)
        output2 = generator(z)

        assert jnp.allclose(output1, output2, atol=1e-6)

    def test_generator_different_output_shapes(self, rng):
        """Test generator with different output shapes."""
        # Each output size needs specific number of hidden dims
        test_configs = [
            ((3, 32, 32), (128, 64, 32)),  # 32x32: needs 3 hidden dims
            ((1, 64, 64), (256, 128, 64, 32)),  # 64x64: needs 4 hidden dims
        ]

        for output_shape, hidden_dims in test_configs:
            config = ConvGeneratorConfig(
                name=f"test_gen_{output_shape[1]}",
                latent_dim=64,
                output_shape=output_shape,
                hidden_dims=hidden_dims,
                activation="relu",
                batch_norm=True,
                dropout_rate=0.0,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding="SAME",
                batch_norm_momentum=0.9,
                batch_norm_use_running_avg=False,
            )
            generator = WGANGenerator(config=config, rngs=nnx.Rngs(rng))

            batch_size = 2
            z = jax.random.normal(rng, (batch_size, 64))
            output = generator(z)

            assert output.shape == (batch_size, *output_shape)

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
            WGANGenerator(config=base_config, rngs=nnx.Rngs(rng))


class TestWGANDiscriminator:
    """Test cases for WGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture."""
        return (3, 64, 64)  # C, H, W format

    @pytest.fixture
    def discriminator_config(self, input_shape):
        """Discriminator config fixture."""
        return ConvDiscriminatorConfig(
            name="test_wgan_disc",
            input_shape=input_shape,
            hidden_dims=(64, 128, 256),  # Small for testing
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )

    @pytest.fixture
    def discriminator(self, rng, discriminator_config):
        """Discriminator fixture."""
        return WGANDiscriminator(config=discriminator_config, rngs=nnx.Rngs(rng))

    def test_discriminator_initialization(self, discriminator, input_shape):
        """Test discriminator initialization."""
        assert discriminator.input_shape == input_shape
        assert discriminator.hidden_dims == [64, 128, 256]
        assert discriminator.wgan_use_instance_norm is True
        assert discriminator.dropout_rate == 0.0

    def test_discriminator_forward_pass(self, discriminator, rng, input_shape):
        """Test discriminator forward pass."""
        batch_size = 4
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        output = discriminator(x)

        # Check output shape (should be scalar per sample)
        assert output.shape == (batch_size,)
        assert output.dtype == jnp.float32

        # Check output is finite (no sigmoid, raw scores)
        assert jnp.all(jnp.isfinite(output))

    def test_discriminator_eval_mode_deterministic(self, discriminator, rng, input_shape):
        """Test discriminator deterministic behavior in eval mode."""
        batch_size = 2
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Two forward passes in eval mode should be identical
        discriminator.eval()
        output1 = discriminator(x)
        output2 = discriminator(x)

        assert jnp.allclose(output1, output2, atol=1e-6)

    def test_discriminator_no_sigmoid(self, discriminator, rng, input_shape):
        """Test that discriminator outputs raw scores (no sigmoid)."""
        batch_size = 4
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        discriminator.eval()
        output = discriminator(x)

        # WGAN discriminator should output raw scores, not probabilities
        # So values can be any real number
        assert output.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(output))

    def test_discriminator_with_instance_norm(self, rng, input_shape):
        """Test discriminator with instance normalization."""
        config = ConvDiscriminatorConfig(
            name="test_disc_instance_norm",
            input_shape=input_shape,
            hidden_dims=(64, 128),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )
        discriminator = WGANDiscriminator(config=config, rngs=nnx.Rngs(rng))

        # Should have instance norm layers
        assert len(discriminator.norm_layers) > 0

        batch_size = 4
        x = jax.random.normal(rng, (batch_size, *input_shape))

        discriminator.eval()
        output = discriminator(x)

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(output))

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
            WGANDiscriminator(config=base_config, rngs=nnx.Rngs(rng))


class TestGradientPenalty:
    """Test cases for gradient penalty computation."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture."""
        return (3, 32, 32)  # Smaller for faster testing

    @pytest.fixture
    def discriminator(self, rng, input_shape):
        """Discriminator fixture."""
        config = ConvDiscriminatorConfig(
            name="test_gp_disc",
            input_shape=input_shape,
            hidden_dims=(32, 64),  # Small for testing
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )
        return WGANDiscriminator(config=config, rngs=nnx.Rngs(rng))

    def test_gradient_penalty_computation(self, discriminator, rng, input_shape):
        """Test gradient penalty computation."""
        batch_size = 4
        real_samples = jax.random.normal(rng, (batch_size, *input_shape))
        fake_samples = jax.random.normal(jax.random.split(rng)[0], (batch_size, *input_shape))

        # Compute gradient penalty
        gp = compute_gradient_penalty(
            discriminator=discriminator,
            real_samples=real_samples,
            fake_samples=fake_samples,
            rngs=nnx.Rngs(jax.random.split(rng)[1]),
            lambda_gp=10.0,
        )

        # Check that gradient penalty is finite and non-negative
        assert jnp.isfinite(gp)
        assert gp >= 0.0

    def test_gradient_penalty_different_lambda(self, discriminator, rng, input_shape):
        """Test gradient penalty with different lambda values."""
        batch_size = 2
        real_samples = jax.random.normal(rng, (batch_size, *input_shape))
        fake_samples = jax.random.normal(jax.random.split(rng)[0], (batch_size, *input_shape))

        # Test with different lambda values
        for lambda_gp in [1.0, 10.0, 100.0]:
            gp = compute_gradient_penalty(
                discriminator=discriminator,
                real_samples=real_samples,
                fake_samples=fake_samples,
                rngs=nnx.Rngs(jax.random.split(rng)[1]),
                lambda_gp=lambda_gp,
            )

            assert jnp.isfinite(gp)
            assert gp >= 0.0


class TestWGAN:
    """Test cases for complete WGAN model."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def image_shape(self):
        """Image shape fixture."""
        return (3, 32, 32)

    @pytest.fixture
    def wgan_config(self, image_shape):
        """WGAN configuration fixture."""
        generator = ConvGeneratorConfig(
            name="test_wgan_gen",
            latent_dim=64,
            output_shape=image_shape,
            hidden_dims=(128, 64, 32),  # 3 dims for 32x32 output
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
            name="test_wgan_disc",
            input_shape=image_shape,
            hidden_dims=(64, 128, 256),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )
        return WGANConfig(
            name="test_wgan",
            generator=generator,
            discriminator=discriminator,
            critic_iterations=5,
            use_gradient_penalty=True,
        )

    @pytest.fixture
    def wgan(self, rng, wgan_config):
        """WGAN fixture."""
        return WGAN(config=wgan_config, rngs=nnx.Rngs(rng))

    def test_wgan_initialization(self, wgan):
        """Test WGAN initialization."""
        assert wgan.lambda_gp == 10.0
        assert wgan.n_critic == 5
        assert isinstance(wgan.generator, WGANGenerator)
        assert isinstance(wgan.discriminator, WGANDiscriminator)

    def test_wgan_generation(self, wgan, rng):
        """Test WGAN sample generation."""
        batch_size = 4

        # Generate samples
        wgan.eval()
        samples = wgan.generate(batch_size, rngs=nnx.Rngs(sample=rng))

        # Check shape
        assert samples.shape == (batch_size, 3, 32, 32)

        # Check range (tanh activation)
        assert jnp.all(samples >= -1.0)
        assert jnp.all(samples <= 1.0)

    def test_wgan_discriminator_forward(self, wgan, rng):
        """Test WGAN discriminator forward pass."""
        batch_size = 4
        real_samples = jax.random.normal(rng, (batch_size, 3, 32, 32))

        # Get discriminator scores
        wgan.eval()
        scores = wgan.discriminator(real_samples)

        # Check shape
        assert scores.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(scores))

    def test_wgan_discriminator_loss(self, wgan, rng):
        """Test WGAN discriminator loss computation."""
        batch_size = 4
        real_samples = jax.random.normal(rng, (batch_size, 3, 32, 32))
        fake_samples = jax.random.normal(jax.random.split(rng)[0], (batch_size, 3, 32, 32))

        # Compute discriminator loss
        wgan.train()
        disc_loss = wgan.discriminator_loss(
            real_samples, fake_samples, rngs=nnx.Rngs(jax.random.split(rng)[1])
        )

        # Loss should be finite
        assert jnp.isfinite(disc_loss)

    def test_wgan_generator_loss(self, wgan, rng):
        """Test WGAN generator loss computation."""
        batch_size = 4
        fake_samples = jax.random.normal(rng, (batch_size, 3, 32, 32))

        # Compute generator loss
        wgan.train()
        gen_loss = wgan.generator_loss(fake_samples)

        # Loss should be finite
        assert jnp.isfinite(gen_loss)

    def test_wgan_different_image_sizes(self, rng):
        """Test WGAN with different image sizes."""
        test_configs = [
            (32, (128, 64, 32), (64, 128)),  # 32x32: needs 3 hidden dims
            (64, (256, 128, 64, 32), (64, 128, 256)),  # 64x64: needs 4 hidden dims
        ]

        for image_size, gen_hidden_dims, disc_hidden_dims in test_configs:
            generator = ConvGeneratorConfig(
                name=f"test_wgan_gen_{image_size}",
                latent_dim=64,
                output_shape=(3, image_size, image_size),
                hidden_dims=gen_hidden_dims,
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
                name=f"test_wgan_disc_{image_size}",
                input_shape=(3, image_size, image_size),
                hidden_dims=disc_hidden_dims,
                activation="leaky_relu",
                batch_norm=False,
                dropout_rate=0.0,
                leaky_relu_slope=0.2,
                use_spectral_norm=False,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding="SAME",
                use_instance_norm=True,
            )
            config = WGANConfig(
                name=f"test_wgan_{image_size}",
                generator=generator,
                discriminator=discriminator,
                critic_iterations=5,
                use_gradient_penalty=True,
            )

            model = WGAN(config=config, rngs=nnx.Rngs(rng))

            # Test generation
            batch_size = 2
            model.eval()
            samples = model.generate(batch_size, rngs=nnx.Rngs(sample=rng))

            assert samples.shape == (batch_size, 3, image_size, image_size)

    def test_wgan_generator_discriminator_consistency(self, wgan, rng):
        """Test that generator output is compatible with discriminator input."""
        batch_size = 4

        # Generate fake samples
        wgan.eval()
        fake_samples = wgan.generate(batch_size, rngs=nnx.Rngs(sample=rng))

        # Discriminate fake samples
        fake_scores = wgan.discriminator(fake_samples)

        # Check shapes are compatible
        assert fake_samples.shape == (batch_size, 3, 32, 32)
        assert fake_scores.shape == (batch_size,)

        # All scores should be valid
        assert jnp.all(jnp.isfinite(fake_scores))


class TestWGANIntegration:
    """Integration tests for WGAN."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(123)

    def test_wgan_end_to_end(self, rng):
        """Test end-to-end WGAN training step."""
        latent_dim = 32
        batch_size = 2

        generator = ConvGeneratorConfig(
            name="test_wgan_e2e_gen",
            latent_dim=latent_dim,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32, 16),  # 3 dims for 32x32 output
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
            name="test_wgan_e2e_disc",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )
        config = WGANConfig(
            name="test_wgan_e2e",
            generator=generator,
            discriminator=discriminator,
            critic_iterations=5,
            use_gradient_penalty=True,
        )

        wgan = WGAN(config=config, rngs=nnx.Rngs(rng))
        image_shape = (3, 32, 32)

        # Create training data
        real_images = jax.random.normal(rng, (batch_size, *image_shape))

        # Generate fake images
        wgan.train()
        fake_images = wgan.generate(batch_size, rngs=nnx.Rngs(sample=jax.random.split(rng)[0]))

        # Compute losses
        disc_loss = wgan.discriminator_loss(
            real_images, fake_images, rngs=nnx.Rngs(jax.random.split(rng)[1])
        )
        gen_loss = wgan.generator_loss(fake_images)

        # Verify all outputs are valid
        assert jnp.isfinite(disc_loss)
        assert jnp.isfinite(gen_loss)
        assert jnp.all(fake_images >= -1.0)
        assert jnp.all(fake_images <= 1.0)


# ============================================================================
# JIT Compatibility Tests
# ============================================================================


class TestWGANGeneratorJIT:
    """JIT compatibility tests for WGAN Generator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def generator(self, rng):
        """Generator fixture for JIT tests."""
        config = ConvGeneratorConfig(
            name="test_wgan_jit_gen",
            latent_dim=32,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32),  # 2 hidden dims
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )
        return WGANGenerator(config=config, rngs=nnx.Rngs(rng))

    def test_jit_forward_pass(self, generator, rng):
        """Test that generator forward pass is JIT compatible."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        z = jax.random.normal(rng, (4, 32))
        generator.eval()
        output_regular = generator(z)
        output_jit = generate_jit(generator, z)

        # GPU floating-point arithmetic with JIT can produce small numerical differences
        # due to XLA operation reordering and fusion. Use realistic tolerances for GPU.
        # Generator has multiple conv + batch norm layers which accumulate numerical error.
        assert jnp.allclose(output_regular, output_jit, rtol=1e-2, atol=1e-3)

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


class TestWGANDiscriminatorJIT:
    """JIT compatibility tests for WGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def discriminator(self, rng):
        """Discriminator fixture for JIT tests."""
        config = ConvDiscriminatorConfig(
            name="test_wgan_jit_disc",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )
        return WGANDiscriminator(config=config, rngs=nnx.Rngs(rng))

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
        assert output.shape == (4,)

    def test_jit_multiple_calls_consistent(self, discriminator, rng):
        """Test that multiple JIT calls produce consistent results."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        discriminator.eval()

        output1 = discriminate_jit(discriminator, x)
        output2 = discriminate_jit(discriminator, x)

        # JIT calls should be exactly consistent
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
            assert output.shape == (batch_size,)


class TestWGANJIT:
    """JIT compatibility tests for complete WGAN model."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def wgan(self, rng):
        """WGAN fixture for JIT tests."""
        generator = ConvGeneratorConfig(
            name="test_wgan_jit_gen",
            latent_dim=32,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32, 16),  # 3 hidden dims for 32x32 output
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
            name="test_wgan_jit_disc",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            use_instance_norm=True,
        )
        config = WGANConfig(
            name="test_wgan_jit",
            generator=generator,
            discriminator=discriminator,
            critic_iterations=5,
            use_gradient_penalty=True,
        )
        return WGAN(config=config, rngs=nnx.Rngs(rng))

    def test_jit_generator_loss(self, wgan, rng):
        """Test that generator loss computation is JIT compatible."""

        @nnx.jit
        def compute_gen_loss_jit(model, fake_samples):
            return model.generator_loss(fake_samples)

        fake_samples = jax.random.normal(rng, (4, 3, 32, 32))
        wgan.train()

        loss_regular = wgan.generator_loss(fake_samples)
        loss_jit = compute_gen_loss_jit(wgan, fake_samples)

        assert jnp.allclose(loss_regular, loss_jit, rtol=1e-3, atol=1e-4)

    def test_jit_discriminator_loss(self, wgan, rng):
        """Test that discriminator loss computation is JIT compatible."""

        @nnx.jit
        def compute_disc_loss_jit(model, real_samples, fake_samples, rngs):
            return model.discriminator_loss(real_samples, fake_samples, rngs)

        real_samples = jax.random.normal(rng, (4, 3, 32, 32))
        fake_samples = jax.random.normal(jax.random.split(rng)[0], (4, 3, 32, 32))
        wgan.train()

        loss_regular = wgan.discriminator_loss(
            real_samples, fake_samples, rngs=nnx.Rngs(jax.random.split(rng)[1])
        )
        loss_jit = compute_disc_loss_jit(
            wgan, real_samples, fake_samples, rngs=nnx.Rngs(jax.random.split(rng)[1])
        )

        # Allow more tolerance for discriminator loss due to gradient penalty
        assert jnp.allclose(loss_regular, loss_jit, rtol=1e-3, atol=1e-4)

    def test_jit_full_training_step(self, wgan, rng):
        """Test that a full training step is JIT compatible."""

        @nnx.jit
        def training_step_jit(model, real_samples, z, rngs):
            fake_samples = model.generator(z)
            disc_loss = model.discriminator_loss(real_samples, fake_samples, rngs)
            gen_loss = model.generator_loss(fake_samples)
            return disc_loss, gen_loss

        real_samples = jax.random.normal(rng, (4, 3, 32, 32))
        z = jax.random.normal(jax.random.split(rng)[0], (4, 32))
        wgan.train()

        disc_loss, gen_loss = training_step_jit(
            wgan, real_samples, z, rngs=nnx.Rngs(jax.random.split(rng)[1])
        )
        assert jnp.isfinite(disc_loss)
        assert jnp.isfinite(gen_loss)

    def test_jit_gradient_computation(self, wgan, rng):
        """Test that gradient computation is JIT compatible."""

        def loss_fn(model, real_samples, fake_samples, rngs):
            return model.discriminator_loss(real_samples, fake_samples, rngs)

        @nnx.jit
        def compute_gradients_jit(model, real_samples, fake_samples, rngs):
            grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
            loss, _ = grad_fn(model, real_samples, fake_samples, rngs)
            return loss

        real_samples = jax.random.normal(rng, (4, 3, 32, 32))
        fake_samples = jax.random.normal(jax.random.split(rng)[0], (4, 3, 32, 32))
        wgan.train()

        loss = compute_gradients_jit(
            wgan, real_samples, fake_samples, rngs=nnx.Rngs(jax.random.split(rng)[1])
        )
        assert jnp.isfinite(loss)
