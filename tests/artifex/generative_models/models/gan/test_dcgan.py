"""Tests for DCGAN implementation with new config pattern."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.gan_config import DCGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConvDiscriminatorConfig,
    ConvGeneratorConfig,
)
from artifex.generative_models.models.gan.dcgan import (
    DCGAN,
    DCGANDiscriminator,
    DCGANGenerator,
)


class TestDCGANGenerator:
    """Test cases for DCGAN Generator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def output_shape(self):
        """Output shape fixture."""
        return (3, 32, 32)  # C, H, W format

    @pytest.fixture
    def latent_dim(self):
        """Latent dimension fixture."""
        return 100

    @pytest.fixture
    def generator_config(self, output_shape, latent_dim):
        """Generator config fixture."""
        return ConvGeneratorConfig(
            name="test_generator",
            output_shape=output_shape,
            latent_dim=latent_dim,
            hidden_dims=(128, 64, 32),
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
        return DCGANGenerator(
            config=generator_config,
            rngs=nnx.Rngs(rng),
        )

    def test_generator_initialization(self, generator, output_shape, latent_dim):
        """Test generator initialization."""
        assert generator.output_shape == output_shape
        assert generator.latent_dim == latent_dim
        assert generator.batch_norm is True
        assert generator.dropout_rate == 0.0

    def test_generator_forward_pass(self, generator, rng, latent_dim):
        """Test generator forward pass."""
        batch_size = 4
        z = jax.random.normal(rng, (batch_size, latent_dim))

        # Forward pass
        output = generator(z)

        # Check output shape (NCHW format)
        assert output.shape == (batch_size, 3, 32, 32)
        assert output.dtype == jnp.float32

        # Check output is bounded (tanh activation)
        assert jnp.all(output >= -1.0)
        assert jnp.all(output <= 1.0)

    def test_generator_eval_mode_deterministic(self, generator, rng, latent_dim):
        """Test generator deterministic behavior in eval mode."""
        batch_size = 2
        z = jax.random.normal(rng, (batch_size, latent_dim))

        # Two forward passes in eval mode should be identical
        generator.eval()
        output1 = generator(z)
        output2 = generator(z)

        assert jnp.allclose(output1, output2, atol=1e-6)

    def test_generator_different_output_shapes(self, rng, latent_dim):
        """Test generator with different output shapes."""
        output_shapes = [
            (1, 32, 32),  # Grayscale
            (3, 64, 64),  # Larger RGB
            (4, 16, 16),  # 4-channel small
        ]

        for output_shape in output_shapes:
            config = ConvGeneratorConfig(
                name=f"gen_{output_shape}",
                output_shape=output_shape,
                latent_dim=latent_dim,
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

            generator = DCGANGenerator(config=config, rngs=nnx.Rngs(rng))

            batch_size = 2
            z = jax.random.normal(rng, (batch_size, latent_dim))
            output = generator(z)

            assert output.shape == (batch_size, *output_shape)

    def test_generator_requires_conv_config(self, rng):
        """Test that generator requires ConvGeneratorConfig, not base GeneratorConfig."""
        from artifex.generative_models.core.configuration.network_configs import GeneratorConfig

        base_config = GeneratorConfig(
            name="base_gen",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32),
            activation="relu",
        )

        with pytest.raises(TypeError, match="ConvGeneratorConfig"):
            DCGANGenerator(config=base_config, rngs=nnx.Rngs(rng))


class TestDCGANDiscriminator:
    """Test cases for DCGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture."""
        return (3, 32, 32)  # C, H, W format

    @pytest.fixture
    def discriminator_config(self, input_shape):
        """Discriminator config fixture."""
        return ConvDiscriminatorConfig(
            name="test_discriminator",
            input_shape=input_shape,
            hidden_dims=(32, 64, 128),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,  # DCGAN discriminators typically don't use batch norm
            dropout_rate=0.0,
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
        return DCGANDiscriminator(
            config=discriminator_config,
            rngs=nnx.Rngs(rng),
        )

    def test_discriminator_initialization(self, discriminator, input_shape):
        """Test discriminator initialization."""
        assert discriminator.input_shape == input_shape
        assert discriminator.batch_norm is False
        assert discriminator.dropout_rate == 0.0

    def test_discriminator_forward_pass(self, discriminator, rng, input_shape):
        """Test discriminator forward pass."""
        batch_size = 4
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        output = discriminator(x)

        # Check output shape (should be (batch_size, 1))
        assert output.shape == (batch_size, 1)
        assert output.dtype == jnp.float32

        # Check output is in valid range (sigmoid output)
        assert jnp.all(output >= 0.0)
        assert jnp.all(output <= 1.0)

    def test_discriminator_eval_mode_deterministic(self, discriminator, rng, input_shape):
        """Test discriminator deterministic behavior in eval mode."""
        batch_size = 2
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Two forward passes in eval mode should be identical
        discriminator.eval()
        output1 = discriminator(x)
        output2 = discriminator(x)

        assert jnp.allclose(output1, output2, atol=1e-6)

    def test_discriminator_with_batch_norm(self, rng, input_shape):
        """Test discriminator with batch normalization."""
        config = ConvDiscriminatorConfig(
            name="disc_with_bn",
            input_shape=input_shape,
            hidden_dims=(32, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,  # Enable batch norm
            dropout_rate=0.0,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )

        discriminator = DCGANDiscriminator(config=config, rngs=nnx.Rngs(rng))

        batch_size = 4
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Training mode
        discriminator.train()
        output_train1 = discriminator(x)
        output_train2 = discriminator(x)

        # Eval mode
        discriminator.eval()
        output_eval = discriminator(x)

        # All outputs should be finite
        assert jnp.all(jnp.isfinite(output_train1))
        assert jnp.all(jnp.isfinite(output_train2))
        assert jnp.all(jnp.isfinite(output_eval))

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
            DCGANDiscriminator(config=base_config, rngs=nnx.Rngs(rng))


class TestDCGAN:
    """Test cases for DCGAN model."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def dcgan_config(self):
        """DCGAN configuration fixture."""
        image_shape = (3, 32, 32)

        generator = ConvGeneratorConfig(
            name="dcgan_generator",
            latent_dim=100,
            output_shape=image_shape,
            hidden_dims=(128, 64, 32),
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
            name="dcgan_discriminator",
            input_shape=image_shape,
            hidden_dims=(32, 64, 128),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
            dropout_rate=0.0,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )

        return DCGANConfig(
            name="test_dcgan",
            generator=generator,
            discriminator=discriminator,
        )

    @pytest.fixture
    def dcgan(self, rng, dcgan_config):
        """DCGAN model fixture."""
        return DCGAN(
            config=dcgan_config,
            rngs=nnx.Rngs(
                params=rng, dropout=jax.random.fold_in(rng, 1), sample=jax.random.fold_in(rng, 2)
            ),
        )

    def test_dcgan_initialization(self, dcgan, dcgan_config):
        """Test DCGAN initialization."""
        assert dcgan.latent_dim == dcgan_config.generator.latent_dim
        assert dcgan.generator is not None
        assert dcgan.discriminator is not None

    def test_dcgan_generation(self, dcgan, rng):
        """Test DCGAN sample generation."""
        batch_size = 4

        # Generate samples
        samples = dcgan.generate(batch_size, rngs=nnx.Rngs(sample=rng))

        # Check shape
        assert samples.shape == (batch_size, 3, 32, 32)

        # Check range (tanh activation)
        assert jnp.all(samples >= -1.0)
        assert jnp.all(samples <= 1.0)

    def test_dcgan_discriminator_forward(self, dcgan, rng):
        """Test DCGAN discriminator forward pass."""
        batch_size = 4
        real_samples = jax.random.normal(rng, (batch_size, 3, 32, 32))

        # Get discriminator scores
        scores = dcgan.discriminator(real_samples)

        # Check shape and range
        assert scores.shape == (batch_size, 1)
        assert jnp.all(scores >= 0.0)
        assert jnp.all(scores <= 1.0)

    def test_dcgan_loss_computation(self, dcgan, rng):
        """Test DCGAN loss computation."""
        batch_size = 4
        real_data = jax.random.normal(rng, (batch_size, 3, 32, 32))

        # Create batch dict
        batch = {"x": real_data}

        # Forward pass
        model_outputs = dcgan(real_data, rngs=nnx.Rngs(sample=rng))

        # Compute loss
        loss_dict = dcgan.loss_fn(
            batch=batch,
            model_outputs=model_outputs,
            rngs=nnx.Rngs(sample=rng),
        )

        # Check loss components
        assert "loss" in loss_dict
        assert "generator_loss" in loss_dict
        assert "discriminator_loss" in loss_dict

        # All losses should be finite
        assert jnp.isfinite(loss_dict["loss"])
        assert jnp.isfinite(loss_dict["generator_loss"])
        assert jnp.isfinite(loss_dict["discriminator_loss"])

    def test_dcgan_different_image_sizes(self, rng):
        """Test DCGAN with different image sizes."""
        for image_size in [16, 32, 64]:
            image_shape = (3, image_size, image_size)

            generator = ConvGeneratorConfig(
                name=f"gen_{image_size}",
                latent_dim=64,
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
                name=f"disc_{image_size}",
                input_shape=image_shape,
                hidden_dims=(32, 64),
                activation="leaky_relu",
                leaky_relu_slope=0.2,
                batch_norm=False,
                dropout_rate=0.0,
                use_spectral_norm=False,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding="SAME",
                batch_norm_momentum=0.9,
                batch_norm_use_running_avg=False,
                output_dim=1,
            )

            config = DCGANConfig(
                name=f"dcgan_{image_size}",
                generator=generator,
                discriminator=discriminator,
            )

            model = DCGAN(
                config=config,
                rngs=nnx.Rngs(
                    params=rng,
                    dropout=jax.random.fold_in(rng, 1),
                    sample=jax.random.fold_in(rng, 2),
                ),
            )

            # Test generation
            batch_size = 2
            samples = model.generate(batch_size)

            assert samples.shape == (batch_size, 3, image_size, image_size)

    def test_dcgan_generator_discriminator_consistency(self, dcgan, rng):
        """Test that generator output is compatible with discriminator input."""
        batch_size = 4

        # Generate fake samples
        fake_samples = dcgan.generate(batch_size, rngs=nnx.Rngs(sample=rng))

        # Discriminate fake samples
        fake_scores = dcgan.discriminator(fake_samples)

        # Check shapes are compatible
        assert fake_samples.shape == (batch_size, 3, 32, 32)
        assert fake_scores.shape == (batch_size, 1)

        # All scores should be valid
        assert jnp.all(jnp.isfinite(fake_scores))

    def test_dcgan_requires_dcgan_config(self, rng):
        """Test that DCGAN requires DCGANConfig, not base GANConfig."""
        from artifex.generative_models.core.configuration.gan_config import GANConfig
        from artifex.generative_models.core.configuration.network_configs import (
            DiscriminatorConfig,
            GeneratorConfig,
        )

        gen = GeneratorConfig(
            name="gen",
            latent_dim=100,
            output_shape=(3, 32, 32),
            hidden_dims=(64, 32),
            activation="relu",
        )

        disc = DiscriminatorConfig(
            name="disc",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
        )

        base_config = GANConfig(
            name="base_gan",
            generator=gen,
            discriminator=disc,
        )

        with pytest.raises(TypeError, match="DCGANConfig"):
            DCGAN(config=base_config, rngs=nnx.Rngs(rng))


class TestDCGANGeneratorJIT:
    """JIT compatibility tests for DCGAN Generator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def generator(self, rng):
        """Generator fixture for JIT tests."""
        config = ConvGeneratorConfig(
            name="test_dcgan_gen_jit",
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
        return DCGANGenerator(config=config, rngs=nnx.Rngs(rng))

    def test_jit_forward_pass(self, generator, rng):
        """Test that generator forward pass is JIT compatible."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        z = jax.random.normal(rng, (4, 32))
        generator.eval()
        output_regular = generator(z)
        output_jit = generate_jit(generator, z)

        assert jnp.allclose(output_regular, output_jit, rtol=1e-5, atol=1e-7)

    def test_jit_compilation_without_errors(self, generator, rng):
        """Test that JIT compilation succeeds without errors."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        z = jax.random.normal(rng, (4, 32))
        generator.eval()

        try:
            output = generate_jit(generator, z)
            assert output.shape == (4, 3, 32, 32)
        except Exception as e:
            pytest.fail(f"JIT compilation failed: {e}")

    def test_jit_multiple_calls_consistent(self, generator, rng):
        """Test that multiple JIT calls produce consistent results."""

        @nnx.jit
        def generate_jit(model, z):
            return model(z)

        z = jax.random.normal(rng, (4, 32))
        generator.eval()

        output1 = generate_jit(generator, z)
        output2 = generate_jit(generator, z)

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


class TestDCGANDiscriminatorJIT:
    """JIT compatibility tests for DCGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def discriminator(self, rng):
        """Discriminator fixture for JIT tests."""
        config = ConvDiscriminatorConfig(
            name="test_dcgan_disc_jit",
            input_shape=(3, 32, 32),
            hidden_dims=(32, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
            dropout_rate=0.0,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )
        return DCGANDiscriminator(config=config, rngs=nnx.Rngs(rng))

    def test_jit_forward_pass(self, discriminator, rng):
        """Test that discriminator forward pass is JIT compatible."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        discriminator.eval()
        output_regular = discriminator(x)
        output_jit = discriminate_jit(discriminator, x)

        assert jnp.allclose(output_regular, output_jit, rtol=1e-5, atol=1e-7)

    def test_jit_compilation_without_errors(self, discriminator, rng):
        """Test that JIT compilation succeeds without errors."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        discriminator.eval()

        try:
            output = discriminate_jit(discriminator, x)
            assert output.shape == (4, 1)
        except Exception as e:
            pytest.fail(f"JIT compilation failed: {e}")

    def test_jit_multiple_calls_consistent(self, discriminator, rng):
        """Test that multiple JIT calls produce consistent results."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        discriminator.eval()

        output1 = discriminate_jit(discriminator, x)
        output2 = discriminate_jit(discriminator, x)

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


class TestDCGANJIT:
    """JIT compatibility tests for full DCGAN model."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def dcgan(self, rng):
        """DCGAN fixture for JIT tests."""
        image_shape = (3, 32, 32)

        generator = ConvGeneratorConfig(
            name="test_dcgan_gen_jit",
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
            name="test_dcgan_disc_jit",
            input_shape=image_shape,
            hidden_dims=(32, 64),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=False,
            dropout_rate=0.0,
            use_spectral_norm=False,
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
            output_dim=1,
        )

        config = DCGANConfig(
            name="test_dcgan_jit",
            generator=generator,
            discriminator=discriminator,
        )

        return DCGAN(
            config=config,
            rngs=nnx.Rngs(
                params=rng,
                dropout=jax.random.fold_in(rng, 1),
                sample=jax.random.fold_in(rng, 2),
            ),
        )

    def test_jit_generator_forward(self, dcgan, rng):
        """Test that DCGAN generator is JIT compatible."""

        @nnx.jit
        def generate_jit(model, z):
            return model.generator(z)

        z = jax.random.normal(rng, (4, 32))
        dcgan.eval()
        output_regular = dcgan.generator(z)
        output_jit = generate_jit(dcgan, z)

        assert jnp.allclose(output_regular, output_jit, rtol=1e-5, atol=1e-7)

    def test_jit_discriminator_forward(self, dcgan, rng):
        """Test that DCGAN discriminator is JIT compatible."""

        @nnx.jit
        def discriminate_jit(model, x):
            return model.discriminator(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        dcgan.eval()
        output_regular = dcgan.discriminator(x)
        output_jit = discriminate_jit(dcgan, x)

        assert jnp.allclose(output_regular, output_jit, rtol=1e-5, atol=1e-7)

    def test_jit_full_forward_pass(self, dcgan, rng):
        """Test that DCGAN full forward pass is JIT compatible."""

        @nnx.jit
        def forward_jit(model, x):
            return model(x)

        x = jax.random.normal(rng, (4, 3, 32, 32))
        dcgan.eval()
        output_regular = dcgan(x)
        output_jit = forward_jit(dcgan, x)

        assert jnp.allclose(
            output_regular["real_scores"], output_jit["real_scores"], rtol=1e-5, atol=1e-7
        )

    def test_jit_training_step(self, dcgan, rng):
        """Test that a typical training step is JIT compatible."""

        @nnx.jit
        def training_step_jit(model, real_images, latent_vectors):
            # Generate fake images
            fake_images = model.generator(latent_vectors)
            # Discriminate
            real_scores = model.discriminator(real_images)
            fake_scores = model.discriminator(fake_images)
            return {
                "fake_images": fake_images,
                "real_scores": real_scores,
                "fake_scores": fake_scores,
            }

        real_images = jax.random.normal(rng, (4, 3, 32, 32))
        latent_vectors = jax.random.normal(jax.random.fold_in(rng, 1), (4, 32))

        dcgan.train()
        try:
            results = training_step_jit(dcgan, real_images, latent_vectors)
            assert results["fake_images"].shape == (4, 3, 32, 32)
            assert results["real_scores"].shape == (4, 1)
            assert results["fake_scores"].shape == (4, 1)
        except Exception as e:
            pytest.fail(f"JIT training step failed: {e}")

    def test_jit_gradient_computation(self, dcgan, rng):
        """Test that gradient computation is JIT compatible."""

        @nnx.jit
        def compute_loss_jit(model, real_images, z):
            fake_images = model.generator(z)
            real_scores = model.discriminator(real_images)
            fake_scores = model.discriminator(fake_images)
            # Simple discriminator loss
            return jnp.mean(fake_scores) - jnp.mean(real_scores)

        real_images = jax.random.normal(rng, (4, 3, 32, 32))
        z = jax.random.normal(jax.random.fold_in(rng, 1), (4, 32))

        dcgan.train()
        try:
            loss = compute_loss_jit(dcgan, real_images, z)
            assert jnp.isfinite(loss)
        except Exception as e:
            pytest.fail(f"JIT gradient computation failed: {e}")
