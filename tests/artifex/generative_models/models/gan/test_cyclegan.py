"""Tests for CycleGAN implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.gan_config import CycleGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    CycleGANGeneratorConfig,
    PatchGANDiscriminatorConfig,
)
from artifex.generative_models.models.gan.cyclegan import (
    CycleGAN,
    CycleGANDiscriminator,
    CycleGANGenerator,
)


class TestCycleGANGenerator:
    """Test cases for CycleGAN Generator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture."""
        return (32, 32, 3)  # NHWC format for Flax compatibility

    @pytest.fixture
    def generator_config(self, input_shape):
        """Generator config fixture."""
        return CycleGANGeneratorConfig(
            name="test_generator",
            latent_dim=0,  # Not used for image-to-image
            hidden_dims=(16, 32),  # Small for testing
            output_shape=input_shape,  # Same shape for image-to-image translation
            input_shape=input_shape,
            n_residual_blocks=2,  # Fewer residual blocks for testing
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )

    @pytest.fixture
    def generator(self, rng, generator_config):
        """Generator fixture."""
        return CycleGANGenerator(
            config=generator_config,
            rngs=nnx.Rngs(rng),
        )

    def test_generator_initialization(self, generator, input_shape):
        """Test generator initialization."""
        assert generator.input_shape == input_shape
        assert generator.output_shape == input_shape
        assert generator.hidden_dims == [16, 32]
        assert generator.use_skip_connections is True
        assert generator.dropout_rate == 0.0

    def test_generator_forward_pass(self, generator, rng, input_shape):
        """Test generator forward pass."""
        batch_size = 4

        # Create input image in NHWC format
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        output = generator(x)

        # Check output shape
        assert output.shape == (batch_size, *input_shape)
        assert output.dtype == jnp.float32

        # Check output is bounded (tanh activation)
        assert jnp.all(output >= -1.0)
        assert jnp.all(output <= 1.0)

    def test_generator_deterministic_mode(self, generator, rng, input_shape):
        """Test generator deterministic behavior."""
        batch_size = 2
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Two forward passes in eval mode should be identical
        generator.eval()
        output1 = generator(x)
        output2 = generator(x)

        assert jnp.allclose(output1, output2, atol=1e-6)

    def test_generator_different_shapes(self, rng):
        """Test generator with different input/output shapes."""
        input_shape = (64, 64, 3)  # NHWC format: (H, W, C)
        output_shape = (64, 64, 1)  # Different number of channels, NHWC format

        config = CycleGANGeneratorConfig(
            name="test_gen_diff_shapes",
            latent_dim=0,
            input_shape=input_shape,
            output_shape=output_shape,
            hidden_dims=(8, 16),  # Even smaller for testing
            n_residual_blocks=1,  # Minimal residual blocks
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )

        generator = CycleGANGenerator(
            config=config,
            rngs=nnx.Rngs(rng),
        )

        batch_size = 2
        x = jax.random.normal(rng, (batch_size, *input_shape))
        output = generator(x)

        assert output.shape == (batch_size, *output_shape)


class TestCycleGANDiscriminator:
    """Test cases for CycleGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def input_shape(self):
        """Input shape fixture."""
        return (32, 32, 3)  # Small for testing

    @pytest.fixture
    def discriminator_config(self, input_shape):
        """Discriminator config fixture."""
        return PatchGANDiscriminatorConfig(
            name="test_discriminator",
            input_shape=input_shape,
            hidden_dims=(16, 32),  # Small for testing
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            dropout_rate=0.0,
            use_spectral_norm=False,
        )

    @pytest.fixture
    def discriminator(self, rng, discriminator_config):
        """Discriminator fixture."""
        return CycleGANDiscriminator(
            config=discriminator_config,
            rngs=nnx.Rngs(rng),
        )

    def test_discriminator_initialization(self, discriminator, input_shape):
        """Test discriminator initialization."""
        assert discriminator.input_shape == input_shape
        assert discriminator.hidden_dims == [16, 32]
        assert discriminator.dropout_rate == 0.0

    def test_discriminator_forward_pass(self, discriminator, rng, input_shape):
        """Test discriminator forward pass."""
        batch_size = 4

        # Create input image in NHWC format
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward pass
        output = discriminator(x)

        # PatchGAN returns spatial patch map: (batch, H', W', 1)
        assert output.ndim == 4
        assert output.shape[0] == batch_size
        assert output.shape[-1] == 1
        assert output.dtype == jnp.float32

    def test_discriminator_deterministic_mode(self, discriminator, rng, input_shape):
        """Test discriminator deterministic behavior."""
        batch_size = 2
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Two forward passes in eval mode should be identical
        discriminator.eval()
        output1 = discriminator(x)
        output2 = discriminator(x)

        assert jnp.allclose(output1, output2, atol=1e-6)

    def test_discriminator_batch_norm_training(self, discriminator, rng, input_shape):
        """Test discriminator batch norm behavior in training mode."""
        batch_size = 4
        x = jax.random.normal(rng, (batch_size, *input_shape))

        # Forward passes in training mode might give different results
        # due to batch normalization updates
        discriminator.train()
        output1 = discriminator(x)
        output2 = discriminator(x)

        # Outputs should be finite
        assert jnp.isfinite(output1).all()
        assert jnp.isfinite(output2).all()


class TestCycleGAN:
    """Test cases for CycleGAN model."""

    @pytest.fixture
    def rng(self):
        """Random number generator fixture."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def input_shape_a(self):
        """Domain A input shape fixture."""
        return (32, 32, 3)  # NHWC format

    @pytest.fixture
    def input_shape_b(self):
        """Domain B input shape fixture."""
        return (32, 32, 3)  # NHWC format

    @pytest.fixture
    def cyclegan(self, rng, input_shape_a, input_shape_b):
        """CycleGAN fixture."""
        # Create nested configs for both generators
        generators = {
            "a_to_b": CycleGANGeneratorConfig(
                name="gen_a_to_b",
                latent_dim=0,  # Not used for image-to-image
                input_shape=input_shape_a,
                output_shape=input_shape_b,
                hidden_dims=(16, 32),  # Small for testing
                n_residual_blocks=2,
                activation="relu",
                batch_norm=True,
                dropout_rate=0.0,
            ),
            "b_to_a": CycleGANGeneratorConfig(
                name="gen_b_to_a",
                latent_dim=0,  # Not used for image-to-image
                input_shape=input_shape_b,
                output_shape=input_shape_a,
                hidden_dims=(16, 32),  # Small for testing
                n_residual_blocks=2,
                activation="relu",
                batch_norm=True,
                dropout_rate=0.0,
            ),
        }

        # Create nested configs for both discriminators
        discriminators = {
            "disc_a": PatchGANDiscriminatorConfig(
                name="disc_a",
                input_shape=input_shape_a,
                hidden_dims=(16, 32),  # Small for testing
                activation="leaky_relu",
                batch_norm=False,
                dropout_rate=0.0,
                leaky_relu_slope=0.2,
                use_spectral_norm=False,
            ),
            "disc_b": PatchGANDiscriminatorConfig(
                name="disc_b",
                input_shape=input_shape_b,
                hidden_dims=(16, 32),  # Small for testing
                activation="leaky_relu",
                batch_norm=False,
                dropout_rate=0.0,
                leaky_relu_slope=0.2,
                use_spectral_norm=False,
            ),
        }

        config = CycleGANConfig(
            name="test_cyclegan",
            generator=generators,
            discriminator=discriminators,
            input_shape_a=input_shape_a,
            input_shape_b=input_shape_b,
            lambda_cycle=10.0,
            lambda_identity=0.5,
        )
        return CycleGAN(
            config=config,
            rngs=nnx.Rngs(rng),
        )

    def test_cyclegan_initialization(self, cyclegan, input_shape_a, input_shape_b):
        """Test CycleGAN initialization."""
        assert cyclegan.input_shape_a == input_shape_a
        assert cyclegan.input_shape_b == input_shape_b
        assert cyclegan.lambda_cycle == 10.0
        assert cyclegan.lambda_identity == 0.5

        # Check that all networks are initialized
        assert cyclegan.generator_a_to_b is not None
        assert cyclegan.generator_b_to_a is not None
        assert cyclegan.discriminator_a is not None
        assert cyclegan.discriminator_b is not None

    def test_cyclegan_forward_cycle(self, cyclegan, rng, input_shape_a):
        """Test forward cycle: A -> B -> A."""
        batch_size = 2
        real_a = jax.random.normal(rng, (batch_size, *input_shape_a))

        # Forward cycle
        fake_b = cyclegan.generator_a_to_b(real_a)
        reconstructed_a = cyclegan.generator_b_to_a(fake_b)

        # Check shapes
        assert fake_b.shape == (batch_size, *input_shape_a)
        assert reconstructed_a.shape == (batch_size, *input_shape_a)

        # Check outputs are bounded
        assert jnp.all(fake_b >= -1.0) and jnp.all(fake_b <= 1.0)
        assert jnp.all(reconstructed_a >= -1.0) and jnp.all(reconstructed_a <= 1.0)

    def test_cyclegan_backward_cycle(self, cyclegan, rng, input_shape_b):
        """Test backward cycle: B -> A -> B."""
        batch_size = 2
        real_b = jax.random.normal(rng, (batch_size, *input_shape_b))

        # Backward cycle
        fake_a = cyclegan.generator_b_to_a(real_b)
        reconstructed_b = cyclegan.generator_a_to_b(fake_a)

        # Check shapes
        assert fake_a.shape == (batch_size, *input_shape_b)
        assert reconstructed_b.shape == (batch_size, *input_shape_b)

        # Check outputs are bounded
        assert jnp.all(fake_a >= -1.0) and jnp.all(fake_a <= 1.0)
        assert jnp.all(reconstructed_b >= -1.0) and jnp.all(reconstructed_b <= 1.0)

    def test_cyclegan_discriminators(self, cyclegan, rng, input_shape_a, input_shape_b):
        """Test discriminator forward passes."""
        batch_size = 2
        real_a = jax.random.normal(rng, (batch_size, *input_shape_a))
        real_b = jax.random.normal(rng, (batch_size, *input_shape_b))

        # Test discriminators
        score_a = cyclegan.discriminator_a(real_a)
        score_b = cyclegan.discriminator_b(real_b)

        # Check shapes and types — PatchGAN returns spatial patch map (batch, H', W', 1)
        assert score_a.ndim == 4
        assert score_a.shape[0] == batch_size
        assert score_a.shape[-1] == 1
        assert score_b.ndim == 4
        assert score_b.shape[0] == batch_size
        assert score_b.shape[-1] == 1
        assert score_a.dtype == jnp.float32
        assert score_b.dtype == jnp.float32

        # Check scores are finite
        assert jnp.isfinite(score_a).all()
        assert jnp.isfinite(score_b).all()

    def test_cyclegan_loss_computation(self, cyclegan, rng, input_shape_a, input_shape_b):
        """Test loss computation with cycle consistency."""
        batch_size = 2
        real_a = jax.random.normal(rng, (batch_size, *input_shape_a))
        real_b = jax.random.normal(rng, (batch_size, *input_shape_b))

        # Compute full cycle
        cycle_loss_a, cycle_loss_b = cyclegan.compute_cycle_loss(real_a, real_b)
        total_cycle_loss = cycle_loss_a + cycle_loss_b

        # Check that losses are valid
        assert jnp.isfinite(cycle_loss_a) and cycle_loss_a >= 0
        assert jnp.isfinite(cycle_loss_b) and cycle_loss_b >= 0
        assert jnp.isfinite(total_cycle_loss) and total_cycle_loss >= 0

    def test_cyclegan_identity_loss(self, cyclegan, rng, input_shape_a, input_shape_b):
        """Test identity loss computation."""
        batch_size = 2
        real_a = jax.random.normal(rng, (batch_size, *input_shape_a))
        real_b = jax.random.normal(rng, (batch_size, *input_shape_b))

        # Identity mappings (should preserve input when translating to same domain)
        identity_a = cyclegan.generator_b_to_a(real_a)  # A -> A via B->A generator
        identity_b = cyclegan.generator_a_to_b(real_b)  # B -> B via A->B generator

        # Compute identity losses
        identity_loss_a = jnp.mean(jnp.abs(real_a - identity_a))
        identity_loss_b = jnp.mean(jnp.abs(real_b - identity_b))

        # Check losses are finite and non-negative
        assert jnp.isfinite(identity_loss_a) and identity_loss_a >= 0
        assert jnp.isfinite(identity_loss_b) and identity_loss_b >= 0

    def test_cyclegan_different_domain_shapes(self, rng):
        """Test CycleGAN with different domain shapes."""
        input_shape_a = (32, 32, 3)  # RGB images, NHWC format
        input_shape_b = (32, 32, 1)  # Grayscale images, NHWC format

        # Create nested configs for both generators
        generators = {
            "a_to_b": CycleGANGeneratorConfig(
                name="gen_a_to_b_diff",
                latent_dim=0,
                input_shape=input_shape_a,
                output_shape=input_shape_b,
                hidden_dims=(8, 16),
                n_residual_blocks=1,
                activation="relu",
                batch_norm=True,
                dropout_rate=0.0,
            ),
            "b_to_a": CycleGANGeneratorConfig(
                name="gen_b_to_a_diff",
                latent_dim=0,
                input_shape=input_shape_b,
                output_shape=input_shape_a,
                hidden_dims=(8, 16),
                n_residual_blocks=1,
                activation="relu",
                batch_norm=True,
                dropout_rate=0.0,
            ),
        }

        # Create nested configs for both discriminators
        discriminators = {
            "disc_a": PatchGANDiscriminatorConfig(
                name="disc_a_diff",
                input_shape=input_shape_a,
                hidden_dims=(4, 8),
                activation="leaky_relu",
                batch_norm=False,
                dropout_rate=0.0,
                leaky_relu_slope=0.2,
                use_spectral_norm=False,
            ),
            "disc_b": PatchGANDiscriminatorConfig(
                name="disc_b_diff",
                input_shape=input_shape_b,
                hidden_dims=(4, 8),
                activation="leaky_relu",
                batch_norm=False,
                dropout_rate=0.0,
                leaky_relu_slope=0.2,
                use_spectral_norm=False,
            ),
        }

        config = CycleGANConfig(
            name="test_cyclegan_different_shapes",
            generator=generators,
            discriminator=discriminators,
            input_shape_a=input_shape_a,
            input_shape_b=input_shape_b,
            lambda_cycle=10.0,
            lambda_identity=0.5,
        )

        cyclegan = CycleGAN(
            config=config,
            rngs=nnx.Rngs(rng),
        )

        batch_size = 2
        real_a = jax.random.normal(rng, (batch_size, *input_shape_a))
        real_b = jax.random.normal(rng, (batch_size, *input_shape_b))

        # Test cross-domain translation
        fake_b = cyclegan.generator_a_to_b(real_a)
        fake_a = cyclegan.generator_b_to_a(real_b)

        assert fake_b.shape == (batch_size, *input_shape_b)
        assert fake_a.shape == (batch_size, *input_shape_a)

    def test_cyclegan_loss_weights(self, rng):
        """Test CycleGAN with different loss weights."""
        input_shape = (32, 32, 3)  # NHWC format

        # Create nested configs for both generators
        generators = {
            "a_to_b": CycleGANGeneratorConfig(
                name="gen_a_to_b_weights",
                latent_dim=0,
                input_shape=input_shape,
                output_shape=input_shape,
                hidden_dims=(8, 16),
                n_residual_blocks=1,
                activation="relu",
                batch_norm=True,
                dropout_rate=0.0,
            ),
            "b_to_a": CycleGANGeneratorConfig(
                name="gen_b_to_a_weights",
                latent_dim=0,
                input_shape=input_shape,
                output_shape=input_shape,
                hidden_dims=(8, 16),
                n_residual_blocks=1,
                activation="relu",
                batch_norm=True,
                dropout_rate=0.0,
            ),
        }

        # Create nested configs for both discriminators
        discriminators = {
            "disc_a": PatchGANDiscriminatorConfig(
                name="disc_a_weights",
                input_shape=input_shape,
                hidden_dims=(4, 8),
                activation="leaky_relu",
                batch_norm=False,
                dropout_rate=0.0,
                leaky_relu_slope=0.2,
                use_spectral_norm=False,
            ),
            "disc_b": PatchGANDiscriminatorConfig(
                name="disc_b_weights",
                input_shape=input_shape,
                hidden_dims=(4, 8),
                activation="leaky_relu",
                batch_norm=False,
                dropout_rate=0.0,
                leaky_relu_slope=0.2,
                use_spectral_norm=False,
            ),
        }

        config = CycleGANConfig(
            name="test_cyclegan_loss_weights",
            generator=generators,
            discriminator=discriminators,
            input_shape_a=input_shape,
            input_shape_b=input_shape,
            lambda_cycle=5.0,  # Different cycle loss weight
            lambda_identity=1.0,  # Different identity loss weight
        )

        cyclegan = CycleGAN(
            config=config,
            rngs=nnx.Rngs(rng),
        )

        assert cyclegan.lambda_cycle == 5.0
        assert cyclegan.lambda_identity == 1.0


class TestCycleGANGeneratorJIT:
    """JIT compatibility tests for CycleGAN Generator."""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def generator(self, rng):
        config = CycleGANGeneratorConfig(
            name="jit_gen",
            latent_dim=0,
            input_shape=(32, 32, 3),
            output_shape=(32, 32, 3),
            hidden_dims=(16, 32),
            n_residual_blocks=2,
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
        )
        return CycleGANGenerator(config=config, rngs=nnx.Rngs(rng))

    def test_jit_call(self, generator, rng):
        """Test that Generator __call__ is JIT compatible."""

        @nnx.jit
        def jit_generate(model, x):
            return model(x)

        x = jax.random.normal(rng, (2, 32, 32, 3))
        output = jit_generate(generator, x)

        assert output.shape == (2, 32, 32, 3)
        assert jnp.all(output >= -1.0) and jnp.all(output <= 1.0)


class TestCycleGANDiscriminatorJIT:
    """JIT compatibility tests for CycleGAN Discriminator."""

    @pytest.fixture
    def rng(self):
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def discriminator(self, rng):
        config = PatchGANDiscriminatorConfig(
            name="jit_disc",
            input_shape=(32, 32, 3),
            hidden_dims=(16, 32),
            activation="leaky_relu",
            leaky_relu_slope=0.2,
            batch_norm=True,
            dropout_rate=0.0,
            use_spectral_norm=False,
        )
        return CycleGANDiscriminator(config=config, rngs=nnx.Rngs(rng))

    def test_jit_call(self, discriminator, rng):
        """Test that Discriminator __call__ is JIT compatible."""

        @nnx.jit
        def jit_discriminate(model, x):
            return model(x)

        x = jax.random.normal(rng, (2, 32, 32, 3))
        output = jit_discriminate(discriminator, x)

        # PatchGAN returns spatial patch map: (batch, H', W', 1)
        assert output.ndim == 4
        assert output.shape[0] == 2
        assert output.shape[-1] == 1
        assert jnp.isfinite(output).all()
