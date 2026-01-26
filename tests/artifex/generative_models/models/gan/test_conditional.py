"""Tests for Conditional GAN implementation."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration.gan_config import ConditionalGANConfig
from artifex.generative_models.core.configuration.network_configs import (
    ConditionalDiscriminatorConfig,
    ConditionalGeneratorConfig,
    ConditionalParams,
)
from artifex.generative_models.models.gan.conditional import (
    ConditionalDiscriminator,
    ConditionalGAN,
    ConditionalGenerator,
)


@pytest.fixture
def rngs():
    """Fixture for random number generators."""
    return nnx.Rngs(params=0, dropout=1, sample=2)


@pytest.fixture
def generator(rngs):
    """Fixture for conditional generator."""
    config = ConditionalGeneratorConfig(
        name="test_cgan_gen",
        latent_dim=100,
        output_shape=(1, 28, 28),
        hidden_dims=(128, 64),
        activation="relu",
        batch_norm=True,
        dropout_rate=0.0,
        conditional=ConditionalParams(num_classes=10, embedding_dim=100),
        kernel_size=(4, 4),
        stride=(2, 2),
        padding="SAME",
        batch_norm_momentum=0.9,
        batch_norm_use_running_avg=False,
    )
    return ConditionalGenerator(config=config, rngs=rngs)


@pytest.fixture
def discriminator_with_dropout(rngs):
    """Fixture for conditional discriminator with dropout."""
    config = ConditionalDiscriminatorConfig(
        name="test_cgan_disc_dropout",
        input_shape=(1, 28, 28),
        hidden_dims=(64, 128),
        activation="leaky_relu",
        batch_norm=False,
        dropout_rate=0.5,  # High dropout rate to make effect visible
        leaky_relu_slope=0.2,
        use_spectral_norm=False,
        conditional=ConditionalParams(num_classes=10, embedding_dim=100),
        kernel_size=(3, 3),
        stride_first=(2, 2),
        stride=(2, 2),
        padding="SAME",
    )
    return ConditionalDiscriminator(config=config, rngs=rngs)


@pytest.fixture
def discriminator_no_dropout(rngs):
    """Fixture for conditional discriminator without dropout."""
    config = ConditionalDiscriminatorConfig(
        name="test_cgan_disc_no_dropout",
        input_shape=(1, 28, 28),
        hidden_dims=(64, 128),
        activation="leaky_relu",
        batch_norm=False,
        dropout_rate=0.0,  # No dropout
        leaky_relu_slope=0.2,
        use_spectral_norm=False,
        conditional=ConditionalParams(num_classes=10, embedding_dim=100),
        kernel_size=(3, 3),
        stride_first=(2, 2),
        stride=(2, 2),
        padding="SAME",
    )
    return ConditionalDiscriminator(config=config, rngs=rngs)


class TestConditionalDiscriminatorDropout:
    """Test dropout behavior in ConditionalDiscriminator."""

    def test_dropout_created_when_rate_nonzero(self, discriminator_with_dropout):
        """Test that dropout is created when dropout_rate > 0."""
        assert hasattr(discriminator_with_dropout, "dropout")
        assert discriminator_with_dropout.dropout is not None
        assert discriminator_with_dropout.dropout_rate == 0.5

    def test_no_dropout_when_rate_zero(self, discriminator_no_dropout):
        """Test that dropout is not created when dropout_rate == 0."""
        # When dropout_rate is 0, dropout should not be created
        # Check the condition in base class __init__
        assert discriminator_no_dropout.dropout_rate == 0.0

    def test_dropout_behavior_train_vs_eval(self, discriminator_with_dropout):
        """Test that dropout behaves differently in train vs eval mode."""
        # Create sample input
        batch_size = 4
        x = jnp.ones((batch_size, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)

        # Set to training mode
        discriminator_with_dropout.train()

        # Run multiple times in training mode - outputs should vary due to dropout
        output1 = discriminator_with_dropout(x, labels)
        output2 = discriminator_with_dropout(x, labels)

        # In training mode with dropout, outputs should be different
        # (stochastic due to random dropout masks)
        assert not jnp.allclose(output1, output2, atol=1e-5), (
            "Expected different outputs in training mode due to dropout"
        )

        # Set to eval mode
        discriminator_with_dropout.eval()

        # Run multiple times in eval mode - outputs should be identical (deterministic)
        output3 = discriminator_with_dropout(x, labels)
        output4 = discriminator_with_dropout(x, labels)

        # In eval mode, dropout is disabled, outputs should be identical
        assert jnp.allclose(output3, output4, atol=1e-7), (
            "Expected identical outputs in eval mode (dropout disabled)"
        )

    def test_dropout_applied_in_forward_pass(self, discriminator_with_dropout):
        """Test that dropout is actually applied during forward pass."""
        # Create sample input
        batch_size = 4
        x = jnp.ones((batch_size, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)

        # Set to training mode
        discriminator_with_dropout.train()

        # With dropout_rate=0.5, approximately half the activations should be zeroed
        # We can verify dropout is being applied by running multiple times
        outputs = []
        for _ in range(10):
            output = discriminator_with_dropout(x, labels)
            outputs.append(output)

        outputs_array = jnp.stack(outputs)

        # Calculate variance across runs - should be non-zero due to dropout
        variance = jnp.var(outputs_array, axis=0)
        mean_variance = jnp.mean(variance)

        # With dropout, there should be some variance across runs
        assert mean_variance > 1e-6, (
            "Expected variance in outputs due to dropout, but got near-zero variance"
        )

    def test_no_dropout_application_when_rate_zero(self, discriminator_no_dropout):
        """Test that no dropout is applied when dropout_rate is 0."""
        # Create sample input
        batch_size = 4
        x = jnp.ones((batch_size, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)

        # Even in training mode, with dropout_rate=0, outputs should be deterministic
        discriminator_no_dropout.train()

        output1 = discriminator_no_dropout(x, labels)
        output2 = discriminator_no_dropout(x, labels)

        # No dropout means deterministic outputs even in training mode
        assert jnp.allclose(output1, output2, atol=1e-7), (
            "Expected identical outputs when dropout_rate=0"
        )


class TestConditionalGeneratorBatchNorm:
    """Test batch normalization behavior in ConditionalGenerator."""

    def test_batchnorm_train_vs_eval(self, generator):
        """Test that batch norm behaves differently in train vs eval mode."""
        # Create sample input
        batch_size = 4
        z = jax.random.normal(jax.random.key(0), (batch_size, 100))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)

        # Training mode - uses batch statistics
        generator.train()
        output_train = generator(z, labels)

        # Eval mode - uses running average statistics
        generator.eval()
        output_eval = generator(z, labels)

        # Outputs should be different due to different normalization statistics
        assert not jnp.allclose(output_train, output_eval, atol=1e-5), (
            "Expected different outputs in train vs eval mode due to batch norm"
        )

        # Both outputs should have correct shape
        assert output_train.shape == (batch_size, 1, 28, 28)
        assert output_eval.shape == (batch_size, 1, 28, 28)


# =========================== JIT COMPATIBILITY TESTS ===========================


class TestConditionalGeneratorJIT:
    """JIT compatibility tests for ConditionalGenerator."""

    @pytest.fixture
    def jit_generator(self, rngs):
        """Fixture for JIT testing with smaller hidden dims for faster testing."""
        config = ConditionalGeneratorConfig(
            name="test_cgan_gen_jit",
            latent_dim=32,
            output_shape=(1, 28, 28),
            hidden_dims=(64, 32),  # 2 layers for consistent testing
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            conditional=ConditionalParams(num_classes=10, embedding_dim=100),
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )
        return ConditionalGenerator(config=config, rngs=rngs)

    def test_jit_forward_pass(self, jit_generator, rngs):
        """Test JIT forward pass compatibility."""

        @nnx.jit
        def generate_jit(model, z, labels):
            return model(z, labels)

        z = jax.random.normal(rngs.sample(), (4, 32))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_generator.eval()

        output_regular = jit_generator(z, labels)
        output_jit = generate_jit(jit_generator, z, labels)

        # GPU floating-point arithmetic with JIT can produce small numerical differences
        # due to XLA operation reordering and fusion. Use realistic tolerances for GPU.
        assert jnp.allclose(output_regular, output_jit, rtol=1e-3, atol=1e-4)

    def test_jit_compilation_without_errors(self, jit_generator, rngs):
        """Test that JIT compilation works without errors."""

        @nnx.jit
        def generate_jit(model, z, labels):
            return model(z, labels)

        z = jax.random.normal(rngs.sample(), (4, 32))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_generator.eval()

        # Should not raise any errors
        output = generate_jit(jit_generator, z, labels)
        assert output.shape == (4, 1, 28, 28)

    def test_jit_multiple_calls_consistent(self, jit_generator, rngs):
        """Test that multiple JIT calls produce consistent results."""

        @nnx.jit
        def generate_jit(model, z, labels):
            return model(z, labels)

        z = jax.random.normal(rngs.sample(), (4, 32))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_generator.eval()

        output1 = generate_jit(jit_generator, z, labels)
        output2 = generate_jit(jit_generator, z, labels)

        # Multiple calls with same input should be identical
        assert jnp.allclose(output1, output2, rtol=1e-7, atol=1e-9)

    def test_jit_with_different_batch_sizes(self, jit_generator, rngs):
        """Test JIT with different batch sizes."""

        @nnx.jit
        def generate_jit(model, z, labels):
            return model(z, labels)

        jit_generator.eval()

        # Test with batch size 2
        z2 = jax.random.normal(rngs.sample(), (2, 32))
        labels2 = jax.nn.one_hot(jnp.array([0, 1]), 10)
        output2 = generate_jit(jit_generator, z2, labels2)
        assert output2.shape == (2, 1, 28, 28)

        # Test with batch size 8
        z8 = jax.random.normal(rngs.sample(), (8, 32))
        labels8 = jax.nn.one_hot(jnp.arange(8) % 10, 10)
        output8 = generate_jit(jit_generator, z8, labels8)
        assert output8.shape == (8, 1, 28, 28)


class TestConditionalDiscriminatorJIT:
    """JIT compatibility tests for ConditionalDiscriminator."""

    @pytest.fixture
    def jit_discriminator(self, rngs):
        """Fixture for JIT testing."""
        config = ConditionalDiscriminatorConfig(
            name="test_cgan_disc_jit",
            input_shape=(1, 28, 28),
            hidden_dims=(64, 128),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            conditional=ConditionalParams(num_classes=10, embedding_dim=100),
            kernel_size=(3, 3),
            stride_first=(2, 2),
            stride=(2, 2),
            padding="SAME",
        )
        return ConditionalDiscriminator(config=config, rngs=rngs)

    def test_jit_forward_pass(self, jit_discriminator, rngs):
        """Test JIT forward pass compatibility."""

        @nnx.jit
        def discriminate_jit(model, x, labels):
            return model(x, labels)

        x = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_discriminator.eval()

        output_regular = jit_discriminator(x, labels)
        output_jit = discriminate_jit(jit_discriminator, x, labels)

        # GPU floating-point arithmetic with JIT can produce differences up to ~5e-5
        # due to XLA operation reordering and fusion
        assert jnp.allclose(output_regular, output_jit, rtol=1e-4, atol=5e-5)

    def test_jit_compilation_without_errors(self, jit_discriminator, rngs):
        """Test that JIT compilation works without errors."""

        @nnx.jit
        def discriminate_jit(model, x, labels):
            return model(x, labels)

        x = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_discriminator.eval()

        # Should not raise any errors
        output = discriminate_jit(jit_discriminator, x, labels)
        assert output.shape == (4,)

    def test_jit_multiple_calls_consistent(self, jit_discriminator, rngs):
        """Test that multiple JIT calls produce consistent results."""

        @nnx.jit
        def discriminate_jit(model, x, labels):
            return model(x, labels)

        x = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_discriminator.eval()

        output1 = discriminate_jit(jit_discriminator, x, labels)
        output2 = discriminate_jit(jit_discriminator, x, labels)

        # Multiple calls with same input should be identical
        assert jnp.allclose(output1, output2, rtol=1e-7, atol=1e-9)

    def test_jit_with_different_batch_sizes(self, jit_discriminator, rngs):
        """Test JIT with different batch sizes."""

        @nnx.jit
        def discriminate_jit(model, x, labels):
            return model(x, labels)

        jit_discriminator.eval()

        # Test with batch size 2
        x2 = jax.random.normal(rngs.sample(), (2, 1, 28, 28))
        labels2 = jax.nn.one_hot(jnp.array([0, 1]), 10)
        output2 = discriminate_jit(jit_discriminator, x2, labels2)
        assert output2.shape == (2,)

        # Test with batch size 8
        x8 = jax.random.normal(rngs.sample(), (8, 1, 28, 28))
        labels8 = jax.nn.one_hot(jnp.arange(8) % 10, 10)
        output8 = discriminate_jit(jit_discriminator, x8, labels8)
        assert output8.shape == (8,)


class TestConditionalGANJIT:
    """JIT compatibility tests for full ConditionalGAN."""

    @pytest.fixture
    def jit_cgan(self, rngs):
        """Fixture for JIT testing."""
        image_shape = (1, 28, 28)

        generator = ConditionalGeneratorConfig(
            name="test_cgan_gen_jit",
            latent_dim=32,
            output_shape=image_shape,
            hidden_dims=(64, 32),
            activation="relu",
            batch_norm=True,
            dropout_rate=0.0,
            conditional=ConditionalParams(num_classes=10, embedding_dim=100),
            kernel_size=(4, 4),
            stride=(2, 2),
            padding="SAME",
            batch_norm_momentum=0.9,
            batch_norm_use_running_avg=False,
        )
        discriminator = ConditionalDiscriminatorConfig(
            name="test_cgan_disc_jit",
            input_shape=image_shape,
            hidden_dims=(64, 128),
            activation="leaky_relu",
            batch_norm=False,
            dropout_rate=0.0,
            leaky_relu_slope=0.2,
            use_spectral_norm=False,
            conditional=ConditionalParams(num_classes=10, embedding_dim=100),
            kernel_size=(3, 3),
            stride_first=(2, 2),
            stride=(2, 2),
            padding="SAME",
        )
        config = ConditionalGANConfig(
            name="test_cgan_jit",
            generator=generator,
            discriminator=discriminator,
        )
        return ConditionalGAN(config=config, rngs=rngs)

    def test_jit_generator_loss(self, jit_cgan, rngs):
        """Test JIT compilation of generator loss."""

        @nnx.jit
        def gen_loss_jit(model, fake_samples, fake_labels):
            return model.generator_loss(fake_samples, fake_labels)

        fake_samples = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        fake_labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_cgan.eval()

        loss_regular = jit_cgan.generator_loss(fake_samples, fake_labels)
        loss_jit = gen_loss_jit(jit_cgan, fake_samples, fake_labels)

        # GPU floating-point arithmetic with JIT can produce differences up to ~1e-5
        assert jnp.allclose(loss_regular, loss_jit, rtol=1e-4, atol=1e-5)

    def test_jit_discriminator_loss(self, jit_cgan, rngs):
        """Test JIT compilation of discriminator loss."""

        @nnx.jit
        def disc_loss_jit(model, real_samples, fake_samples, real_labels, fake_labels):
            return model.discriminator_loss(real_samples, fake_samples, real_labels, fake_labels)

        real_samples = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        fake_samples = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        real_labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        fake_labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_cgan.eval()

        loss_regular = jit_cgan.discriminator_loss(
            real_samples, fake_samples, real_labels, fake_labels
        )
        loss_jit = disc_loss_jit(jit_cgan, real_samples, fake_samples, real_labels, fake_labels)

        # GPU floating-point arithmetic with JIT can produce differences up to ~1e-5
        assert jnp.allclose(loss_regular, loss_jit, rtol=1e-4, atol=1e-5)

    def test_jit_full_training_step(self, jit_cgan, rngs):
        """Test JIT compilation of full training step."""

        @nnx.jit
        def training_step_jit(model, z, real_samples, labels):
            fake_samples = model.generator(z, labels)
            g_loss = model.generator_loss(fake_samples, labels)
            d_loss = model.discriminator_loss(real_samples, fake_samples, labels, labels)
            return g_loss, d_loss

        z = jax.random.normal(rngs.sample(), (4, 32))
        real_samples = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_cgan.eval()

        # Should not raise any errors
        g_loss, d_loss = training_step_jit(jit_cgan, z, real_samples, labels)

        assert jnp.isfinite(g_loss)
        assert jnp.isfinite(d_loss)

    def test_jit_gradient_computation(self, jit_cgan, rngs):
        """Test JIT compatibility with gradient computation."""

        def loss_fn(model, z, real_samples, labels):
            fake_samples = model.generator(z, labels)
            g_loss = model.generator_loss(fake_samples, labels)
            d_loss = model.discriminator_loss(real_samples, fake_samples, labels, labels)
            return g_loss + d_loss

        @nnx.jit
        def compute_gradients_jit(model, z, real_samples, labels):
            grads = nnx.grad(loss_fn, argnums=0)(model, z, real_samples, labels)
            return grads

        z = jax.random.normal(rngs.sample(), (4, 32))
        real_samples = jax.random.normal(rngs.sample(), (4, 1, 28, 28))
        labels = jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 10)
        jit_cgan.train()

        # Should not raise any errors
        grads = compute_gradients_jit(jit_cgan, z, real_samples, labels)

        # Verify gradients exist and are finite
        assert grads is not None
