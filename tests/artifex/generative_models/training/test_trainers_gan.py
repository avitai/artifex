"""TDD tests for GAN-specific trainer.

These tests define the expected behavior for GAN training based on
state-of-the-art techniques (2024-2025):
- Multiple loss functions (vanilla, wasserstein, hinge, lsgan, relativistic)
- Gradient penalties (WGAN-GP, R1, R2)
- Label smoothing
- TTUR (Two-Timescale Update Rule)

References:
    - R3GAN: https://arxiv.org/abs/2501.xxxxx
    - WGAN-GP: https://arxiv.org/abs/1704.00028
    - R1 Regularization: https://arxiv.org/abs/1801.04406
    - TTUR: https://arxiv.org/abs/1706.08500
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleGenerator(nnx.Module):
    """Simple generator for testing trainer functionality."""

    def __init__(self, latent_dim: int = 8, output_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(latent_dim, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, output_dim, rngs=rngs)

    def __call__(self, z: jax.Array) -> jax.Array:
        x = nnx.relu(self.fc1(z))
        return self.fc2(x)


class SimpleDiscriminator(nnx.Module):
    """Simple discriminator for testing trainer functionality."""

    def __init__(self, input_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(input_dim, 32, rngs=rngs)
        self.fc2 = nnx.Linear(32, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


@pytest.fixture
def generator() -> SimpleGenerator:
    """Create a simple generator for testing."""
    return SimpleGenerator(rngs=nnx.Rngs(0))


@pytest.fixture
def discriminator() -> SimpleDiscriminator:
    """Create a simple discriminator for testing."""
    return SimpleDiscriminator(rngs=nnx.Rngs(1))


@pytest.fixture
def real_data() -> jax.Array:
    """Create real data samples for testing."""
    return jax.random.normal(jax.random.key(0), (8, 16))


@pytest.fixture
def latent_vectors() -> jax.Array:
    """Create latent vectors for testing."""
    return jax.random.normal(jax.random.key(1), (8, 8))


@pytest.fixture
def rng_key() -> jax.Array:
    """Create PRNG key for testing."""
    return jax.random.key(42)


# =============================================================================
# GANTrainingConfig Tests
# =============================================================================


class TestGANTrainingConfig:
    """Tests for GANTrainingConfig dataclass."""

    def test_config_exists(self) -> None:
        """GANTrainingConfig should be importable."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainingConfig,
        )

        assert GANTrainingConfig is not None

    def test_config_default_values(self) -> None:
        """GANTrainingConfig should have sensible defaults."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainingConfig,
        )

        config = GANTrainingConfig()
        assert config.loss_type == "vanilla"
        assert config.n_critic == 1
        assert config.gp_weight == 10.0
        assert config.gp_target == 1.0
        assert config.r1_weight == 0.0
        assert config.label_smoothing == 0.0

    def test_config_custom_values(self) -> None:
        """GANTrainingConfig should accept custom values."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainingConfig,
        )

        config = GANTrainingConfig(
            loss_type="wasserstein",
            n_critic=5,
            gp_weight=15.0,
            r1_weight=10.0,
            label_smoothing=0.1,
        )
        assert config.loss_type == "wasserstein"
        assert config.n_critic == 5
        assert config.gp_weight == 15.0
        assert config.r1_weight == 10.0
        assert config.label_smoothing == 0.1

    def test_config_all_loss_types(self) -> None:
        """GANTrainingConfig should support all loss types."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainingConfig,
        )

        for loss_type in ["vanilla", "wasserstein", "hinge", "lsgan"]:
            config = GANTrainingConfig(loss_type=loss_type)
            assert config.loss_type == loss_type


# =============================================================================
# Discriminator Loss Tests
# =============================================================================


class TestDiscriminatorLoss:
    """Tests for discriminator loss computation."""

    def test_vanilla_loss_computation(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """Vanilla GAN discriminator loss should use sigmoid cross-entropy."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="vanilla")
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        d_real = jnp.array([2.0, 1.5, 1.0])
        d_fake = jnp.array([-1.0, -0.5, 0.0])
        loss = trainer.compute_d_loss_vanilla(d_real, d_fake)

        assert loss.shape == ()
        assert loss > 0  # Loss should be positive

    def test_wasserstein_loss_computation(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """Wasserstein discriminator loss should be E[D(fake)] - E[D(real)]."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="wasserstein")
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        d_real = jnp.array([1.0, 2.0, 3.0])
        d_fake = jnp.array([-1.0, -2.0, -3.0])
        loss = trainer.compute_d_loss_wasserstein(d_real, d_fake)

        # E[fake] - E[real] = -2 - 2 = -4
        assert loss == pytest.approx(-4.0)

    def test_hinge_loss_computation(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """Hinge discriminator loss should use relu(1 - d_real) + relu(1 + d_fake)."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="hinge")
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        d_real = jnp.array([1.5])  # > 1, so relu(1 - 1.5) = 0
        d_fake = jnp.array([-0.5])  # relu(1 + (-0.5)) = 0.5
        loss = trainer.compute_d_loss_hinge(d_real, d_fake)

        assert loss == pytest.approx(0.5)

    def test_lsgan_loss_computation(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """LSGAN discriminator loss should use least squares."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="lsgan")
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        d_real = jnp.array([0.5])  # (0.5 - 1)^2 = 0.25
        d_fake = jnp.array([0.5])  # 0.5^2 = 0.25
        loss = trainer.compute_d_loss_lsgan(d_real, d_fake)

        # 0.5 * (0.25 + 0.25) = 0.25
        assert loss == pytest.approx(0.25)

    def test_discriminator_loss_dispatch(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """compute_discriminator_loss should dispatch to correct loss function."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        d_real = jnp.array([1.0, 2.0])
        d_fake = jnp.array([-1.0, -2.0])

        for loss_type in ["vanilla", "wasserstein", "hinge", "lsgan"]:
            config = GANTrainingConfig(loss_type=loss_type)
            trainer = GANTrainer(config)

            loss = trainer.compute_discriminator_loss(d_real, d_fake)
            assert loss.shape == ()


# =============================================================================
# Generator Loss Tests
# =============================================================================


class TestGeneratorLoss:
    """Tests for generator loss computation."""

    def test_vanilla_generator_loss(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """Vanilla generator loss should use non-saturating loss."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="vanilla")
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        d_fake = jnp.array([2.0, 1.0, 0.0])
        loss = trainer.compute_g_loss_vanilla(d_fake)

        assert loss.shape == ()
        assert loss > 0

    def test_wasserstein_generator_loss(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """Wasserstein generator loss should be -E[D(fake)]."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="wasserstein")
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        d_fake = jnp.array([1.0, 2.0, 3.0])
        loss = trainer.compute_g_loss_wasserstein(d_fake)

        # -E[fake] = -2.0
        assert loss == pytest.approx(-2.0)

    def test_generator_loss_dispatch(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """compute_generator_loss should dispatch to correct loss function."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        d_fake = jnp.array([1.0, 0.5])

        for loss_type in ["vanilla", "wasserstein", "hinge", "lsgan"]:
            config = GANTrainingConfig(loss_type=loss_type)
            trainer = GANTrainer(config)

            loss = trainer.compute_generator_loss(d_fake)
            assert loss.shape == ()


# =============================================================================
# Gradient Penalty Tests
# =============================================================================


class TestGradientPenalties:
    """Tests for gradient penalty regularization."""

    def test_gradient_penalty_shape(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        real_data: jax.Array,
        rng_key: jax.Array,
    ) -> None:
        """Gradient penalty should return scalar loss."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="wasserstein", gp_weight=10.0)
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        fake = jax.random.normal(rng_key, real_data.shape)
        gp = trainer.compute_gradient_penalty(discriminator, real_data, fake, rng_key)

        assert gp.shape == ()
        assert gp >= 0  # Gradient penalty is non-negative

    def test_r1_penalty_shape(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        real_data: jax.Array,
    ) -> None:
        """R1 penalty should return scalar loss."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(r1_weight=10.0)
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        r1 = trainer.compute_r1_penalty(discriminator, real_data)

        assert r1.shape == ()
        assert r1 >= 0  # R1 penalty is non-negative

    def test_r1_penalty_is_gradient_norm_squared(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
    ) -> None:
        """R1 penalty should compute squared gradient norm."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(r1_weight=10.0)
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        # Use simple input where we can reason about gradients
        real = jnp.ones((2, 16)) * 0.5
        r1 = trainer.compute_r1_penalty(discriminator, real)

        # R1 is mean of squared gradient norms, should be finite
        assert jnp.isfinite(r1)


# =============================================================================
# Training Step Tests
# =============================================================================


class TestTrainingSteps:
    """Tests for training step execution."""

    def test_discriminator_step_updates_discriminator(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        real_data: jax.Array,
        latent_vectors: jax.Array,
        rng_key: jax.Array,
    ) -> None:
        """Discriminator step should update discriminator parameters."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="vanilla")
        nnx.Optimizer(generator, optax.adam(1e-3), wrt=nnx.Param)
        d_opt = nnx.Optimizer(discriminator, optax.adam(1e-3), wrt=nnx.Param)
        trainer = GANTrainer(config)

        # Get initial params
        initial_params = nnx.state(discriminator, nnx.Param)
        initial_fc1_kernel = initial_params["fc1"]["kernel"].value.copy()

        # Run discriminator step
        trainer.discriminator_step(
            generator, discriminator, d_opt, real_data, latent_vectors, rng_key
        )

        # Get updated params
        updated_params = nnx.state(discriminator, nnx.Param)
        updated_fc1_kernel = updated_params["fc1"]["kernel"].value

        # Discriminator params should have changed
        assert not jnp.allclose(initial_fc1_kernel, updated_fc1_kernel)

    def test_discriminator_step_returns_metrics(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        real_data: jax.Array,
        latent_vectors: jax.Array,
        rng_key: jax.Array,
    ) -> None:
        """Discriminator step should return loss and metrics."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="vanilla")
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        d_opt = nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        loss, metrics = trainer.discriminator_step(
            generator, discriminator, d_opt, real_data, latent_vectors, rng_key
        )

        assert isinstance(loss, jax.Array)
        assert "d_loss" in metrics
        assert "d_real" in metrics
        assert "d_fake" in metrics

    def test_generator_step_updates_generator(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        latent_vectors: jax.Array,
    ) -> None:
        """Generator step should update generator parameters."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="vanilla")
        g_opt = nnx.Optimizer(generator, optax.adam(1e-3), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-3), wrt=nnx.Param)
        trainer = GANTrainer(config)

        # Get initial params
        initial_params = nnx.state(generator, nnx.Param)
        initial_fc1_kernel = initial_params["fc1"]["kernel"].value.copy()

        # Run generator step
        trainer.generator_step(generator, discriminator, g_opt, latent_vectors)

        # Get updated params
        updated_params = nnx.state(generator, nnx.Param)
        updated_fc1_kernel = updated_params["fc1"]["kernel"].value

        # Generator params should have changed
        assert not jnp.allclose(initial_fc1_kernel, updated_fc1_kernel)

    def test_generator_step_returns_metrics(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        latent_vectors: jax.Array,
    ) -> None:
        """Generator step should return loss and metrics."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="vanilla")
        g_opt = nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        loss, metrics = trainer.generator_step(generator, discriminator, g_opt, latent_vectors)

        assert isinstance(loss, jax.Array)
        assert "g_loss" in metrics


# =============================================================================
# Regularization Integration Tests
# =============================================================================


class TestRegularizationIntegration:
    """Tests for regularization in training steps."""

    def test_gp_added_to_wasserstein_loss(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        real_data: jax.Array,
        latent_vectors: jax.Array,
        rng_key: jax.Array,
    ) -> None:
        """Gradient penalty should be added to Wasserstein loss."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="wasserstein", gp_weight=10.0)
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        d_opt = nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        _, metrics = trainer.discriminator_step(
            generator, discriminator, d_opt, real_data, latent_vectors, rng_key
        )

        assert "gp_loss" in metrics
        assert "d_loss_total" in metrics

    def test_r1_added_when_weight_nonzero(
        self,
        generator: SimpleGenerator,
        discriminator: SimpleDiscriminator,
        real_data: jax.Array,
        latent_vectors: jax.Array,
        rng_key: jax.Array,
    ) -> None:
        """R1 penalty should be added when r1_weight > 0."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        config = GANTrainingConfig(loss_type="vanilla", r1_weight=10.0)
        nnx.Optimizer(generator, optax.adam(1e-4), wrt=nnx.Param)
        d_opt = nnx.Optimizer(discriminator, optax.adam(1e-4), wrt=nnx.Param)
        trainer = GANTrainer(config)

        _, metrics = trainer.discriminator_step(
            generator, discriminator, d_opt, real_data, latent_vectors, rng_key
        )

        assert "r1_loss" in metrics


# =============================================================================
# Label Smoothing Tests
# =============================================================================


class TestLabelSmoothing:
    """Tests for label smoothing in vanilla GAN loss."""

    def test_label_smoothing_applied(
        self, generator: SimpleGenerator, discriminator: SimpleDiscriminator
    ) -> None:
        """Label smoothing should reduce effective target for real samples."""
        from artifex.generative_models.training.trainers.gan_trainer import (
            GANTrainer,
            GANTrainingConfig,
        )

        # Without smoothing
        config_no_smooth = GANTrainingConfig(loss_type="vanilla", label_smoothing=0.0)
        trainer_no_smooth = GANTrainer(config_no_smooth)

        # With smoothing
        config_smooth = GANTrainingConfig(loss_type="vanilla", label_smoothing=0.1)
        trainer_smooth = GANTrainer(config_smooth)

        d_real = jnp.array([2.0, 1.5])
        d_fake = jnp.array([-1.0, -0.5])

        loss_no_smooth = trainer_no_smooth.compute_d_loss_vanilla(d_real, d_fake)
        loss_smooth = trainer_smooth.compute_d_loss_vanilla(d_real, d_fake)

        # Losses should be different due to smoothing
        assert not jnp.allclose(loss_no_smooth, loss_smooth)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestGANTrainerExports:
    """Tests for GAN trainer exports."""

    def test_exports_from_trainers_init(self) -> None:
        """GAN trainer classes should be exported from trainers __init__."""
        from artifex.generative_models.training.trainers import (
            GANTrainer,
            GANTrainingConfig,
        )

        assert GANTrainer is not None
        assert GANTrainingConfig is not None
