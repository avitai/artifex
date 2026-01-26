"""Tests for adversarial loss functions."""

import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.losses.adversarial import (
    hinge_discriminator_loss,
    hinge_generator_loss,
    least_squares_discriminator_loss,
    least_squares_generator_loss,
    vanilla_discriminator_loss,
    vanilla_generator_loss,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
)


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def real_scores():
    """Fixture for real sample scores."""
    return jnp.array([0.9, 0.8, 0.7, 0.6])


@pytest.fixture
def fake_scores():
    """Fixture for fake sample scores."""
    return jnp.array([0.3, 0.2, 0.1, 0.4])


@pytest.fixture
def random_scores(key):
    """Fixture for random scores."""
    return jax.random.uniform(key, (8,))


class TestVanillaGANLoss:
    """Test cases for vanilla GAN loss."""

    def test_generator_loss(self, fake_scores):
        """Test generator loss computation."""
        # Compute loss
        loss = vanilla_generator_loss(fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Higher fake scores (closer to 1) should give lower loss
        high_scores = jnp.array([0.9, 0.8, 0.7, 0.6])
        high_loss = vanilla_generator_loss(high_scores)
        assert high_loss < loss

        # Test reduction methods
        loss_none = vanilla_generator_loss(fake_scores, reduction="none")
        assert loss_none.shape == fake_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = vanilla_generator_loss(fake_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.array([0.5, 1.0, 0.8, 0.3])
        weighted_loss = vanilla_generator_loss(fake_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)

    def test_discriminator_loss(self, real_scores, fake_scores):
        """Test discriminator loss computation."""
        # Compute loss
        loss = vanilla_discriminator_loss(real_scores, fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Better discrimination should give lower loss
        better_real_scores = jnp.array([0.99, 0.98, 0.97, 0.96])
        better_fake_scores = jnp.array([0.01, 0.02, 0.03, 0.04])
        better_loss = vanilla_discriminator_loss(better_real_scores, better_fake_scores)
        assert better_loss < loss

        # Test reduction methods
        loss_none = vanilla_discriminator_loss(real_scores, fake_scores, reduction="none")
        assert loss_none.shape == real_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = vanilla_discriminator_loss(real_scores, fake_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.array([0.5, 1.0, 0.8, 0.3])
        weighted_loss = vanilla_discriminator_loss(real_scores, fake_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)


class TestLeastSquaresGANLoss:
    """Test cases for Least Squares GAN loss."""

    def test_generator_loss(self, fake_scores):
        """Test generator loss computation."""
        # Compute loss
        loss = least_squares_generator_loss(fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Higher fake scores (closer to 1) should give lower loss
        high_scores = jnp.array([0.9, 0.8, 0.7, 0.6])
        high_loss = least_squares_generator_loss(high_scores)
        assert high_loss < loss

        # Test reduction methods
        loss_none = least_squares_generator_loss(fake_scores, reduction="none")
        assert loss_none.shape == fake_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = least_squares_generator_loss(fake_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.array([0.5, 1.0, 0.8, 0.3])
        weighted_loss = least_squares_generator_loss(fake_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)

        # Test with different target
        target_loss = least_squares_generator_loss(fake_scores, target_real=0.9)
        assert jnp.isfinite(target_loss)

    def test_discriminator_loss(self, real_scores, fake_scores):
        """Test discriminator loss computation."""
        # Compute loss
        loss = least_squares_discriminator_loss(real_scores, fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Better discrimination should give lower loss
        better_real_scores = jnp.array([0.99, 0.98, 0.97, 0.96])
        better_fake_scores = jnp.array([0.01, 0.02, 0.03, 0.04])
        better_loss = least_squares_discriminator_loss(better_real_scores, better_fake_scores)
        assert better_loss < loss

        # Test reduction methods
        loss_none = least_squares_discriminator_loss(real_scores, fake_scores, reduction="none")
        assert loss_none.shape == real_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = least_squares_discriminator_loss(real_scores, fake_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.array([0.5, 1.0, 0.8, 0.3])
        weighted_loss = least_squares_discriminator_loss(real_scores, fake_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)

        # Test with different targets
        target_loss = least_squares_discriminator_loss(
            real_scores, fake_scores, target_real=0.9, target_fake=0.1
        )
        assert jnp.isfinite(target_loss)


class TestWassersteinGANLoss:
    """Test cases for Wasserstein GAN loss."""

    def test_generator_loss(self, random_scores):
        """Test generator loss computation."""
        # Compute loss
        loss = wasserstein_generator_loss(random_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Higher critic scores should give lower generator loss
        high_scores = random_scores + 1.0
        high_loss = wasserstein_generator_loss(high_scores)
        assert high_loss < loss

        # Test reduction methods
        loss_none = wasserstein_generator_loss(random_scores, reduction="none")
        assert loss_none.shape == random_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = wasserstein_generator_loss(random_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.ones_like(random_scores) * 0.5
        weighted_loss = wasserstein_generator_loss(random_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)

    def test_discriminator_loss(self, key):
        """Test discriminator/critic loss computation."""
        # Generate critic scores for real and fake samples
        key1, key2 = jax.random.split(key)
        real_scores = jax.random.normal(key1, (4,))
        # Make fake scores lower
        fake_scores = jax.random.normal(key2, (4,)) - 1.0

        # Compute loss
        loss = wasserstein_discriminator_loss(real_scores, fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Higher real scores and lower fake scores should give lower loss
        better_real_scores = real_scores + 1.0
        better_fake_scores = fake_scores - 1.0
        better_loss = wasserstein_discriminator_loss(better_real_scores, better_fake_scores)
        assert better_loss < loss

        # Test reduction methods
        loss_none = wasserstein_discriminator_loss(real_scores, fake_scores, reduction="none")
        assert loss_none.shape == real_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = wasserstein_discriminator_loss(real_scores, fake_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.array([0.5, 1.0, 0.8, 0.3])
        weighted_loss = wasserstein_discriminator_loss(real_scores, fake_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)


class TestHingeGANLoss:
    """Test cases for Hinge GAN loss."""

    def test_generator_loss(self, random_scores):
        """Test generator loss computation."""
        # Compute loss
        loss = hinge_generator_loss(random_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Higher scores should give lower generator loss
        high_scores = random_scores + 1.0
        high_loss = hinge_generator_loss(high_scores)
        assert high_loss < loss

        # Test reduction methods
        loss_none = hinge_generator_loss(random_scores, reduction="none")
        assert loss_none.shape == random_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = hinge_generator_loss(random_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.ones_like(random_scores) * 0.5
        weighted_loss = hinge_generator_loss(random_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)

    def test_discriminator_loss(self, key):
        """Test discriminator loss computation."""
        # Generate scores for real and fake samples
        key1, key2 = jax.random.split(key)
        real_scores = jax.random.normal(key1, (4,))
        # Make fake scores lower
        fake_scores = jax.random.normal(key2, (4,)) - 1.0

        # Compute loss
        loss = hinge_discriminator_loss(real_scores, fake_scores)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Higher real scores and lower fake scores should give lower loss
        better_real_scores = real_scores + 1.0
        better_fake_scores = fake_scores - 1.0
        better_loss = hinge_discriminator_loss(better_real_scores, better_fake_scores)
        assert better_loss < loss

        # Test reduction methods
        loss_none = hinge_discriminator_loss(real_scores, fake_scores, reduction="none")
        assert loss_none.shape == real_scores.shape
        assert jnp.isclose(loss, jnp.mean(loss_none))

        loss_sum = hinge_discriminator_loss(real_scores, fake_scores, reduction="sum")
        assert jnp.isclose(loss_sum, jnp.sum(loss_none))

        # Test with weights
        weights = jnp.array([0.5, 1.0, 0.8, 0.3])
        weighted_loss = hinge_discriminator_loss(real_scores, fake_scores, weights=weights)
        assert jnp.isfinite(weighted_loss)
