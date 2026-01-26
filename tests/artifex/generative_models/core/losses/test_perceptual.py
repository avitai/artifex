"""Tests for perceptual loss functions."""

import jax
import jax.numpy as jnp
import pytest

from artifex.generative_models.core.losses.perceptual import (
    contextual_loss,
    feature_reconstruction_loss,
    style_loss,
)


@pytest.fixture
def key():
    """Fixture for JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def image_features(key):
    """Fixture for image feature maps."""
    key1, key2, key3 = jax.random.split(key, 3)
    return {
        "conv1": jax.random.normal(key1, (2, 16, 16, 32)),
        "conv2": jax.random.normal(key2, (2, 8, 8, 64)),
        "conv3": jax.random.normal(key3, (2, 4, 4, 128)),
    }


@pytest.fixture
def image_features_list(key):
    """Fixture for image feature maps as list."""
    key1, key2, key3 = jax.random.split(key, 3)
    return [
        jax.random.normal(key1, (2, 16, 16, 32)),
        jax.random.normal(key2, (2, 8, 8, 64)),
        jax.random.normal(key3, (2, 4, 4, 128)),
    ]


@pytest.fixture
def single_feature_map(key):
    """Fixture for a single feature map."""
    return jax.random.normal(key, (2, 16, 16, 32))


class TestFeatureReconstructionLoss:
    """Test cases for feature reconstruction loss."""

    def test_with_dict_features(self, image_features):
        """Test feature reconstruction loss with dictionary features."""
        # Create slightly different predicted features (JAX convention: predictions, targets)
        pred_features = {k: v + 0.1 for k, v in image_features.items()}

        # Compute loss (predictions first, then targets)
        loss = feature_reconstruction_loss(pred_features, image_features)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test with weights
        weights = {"conv1": 1.0, "conv2": 0.5, "conv3": 0.25}
        weighted_loss = feature_reconstruction_loss(pred_features, image_features, weights=weights)
        assert jnp.isfinite(weighted_loss)

        # Loss should be higher when features are more different
        very_different = {k: v + 1.0 for k, v in image_features.items()}
        higher_loss = feature_reconstruction_loss(very_different, image_features)
        assert higher_loss > loss

    def test_with_list_features(self, image_features_list):
        """Test feature reconstruction loss with list features."""
        # Create slightly different predicted features
        pred_features = [v + 0.1 for v in image_features_list]

        # Compute loss (predictions first, then targets)
        loss = feature_reconstruction_loss(pred_features, image_features_list)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test with weights
        weights = [1.0, 0.5, 0.25]
        weighted_loss = feature_reconstruction_loss(
            pred_features, image_features_list, weights=weights
        )
        assert jnp.isfinite(weighted_loss)

        # Loss should be higher when features are more different
        very_different = [v + 1.0 for v in image_features_list]
        higher_loss = feature_reconstruction_loss(very_different, image_features_list)
        assert higher_loss > loss

    def test_with_single_feature(self, single_feature_map):
        """Test feature reconstruction loss with a single feature map."""
        # Create slightly different predicted features
        pred_features = single_feature_map + 0.1

        # Compute loss (predictions first, then targets)
        loss = feature_reconstruction_loss(pred_features, single_feature_map)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test reduction methods
        loss_none = feature_reconstruction_loss(pred_features, single_feature_map, reduction="none")
        assert loss_none.shape == single_feature_map.shape

        loss_sum = feature_reconstruction_loss(pred_features, single_feature_map, reduction="sum")
        assert jnp.isfinite(loss_sum)


class TestStyleLoss:
    """Test cases for style loss."""

    def test_with_dict_features(self, image_features):
        """Test style loss with dictionary features."""
        # Create slightly different predicted features (JAX convention: predictions, targets)
        pred_features = {k: v + 0.1 for k, v in image_features.items()}

        # Compute loss (predictions first, then targets)
        loss = style_loss(pred_features, image_features)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test with weights
        weights = {"conv1": 1.0, "conv2": 0.5, "conv3": 0.25}
        weighted_loss = style_loss(pred_features, image_features, weights=weights)
        assert jnp.isfinite(weighted_loss)

        # Loss should be higher when features are more different
        very_different = {k: v + 1.0 for k, v in image_features.items()}
        higher_loss = style_loss(very_different, image_features)
        assert higher_loss > loss

    def test_with_list_features(self, image_features_list):
        """Test style loss with list features."""
        # Create slightly different predicted features
        pred_features = [v + 0.1 for v in image_features_list]

        # Compute loss (predictions first, then targets)
        loss = style_loss(pred_features, image_features_list)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test with weights
        weights = [1.0, 0.5, 0.25]
        weighted_loss = style_loss(pred_features, image_features_list, weights=weights)
        assert jnp.isfinite(weighted_loss)

        # Loss should be higher when features are more different
        very_different = [v + 1.0 for v in image_features_list]
        higher_loss = style_loss(very_different, image_features_list)
        assert higher_loss > loss

    def test_with_single_feature(self, single_feature_map):
        """Test style loss with a single feature map."""
        # Create slightly different predicted features
        pred_features = single_feature_map + 0.1

        # Compute loss (predictions first, then targets)
        loss = style_loss(pred_features, single_feature_map)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Same features should give very small loss
        same_loss = style_loss(single_feature_map, single_feature_map)
        assert same_loss < 1e-6


class TestContextualLoss:
    """Test cases for contextual loss."""

    def test_basic_functionality(self, single_feature_map):
        """Test basic functionality of contextual loss."""
        # Create slightly different predicted features (JAX convention: predictions, targets)
        pred_features = single_feature_map + 0.1

        # Compute loss (predictions first, then targets)
        loss = contextual_loss(pred_features, single_feature_map)

        # Loss should be scalar and finite
        assert loss.shape == ()
        assert jnp.isfinite(loss)

        # Test reduction methods
        loss_none = contextual_loss(pred_features, single_feature_map, reduction="none")
        assert loss_none.shape == (2,)  # Batch dimension

        loss_sum = contextual_loss(pred_features, single_feature_map, reduction="sum")
        assert jnp.isfinite(loss_sum)

        # Test with weights
        weights = jnp.array([0.5, 1.0])
        weighted_loss = contextual_loss(pred_features, single_feature_map, weights=weights)
        assert jnp.isfinite(weighted_loss)

    def test_with_different_bandwidths(self, single_feature_map):
        """Test contextual loss with different bandwidth values."""
        # Create predicted features
        pred_features = single_feature_map + 0.1

        # Compute loss with default bandwidth
        default_loss = contextual_loss(pred_features, single_feature_map)

        # Compute loss with different bandwidth values
        small_bw_loss = contextual_loss(pred_features, single_feature_map, band_width=0.01)
        large_bw_loss = contextual_loss(pred_features, single_feature_map, band_width=1.0)

        # All losses should be finite
        assert jnp.isfinite(default_loss)
        assert jnp.isfinite(small_bw_loss)
        assert jnp.isfinite(large_bw_loss)

        # Bandwidth affects the sensitivity of the loss function
        # Smaller bandwidth makes the loss more sensitive to differences
        assert small_bw_loss != default_loss
        assert large_bw_loss != default_loss

    def test_with_identical_features(self, single_feature_map):
        """Test contextual loss with identical features."""
        # Same features should give very small loss
        loss = contextual_loss(single_feature_map, single_feature_map)
        assert loss < 0.1  # Should be close to 0
