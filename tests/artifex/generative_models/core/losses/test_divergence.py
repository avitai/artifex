"""Tests for the divergence losses module."""

import jax.numpy as jnp
import numpy as np
import pytest
from calibrax.metrics.functional.divergence import (
    js_divergence as calibrax_js_divergence,
    kl_divergence as calibrax_kl_divergence,
    reverse_kl_divergence as calibrax_reverse_kl_divergence,
    wasserstein_1d as calibrax_wasserstein_1d,
)
from distrax import Normal

from artifex.generative_models.core.losses.divergence import (
    js_divergence,
    kl_divergence,
    reverse_kl_divergence,
    wasserstein_distance,
)


class TestKLDivergence:
    """Tests for the KL divergence function."""

    def test_identical_distributions(self):
        """Test KL divergence with identical distributions."""
        p = jnp.array([0.5, 0.5])
        q = jnp.array([0.5, 0.5])
        result = kl_divergence(p, q)
        # KL divergence between identical distributions should be 0
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_different_distributions(self):
        """Test KL divergence with different distributions."""
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.1, 0.9])
        result = kl_divergence(p, q)
        np.testing.assert_allclose(result, calibrax_kl_divergence(p, q), rtol=1e-6)

    def test_with_log_inputs(self):
        """Test KL divergence with pre-computed log probabilities."""
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.1, 0.9])
        log_p = jnp.log(p)
        log_q = jnp.log(q)
        result = kl_divergence(p, q, log_predictions=log_p, log_targets=log_q)
        np.testing.assert_allclose(result, calibrax_kl_divergence(p, q), rtol=1e-6)

    def test_distrax_distributions(self):
        """Test KL divergence with distrax distributions."""
        p = Normal(loc=0.0, scale=1.0)
        q = Normal(loc=1.0, scale=2.0)
        result = kl_divergence(p, q)
        # For two normal distributions, KL has analytical solution
        expected = p.kl_divergence(q)
        np.testing.assert_allclose(result, expected)

    def test_default_array_path_matches_calibrax_kl(self):
        """Default array KL should track the shared CalibraX primitive."""
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.1, 0.9])

        np.testing.assert_allclose(kl_divergence(p, q), calibrax_kl_divergence(p, q))


class TestReverseKLDivergence:
    """Tests for the reverse KL divergence function."""

    def test_identical_distributions(self):
        """Test reverse KL divergence with identical distributions."""
        p = jnp.array([0.5, 0.5])
        q = jnp.array([0.5, 0.5])
        result = reverse_kl_divergence(p, q)
        # KL divergence between identical distributions should be 0
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_different_distributions(self):
        """Test reverse KL divergence with different distributions."""
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.1, 0.9])
        result = reverse_kl_divergence(p, q)
        np.testing.assert_allclose(result, calibrax_reverse_kl_divergence(p, q), rtol=1e-6)

    def test_distrax_distributions(self):
        """Test reverse KL divergence with distrax distributions."""
        p = Normal(loc=0.0, scale=1.0)
        q = Normal(loc=1.0, scale=2.0)
        result = reverse_kl_divergence(p, q)
        # For two normal distributions, KL has analytical solution
        expected = q.kl_divergence(p)
        np.testing.assert_allclose(result, expected)

    def test_default_array_path_matches_calibrax_reverse_kl(self):
        """Default array reverse KL should track the shared CalibraX primitive."""
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.1, 0.9])

        np.testing.assert_allclose(
            reverse_kl_divergence(p, q),
            calibrax_reverse_kl_divergence(p, q),
        )


class TestJSDivergence:
    """Tests for the Jensen-Shannon divergence function."""

    def test_identical_distributions(self):
        """Test JS divergence with identical distributions."""
        p = jnp.array([0.5, 0.5])
        q = jnp.array([0.5, 0.5])
        result = js_divergence(p, q)
        # JS divergence between identical distributions should be 0
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_different_distributions(self):
        """Test JS divergence with different distributions."""
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.1, 0.9])
        result = js_divergence(p, q)

        # Manually compute JS: 0.5 * (KL(p||m) + KL(q||m)) where m = 0.5 * (p + q)
        m = 0.5 * (p + q)
        kl_p_m = jnp.sum(p * (jnp.log(p) - jnp.log(m)))
        kl_q_m = jnp.sum(q * (jnp.log(q) - jnp.log(m)))
        expected = 0.5 * (kl_p_m + kl_q_m)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_distrax_error(self):
        """
        Test that JS divergence raises NotImplementedError
        for distrax distributions.
        """
        p = Normal(loc=0.0, scale=1.0)
        q = Normal(loc=1.0, scale=2.0)
        with pytest.raises(NotImplementedError):
            js_divergence(p, q)

    def test_default_array_path_matches_calibrax_js(self):
        """Default array JS should track the shared CalibraX primitive."""
        p = jnp.array([0.9, 0.1])
        q = jnp.array([0.1, 0.9])

        np.testing.assert_allclose(js_divergence(p, q), calibrax_js_divergence(p, q), rtol=1e-5)


class TestWassersteinDistance:
    """Tests for the Wasserstein distance function."""

    def test_identical_distributions(self):
        """Test Wasserstein distance with identical distributions."""
        p = jnp.array([1.0, 2.0, 3.0])
        q = jnp.array([1.0, 2.0, 3.0])
        result = wasserstein_distance(p, q)
        # Wasserstein distance between identical distributions should be 0
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_shifted_distributions(self):
        """Test Wasserstein distance with shifted distributions."""
        p = jnp.array([1.0, 2.0, 3.0])
        q = jnp.array([2.0, 3.0, 4.0])
        # L1 Wasserstein is just mean absolute difference between sorted samples
        result = wasserstein_distance(p, q, p=1)
        expected = 1.0  # Each point is shifted by 1.0
        np.testing.assert_allclose(result, expected)

    def test_l2_wasserstein(self):
        """Test L2 Wasserstein distance."""
        p = jnp.array([1.0, 2.0, 3.0])
        q = jnp.array([2.0, 3.0, 4.0])
        # L2 Wasserstein is sqrt of mean squared difference between sorted samples
        result = wasserstein_distance(p, q, p=2)
        expected = 1.0  # sqrt(1^2) = 1.0
        np.testing.assert_allclose(result, expected)

    def test_distrax_error(self):
        """
        Test that Wasserstein distance raises NotImplementedError
        for distrax distributions.
        """
        p = Normal(loc=0.0, scale=1.0)
        q = Normal(loc=1.0, scale=2.0)
        with pytest.raises(NotImplementedError):
            wasserstein_distance(p, q)

    def test_batched_input(self):
        """Test Wasserstein distance with batched input."""
        p = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        q = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        result = wasserstein_distance(p, q, p=1, axis=1)
        expected = jnp.array([1.0, 1.0])  # Each point is shifted by 1.0
        np.testing.assert_allclose(result, expected)

    def test_default_l1_path_matches_calibrax_wasserstein(self):
        """Default 1D Wasserstein should track the shared CalibraX primitive."""
        p = jnp.array([1.0, 2.0, 3.0])
        q = jnp.array([2.0, 3.0, 4.0])

        np.testing.assert_allclose(wasserstein_distance(p, q), calibrax_wasserstein_1d(p, q))
