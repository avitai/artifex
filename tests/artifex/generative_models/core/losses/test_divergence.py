"""Tests for the divergence losses module."""

import jax
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
    energy_distance,
    gaussian_kl_divergence,
    js_divergence,
    kl_divergence,
    maximum_mean_discrepancy,
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

    def test_general_lp_path_supports_tuple_axis_and_weighted_reduction(self):
        """The local Wasserstein path should support higher-order distances."""
        p = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        q = p + 1.0
        weights = jnp.array([1.0, 0.5])

        result = wasserstein_distance(p, q, p=3, axis=(1,), weights=weights)

        np.testing.assert_allclose(result, 0.75)


class TestMaximumMeanDiscrepancy:
    """Tests for MMD divergence paths."""

    def test_rbf_kernel_returns_per_batch_values(self):
        """RBF MMD should delegate to the shared CalibraX estimator per batch."""
        predictions = jnp.array([[[0.0], [1.0]], [[1.0], [2.0]]])
        targets = predictions + 0.25

        result = maximum_mean_discrepancy(
            predictions,
            targets,
            kernel_type="rbf",
            kernel_bandwidth=1.5,
            reduction="none",
        )

        assert result.shape == (2,)
        assert jnp.isfinite(result).all()

    @pytest.mark.parametrize("kernel_type", ["linear", "polynomial"])
    def test_local_kernel_estimators_return_weighted_means(self, kernel_type):
        """Local MMD estimators should support weighted reductions."""
        predictions = jnp.array([[[0.0, 1.0], [1.0, 0.0]], [[2.0, 1.0], [3.0, 0.0]]])
        targets = predictions + 0.5
        weights = jnp.array([1.0, 0.25])

        result = maximum_mean_discrepancy(
            predictions,
            targets,
            kernel_type=kernel_type,
            reduction="mean",
            weights=weights,
        )

        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_unknown_kernel_type_raises(self):
        """Unsupported kernels should fail clearly."""
        samples = jnp.zeros((1, 2, 1))

        with pytest.raises(ValueError, match="Unknown kernel type"):
            maximum_mean_discrepancy(samples, samples, kernel_type="unknown")


class TestEnergyDistance:
    """Tests for energy distance."""

    def test_identical_samples_have_zero_energy_distance(self):
        """Matching empirical distributions should have zero energy distance."""
        samples = jnp.array([[[0.0], [1.0]], [[2.0], [3.0]]])

        result = energy_distance(samples, samples, reduction="none")

        np.testing.assert_allclose(result, jnp.zeros((2,)), atol=1e-6)

    def test_energy_distance_supports_beta_and_weights(self):
        """Energy distance should support non-default beta and weighted reductions."""
        predictions = jnp.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]])
        targets = predictions + 0.5
        weights = jnp.array([1.0, 0.5])

        result = energy_distance(predictions, targets, beta=1.5, weights=weights)

        assert result.shape == ()
        assert jnp.isfinite(result)
        assert result >= 0.0


class TestGaussianKLDivergence:
    """Tests for closed-form Gaussian KL divergence."""

    def test_standard_normal_has_zero_kl(self):
        """A unit Gaussian posterior should match the unit Gaussian prior."""
        mean = jnp.zeros((3, 4))
        logvar = jnp.zeros((3, 4))

        result = gaussian_kl_divergence(mean, logvar, reduction="none")

        np.testing.assert_allclose(result, jnp.zeros((3,)))

    def test_gaussian_kl_supports_custom_axis_and_weights(self):
        """Gaussian KL should reduce requested axes before applying sample weights."""
        mean = jnp.ones((2, 2, 2))
        logvar = jnp.zeros((2, 2, 2))
        weights = jnp.array([1.0, 0.5])

        result = gaussian_kl_divergence(mean, logvar, axis=(1, 2), weights=weights)

        np.testing.assert_allclose(result, 1.5)


class TestDivergenceJAXTransformCompatibility:
    """JIT and differentiation checks for divergence paths used in training losses."""

    @pytest.mark.parametrize(
        ("name", "loss_fn"),
        [
            (
                "kl",
                lambda logits, target: kl_divergence(jax.nn.softmax(logits), target),
            ),
            (
                "reverse_kl",
                lambda logits, target: reverse_kl_divergence(jax.nn.softmax(logits), target),
            ),
            (
                "js",
                lambda logits, target: js_divergence(jax.nn.softmax(logits), target),
            ),
            (
                "wasserstein",
                lambda logits, target: wasserstein_distance(logits, target, p=2),
            ),
        ],
    )
    def test_probability_divergences_are_jittable_and_differentiable(self, name, loss_fn):
        """Probability-style divergences should compile and expose finite input gradients."""
        logits = jnp.array([0.2, -0.1, 0.5])
        target = jnp.array([0.2, 0.5, 0.3])

        compiled_value = jax.jit(loss_fn)(logits, target)
        gradients = jax.grad(loss_fn)(logits, target)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value), name
        assert jnp.isfinite(gradients).all(), name

    @pytest.mark.parametrize("kernel_type", ["rbf", "linear", "polynomial"])
    def test_mmd_kernels_are_jittable_and_differentiable(self, kernel_type):
        """MMD kernels used as losses should compile and expose finite sample gradients."""
        predictions = jnp.array([[[0.0, 0.1], [1.0, 0.2]], [[2.0, 1.0], [3.0, 0.5]]])
        targets = predictions + 1.0

        def loss_fn(samples):
            return maximum_mean_discrepancy(
                samples,
                targets,
                kernel_type=kernel_type,
                kernel_bandwidth=1.5,
            )

        compiled_value = jax.jit(loss_fn)(predictions)
        gradients = jax.grad(loss_fn)(predictions)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert jnp.isfinite(gradients).all()

    def test_energy_distance_is_jittable_and_differentiable(self):
        """Energy distance should compile and keep finite gradients at self-pair distances."""
        predictions = jnp.array([[[0.0, 0.1], [1.0, 0.2]], [[2.0, 1.0], [3.0, 0.5]]])
        targets = predictions + 0.25

        def loss_fn(samples):
            return energy_distance(samples, targets, beta=1.5)

        compiled_value = jax.jit(loss_fn)(predictions)
        gradients = jax.grad(loss_fn)(predictions)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert jnp.isfinite(gradients).all()

    def test_gaussian_kl_is_jittable_and_differentiable(self):
        """Closed-form Gaussian KL should compile and expose finite latent gradients."""
        mean = jnp.array([[0.2, -0.1], [0.5, 0.3]])
        logvar = jnp.array([[0.0, 0.1], [-0.2, 0.2]])

        def loss_fn(latent_mean, latent_logvar):
            return gaussian_kl_divergence(latent_mean, latent_logvar)

        compiled_value = jax.jit(loss_fn)(mean, logvar)
        mean_grad, logvar_grad = jax.grad(loss_fn, argnums=(0, 1))(mean, logvar)

        assert compiled_value.shape == ()
        assert jnp.isfinite(compiled_value)
        assert jnp.isfinite(mean_grad).all()
        assert jnp.isfinite(logvar_grad).all()
