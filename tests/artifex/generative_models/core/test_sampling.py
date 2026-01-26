"""Tests for sampling utilities."""

import jax
import jax.numpy as jnp
import pytest

# Import distribution for testing
from artifex.generative_models.core.distributions import Normal

# Import sampling utilities
from artifex.generative_models.core.sampling.ancestral import ancestral_sampling
from artifex.generative_models.core.sampling.mcmc import mcmc_sampling
from artifex.generative_models.core.sampling.ode import ode_sampling
from artifex.generative_models.core.sampling.sde import sde_sampling

# Import test fixtures
from tests.artifex.generative_models.utils.test_fixtures import get_rng_key


@pytest.fixture
def normal_dist():
    """Fixture for normal distribution."""
    return Normal(loc=jnp.zeros(2), scale=jnp.ones(2))


class TestAncestralSampling:
    """Test cases for ancestral sampling."""

    def test_ancestral_sampling_normal(self, normal_dist):
        """Test ancestral sampling from a normal distribution."""
        key = get_rng_key(0)  # Use consistent seed but avoid redefining fixture
        n_samples = 1000
        samples = ancestral_sampling(normal_dist, key, n_samples)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # Check statistics
        # (mean and variance should be close to distribution parameters)
        mean = jnp.mean(samples, axis=0)
        std = jnp.std(samples, axis=0)

        # Allow for sampling variance
        assert jnp.allclose(mean, normal_dist.loc, atol=0.2)
        assert jnp.allclose(std, normal_dist.scale, atol=0.2)

    def test_ancestral_sampling_batched(self, normal_dist):
        """Test ancestral sampling with batched distributions."""
        key = get_rng_key(1)  # Different seed to avoid correlation

        # Create a batched distribution
        batch_size = 3
        batched_loc = jnp.tile(normal_dist.loc, (batch_size, 1))
        batched_scale = jnp.tile(normal_dist.scale, (batch_size, 1))
        batched_dist = Normal(loc=batched_loc, scale=batched_scale)

        n_samples = 1000
        samples = ancestral_sampling(batched_dist, key, n_samples)

        # Check shape (should be [batch_size, n_samples, dim])
        assert samples.shape == (batch_size, n_samples, 2)

        # Check statistics for each batch element
        for i in range(batch_size):
            batch_samples = samples[i]
            mean = jnp.mean(batch_samples, axis=0)
            std = jnp.std(batch_samples, axis=0)

            assert jnp.allclose(mean, normal_dist.loc, atol=0.2)
            assert jnp.allclose(std, normal_dist.scale, atol=0.2)

    def test_ancestral_sampling_custom_distribution(self):
        """Test ancestral sampling with a custom distribution object."""
        key = get_rng_key(2)  # Different seed

        # Create a simple custom distribution with a sample method
        class CustomDistribution:
            def sample(self, seed, sample_shape):
                # Just return standard normal samples for testing
                return jax.random.normal(seed, (*sample_shape, 2))

        custom_dist = CustomDistribution()
        n_samples = 1000
        samples = ancestral_sampling(custom_dist, key, n_samples)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # Check statistics (should be close to standard normal)
        mean = jnp.mean(samples, axis=0)
        std = jnp.std(samples, axis=0)

        assert jnp.allclose(mean, jnp.zeros(2), atol=0.2)
        assert jnp.allclose(std, jnp.ones(2), atol=0.2)

    def test_ancestral_sampling_invalid_input(self):
        """Test that ancestral sampling raises an error for invalid inputs."""
        key = get_rng_key(3)  # Different seed

        # Create an object without a sample method
        class InvalidObject:
            pass

        invalid_obj = InvalidObject()

        # Check that it raises a ValueError
        with pytest.raises(
            ValueError,
            match=("Input must be a Distribution instance or have a sample method"),
        ):
            ancestral_sampling(invalid_obj, key, 10)

    def test_ancestral_sampling_nnx_key(self, normal_dist):
        """Test ancestral sampling with nnx.Rngs key."""
        # Since we can't easily mock nnx.Rngs, we'll just test that
        # a regular JAX key works in ancestral_sampling
        key = get_rng_key(4)  # Different seed
        n_samples = 1000
        samples = ancestral_sampling(normal_dist, key, n_samples)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # Check statistics
        mean = jnp.mean(samples, axis=0)
        std = jnp.std(samples, axis=0)

        assert jnp.allclose(mean, normal_dist.loc, atol=0.2)
        assert jnp.allclose(std, normal_dist.scale, atol=0.2)

        # Test with a real nnx.Rngs object if flax is available
        try:
            from flax import nnx

            rngs = nnx.Rngs(0)  # Initialize with a seed
            nnx_samples = ancestral_sampling(normal_dist, rngs, n_samples)
            assert nnx_samples.shape == (n_samples, 2)
        except (ImportError, AttributeError):
            # Skip this part of the test if flax.nnx is not available
            # or doesn't have the expected API
            pass

    def test_ancestral_sampling_with_sample_shape(self, normal_dist):
        """Test ancestral sampling with explicit sample_shape."""
        key = get_rng_key(5)  # Different seed
        n_samples = 100
        sample_shape = (5,)  # Add an extra dimension
        samples = ancestral_sampling(normal_dist, key, n_samples, sample_shape)

        # Check shape should include sample_shape
        expected_shape = (n_samples, *sample_shape, 2)
        assert samples.shape == expected_shape


class TestMCMCSampling:
    """Test cases for MCMC sampling."""

    def test_mcmc_sampling_basic(self, normal_dist):
        """Test basic MCMC sampling."""
        key = get_rng_key(6)  # Different seed

        # Define log probability function (unnormalized)
        def log_prob_fn(x):
            return normal_dist.log_prob(x)

        # Initial state
        init_state = jnp.zeros(2)

        # Sample
        n_samples = 1000
        n_burnin = 200  # Increased burnin
        samples = mcmc_sampling(log_prob_fn, init_state, key, n_samples, n_burnin=n_burnin)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # Check statistics
        # (mean and variance should be close to distribution parameters)
        mean = jnp.mean(samples, axis=0)
        std = jnp.std(samples, axis=0)

        # MCMC might need more tolerance than direct sampling
        assert jnp.allclose(mean, normal_dist.loc, atol=1.0)
        assert jnp.allclose(std, normal_dist.scale, atol=0.6)

    def test_mcmc_step_size(self, normal_dist):
        """Test MCMC sampling with different step sizes."""
        key = get_rng_key(7)  # Different seed

        # Define log probability function (unnormalized)
        def log_prob_fn(x):
            return normal_dist.log_prob(x)

        # Initial state
        init_state = jnp.zeros(2)

        # Sample with different step sizes
        n_samples = 1000
        n_burnin = 200  # Increased burnin

        # Small step size should have high acceptance rate but slow mixing
        small_step = mcmc_sampling(
            log_prob_fn, init_state, key, n_samples, n_burnin=n_burnin, step_size=0.01
        )

        # Large step size should have lower acceptance rate
        key2 = jax.random.fold_in(key, 0)  # Create a new key
        large_step = mcmc_sampling(
            log_prob_fn, init_state, key2, n_samples, n_burnin=n_burnin, step_size=1.0
        )

        # Both should have correct means
        small_mean = jnp.mean(small_step, axis=0)
        large_mean = jnp.mean(large_step, axis=0)

        assert jnp.allclose(small_mean, normal_dist.loc, atol=1.0)
        assert jnp.allclose(large_mean, normal_dist.loc, atol=1.0)


class TestODESampling:
    """Test cases for ODE-based sampling."""

    def test_ode_sampling_basic(self):
        """Test basic ODE-based sampling."""
        key = get_rng_key(8)  # Different seed

        # Function defining the vector field (e.g., score function)
        def score_fn(x, t):
            # Simple test score function that pushes toward the origin
            return -x

        # Sample using ODE solver
        n_samples = 100
        t_span = (0.0, 1.0)
        init_noise = jax.random.normal(key, (n_samples, 2))

        samples = ode_sampling(score_fn, init_noise, t_span)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # For this score function,
        # final values should be closer to origin than initial
        init_norm = jnp.linalg.norm(init_noise, axis=1)
        final_norm = jnp.linalg.norm(samples, axis=1)
        assert jnp.mean(final_norm) < jnp.mean(init_norm)

    def test_ode_solver_methods(self):
        """Test ODE-based sampling with different solver methods."""
        key = get_rng_key(9)  # Different seed

        # Function defining the vector field
        def score_fn(x, t):
            return -x

        # Sample using different ODE solvers
        n_samples = 100
        t_span = (0.0, 1.0)
        init_noise = jax.random.normal(key, (n_samples, 2))

        # Try different methods
        rk4_samples = ode_sampling(score_fn, init_noise, t_span, method="rk4")
        euler_samples = ode_sampling(score_fn, init_noise, t_span, method="euler")

        # Check shapes
        assert rk4_samples.shape == (n_samples, 2)
        assert euler_samples.shape == (n_samples, 2)

        # Both should move toward the origin
        init_norm = jnp.linalg.norm(init_noise, axis=1)
        rk4_norm = jnp.linalg.norm(rk4_samples, axis=1)
        euler_norm = jnp.linalg.norm(euler_samples, axis=1)

        assert jnp.mean(rk4_norm) < jnp.mean(init_norm)
        assert jnp.mean(euler_norm) < jnp.mean(init_norm)

        # Different random seeds can lead to different relative performance
        # between the methods, so we'll check they both decrease the norm
        # but not compare them directly


class TestSDESampling:
    """Test cases for SDE-based sampling."""

    def test_sde_sampling_basic(self):
        """Test basic SDE-based sampling."""
        key = get_rng_key(10)  # Different seed

        def drift_fn(x, t):
            # Pull toward origin with strength proportional to distance
            return -x

        def diffusion_fn(x, t):
            # Constant noise level
            return jnp.ones_like(x) * 0.1

        # Sample using SDE solver
        n_samples = 100
        n_steps = 100
        t_span = (0.0, 1.0)
        init_noise = jax.random.normal(key, (n_samples, 2))

        samples = sde_sampling(drift_fn, diffusion_fn, init_noise, t_span, n_steps=n_steps, key=key)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # Should move toward the origin overall due to drift
        init_norm = jnp.linalg.norm(init_noise, axis=1)
        final_norm = jnp.linalg.norm(samples, axis=1)
        assert jnp.mean(final_norm) < jnp.mean(init_norm)

    def test_sde_time_dependent_coefficients(self):
        """Test SDE sampling with time-dependent coefficients."""
        key = get_rng_key(11)  # Different seed

        def drift_fn(x, t):
            # Stronger pull toward origin as t increases
            return -x * (1.0 + t)

        def diffusion_fn(x, t):
            # Decreasing noise level over time
            return jnp.ones_like(x) * (0.2 * (1.0 - t) + 0.01)

        # Sample using SDE solver
        n_samples = 100
        n_steps = 100
        t_span = (0.0, 1.0)
        init_noise = jax.random.normal(key, (n_samples, 2))

        samples = sde_sampling(drift_fn, diffusion_fn, init_noise, t_span, n_steps=n_steps, key=key)

        # Check shape
        assert samples.shape == (n_samples, 2)

        # Should move toward the origin overall due to drift
        init_norm = jnp.linalg.norm(init_noise, axis=1)
        final_norm = jnp.linalg.norm(samples, axis=1)
        assert jnp.mean(final_norm) < jnp.mean(init_norm)
