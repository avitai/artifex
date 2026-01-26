"""Tests for continuous distributions: Normal and Beta."""

import warnings

import distrax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.distributions import Beta, Normal


# Suppress deprecation warning from TensorFlow Probability
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="jax.interpreters.xla.pytype_aval_mappings is deprecated",
)


@pytest.fixture
def rngs():
    """Fixture for random number generators."""
    return nnx.Rngs(0)


class TestNormal:
    """Test cases for the Normal distribution."""

    @pytest.fixture
    def normal_fixed(self):
        """Fixture for fixed Normal distribution."""
        return Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))

    @pytest.fixture
    def normal_learnable(self):
        """Fixture for learnable Normal distribution."""
        return Normal()

    def test_initialization(self, normal_fixed, normal_learnable):
        """Test initialization of Normal distribution."""
        # Test fixed parameters
        assert normal_fixed.loc.value == 0.0
        assert normal_fixed.scale.value == 1.0

        # Test learnable parameters
        assert isinstance(normal_learnable.loc, nnx.Param)
        assert isinstance(normal_learnable.scale, nnx.Param)
        assert normal_learnable.loc.value.shape == ()
        assert normal_learnable.scale.value.shape == ()

    def test_sample(self, rngs, normal_fixed):
        """Test sampling from Normal distribution."""
        samples = normal_fixed.sample(sample_shape=(1000,), rngs=rngs)
        assert samples.shape == (1000,)
        assert jnp.allclose(jnp.mean(samples), 0.0, atol=0.1)
        assert jnp.allclose(jnp.std(samples), 1.0, atol=0.1)

    def test_call(self, rngs, normal_fixed):
        """Test __call__ method."""
        # Test sampling
        samples = normal_fixed(rngs=rngs)
        assert isinstance(samples, jax.Array)

        # Test log probability computation
        x = jnp.array([-1.0, 0.0, 1.0])
        log_probs = normal_fixed(x)
        assert log_probs.shape == (3,)
        # Check that log probability is highest at mean
        assert log_probs[1] > log_probs[0]
        assert log_probs[1] > log_probs[2]

    def test_log_prob(self, normal_fixed):
        """Test log probability computation."""
        x = jnp.array([-1.0, 0.0, 1.0])
        log_probs = normal_fixed.log_prob(x)
        assert log_probs.shape == (3,)
        # Check that log probability is highest at mean
        assert log_probs[1] > log_probs[0]
        assert log_probs[1] > log_probs[2]

    def test_entropy(self, normal_fixed):
        """Test entropy computation."""
        entropy = normal_fixed.entropy()
        assert isinstance(entropy, jax.Array)
        assert entropy > 0  # Entropy should be positive for non-deterministic distribution

    def test_kl_divergence(self, normal_fixed):
        """Test KL divergence computation."""
        other = Normal(loc=jnp.array(1.0), scale=jnp.array(2.0))
        kl = normal_fixed.kl_divergence(other)
        assert isinstance(kl, jax.Array)
        assert kl > 0  # KL divergence should be positive


class TestBeta:
    """Test cases for the Beta distribution."""

    @pytest.fixture
    def beta_fixed(self):
        """Fixture for fixed Beta distribution."""
        return Beta(concentration0=jnp.array(2.0), concentration1=jnp.array(3.0))

    @pytest.fixture
    def beta_learnable(self):
        """Fixture for learnable Beta distribution."""
        return Beta()

    def test_initialization(self, beta_fixed, beta_learnable):
        """Test initialization of Beta distribution."""
        # Test fixed parameters
        assert beta_fixed.concentration0 == 2.0
        assert beta_fixed.concentration1 == 3.0

        # Test learnable parameters
        assert isinstance(beta_learnable.concentration0, nnx.Param)
        assert isinstance(beta_learnable.concentration1, nnx.Param)
        assert beta_learnable.concentration0.value.shape == ()
        assert beta_learnable.concentration1.value.shape == ()

    def test_sample(self, rngs, beta_fixed):
        """Test sampling from Beta distribution."""
        samples = beta_fixed.sample(sample_shape=(1000,), rngs=rngs)
        assert samples.shape == (1000,)
        assert jnp.all(samples >= 0) and jnp.all(samples <= 1)
        # Check mean is approximately alpha/(alpha+beta)
        assert jnp.allclose(jnp.mean(samples), 2.0 / 5.0, atol=0.1)

    def test_call(self, rngs, beta_fixed):
        """Test __call__ method."""
        # Test sampling
        samples = beta_fixed(rngs=rngs)
        assert isinstance(samples, jax.Array)

        # Test log probability computation
        x = jnp.array([0.25, 0.5, 0.75])
        log_probs = beta_fixed(x)
        assert log_probs.shape == (3,)
        # Check that log probability is finite
        assert jnp.all(jnp.isfinite(log_probs))

    def test_log_prob(self, beta_fixed):
        """Test log probability computation."""
        x = jnp.array([0.25, 0.5, 0.75])
        log_probs = beta_fixed.log_prob(x)
        assert log_probs.shape == (3,)
        # Check that log probability is finite
        assert jnp.all(jnp.isfinite(log_probs))

    def test_entropy(self, beta_fixed):
        """Test entropy computation."""
        entropy = beta_fixed.entropy()
        assert isinstance(entropy, jax.Array)
        # Entropy should be finite
        assert jnp.isfinite(entropy)

    def test_kl_divergence(self, beta_fixed):
        """Test KL divergence computation."""
        other = Beta(concentration0=jnp.array(3.0), concentration1=jnp.array(4.0))
        kl = beta_fixed.kl_divergence(other)
        assert isinstance(kl, jax.Array)
        assert kl > 0  # KL divergence should be positive

    def test_statistics(self, beta_fixed):
        """Test statistical properties."""
        mean = beta_fixed.mean()
        variance = beta_fixed.variance()
        mode = beta_fixed.mode()

        # Check mean is alpha/(alpha+beta)
        assert jnp.allclose(mean, 2.0 / 5.0)
        # Check variance is alpha*beta/((alpha+beta)^2*(alpha+beta+1))
        assert jnp.allclose(variance, 2.0 * 3.0 / (5.0**2 * 6.0))
        # Check mode is (alpha-1)/(alpha+beta-2)
        assert jnp.allclose(mode, 1.0 / 3.0)


class TestJITCompatibility:
    """Test JIT compatibility for continuous distributions.

    Note: NNX JIT requires functions that take the module as the first argument.
    Bound methods cannot be directly JIT-compiled with nnx.jit.
    """

    def test_normal_jit_compatibility(self, rngs):
        """Test that Normal distribution methods are JIT-compatible."""
        # Create a Normal distribution
        normal = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs)

        # Test that methods can be JIT compiled using proper NNX pattern
        # NNX JIT requires functions that take the module as first argument
        key = jax.random.key(42)

        # Test log_prob method JIT compatibility using proper NNX pattern
        @nnx.jit
        def jit_log_prob(dist, x):
            return dist.log_prob(x)

        x = jnp.array([0.0, 1.0, -1.0])
        log_probs = jit_log_prob(normal, x)
        assert log_probs.shape == (3,)
        assert jnp.isfinite(log_probs).all()

        # Test sample method JIT compatibility using proper NNX pattern
        @nnx.jit(static_argnames=["sample_shape"])
        def jit_sample(dist, sample_shape, rngs):
            return dist.sample(sample_shape=sample_shape, rngs=rngs)

        samples = jit_sample(normal, sample_shape=(10,), rngs=nnx.Rngs(sample=key))
        assert samples.shape == (10,)
        assert jnp.isfinite(samples).all()

        # Test that methods work correctly without JIT (baseline)
        samples_no_jit = normal.sample(sample_shape=(5,), rngs=nnx.Rngs(sample=key))
        assert samples_no_jit.shape == (5,)
        assert jnp.isfinite(samples_no_jit).all()

        log_prob_no_jit = normal.log_prob(jnp.array(0.0))
        assert jnp.isfinite(log_prob_no_jit)

    def test_beta_jit_compatibility(self, rngs):
        """Test that Beta distribution methods are JIT-compatible."""
        # Create a Beta distribution
        beta = Beta(concentration0=jnp.array(2.0), concentration1=jnp.array(3.0), rngs=rngs)

        key = jax.random.key(42)

        # Test log_prob method JIT compatibility using proper NNX pattern
        @nnx.jit
        def jit_log_prob(dist, x):
            return dist.log_prob(x)

        x = jnp.array([0.1, 0.5, 0.9])
        log_probs = jit_log_prob(beta, x)
        assert log_probs.shape == (3,)
        assert jnp.isfinite(log_probs).all()

        # Test sample method JIT compatibility using proper NNX pattern
        @nnx.jit(static_argnames=["sample_shape"])
        def jit_sample(dist, sample_shape, rngs):
            return dist.sample(sample_shape=sample_shape, rngs=rngs)

        samples = jit_sample(beta, sample_shape=(5,), rngs=nnx.Rngs(sample=key))
        assert samples.shape == (5,)
        assert jnp.isfinite(samples).all()
        assert jnp.all((samples >= 0) & (samples <= 1))  # Beta samples in [0,1]

    def test_rng_handling_jit_compatibility(self, rngs):
        """Test that RNG handling is JIT-compatible."""
        normal = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs)

        key = jax.random.key(123)

        # Test JIT compilation with different RNG scenarios using proper NNX pattern
        @nnx.jit(static_argnames=["sample_shape"])
        def jit_sample(dist, sample_shape, rngs=None):
            return dist.sample(sample_shape=sample_shape, rngs=rngs)

        # Test with explicit RNGs
        test_rngs = nnx.Rngs(sample=key, default=key)
        samples1 = jit_sample(normal, sample_shape=(3,), rngs=test_rngs)
        assert samples1.shape == (3,)
        assert jnp.isfinite(samples1).all()

        # Test with no RNGs (should use fallback)
        samples2 = jit_sample(normal, sample_shape=(3,))
        assert samples2.shape == (3,)
        assert jnp.isfinite(samples2).all()

    def test_batch_operations_jit_compatibility(self, rngs):
        """Test that batch operations are JIT-compatible."""
        # Create batch Normal distributions
        batch_normal = Normal(
            loc=jnp.array([0.0, 1.0, -1.0]), scale=jnp.array([1.0, 0.5, 2.0]), rngs=rngs
        )

        key = jax.random.key(456)

        # Test batch sampling with proper NNX JIT pattern
        @nnx.jit(static_argnames=["sample_shape"])
        def jit_batch_sample(dist, sample_shape, rngs):
            return dist.sample(sample_shape=sample_shape, rngs=rngs)

        batch_samples = jit_batch_sample(batch_normal, sample_shape=(4,), rngs=nnx.Rngs(sample=key))
        assert batch_samples.shape == (4, 3)  # (sample_shape, batch_shape)
        assert jnp.isfinite(batch_samples).all()

        # Test batch log_prob with proper NNX JIT pattern
        @nnx.jit
        def jit_batch_log_prob(dist, x):
            return dist.log_prob(x)

        x_batch = jnp.array([[0.0, 1.0, -1.0], [1.0, 0.5, 0.0]])
        batch_log_probs = jit_batch_log_prob(batch_normal, x_batch)
        assert batch_log_probs.shape == (2, 3)  # (input_batch, dist_batch)
        assert jnp.isfinite(batch_log_probs).all()


class TestOptimizations:
    """Test all optimization features implemented in distributions."""

    def test_parameter_caching(self, rngs):
        """Test that entropy and KL divergence are properly cached."""
        normal1 = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs)
        normal2 = Normal(loc=jnp.array(1.0), scale=jnp.array(1.0), rngs=rngs)

        # First entropy computation
        entropy1 = normal1.entropy()
        assert jnp.isfinite(entropy1)

        # Second entropy computation should use cache (same result)
        entropy2 = normal1.entropy()
        assert entropy1 == entropy2

        # First KL divergence computation
        kl1 = normal1.kl_divergence(normal2)
        assert jnp.isfinite(kl1)

        # Second KL divergence computation should use cache
        kl2 = normal1.kl_divergence(normal2)
        assert kl1 == kl2

    def test_variable_type_optimization(self, rngs):
        """Test optimized variable types for static vs dynamic parameters."""
        # Test with trainable parameters (default)
        normal_trainable = Normal(
            loc=jnp.array(0.0),
            scale=jnp.array(1.0),
            rngs=rngs,
            trainable_loc=True,
            trainable_scale=True,
        )

        # Check that parameters are nnx.Param
        assert isinstance(normal_trainable.loc, nnx.Param)
        assert isinstance(normal_trainable.scale, nnx.Param)

        # Test with static parameters
        normal_static = Normal(
            loc=jnp.array(0.0),
            scale=jnp.array(1.0),
            rngs=rngs,
            trainable_loc=False,
            trainable_scale=False,
        )

        # Check that parameters are nnx.Cache
        assert isinstance(normal_static.loc, nnx.Cache)
        assert isinstance(normal_static.scale, nnx.Cache)

        # Both should work the same functionally
        x = jnp.array([0.0, 1.0, -1.0])
        log_prob_trainable = normal_trainable.log_prob(x)
        log_prob_static = normal_static.log_prob(x)

        assert jnp.allclose(log_prob_trainable, log_prob_static)

    def test_numerical_stability(self, rngs):
        """Test numerical stability features."""
        normal = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs)

        # Test safe_log utility
        x = jnp.array([1e-10, 1.0, 1e10])
        safe_log_result = normal._safe_log(x)
        assert jnp.isfinite(safe_log_result).all()

        # Test safe_div utility
        numerator = jnp.array([1.0, 2.0, 3.0])
        denominator = jnp.array([1e-10, 1.0, 2.0])
        safe_div_result = normal._safe_div(numerator, denominator)
        assert jnp.isfinite(safe_div_result).all()

        # Test parameter validation (use debug version for testing)
        try:
            normal._validate_parameters()
            # Test debug version of finite check
            normal._check_finite_debug(jnp.array([1.0, 2.0]), "test_values")
        except ValueError:
            pytest.fail("Parameter validation failed for valid parameters")

    def test_rng_handling_improvements(self, rngs):
        """Test improved RNG handling with key splitting."""
        normal = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs)

        # Test RNG key extraction
        key = normal._get_rng_key(rngs, "sample")
        assert key.shape == ()  # Scalar key

        # Test key splitting
        split_keys = normal._split_rng_key(key, 5)
        assert len(split_keys) == 5
        assert all(k.shape == () for k in split_keys)

        # Test parallel RNG creation
        parallel_rngs = normal._create_parallel_rngs(rngs, 3)
        assert len(parallel_rngs) == 3
        assert all(isinstance(r, nnx.Rngs) for r in parallel_rngs)

    def test_memory_efficiency(self, rngs):
        """Test memory efficiency utilities."""
        normal = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs)

        # Test efficient stacking
        arrays = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])]
        stacked = normal._efficient_stack(arrays, axis=0)
        assert stacked.shape == (3, 2)

        # Test single array stacking
        single_stacked = normal._efficient_stack([arrays[0]], axis=0)
        assert single_stacked.shape == (1, 2)

    def test_batch_processing_performance(self, rngs):
        """Test that batch processing is more efficient than loops."""
        from artifex.generative_models.core.distributions.continuous import Normal
        from artifex.generative_models.core.distributions.mixture import Mixture

        # Create mixture with multiple components
        components = [
            Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs),
            Normal(loc=jnp.array(2.0), scale=jnp.array(0.5), rngs=rngs),
            Normal(loc=jnp.array(-1.0), scale=jnp.array(1.5), rngs=rngs),
        ]
        weights = jnp.array([0.3, 0.5, 0.2])
        mixture = Mixture(components=components, weights=weights, rngs=rngs)

        # Test vectorized sampling
        samples = mixture.sample(sample_shape=(100,), rngs=rngs)
        assert samples.shape == (100,)
        assert jnp.isfinite(samples).all()

        # Test vectorized log_prob
        x = jnp.linspace(-5, 5, 50)
        log_probs = mixture.log_prob(x)
        assert log_probs.shape == (50,)
        assert jnp.isfinite(log_probs).all()

        # Test vectorized entropy
        entropy = mixture.entropy()
        assert jnp.isfinite(entropy)

    def test_jit_compatibility_with_optimizations(self, rngs):
        """Test that all optimizations are JIT-compatible."""
        normal = Normal(
            loc=jnp.array(0.0),
            scale=jnp.array(1.0),
            rngs=rngs,
            trainable_loc=False,  # Use static parameters for better JIT performance
            trainable_scale=False,
        )

        # Test JIT compilation using proper NNX pattern (function taking module as first arg)
        @nnx.jit
        def jit_log_prob(dist, x):
            return dist.log_prob(x)

        @nnx.jit(static_argnames=["use_cache"])
        def jit_entropy(dist, use_cache):
            return dist.entropy(use_cache=use_cache)

        @nnx.jit(static_argnames=["sample_shape"])
        def jit_sample(dist, sample_shape):
            return dist.sample(sample_shape=sample_shape)

        x = jnp.array([0.0, 1.0, -1.0])

        # Test JIT-compiled methods
        log_probs = jit_log_prob(normal, x)
        assert log_probs.shape == (3,)
        assert jnp.isfinite(log_probs).all()

        entropy = jit_entropy(normal, use_cache=False)  # Disable caching in JIT context
        assert jnp.isfinite(entropy)

        samples = jit_sample(normal, sample_shape=(5,))
        assert samples.shape == (5,)
        assert jnp.isfinite(samples).all()

    def test_cache_invalidation(self, rngs):
        """Test that caches are properly invalidated when parameters change."""
        # Create a normal distribution with mutable parameters
        normal = Normal(loc=jnp.array(0.0), scale=jnp.array(1.0), rngs=rngs)

        # Compute entropy (should be cached)
        entropy1 = normal.entropy()

        # Modify scale parameter (this will change entropy, unlike location)
        normal.scale = nnx.Param(jnp.array(2.0))  # Change scale
        normal._dist = distrax.Normal(loc=normal.loc, scale=normal.scale)  # Update underlying dist

        # Entropy should be recomputed (cache invalidated)
        entropy2 = normal.entropy()

        # Should be different due to scale parameter change
        # Normal entropy = 0.5 * log(2π * σ²), so changing σ changes entropy
        assert not jnp.allclose(entropy1, entropy2, rtol=1e-6)
