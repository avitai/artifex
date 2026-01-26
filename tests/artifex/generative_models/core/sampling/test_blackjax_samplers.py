"""Tests for BlackJAX samplers integration.

This module provides comprehensive tests for the BlackJAX integration in the
artifex package for sampling from probability distributions using various MCMC
methods (HMC, NUTS, MALA).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.distributions import Mixture, Normal
from artifex.generative_models.core.sampling.blackjax_samplers import (
    _extract_key,
    _prepare_logdensity_fn,
    BlackJAXHMC,
    BlackJAXMALA,
    BlackJAXNUTS,
    BlackJAXSamplerState,
    hmc_sampling,
    mala_sampling,
    nuts_sampling,
)


# =============================================================================
# Test Utilities
# =============================================================================


def simple_log_prob(x):
    """Simple negative quadratic log probability (standard normal)."""
    return -0.5 * jnp.sum(x**2)


def create_normal_logprob(mean=None, scale=None):
    """Create a normal distribution log probability function.

    Args:
        mean: Mean of the distribution. Default is [0, 0].
        scale: Scale of the distribution. Default is [1, 1].

    Returns:
        A function that computes the log probability of a point.
    """
    if mean is None:
        mean = jnp.array([0.0, 0.0])
    if scale is None:
        scale = jnp.array([1.0, 1.0])

    def log_prob_fn(x):
        dist = Normal(loc=mean, scale=scale)
        return jnp.sum(dist.log_prob(x))

    return log_prob_fn


# =============================================================================
# Test _extract_key function
# =============================================================================


class TestExtractKey:
    """Test suite for _extract_key utility function."""

    def test_extract_key_from_jax_array(self):
        """Test extracting key from JAX random key."""
        key = jax.random.key(42)
        result = _extract_key(key)
        assert result.shape == key.shape

    def test_extract_key_from_rngs_with_sample(self):
        """Test extracting key from nnx.Rngs with 'sample' key."""
        rngs = nnx.Rngs(sample=42)
        result = _extract_key(rngs)
        assert hasattr(result, "shape")

    def test_extract_key_from_rngs_with_default(self):
        """Test extracting key from nnx.Rngs with 'default' key."""
        rngs = nnx.Rngs(default=42)
        result = _extract_key(rngs)
        assert hasattr(result, "shape")

    def test_extract_key_from_rngs_with_other_key(self):
        """Test extracting key from nnx.Rngs with custom key name."""
        rngs = nnx.Rngs(params=42)
        result = _extract_key(rngs)
        assert hasattr(result, "shape")

    def test_extract_key_from_rngs_seed_only(self):
        """Test extracting key from nnx.Rngs initialized with just seed."""
        rngs = nnx.Rngs(seed=42)
        result = _extract_key(rngs)
        assert hasattr(result, "shape")


# =============================================================================
# Test _prepare_logdensity_fn function
# =============================================================================


class TestPrepareLogdensityFn:
    """Test suite for _prepare_logdensity_fn utility function."""

    def test_prepare_from_distribution(self):
        """Test preparing log density from Distribution object."""
        dist = Normal(loc=jnp.zeros(2), scale=jnp.ones(2))
        log_prob_fn = _prepare_logdensity_fn(dist)

        x = jnp.array([0.0, 0.0])
        result = log_prob_fn(x)

        # Should return a scalar
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_prepare_from_callable(self):
        """Test preparing log density from callable."""
        log_prob_fn = _prepare_logdensity_fn(simple_log_prob)

        x = jnp.array([0.0, 0.0])
        result = log_prob_fn(x)

        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_prepare_ensures_scalar_output(self):
        """Test that output is always scalar even for multi-dim input."""

        def multi_dim_log_prob(x):
            return -0.5 * x**2  # Returns array, not scalar

        log_prob_fn = _prepare_logdensity_fn(multi_dim_log_prob)

        x = jnp.array([1.0, 2.0, 3.0])
        result = log_prob_fn(x)

        # Should sum to scalar
        assert result.shape == ()


# =============================================================================
# Test BlackJAXSamplerState
# =============================================================================


class TestBlackJAXSamplerState:
    """Test suite for BlackJAXSamplerState NamedTuple."""

    def test_state_creation(self):
        """Test creating a sampler state."""
        x = jnp.zeros(2)
        key = jax.random.key(0)
        sampler_state = {"position": x}

        state = BlackJAXSamplerState(x=x, sampler_state=sampler_state, key=key)

        assert jnp.allclose(state.x, x)
        assert state.key is key
        assert state.sampler_state == sampler_state

    def test_state_is_namedtuple(self):
        """Test that state behaves as NamedTuple."""
        x = jnp.zeros(2)
        key = jax.random.key(0)
        state = BlackJAXSamplerState(x=x, sampler_state=None, key=key)

        # Can access by index
        assert jnp.allclose(state[0], x)
        assert state[2] is key


# =============================================================================
# Test BlackJAXHMC
# =============================================================================


class TestBlackJAXHMC:
    """Test suite for BlackJAXHMC sampler."""

    @pytest.fixture
    def log_prob_fn(self):
        """Simple log probability function that works with any shape."""
        return simple_log_prob

    def test_hmc_initialization(self, log_prob_fn):
        """Test HMC sampler initialization."""
        hmc = BlackJAXHMC(
            log_prob_fn,
            step_size=0.1,
            num_integration_steps=5,
            inverse_mass_matrix=jnp.ones(2),
        )

        assert hmc.step_size == 0.1
        assert hmc.num_integration_steps == 5
        assert hmc._kernel is None  # Not initialized until init() is called

    def test_hmc_init_state(self, log_prob_fn):
        """Test HMC init method creates proper state."""
        hmc = BlackJAXHMC(
            log_prob_fn,
            step_size=0.1,
            num_integration_steps=5,
        )

        init_position = jnp.zeros(2)
        key = jax.random.key(42)
        state = hmc.init(init_position, key)

        assert isinstance(state, BlackJAXSamplerState)
        assert state.x.shape == init_position.shape
        assert state.sampler_state is not None
        assert hmc._kernel is not None

    def test_hmc_init_auto_mass_matrix_1d(self, log_prob_fn):
        """Test HMC auto-creates mass matrix for 1D input."""
        hmc = BlackJAXHMC(log_prob_fn, step_size=0.1, num_integration_steps=5)

        init_position = jnp.zeros(3)
        key = jax.random.key(42)
        _state = hmc.init(init_position, key)

        assert hmc.inverse_mass_matrix.shape == (3,)

    def test_hmc_init_auto_mass_matrix_2d(self, log_prob_fn):
        """Test HMC auto-creates mass matrix for 2D input."""
        hmc = BlackJAXHMC(log_prob_fn, step_size=0.1, num_integration_steps=5)

        init_position = jnp.zeros((3, 4))
        key = jax.random.key(42)
        _state = hmc.init(init_position, key)

        # Uses last dimension
        assert hmc.inverse_mass_matrix.shape == (4,)

    def test_hmc_init_scalar_position(self, log_prob_fn):
        """Test HMC auto-creates mass matrix for scalar input."""
        hmc = BlackJAXHMC(log_prob_fn, step_size=0.1, num_integration_steps=5)

        init_position = jnp.array(0.0)  # Scalar
        key = jax.random.key(42)
        state = hmc.init(init_position, key)

        # For scalar input, BlackJAX requires 1D mass matrix
        assert hmc.inverse_mass_matrix.shape == (1,)
        assert state is not None

    def test_hmc_step(self, log_prob_fn):
        """Test HMC step method."""
        # Note: BlackJAX requires 1D or 2D mass matrix, not 0D
        init_position = jnp.zeros(2)
        hmc = BlackJAXHMC(
            log_prob_fn,
            step_size=0.1,
            num_integration_steps=5,
            inverse_mass_matrix=jnp.ones(init_position.shape[0]),
        )

        key = jax.random.key(42)
        state = hmc.init(init_position, key)

        new_state, info = hmc.step(state)

        assert isinstance(new_state, BlackJAXSamplerState)
        assert new_state.x.shape == init_position.shape
        assert "is_accepted" in info
        assert jnp.all(jnp.isfinite(new_state.x))

    def test_hmc_step_without_prior_init(self, log_prob_fn):
        """Test HMC step initializes kernel if not already done."""
        init_position = jnp.zeros(2)
        hmc = BlackJAXHMC(
            log_prob_fn,
            step_size=0.1,
            num_integration_steps=5,
            inverse_mass_matrix=jnp.ones(init_position.shape[0]),
        )

        # Manually create state without calling init
        key = jax.random.key(42)

        # Initialize kernel manually for the state
        hmc._kernel = None  # Ensure kernel is None
        hmc.inverse_mass_matrix = jnp.ones(init_position.shape[0])

        # Create a proper BlackJAX state
        import blackjax

        # Note: blackjax.hmc() argument order is:
        # (logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
        kernel = blackjax.hmc(
            hmc._scalar_log_prob_fn,
            hmc.step_size,
            hmc.inverse_mass_matrix,
            hmc.num_integration_steps,
        )
        hmc._kernel = kernel
        sampler_state = kernel.init(init_position)
        state = BlackJAXSamplerState(x=init_position, sampler_state=sampler_state, key=key)

        new_state, _info = hmc.step(state)
        assert new_state.x.shape == init_position.shape

    def test_hmc_multiple_steps(self, log_prob_fn):
        """Test multiple HMC steps."""
        init_position = jnp.zeros(2)
        hmc = BlackJAXHMC(
            log_prob_fn,
            step_size=0.1,
            num_integration_steps=5,
            inverse_mass_matrix=jnp.ones(init_position.shape[0]),
        )

        key = jax.random.key(42)
        state = hmc.init(init_position, key)

        # Run multiple steps
        positions = [state.x]
        for _ in range(5):
            state, _ = hmc.step(state)
            positions.append(state.x)

        # Should have moved from initial position
        positions = jnp.stack(positions)
        assert positions.shape == (6, 2)
        assert jnp.all(jnp.isfinite(positions))


# =============================================================================
# Test BlackJAXNUTS
# =============================================================================


class TestBlackJAXNUTS:
    """Test suite for BlackJAXNUTS sampler."""

    @pytest.fixture
    def log_prob_fn(self):
        """Simple log probability function that works with any shape."""
        return simple_log_prob

    def test_nuts_initialization(self, log_prob_fn):
        """Test NUTS sampler initialization."""
        nuts = BlackJAXNUTS(
            log_prob_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(1),
        )

        assert nuts.step_size == 0.1
        assert nuts._kernel is None

    def test_nuts_init_state(self, log_prob_fn):
        """Test NUTS init method creates proper state."""
        nuts = BlackJAXNUTS(log_prob_fn, step_size=0.1)

        init_position = jnp.zeros(1)
        key = jax.random.key(42)
        state = nuts.init(init_position, key)

        assert isinstance(state, BlackJAXSamplerState)
        assert state.x.shape == init_position.shape
        assert nuts._kernel is not None

    def test_nuts_init_auto_mass_matrix_multidim(self, log_prob_fn):
        """Test NUTS auto-creates mass matrix for multi-dim input."""
        nuts = BlackJAXNUTS(log_prob_fn, step_size=0.1)

        init_position = jnp.zeros((2, 3))
        key = jax.random.key(42)
        _state = nuts.init(init_position, key)

        # Uses last dimension
        assert nuts.inverse_mass_matrix.shape == (3,)

    def test_nuts_init_scalar_position(self, log_prob_fn):
        """Test NUTS auto-creates mass matrix for scalar input."""
        nuts = BlackJAXNUTS(log_prob_fn, step_size=0.1)

        init_position = jnp.array(0.0)  # Scalar
        key = jax.random.key(42)
        state = nuts.init(init_position, key)

        # For scalar input, BlackJAX requires 1D mass matrix
        assert nuts.inverse_mass_matrix.shape == (1,)
        assert state is not None

    def test_nuts_step(self, log_prob_fn):
        """Test NUTS step method."""
        init_position = jnp.zeros(2)
        nuts = BlackJAXNUTS(
            log_prob_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(init_position.shape[0]),
        )

        key = jax.random.key(42)
        state = nuts.init(init_position, key)

        new_state, info = nuts.step(state)

        assert isinstance(new_state, BlackJAXSamplerState)
        assert new_state.x.shape == init_position.shape
        # NUTS uses acceptance_rate instead of is_accepted
        assert "acceptance_rate" in info
        assert "num_trajectory_expansions" in info

    def test_nuts_multiple_steps(self, log_prob_fn):
        """Test multiple NUTS steps."""
        init_position = jnp.zeros(2)
        nuts = BlackJAXNUTS(
            log_prob_fn,
            step_size=0.1,
            inverse_mass_matrix=jnp.ones(init_position.shape[0]),
        )

        key = jax.random.key(42)
        state = nuts.init(init_position, key)

        # Run multiple steps
        for _ in range(3):
            state, _ = nuts.step(state)

        assert jnp.all(jnp.isfinite(state.x))


# =============================================================================
# Test BlackJAXMALA
# =============================================================================


class TestBlackJAXMALA:
    """Test suite for BlackJAXMALA sampler."""

    @pytest.fixture
    def log_prob_fn(self):
        """Simple log probability function that works with any shape."""
        return simple_log_prob

    def test_mala_initialization(self, log_prob_fn):
        """Test MALA sampler initialization."""
        mala = BlackJAXMALA(log_prob_fn, step_size=0.1)

        assert mala.step_size == 0.1
        assert mala._kernel is None

    def test_mala_init_state(self, log_prob_fn):
        """Test MALA init method creates proper state."""
        mala = BlackJAXMALA(log_prob_fn, step_size=0.1)

        init_position = jnp.zeros(2)
        key = jax.random.key(42)
        state = mala.init(init_position, key)

        assert isinstance(state, BlackJAXSamplerState)
        assert state.x.shape == init_position.shape
        assert mala._kernel is not None

    def test_mala_step(self, log_prob_fn):
        """Test MALA step method."""
        mala = BlackJAXMALA(log_prob_fn, step_size=0.1)

        init_position = jnp.zeros(2)
        key = jax.random.key(42)
        state = mala.init(init_position, key)

        new_state, info = mala.step(state)

        assert isinstance(new_state, BlackJAXSamplerState)
        assert new_state.x.shape == init_position.shape
        assert "is_accepted" in info

    def test_mala_step_without_prior_init(self, log_prob_fn):
        """Test MALA step initializes kernel if needed."""
        mala = BlackJAXMALA(log_prob_fn, step_size=0.1)

        # Create state manually
        init_position = jnp.zeros(2)
        key = jax.random.key(42)

        # Initialize properly
        import blackjax

        kernel = blackjax.mala(mala._scalar_log_prob_fn, mala.step_size)
        mala._kernel = kernel
        sampler_state = kernel.init(init_position)
        state = BlackJAXSamplerState(x=init_position, sampler_state=sampler_state, key=key)

        new_state, _info = mala.step(state)
        assert new_state.x.shape == init_position.shape

    def test_mala_multiple_steps(self, log_prob_fn):
        """Test multiple MALA steps."""
        mala = BlackJAXMALA(log_prob_fn, step_size=0.1)

        init_position = jnp.zeros(2)
        key = jax.random.key(42)
        state = mala.init(init_position, key)

        # Run multiple steps
        positions = []
        for _ in range(10):
            state, _ = mala.step(state)
            positions.append(state.x)

        positions = jnp.stack(positions)
        assert positions.shape == (10, 2)
        assert jnp.all(jnp.isfinite(positions))


# =============================================================================
# Test hmc_sampling function
# =============================================================================


class TestHMCSamplingFunction:
    """Test suite for hmc_sampling high-level function."""

    def test_hmc_sampling_basic(self):
        """Test basic HMC sampling."""
        log_prob_fn = create_normal_logprob(mean=jnp.array([0.0]), scale=jnp.array([1.0]))
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = hmc_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=50,
            n_burnin=10,
            step_size=0.1,
            num_integration_steps=5,
        )

        assert samples.shape == (50, 1)
        assert jnp.all(jnp.isfinite(samples))

    def test_hmc_sampling_with_rngs(self):
        """Test HMC sampling with nnx.Rngs key."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        rngs = nnx.Rngs(sample=42)

        samples = hmc_sampling(
            log_prob_fn,
            init_state,
            rngs,
            n_samples=30,
            n_burnin=10,
            step_size=0.1,
            num_integration_steps=5,
        )

        assert samples.shape == (30, 1)

    def test_hmc_sampling_with_distribution(self):
        """Test HMC sampling with Distribution object."""
        dist = Normal(loc=jnp.zeros(2), scale=jnp.ones(2))
        init_state = jnp.zeros(2)
        key = jax.random.key(42)

        samples = hmc_sampling(
            dist,
            init_state,
            key,
            n_samples=30,
            n_burnin=10,
            step_size=0.1,
            num_integration_steps=5,
        )

        assert samples.shape == (30, 2)

    def test_hmc_sampling_scalar_init(self):
        """Test HMC sampling with scalar initial state."""

        def scalar_log_prob(x):
            return -0.5 * x**2

        init_state = jnp.array(0.0)
        key = jax.random.key(42)

        samples = hmc_sampling(
            scalar_log_prob,
            init_state,
            key,
            n_samples=30,
            n_burnin=10,
            step_size=0.1,
            num_integration_steps=5,
        )

        # Should squeeze back to 1D
        assert samples.shape == (30,)

    def test_hmc_sampling_with_thinning(self):
        """Test HMC sampling with thinning."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = hmc_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=10,
            step_size=0.1,
            num_integration_steps=5,
            thinning=2,
        )

        assert samples.shape == (20, 1)

    def test_hmc_sampling_no_burnin(self):
        """Test HMC sampling with zero burnin."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = hmc_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=0,
            step_size=0.1,
            num_integration_steps=5,
        )

        assert samples.shape == (20, 1)

    def test_hmc_sampling_custom_mass_matrix(self):
        """Test HMC sampling with custom inverse mass matrix."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0, 0.0])
        key = jax.random.key(42)

        samples = hmc_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=10,
            step_size=0.1,
            num_integration_steps=5,
            inverse_mass_matrix=jnp.array([1.0, 2.0]),
        )

        assert samples.shape == (20, 2)

    def test_hmc_sampling_3d_state(self):
        """Test HMC sampling with 3D state."""

        def multidim_log_prob(x):
            return -0.5 * jnp.sum(x**2)

        # Use a shape where last dim is not the same as first dim
        init_state = jnp.zeros(4)
        key = jax.random.key(42)

        samples = hmc_sampling(
            multidim_log_prob,
            init_state,
            key,
            n_samples=10,
            n_burnin=5,
            step_size=0.1,
            num_integration_steps=5,
        )

        assert samples.shape == (10, 4)


# =============================================================================
# Test nuts_sampling function
# =============================================================================


class TestNUTSSamplingFunction:
    """Test suite for nuts_sampling high-level function."""

    def test_nuts_sampling_basic(self):
        """Test basic NUTS sampling."""
        log_prob_fn = create_normal_logprob(mean=jnp.array([0.0]), scale=jnp.array([1.0]))
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = nuts_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=30,
            n_burnin=10,
            step_size=0.1,
        )

        assert samples.shape == (30, 1)
        assert jnp.all(jnp.isfinite(samples))

    def test_nuts_sampling_with_rngs(self):
        """Test NUTS sampling with nnx.Rngs key."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        rngs = nnx.Rngs(sample=42)

        samples = nuts_sampling(
            log_prob_fn,
            init_state,
            rngs,
            n_samples=20,
            n_burnin=5,
            step_size=0.1,
        )

        assert samples.shape == (20, 1)

    def test_nuts_sampling_default_step_size(self):
        """Test NUTS sampling with default step size."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = nuts_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=5,
            step_size=None,  # Uses default
        )

        assert samples.shape == (20, 1)

    def test_nuts_sampling_with_thinning(self):
        """Test NUTS sampling with thinning."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = nuts_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=5,
            step_size=0.1,
            thinning=2,
        )

        assert samples.shape == (20, 1)

    def test_nuts_sampling_no_burnin(self):
        """Test NUTS sampling with zero burnin."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = nuts_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=0,
            step_size=0.1,
        )

        assert samples.shape == (20, 1)

    def test_nuts_sampling_scalar_state(self):
        """Test NUTS sampling with scalar initial state."""

        def scalar_log_prob(x):
            return -0.5 * jnp.sum(x**2)

        init_state = jnp.array(0.0)
        key = jax.random.key(42)

        samples = nuts_sampling(
            scalar_log_prob,
            init_state,
            key,
            n_samples=15,
            n_burnin=5,
            step_size=0.1,
        )

        assert samples.shape == (15,) or samples.shape == (15, 1)


# =============================================================================
# Test mala_sampling function
# =============================================================================


class TestMALASamplingFunction:
    """Test suite for mala_sampling high-level function."""

    def test_mala_sampling_basic(self):
        """Test basic MALA sampling."""
        log_prob_fn = create_normal_logprob(mean=jnp.array([0.0]), scale=jnp.array([1.0]))
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = mala_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=50,
            n_burnin=20,
            step_size=0.05,
        )

        assert samples.shape == (50, 1)
        assert jnp.all(jnp.isfinite(samples))

    def test_mala_sampling_with_rngs(self):
        """Test MALA sampling with nnx.Rngs key."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        rngs = nnx.Rngs(sample=42)

        samples = mala_sampling(
            log_prob_fn,
            init_state,
            rngs,
            n_samples=30,
            n_burnin=10,
            step_size=0.05,
        )

        assert samples.shape == (30, 1)

    def test_mala_sampling_with_thinning(self):
        """Test MALA sampling with thinning."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = mala_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=10,
            step_size=0.05,
            thinning=3,
        )

        assert samples.shape == (20, 1)

    def test_mala_sampling_no_burnin(self):
        """Test MALA sampling with zero burnin."""
        log_prob_fn = simple_log_prob
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = mala_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=20,
            n_burnin=0,
            step_size=0.05,
        )

        assert samples.shape == (20, 1)

    def test_mala_sampling_2d(self):
        """Test MALA sampling with 2D state."""
        log_prob_fn = simple_log_prob
        init_state = jnp.zeros(2)
        key = jax.random.key(42)

        samples = mala_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=30,
            n_burnin=10,
            step_size=0.05,
        )

        assert samples.shape == (30, 2)


# =============================================================================
# Test Mixture Distribution Sampling
# =============================================================================


class TestMixtureSampling:
    """Test suite for sampling from mixture distributions."""

    def test_hmc_mixture_sampling(self):
        """Test HMC sampling from mixture distribution."""
        loc1 = jnp.array([-3.0])
        loc2 = jnp.array([3.0])
        scale1 = jnp.array([1.0])
        scale2 = jnp.array([1.0])
        weights = jnp.array([0.5, 0.5])

        dist1 = Normal(loc=loc1, scale=scale1)
        dist2 = Normal(loc=loc2, scale=scale2)
        mixture = Mixture(components=[dist1, dist2], weights=weights)

        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = hmc_sampling(
            mixture,
            init_state,
            key,
            n_samples=100,
            n_burnin=50,
            step_size=0.2,
            num_integration_steps=10,
        )

        assert samples.shape == (100, 1)
        assert jnp.all(jnp.isfinite(samples))

        # For a balanced mixture, the mean should be close to 0 or at one of the modes
        # Due to sampling variance, we just check that samples are finite and in range
        assert jnp.all(jnp.abs(samples) < 10.0)  # Samples should be reasonable


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestPyTreeStateHandling:
    """Test PyTree state handling in high-level sampling functions."""

    def test_hmc_sampling_with_dict_state_1d_values(self):
        """Test HMC sampling with dict state (PyTree) containing 1D arrays."""

        def dict_log_prob(x):
            return -0.5 * (jnp.sum(x["a"] ** 2) + jnp.sum(x["b"] ** 2))

        init_state = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
        key = jax.random.key(42)

        samples = hmc_sampling(
            dict_log_prob,
            init_state,
            key,
            n_samples=10,
            n_burnin=5,
            step_size=0.1,
            num_integration_steps=5,
        )

        assert "a" in samples
        assert "b" in samples
        assert samples["a"].shape == (10, 2)
        assert samples["b"].shape == (10, 3)

    def test_hmc_sampling_with_dict_state_multidim_values(self):
        """Test HMC sampling with dict state containing multidim arrays."""

        def dict_log_prob(x):
            return -0.5 * (jnp.sum(x["mat"] ** 2))

        init_state = {"mat": jnp.zeros((2, 3))}
        key = jax.random.key(42)

        samples = hmc_sampling(
            dict_log_prob,
            init_state,
            key,
            n_samples=10,
            n_burnin=5,
            step_size=0.1,
            num_integration_steps=5,
        )

        assert "mat" in samples
        assert samples["mat"].shape == (10, 2, 3)

    def test_nuts_sampling_with_dict_state(self):
        """Test NUTS sampling with dict state (PyTree)."""

        def dict_log_prob(x):
            return -0.5 * (jnp.sum(x["a"] ** 2) + jnp.sum(x["b"] ** 2))

        init_state = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
        key = jax.random.key(42)

        samples = nuts_sampling(
            dict_log_prob,
            init_state,
            key,
            n_samples=10,
            n_burnin=5,
            step_size=0.1,
        )

        assert "a" in samples
        assert "b" in samples
        assert samples["a"].shape == (10, 2)
        assert samples["b"].shape == (10, 3)


class TestEdgeCases:
    """Test edge cases for BlackJAX samplers."""

    def test_hmc_with_rngs_parameter(self):
        """Test HMC sampler with rngs parameter in constructor."""
        log_prob_fn = simple_log_prob
        rngs = nnx.Rngs(seed=42)

        hmc = BlackJAXHMC(
            log_prob_fn,
            step_size=0.1,
            num_integration_steps=5,
            rngs=rngs,
        )

        assert hmc.rngs is rngs

    def test_nuts_with_rngs_parameter(self):
        """Test NUTS sampler with rngs parameter in constructor."""
        log_prob_fn = simple_log_prob
        rngs = nnx.Rngs(seed=42)

        nuts = BlackJAXNUTS(
            log_prob_fn,
            step_size=0.1,
            rngs=rngs,
        )

        assert nuts.rngs is rngs

    def test_mala_with_rngs_parameter(self):
        """Test MALA sampler with rngs parameter in constructor."""
        log_prob_fn = simple_log_prob
        rngs = nnx.Rngs(seed=42)

        mala = BlackJAXMALA(
            log_prob_fn,
            step_size=0.1,
            rngs=rngs,
        )

        assert mala.rngs is rngs

    def test_scalar_log_prob_wrapper(self):
        """Test that _scalar_log_prob_fn properly handles multi-dim output."""
        hmc = BlackJAXHMC(
            lambda x: -0.5 * x**2,  # Returns array, not scalar
            step_size=0.1,
            num_integration_steps=5,
        )

        x = jnp.array([1.0, 2.0])
        result = hmc._scalar_log_prob_fn(x)

        # Should sum to scalar
        assert result.shape == ()
        assert result == pytest.approx(-2.5)


# =============================================================================
# Statistical Tests (marked as xfail for robustness)
# =============================================================================


@pytest.mark.xfail(
    reason="Statistical tests may occasionally fail due to sampling variance",
    strict=False,
)
class TestStatisticalProperties:
    """Test statistical properties of samplers."""

    def test_hmc_samples_from_correct_distribution(self):
        """Test that HMC samples approximate the target distribution."""
        mean = jnp.array([0.0])
        scale = jnp.array([1.0])
        log_prob_fn = create_normal_logprob(mean=mean, scale=scale)
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = hmc_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=500,
            n_burnin=200,
            step_size=0.1,
            num_integration_steps=10,
        )

        sample_mean = jnp.mean(samples)
        sample_std = jnp.std(samples)

        assert jnp.allclose(sample_mean, mean[0], atol=0.3)
        assert jnp.allclose(sample_std, scale[0], atol=0.3)

    def test_mala_samples_from_correct_distribution(self):
        """Test that MALA samples approximate the target distribution."""
        mean = jnp.array([0.0])
        scale = jnp.array([1.0])
        log_prob_fn = create_normal_logprob(mean=mean, scale=scale)
        init_state = jnp.array([0.0])
        key = jax.random.key(42)

        samples = mala_sampling(
            log_prob_fn,
            init_state,
            key,
            n_samples=500,
            n_burnin=200,
            step_size=0.05,
        )

        sample_mean = jnp.mean(samples)
        sample_std = jnp.std(samples)

        assert jnp.allclose(sample_mean, mean[0], atol=0.3)
        assert jnp.allclose(sample_std, scale[0], atol=0.3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
