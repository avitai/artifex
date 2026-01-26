"""BlackJAX integration module.

This module provides integration with BlackJAX, a library of samplers
for JAX. It allows using BlackJAX's advanced MCMC samplers with our
distribution framework.
"""

from typing import Any, Callable, NamedTuple

import blackjax
import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.interfaces import Distribution
from artifex.generative_models.core.sampling.base import SamplingAlgorithm


def _extract_key(rng: jax.Array | nnx.Rngs) -> jax.Array:
    """Extract a JAX random key from either a key or an nnx.Rngs object.

    Args:
        rng: JAX random key or nnx.Rngs object.

    Returns:
        A JAX random key.
    """
    # Handle nnx.Rngs if provided
    if isinstance(rng, nnx.Rngs):
        # Check for specific keys in priority order
        if "sample" in rng:
            return rng.sample()
        if "default" in rng:
            return rng.default()

        # Fallback to first available key
        for k in rng:
            if not k.startswith("_"):
                return getattr(rng, k)()

        # If no keys available, use a default key
        return jax.random.key(0)

    # If it's already a JAX key, return as is
    return rng


def _prepare_logdensity_fn(
    log_prob_fn: Callable[[Any], float] | Distribution,
) -> Callable[[Any], float]:
    """Prepare log density function for BlackJAX.

    Args:
        log_prob_fn: Function that computes the log probability of a state,
            or a Distribution object from which to sample.

    Returns:
        A function that computes the log probability of a state.
    """
    # If log_prob_fn is a Distribution, extract its log_prob method
    if isinstance(log_prob_fn, Distribution):
        # Make sure we return a scalar by summing the log probs if needed
        def scalar_log_prob(x):
            result = log_prob_fn.log_prob(x)
            # If the result is not a scalar, sum it
            if hasattr(result, "shape") and len(result.shape) > 0:
                return jnp.sum(result)
            return result

        return scalar_log_prob

    # If it's a function, wrap it to ensure it returns a scalar
    def ensure_scalar_log_prob(x):
        result = log_prob_fn(x)
        # If the result is not a scalar, sum it
        if hasattr(result, "shape") and len(result.shape) > 0:
            return jnp.sum(result)
        return result

    return ensure_scalar_log_prob


class BlackJAXSamplerState(NamedTuple):
    """State of a BlackJAX sampler."""

    x: jax.Array  # Current position
    sampler_state: Any  # BlackJAX internal state
    key: jax.Array  # Random key


class BlackJAXHMC(SamplingAlgorithm):
    """Hamiltonian Monte Carlo sampler from BlackJAX.

    BlackJAX docs: https://blackjax-devs.github.io/blackjax/
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        step_size: float = 1e-3,
        inverse_mass_matrix: jax.Array | None = None,
        num_integration_steps: int = 10,
        *,
        rngs=None,
    ):
        """Constructor.

        Args:
          log_prob_fn: Function that returns the log probability of a position.
          step_size: Step size for the leapfrog integrator.
          inverse_mass_matrix: Inverse mass matrix for the HMC sampler.
          num_integration_steps: Number of integration steps for leapfrog.
          rngs: Random number generator keys (nnx.Rngs or jax.Array).
        """
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.inverse_mass_matrix = inverse_mass_matrix
        self.num_integration_steps = num_integration_steps
        self.rngs = rngs
        self._kernel = None

    def _scalar_log_prob_fn(self, x):
        """Return scalar log probability by summing over batch dimensions."""
        log_prob = self.log_prob_fn(x)

        # Handle multi-dimensional output by summing
        if hasattr(log_prob, "shape") and len(log_prob.shape) > 0:
            return jnp.sum(log_prob)

        return log_prob

    def init(self, x: jax.Array, key: jax.Array) -> BlackJAXSamplerState:
        """Initialize the sampler state.

        Args:
          x: Initial position.
          key: Random key.

        Returns:
          Initial state.
        """
        # Set up the inverse mass matrix if not provided
        # Note: BlackJAX requires mass matrix to be at least 1D (ndim >= 1)
        if self.inverse_mass_matrix is None:
            if hasattr(x, "shape"):
                if len(x.shape) == 0:
                    # Handle scalar case - BlackJAX requires 1D mass matrix
                    self.inverse_mass_matrix = jnp.array([1.0])
                elif len(x.shape) == 1:
                    # 1D array
                    self.inverse_mass_matrix = jnp.ones(x.shape[0])
                else:
                    # Multi-dimensional array
                    self.inverse_mass_matrix = jnp.ones(x.shape[-1])
            else:
                # Handle scalar case - BlackJAX requires 1D mass matrix
                self.inverse_mass_matrix = jnp.array([1.0])

        # Create the HMC kernel
        logprob_fn = self._scalar_log_prob_fn

        # Note: blackjax.hmc() argument order is:
        # (logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
        self._kernel = blackjax.hmc(
            logprob_fn,
            self.step_size,
            self.inverse_mass_matrix,
            self.num_integration_steps,
        )

        # Create initial state
        sampler_state = self._kernel.init(x)

        return BlackJAXSamplerState(x=x, sampler_state=sampler_state, key=key)

    def step(self, state: BlackJAXSamplerState) -> tuple[BlackJAXSamplerState, dict]:
        """Perform one sampling step.

        Args:
          state: Current state.

        Returns:
          New state and auxiliary information.
        """
        key, subkey = jax.random.split(state.key)

        # Ensure the kernel is initialized
        if self._kernel is None:
            # Create the HMC kernel (should typically be initialized in init())
            # Note: BlackJAX requires mass matrix to be at least 1D (ndim >= 1)
            if hasattr(state.x, "shape") and len(state.x.shape) > 0:
                dim = state.x.shape[-1]
                if self.inverse_mass_matrix is None:
                    self.inverse_mass_matrix = jnp.ones(dim)
            else:
                # Handle scalar case - BlackJAX requires 1D mass matrix
                if self.inverse_mass_matrix is None:
                    self.inverse_mass_matrix = jnp.array([1.0])

            logprob_fn = self._scalar_log_prob_fn

            # Note: blackjax.hmc() argument order is:
            # (logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
            self._kernel = blackjax.hmc(
                logprob_fn,
                self.step_size,
                self.inverse_mass_matrix,
                self.num_integration_steps,
            )

        # Run one step
        sampler_state, info = self._kernel.step(subkey, state.sampler_state)

        # Convert info to a dict if it's not already
        info_dict = dict(is_accepted=info.is_accepted)

        new_state = BlackJAXSamplerState(
            x=sampler_state.position, sampler_state=sampler_state, key=key
        )

        return new_state, info_dict


class BlackJAXNUTS(SamplingAlgorithm):
    """No U-Turn Sampler (NUTS) from BlackJAX.

    BlackJAX docs: https://blackjax-devs.github.io/blackjax/
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        step_size: float = 1e-3,
        inverse_mass_matrix: jax.Array | None = None,
        *,
        rngs=None,
    ):
        """Constructor.

        Args:
          log_prob_fn: Function that returns the log probability of a position.
          step_size: Step size.
          inverse_mass_matrix: Inverse mass matrix for the HMC sampler.
          rngs: Random number generator keys (nnx.Rngs or jax.Array).
        """
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.inverse_mass_matrix = inverse_mass_matrix
        self.rngs = rngs
        self._kernel = None

    def _scalar_log_prob_fn(self, x):
        """Return scalar log probability by summing over batch dimensions."""
        log_prob = self.log_prob_fn(x)

        # Handle multi-dimensional output by summing
        if hasattr(log_prob, "shape") and len(log_prob.shape) > 0:
            return jnp.sum(log_prob)

        return log_prob

    def init(self, x: jax.Array, key: jax.Array) -> BlackJAXSamplerState:
        """Initialize the sampler state.

        Args:
          x: Initial position.
          key: Random key.

        Returns:
          Initial state.
        """
        # Set up the inverse mass matrix if not provided
        # Note: BlackJAX requires mass matrix to be at least 1D (ndim >= 1)
        if self.inverse_mass_matrix is None:
            if hasattr(x, "shape"):
                if len(x.shape) == 0:
                    # Handle scalar case - BlackJAX requires 1D mass matrix
                    self.inverse_mass_matrix = jnp.array([1.0])
                elif len(x.shape) == 1:
                    # 1D array
                    self.inverse_mass_matrix = jnp.ones(x.shape[0])
                else:
                    # Multi-dimensional array
                    self.inverse_mass_matrix = jnp.ones(x.shape[-1])
            else:
                # Handle scalar case - BlackJAX requires 1D mass matrix
                self.inverse_mass_matrix = jnp.array([1.0])

        # Create the NUTS kernel
        logprob_fn = self._scalar_log_prob_fn

        self._kernel = blackjax.nuts(
            logprob_fn,
            self.step_size,
            self.inverse_mass_matrix,
        )

        # Create initial state with just the position
        sampler_state = self._kernel.init(x)

        return BlackJAXSamplerState(x=x, sampler_state=sampler_state, key=key)

    def step(self, state: BlackJAXSamplerState) -> tuple[BlackJAXSamplerState, dict]:
        """Perform one sampling step.

        Args:
          state: Current state.

        Returns:
          New state and auxiliary information.
        """
        key, subkey = jax.random.split(state.key)

        # Run one step
        sampler_state, info = self._kernel.step(subkey, state.sampler_state)

        # Convert info to a dict if it's not already
        # NUTS uses acceptance_rate instead of is_accepted
        info_dict = dict(
            acceptance_rate=info.acceptance_rate,
            num_trajectory_expansions=info.num_trajectory_expansions,
        )

        new_state = BlackJAXSamplerState(
            x=sampler_state.position, sampler_state=sampler_state, key=key
        )

        return new_state, info_dict


class BlackJAXMALA(SamplingAlgorithm):
    """Metropolis-Adjusted Langevin Algorithm sampler from BlackJAX.

    BlackJAX docs: https://blackjax-devs.github.io/blackjax/
    """

    def __init__(self, log_prob_fn: Callable, step_size: float = 1e-3, *, rngs=None):
        """Constructor.

        Args:
            log_prob_fn: Function that returns the log probability of a
                position.
            step_size: Step size for the MALA update.
            rngs: Random number generator keys (nnx.Rngs or jax.Array).
        """
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.rngs = rngs
        self._kernel = None

    def _scalar_log_prob_fn(self, x):
        """Return scalar log probability by summing over batch dimensions."""
        log_prob = self.log_prob_fn(x)

        # Handle multi-dimensional output by summing
        if hasattr(log_prob, "shape") and len(log_prob.shape) > 0:
            return jnp.sum(log_prob)

        return log_prob

    def init(self, x: jax.Array, key: jax.Array) -> BlackJAXSamplerState:
        """Initialize the sampler state.

        Args:
          x: Initial position.
          key: Random key.

        Returns:
          Initial state.
        """
        # Create the MALA kernel
        logprob_fn = self._scalar_log_prob_fn

        self._kernel = blackjax.mala(
            logprob_fn,
            self.step_size,
        )

        # Create initial state
        sampler_state = self._kernel.init(x)

        return BlackJAXSamplerState(x=x, sampler_state=sampler_state, key=key)

    def step(self, state: BlackJAXSamplerState) -> tuple[BlackJAXSamplerState, dict]:
        """Perform one sampling step.

        Args:
          state: Current state.

        Returns:
          New state and auxiliary information.
        """
        key, subkey = jax.random.split(state.key)

        # Ensure the kernel is initialized
        if self._kernel is None:
            # Create the MALA kernel (should typically be initialized
            # in init())
            logprob_fn = self._scalar_log_prob_fn
            self._kernel = blackjax.mala(
                logprob_fn,
                self.step_size,
            )

        # Run one step
        sampler_state, info = self._kernel.step(subkey, state.sampler_state)

        # Convert info to a dict if it's not already
        info_dict = dict(is_accepted=info.is_accepted)

        new_state = BlackJAXSamplerState(
            x=sampler_state.position, sampler_state=sampler_state, key=key
        )

        return new_state, info_dict


def hmc_sampling(
    log_prob_fn: Callable[[Any], float] | Distribution,
    init_state: Any,
    key: jax.Array | nnx.Rngs,
    n_samples: int,
    n_burnin: int = 100,
    step_size: float = 0.1,
    num_integration_steps: int = 10,
    inverse_mass_matrix: float | jax.Array | None = None,
    adapt_step_size: bool = False,
    thinning: int = 1,
    *,
    rngs=None,
) -> jax.Array:
    """Sample from a distribution using Hamiltonian Monte Carlo.

    Args:
        log_prob_fn: Function that computes the log probability of a state,
            or a Distribution object from which to sample.
        init_state: Initial state of the chain.
        key: JAX random key or nnx.Rngs object.
        n_samples: Number of samples to draw.
        n_burnin: Number of burn-in steps to discard.
        step_size: Step size for the leapfrog integrator.
        num_integration_steps: Number of integration steps for leapfrog.
        inverse_mass_matrix: Inverse mass matrix. If None, identity is used.
        adapt_step_size: Whether to adapt the step size.
        thinning: Thinning factor (keep every `thinning` samples).
        rngs: Optional nnx.Rngs for function-specific random operations.

    Returns:
        Array of samples with shape [n_samples, ...].
    """
    # Extract JAX key from nnx.Rngs if needed
    key = _extract_key(key)

    # Handle scalar init_state - blackjax requires at least 1D
    is_scalar = False
    if hasattr(init_state, "shape") and len(init_state.shape) == 0:
        is_scalar = True
        init_state = jnp.array([init_state.item()])
        # Wrap log_prob_fn to handle scalar input
        original_log_prob_fn = log_prob_fn
        log_prob_fn = lambda x: original_log_prob_fn(x[0] if len(x) == 1 else x)

    # Prepare log density function
    logdensity_fn = _prepare_logdensity_fn(log_prob_fn)

    # Set up mass matrix based on parameter dimension
    # Note: BlackJAX requires mass matrix to be at least 1D (ndim >= 1)
    # For PyTree states, BlackJAX uses ravel_pytree internally, so we need
    # a single flattened 1D array, not a PyTree of matrices.
    if inverse_mass_matrix is None:
        if hasattr(init_state, "shape"):
            # Array-like state
            if len(init_state.shape) == 0:
                # Handle scalar case - blackjax requires at least 1D
                inverse_mass_matrix = jnp.array([1.0])
            elif len(init_state.shape) == 1:
                # 1D array
                inverse_mass_matrix = jnp.ones(init_state.shape[0])
            else:
                # Multi-dimensional array - use total flattened size
                inverse_mass_matrix = jnp.ones(init_state.size)
        elif isinstance(init_state, dict):
            # PyTree state (with dict at the top level)
            # BlackJAX uses ravel_pytree internally, so we need a single
            # flattened 1D array with size = total elements across all leaves
            total_size = sum(v.size for v in init_state.values())
            inverse_mass_matrix = jnp.ones(total_size)
        else:
            # Scalar state - BlackJAX requires at least 1D mass matrix
            inverse_mass_matrix = jnp.array([1.0])

    # Create HMC sampler
    hmc = blackjax.hmc(
        logdensity_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        num_integration_steps=num_integration_steps,
    )

    # Initialize state
    state = hmc.init(init_state)

    # Get step function and JIT compile it for performance
    step_fn = jax.jit(hmc.step)

    # Run burn-in using jax.lax.fori_loop for JIT compatibility
    if n_burnin > 0:

        def burnin_body(_i, carry):
            state, key = carry
            key, subkey = jax.random.split(key)
            new_state, _ = step_fn(subkey, state)
            return new_state, key

        state, key = jax.lax.fori_loop(0, n_burnin, burnin_body, (state, key))

    # Sampling using jax.lax.scan for JIT compatibility and efficiency
    def sample_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        new_state, _ = step_fn(subkey, state)
        return (new_state, key), new_state.position

    # Run sampling
    n_steps = n_samples * thinning
    (state, key), all_positions = jax.lax.scan(sample_step, (state, key), jnp.arange(n_steps))

    # Apply thinning by selecting every thinning-th sample
    if thinning > 1:
        samples = all_positions[::thinning]
    else:
        samples = all_positions

    # If we converted scalar to 1D, convert back
    if is_scalar:
        samples = samples.squeeze(axis=-1)

    return samples


def nuts_sampling(
    log_prob_fn: Callable[[Any], float] | Distribution,
    init_state: Any,
    key: jax.Array | nnx.Rngs,
    n_samples: int,
    n_burnin: int = 100,
    step_size: float | None = None,
    inverse_mass_matrix: float | jax.Array | None = None,
    target_acceptance_rate: float = 0.8,
    max_num_doublings: int = 10,
    divergence_threshold: int = 1000,
    thinning: int = 1,
    *,
    rngs=None,
) -> jax.Array:
    """Sample from a distribution using the No-U-Turn Sampler (NUTS).

    Note: This function may require significant memory resources.

    Args:
        log_prob_fn: Function that computes the log probability of a state,
            or a Distribution object from which to sample.
        init_state: Initial state of the chain.
        key: JAX random key or nnx.Rngs object.
        n_samples: Number of samples to draw.
        n_burnin: Number of burn-in steps to discard.
        step_size: Initial step size. If None, it will be set to 1e-3.
        inverse_mass_matrix: Inverse mass matrix. If None, identity is used.
        target_acceptance_rate: Target acceptance rate for adaptation.
        max_num_doublings: Maximum number of trajectory doublings.
        divergence_threshold: Threshold for detecting divergences.
        thinning: Thinning factor (keep every `thinning` samples).
        rngs: Optional nnx.Rngs for function-specific random operations.

    Returns:
        Array of samples with shape [n_samples, ...].
    """
    # Extract JAX key from nnx.Rngs if needed
    key = _extract_key(key)

    # Prepare log density function
    logdensity_fn = _prepare_logdensity_fn(log_prob_fn)

    # Set default step size if None
    if step_size is None:
        step_size = 1e-3

    # Set up mass matrix based on parameter dimension
    # Note: BlackJAX requires mass matrix to be at least 1D (ndim >= 1)
    # For PyTree states, BlackJAX uses ravel_pytree internally, so we need
    # a single flattened 1D array, not a PyTree of matrices.
    if inverse_mass_matrix is None:
        if hasattr(init_state, "shape"):
            # Array-like state
            if len(init_state.shape) == 0:
                # Handle scalar case - blackjax requires at least 1D
                inverse_mass_matrix = jnp.array([1.0])
            elif len(init_state.shape) == 1:
                # 1D array
                inverse_mass_matrix = jnp.ones(init_state.shape[0])
            else:
                # Multi-dimensional array - use total flattened size
                inverse_mass_matrix = jnp.ones(init_state.size)
        elif isinstance(init_state, dict):
            # PyTree state (with dict at the top level)
            # BlackJAX uses ravel_pytree internally, so we need a single
            # flattened 1D array with size = total elements across all leaves
            total_size = sum(v.size for v in init_state.values())
            inverse_mass_matrix = jnp.ones(total_size)
        else:
            # Scalar state - BlackJAX requires at least 1D mass matrix
            inverse_mass_matrix = jnp.array([1.0])

    # Create NUTS sampler
    # Note: blackjax.nuts() only accepts logdensity_fn, step_size, and inverse_mass_matrix
    # max_num_doublings and divergence_threshold are not directly supported in the API
    nuts = blackjax.nuts(
        logdensity_fn,
        step_size,
        inverse_mass_matrix,
    )

    # Initialize state
    state = nuts.init(init_state)

    # Get step function and JIT compile it for performance
    step_fn = jax.jit(nuts.step)

    # Run burn-in using jax.lax.fori_loop for JIT compatibility
    if n_burnin > 0:

        def burnin_body(_i, carry):
            state, key = carry
            key, subkey = jax.random.split(key)
            new_state, _ = step_fn(subkey, state)
            return new_state, key

        state, key = jax.lax.fori_loop(0, n_burnin, burnin_body, (state, key))

    # Sampling using jax.lax.scan for JIT compatibility and efficiency
    def sample_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        new_state, _ = step_fn(subkey, state)
        return (new_state, key), new_state.position

    # Run sampling
    n_steps = n_samples * thinning
    (state, key), all_positions = jax.lax.scan(sample_step, (state, key), jnp.arange(n_steps))

    # Apply thinning by selecting every thinning-th sample
    if thinning > 1:
        samples = all_positions[::thinning]
    else:
        samples = all_positions

    return samples


def mala_sampling(
    log_prob_fn: Callable[[Any], float] | Distribution,
    init_state: Any,
    key: jax.Array | nnx.Rngs,
    n_samples: int,
    n_burnin: int = 100,
    step_size: float = 1e-3,
    thinning: int = 1,
    *,
    rngs=None,
) -> jax.Array:
    """Sample from a distribution using Metropolis-Adjusted Langevin Algorithm.

    Args:
        log_prob_fn: Function that computes the log probability of a state,
            or a Distribution object from which to sample.
        init_state: Initial state of the chain.
        key: JAX random key or nnx.Rngs object.
        n_samples: Number of samples to draw.
        n_burnin: Number of burn-in steps to discard.
        step_size: Step size for the gradient update.
        thinning: Thinning factor (keep every `thinning` samples).
        rngs: Optional nnx.Rngs for function-specific random operations.

    Returns:
        Array of samples with shape [n_samples, ...].
    """
    # Extract JAX key from nnx.Rngs if needed
    key = _extract_key(key)

    # Prepare log density function
    logdensity_fn = _prepare_logdensity_fn(log_prob_fn)

    # Create MALA sampler
    mala = blackjax.mala(logdensity_fn, step_size=step_size)

    # Initialize state
    state = mala.init(init_state)

    # Get step function and JIT compile it for performance
    step_fn = jax.jit(mala.step)

    # Run burn-in using jax.lax.fori_loop for JIT compatibility
    if n_burnin > 0:

        def burnin_body(_i, carry):
            state, key = carry
            key, subkey = jax.random.split(key)
            new_state, _ = step_fn(subkey, state)
            return new_state, key

        state, key = jax.lax.fori_loop(0, n_burnin, burnin_body, (state, key))

    # Sampling using jax.lax.scan for JIT compatibility and efficiency
    def sample_step(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        new_state, _ = step_fn(subkey, state)
        return (new_state, key), new_state.position

    # Run sampling
    n_steps = n_samples * thinning
    (state, key), all_positions = jax.lax.scan(sample_step, (state, key), jnp.arange(n_steps))

    # Apply thinning by selecting every thinning-th sample
    if thinning > 1:
        samples = all_positions[::thinning]
    else:
        samples = all_positions

    return samples
