"""MCMC sampling utilities.

This module provides utilities for Markov Chain Monte Carlo (MCMC) sampling.
Specifically, we implement a basic Metropolis-Hastings algorithm for sampling
from distributions defined by their (unnormalized) log probability functions.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.interfaces import Distribution


def metropolis_step(
    key: jax.Array,
    state: jax.Array,
    log_prob_fn: Callable[[jax.Array], jax.Array],
    step_size: float = 0.1,
) -> tuple[jax.Array, jax.Array]:
    """Perform a single Metropolis-Hastings MCMC step.

    Args:
        key: JAX random key.
        state: Current state of the chain.
        log_prob_fn: Function that computes the log probability of a state.
        step_size: Size of the random proposal step.

    Returns:
        Tuple of (new_state, acceptance_probability).
    """
    # Split key for proposal and acceptance decision
    key_proposal, key_accept = jax.random.split(key)

    # Propose a new state (random walk proposal)
    proposal = state + step_size * jax.random.normal(key_proposal, state.shape)

    # Compute log probabilities
    log_prob_current = log_prob_fn(state)
    log_prob_proposal = log_prob_fn(proposal)

    # Compute acceptance probability
    log_accept_prob = log_prob_proposal - log_prob_current
    accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_prob))

    # Accept or reject
    u = jax.random.uniform(key_accept)
    new_state = jnp.where(u < accept_prob, proposal, state)

    return new_state, accept_prob


def mcmc_sampling(
    log_prob_fn: Callable[[jax.Array], jax.Array] | Distribution,
    init_state: jax.Array,
    key: jax.Array | nnx.Rngs,
    n_samples: int,
    n_burnin: int = 100,
    step_size: float = 0.1,
    thinning: int = 1,
) -> jax.Array:
    """Sample from a distribution using MCMC.

    Args:
        log_prob_fn: Function that computes the log probability of a state,
            or a Distribution object from which to sample.
        init_state: Initial state of the chain.
        key: JAX random key or nnx.Rngs object.
        n_samples: Number of samples to draw.
        n_burnin: Number of burn-in steps to discard.
        step_size: Size of the random proposal step.
        thinning: Thinning factor (keep every `thinning` samples).

    Returns:
        Array of samples with shape [n_samples, ...] where ... is init_state shape.
    """
    # Handle nnx.Rngs if provided
    actual_key: jax.Array
    if isinstance(key, nnx.Rngs):
        if hasattr(key, "params"):
            actual_key = key.params()
        elif hasattr(key, "default"):
            actual_key = key.default()
        else:
            # Use the first available key
            for k in dir(key):
                if not k.startswith("_") and callable(getattr(key, k)):
                    # Get key from first available callable attribute
                    actual_key = getattr(key, k)()
                    break
            else:
                raise ValueError("Could not extract JAX key from nnx.Rngs")
    else:
        actual_key = key

    # If log_prob_fn is a Distribution, extract its log_prob method
    actual_log_prob_fn: Callable[[jax.Array], jax.Array]
    if isinstance(log_prob_fn, Distribution):
        actual_log_prob_fn = log_prob_fn.log_prob
    else:
        actual_log_prob_fn = log_prob_fn

    # Use jitted step function for speed
    step_fn = jax.jit(lambda k, s: metropolis_step(k, s, actual_log_prob_fn, step_size))

    # Total number of steps needed (including burn-in and thinning)
    n_steps = n_burnin + n_samples * thinning

    # Initialize the state with a small random offset for better convergence
    actual_key, subkey = jax.random.split(actual_key)
    state = init_state + jax.random.normal(subkey, init_state.shape) * 0.01

    # Initialize samples array
    samples = jnp.zeros((n_samples, *init_state.shape))

    # Pre-generate all keys for deterministic behavior
    keys = jax.random.split(actual_key, n_steps)

    # Loop through all steps
    for i in range(n_steps):
        # Perform a Metropolis step
        state, _ = step_fn(keys[i], state)

        # After burn-in, collect samples (with thinning)
        if i >= n_burnin and (i - n_burnin) % thinning == 0:
            # Store this sample
            idx = (i - n_burnin) // thinning
            samples = samples.at[idx].set(state)

    return samples
