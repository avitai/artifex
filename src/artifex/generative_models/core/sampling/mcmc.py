"""MCMC sampling utilities.

This module provides utilities for Markov Chain Monte Carlo (MCMC) sampling.
Specifically, we implement a basic Metropolis-Hastings algorithm for sampling
from distributions defined by their (unnormalized) log probability functions.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.distributions.base import Distribution
from artifex.generative_models.core.rng import extract_rng_key


def _require_scalar_joint_log_prob(value: jax.Array) -> jax.Array:
    value = jnp.asarray(value)
    if value.ndim != 0:
        raise ValueError("log_prob_fn must return a scalar joint log probability")
    return value


def metropolis_step(
    key: jax.Array,
    state: jax.Array,
    log_prob_fn: Callable[[jax.Array], jax.Array],
    step_size: float = 0.1,
) -> tuple[jax.Array, jax.Array]:
    """Perform one Metropolis-Hastings transition."""
    key_proposal, key_accept = jax.random.split(key)
    proposal = state + step_size * jax.random.normal(key_proposal, state.shape)

    log_prob_current = _require_scalar_joint_log_prob(log_prob_fn(state))
    log_prob_proposal = _require_scalar_joint_log_prob(log_prob_fn(proposal))

    log_accept_prob = log_prob_proposal - log_prob_current
    accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_prob))

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
    """Run MCMC sampling for a fixed number of steps."""
    actual_key = extract_rng_key(key, streams=("sample", "default"), context="MCMC sampling")

    if isinstance(log_prob_fn, Distribution):
        actual_log_prob_fn = lambda x: jnp.sum(log_prob_fn.log_prob(x))
    else:
        actual_log_prob_fn = log_prob_fn

    step_fn = jax.jit(lambda k, s: metropolis_step(k, s, actual_log_prob_fn, step_size))

    n_steps = n_burnin + n_samples * thinning
    actual_key, subkey = jax.random.split(actual_key)
    state = init_state + jax.random.normal(subkey, init_state.shape) * 0.01
    samples = jnp.zeros((n_samples, *init_state.shape))
    keys = jax.random.split(actual_key, n_steps)

    for i in range(n_steps):
        state, _ = step_fn(keys[i], state)
        if i >= n_burnin and (i - n_burnin) % thinning == 0:
            idx = (i - n_burnin) // thinning
            samples = samples.at[idx].set(state)

    return samples
