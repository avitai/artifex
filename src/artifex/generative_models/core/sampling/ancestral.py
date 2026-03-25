"""Ancestral sampling utilities.

This module provides utilities for ancestral sampling from probabilistic
distributions.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.distributions.base import Distribution
from artifex.generative_models.core.rng import extract_rng_key


def ancestral_sampling(
    distribution: Distribution | jax.Array,
    key: jax.Array | nnx.Rngs,
    n_samples: int,
    sample_shape: tuple[int, ...] | None = None,
) -> jax.Array:
    """Sample from a distribution using ancestral sampling."""
    key = extract_rng_key(key, streams=("sample", "default"), context="ancestral sampling")

    if isinstance(distribution, Distribution):
        is_batched = False
        if hasattr(distribution, "loc") and len(distribution.loc.shape) > 1:
            is_batched = True

        sample_tuple = (n_samples,) if sample_shape is None else (n_samples, *sample_shape)
        samples = distribution.sample(
            sample_shape=sample_tuple,
            rngs=nnx.Rngs(sample=key, default=key),
        )

        if is_batched and samples.shape[0] != distribution.loc.shape[0]:
            samples = jnp.swapaxes(samples, 0, 1)
    else:
        if hasattr(distribution, "sample"):
            samples = distribution.sample(seed=key, sample_shape=(n_samples,))
        else:
            raise ValueError("Input must be a Distribution instance or have a sample method.")

    return samples
