"""Ancestral sampling utilities.

This module provides utilities for ancestral sampling from probabilistic
distributions.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.interfaces import Distribution


def ancestral_sampling(
    distribution: Distribution | jax.Array,
    key: jax.Array | nnx.Rngs,
    n_samples: int,
    sample_shape: tuple[int, ...] | None = None,
) -> jax.Array:
    """Sample from a distribution using ancestral sampling.

    Args:
        distribution: Distribution object from Artifex's distributions.
        key: JAX random key or nnx.Rngs object.
        n_samples: Number of samples to draw.
        sample_shape: Optional shape information for batched distributions.

    Returns:
        Samples from the distribution with shape based on n_samples.
    """
    # Handle nnx.Rngs if provided
    if isinstance(key, nnx.Rngs):
        if hasattr(key, "params"):
            key = key.params()
        elif hasattr(key, "default"):
            key = key.default()
        else:
            # Use the first available key
            for k in dir(key):
                if not k.startswith("_") and callable(getattr(key, k)):
                    key = getattr(key, k)()
                    break

    # Check if we're dealing with a distribution from our library
    if isinstance(distribution, Distribution):
        # Check if we're dealing with a batched distribution
        is_batched = False
        if hasattr(distribution, "loc") and len(distribution.loc.shape) > 1:
            is_batched = True

        # Create the sample shape tuple
        if sample_shape is None:
            sample_tuple = (n_samples,)
        else:
            sample_tuple = (n_samples, *sample_shape)

        # Use the distribution's underlying distrax distribution directly
        # Ensure we pass a valid JAX key directly
        samples = distribution._dist.sample(seed=key, sample_shape=sample_tuple)

        # Rearrange for batched distributions if needed
        if is_batched and samples.shape[0] != distribution.loc.shape[0]:
            samples = jnp.swapaxes(samples, 0, 1)
    else:
        # Handle raw distrax distributions or custom objects
        if hasattr(distribution, "sample"):
            # Here we'll use sample_shape=(n_samples,) consistently
            # We can't check if distribution accepts seed vs key,
            # so we'll pass it using the standard JAX key name
            sample_shape = (n_samples,)
            samples = distribution.sample(seed=key, sample_shape=sample_shape)
        else:
            msg = "Input must be a Distribution instance or have a "
            msg += "sample method."
            raise ValueError(msg)

    return samples
