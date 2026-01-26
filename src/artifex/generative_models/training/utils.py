"""Shared utilities for training modules.

This module provides common utilities used across different trainers to
eliminate code duplication and ensure consistent behavior.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def extract_batch_data(
    batch: dict[str, Any],
    keys: tuple[str, ...] = ("image", "data"),
) -> jax.Array:
    """Extract data from batch dictionary using fallback keys.

    Searches for data in the batch dictionary using the provided keys
    in order, returning the first found value.

    Args:
        batch: Batch dictionary containing training data.
        keys: Tuple of keys to try in order. Defaults to ("image", "data").

    Returns:
        The data array from the batch.

    Raises:
        KeyError: If none of the keys are found in the batch.

    Example:
        >>> batch = {"image": jnp.ones((32, 28, 28, 1))}
        >>> data = extract_batch_data(batch)
        >>> data.shape
        (32, 28, 28, 1)
    """
    for key in keys:
        if key in batch:
            return batch[key]

    msg = f"Batch must contain one of {keys}"
    raise KeyError(msg)


def expand_dims_to_match(
    arr: jax.Array,
    target_ndim: int,
) -> jax.Array:
    """Expand array dimensions to match target number of dimensions.

    Adds trailing dimensions of size 1 until the array has the target
    number of dimensions. Useful for broadcasting operations.

    Args:
        arr: Array to expand.
        target_ndim: Target number of dimensions.

    Returns:
        Array with trailing dimensions added.

    Example:
        >>> t = jnp.array([0.5, 0.3])  # shape (2,)
        >>> expanded = expand_dims_to_match(t, 4)
        >>> expanded.shape
        (2, 1, 1, 1)
    """
    while arr.ndim < target_ndim:
        arr = arr[..., None]
    return arr


def reshape_for_broadcast(
    arr: jax.Array,
    batch_size: int,
    target_ndim: int,
) -> jax.Array:
    """Reshape array to broadcast over batch and spatial dimensions.

    Creates a shape like (batch_size, 1, 1, ...) for broadcasting
    with data of the given number of dimensions.

    Args:
        arr: Array with shape (batch_size,) or (batch_size, 1).
        batch_size: Size of the batch dimension.
        target_ndim: Target number of dimensions for broadcasting.

    Returns:
        Reshaped array ready for broadcasting.

    Example:
        >>> t = jnp.array([[0.5], [0.3]])  # shape (2, 1)
        >>> reshaped = reshape_for_broadcast(t, 2, 4)
        >>> reshaped.shape
        (2, 1, 1, 1)
    """
    target_shape = (batch_size,) + (1,) * (target_ndim - 1)
    return arr.reshape(target_shape)


def sample_logit_normal(
    key: jax.Array,
    shape: tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
) -> jax.Array:
    """Sample from logit-normal distribution.

    Samples from normal distribution and applies sigmoid to get
    values concentrated around the center of [0, 1].

    Args:
        key: PRNG key for random sampling.
        shape: Output shape.
        loc: Location parameter (shifts the distribution).
        scale: Scale parameter (controls spread).

    Returns:
        Samples in (0, 1) interval.

    Example:
        >>> key = jax.random.key(0)
        >>> samples = sample_logit_normal(key, (1000,), loc=-0.5, scale=1.0)
        >>> # Samples will favor middle values
    """
    u = jax.random.normal(key, shape)
    u = u * scale + loc
    return jax.nn.sigmoid(u)


def sample_u_shaped(
    key: jax.Array,
    shape: tuple[int, ...],
) -> jax.Array:
    """Sample from U-shaped distribution in [0, 1].

    Samples favor endpoints (0 and 1), useful for rectified flow
    training where endpoint behavior is critical.

    Uses approximation via sin transform of uniform samples.

    Args:
        key: PRNG key for random sampling.
        shape: Output shape.

    Returns:
        Samples in [0, 1] interval favoring endpoints.

    Example:
        >>> key = jax.random.key(0)
        >>> samples = sample_u_shaped(key, (1000,))
        >>> # More samples near 0 and 1 than 0.5
    """
    u = jax.random.uniform(key, shape)
    # Transform to favor endpoints: t = sin(pi * u / 2)^2
    return jnp.sin(jnp.pi * u / 2) ** 2


def extract_model_prediction(
    output: dict[str, Any] | jax.Array,
    keys: tuple[str, ...] = ("predicted_noise", "prediction", "output", "noise"),
) -> jax.Array:
    """Extract prediction array from model output.

    Diffusion models return dict for extensibility. This utility extracts
    the primary prediction using priority-ordered keys.

    Args:
        output: Model output (dict or direct array).
        keys: Priority-ordered keys to check. First match is returned.

    Returns:
        The prediction array.

    Example:
        >>> output = {"predicted_noise": jnp.ones((2, 28, 28, 1))}
        >>> pred = extract_model_prediction(output)
        >>> pred.shape
        (2, 28, 28, 1)
    """
    if not isinstance(output, dict):
        return output

    for key in keys:
        if key in output:
            return output[key]

    # Fallback to first value
    return next(iter(output.values()))
