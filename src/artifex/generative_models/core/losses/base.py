"""Minimal shared loss utilities.

This module intentionally stays small. Family-specific losses should be built
from explicit JAX and CalibraX primitives rather than additional management
frameworks or registries.
"""

import jax
import jax.numpy as jnp


def reduce_loss(
    loss: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Apply reduction to a loss tensor with optional weighting.

    Args:
        loss: Raw loss values.
        reduction: Reduction method:
            - ``"none"``: no reduction
            - ``"mean"``: mean over all elements or the requested axis
            - ``"sum"``: sum over all elements or the requested axis
            - ``"batch_sum"``: sum over non-batch dims, mean over batch
        weights: Optional weighting factors for the loss values.
        axis: Axis or axes over which to reduce. Ignored for ``"batch_sum"``.

    Returns:
        Reduced loss value(s).

    Raises:
        ValueError: If ``reduction`` is not supported.
    """
    valid_reductions = ["none", "mean", "sum", "batch_sum"]
    if reduction not in valid_reductions:
        raise ValueError(f"Invalid reduction: {reduction}, expected one of {valid_reductions}")

    if weights is not None:
        loss = loss * weights

    if reduction == "mean":
        return jnp.mean(loss, axis=axis)
    if reduction == "sum":
        return jnp.sum(loss, axis=axis)
    if reduction == "batch_sum":
        if loss.ndim <= 1:
            return jnp.mean(loss)
        batch_size = loss.shape[0]
        spatial_sum = jnp.sum(loss.reshape(batch_size, -1), axis=-1)
        return jnp.mean(spatial_sum)
    return loss
