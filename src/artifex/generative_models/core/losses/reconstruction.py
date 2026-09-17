"""Reconstruction losses module.

This module provides loss functions for direct comparison between model outputs
and target values, typically used for reconstruction tasks in autoencoders,
image-to-image translation, and other generative models. The element-wise losses
are calibrax's, taking ``reduction`` positionally as the rest of this package does;
the Charbonnier loss is ``calibrax.metrics.functional.charbonnier_loss``.
"""

import jax
import jax.numpy as jnp
from calibrax.metrics import reduce_values
from calibrax.metrics.functional.regression import (
    huber_loss as _calibrax_huber_loss,
    mae as _calibrax_mae,
    mse as _calibrax_mse,
)


def mse_loss(
    predictions: jax.Array,
    targets: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Mean Squared Error loss (L2 loss).

    Calculates the squared error between predictions and targets:
    MSE = mean((predictions - targets)**2)

    Args:
        predictions: Model output values
        targets: Ground truth values
        reduction: Reduction method ('none', 'mean', 'sum', 'batch_sum')
        weights: Optional weights for each element; a mean is the weighted mean
        axis: Axis or axes over which to reduce

    Returns:
        Loss value(s) after specified reduction

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> targ = jnp.array([0.0, 0.0, 0.0])
        >>> mse_loss(pred, targ)  # Returns 4.66...
    """
    return _calibrax_mse(predictions, targets, weights=weights, reduction=reduction, axis=axis)


def mae_loss(
    predictions: jax.Array,
    targets: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Mean Absolute Error loss (L1 loss).

    Calculates the absolute error between predictions and targets:
    MAE = mean(abs(predictions - targets))

    Args:
        predictions: Model output values
        targets: Ground truth values
        reduction: Reduction method ('none', 'mean', 'sum', 'batch_sum')
        weights: Optional weights for each element; a mean is the weighted mean
        axis: Axis or axes over which to reduce

    Returns:
        Loss value(s) after specified reduction

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> targ = jnp.array([0.0, 0.0, 0.0])
        >>> mae_loss(pred, targ)  # Returns 2.0
    """
    return _calibrax_mae(predictions, targets, weights=weights, reduction=reduction, axis=axis)


def huber_loss(
    predictions: jax.Array,
    targets: jax.Array,
    delta: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Huber loss (smooth L1 loss).

    A loss function that is less sensitive to outliers than MSE:
    - For |predictions - targets| <= delta:
      0.5 * (predictions - targets)**2
    - For |predictions - targets| > delta:
      delta * (|predictions - targets| - 0.5 * delta)

    Args:
        predictions: Model output values
        targets: Ground truth values
        delta: Threshold where the loss changes from quadratic to linear
        reduction: Reduction method ('none', 'mean', 'sum', 'batch_sum')
        weights: Optional weights for each element; a mean is the weighted mean
        axis: Axis or axes over which to reduce

    Returns:
        Loss value(s) after specified reduction

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> targ = jnp.array([0.0, 0.0, 0.0])
        >>> huber_loss(pred, targ, delta=1.5)
    """
    return _calibrax_huber_loss(
        predictions, targets, delta=delta, weights=weights, reduction=reduction, axis=axis
    )


def psnr_loss(
    predictions: jax.Array,
    targets: jax.Array,
    max_value: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Peak Signal-to-Noise Ratio (PSNR) expressed as a loss.

    PSNR is a quality metric for images, converted to a loss:
    loss = -20 * log10(max_value / sqrt(mse))

    Args:
        predictions: Model output values (image)
        targets: Ground truth values (image)
        max_value: Maximum possible pixel value (1.0 for normalized images)
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        PSNR loss value(s) after specified reduction (negative PSNR)

    Example:
        >>> pred = jnp.array([[0.5, 0.6], [0.7, 0.8]])
        >>> targ = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        >>> psnr_loss(pred, targ)
    """
    mse = _calibrax_mse(predictions, targets, axis=axis)

    # Negated PSNR, so that minimising the loss raises the ratio.
    psnr = -20 * jnp.log10(max_value / jnp.sqrt(mse + 1e-8))

    # The axis was consumed by the MSE, so the reduction is over what is left.
    return reduce_values(psnr, weights=weights, reduction=reduction)
