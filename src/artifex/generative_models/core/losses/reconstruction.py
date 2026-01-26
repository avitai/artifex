"""
Reconstruction losses module.

This module provides loss functions for direct comparison between model outputs
and target values, typically used for reconstruction tasks in autoencoders,
image-to-image translation, and other generative models.
"""

import jax
import jax.numpy as jnp

from artifex.generative_models.core.losses.base import reduce_loss


def mse_loss(
    predictions: jax.Array,
    targets: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Mean Squared Error loss (L2 loss).

    Calculates the squared error between predictions and targets:
    MSE = mean((predictions - targets)**2)

    Args:
        predictions: Model output values
        targets: Ground truth values
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        Loss value(s) after specified reduction

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> targ = jnp.array([0.0, 0.0, 0.0])
        >>> mse_loss(pred, targ)  # Returns 4.66...
    """
    squared_error = jnp.square(predictions - targets)
    return reduce_loss(squared_error, reduction, weights, axis)


def mae_loss(
    predictions: jax.Array,
    targets: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Mean Absolute Error loss (L1 loss).

    Calculates the absolute error between predictions and targets:
    MAE = mean(abs(predictions - targets))

    Args:
        predictions: Model output values
        targets: Ground truth values
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        Loss value(s) after specified reduction

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> targ = jnp.array([0.0, 0.0, 0.0])
        >>> mae_loss(pred, targ)  # Returns 2.0
    """
    absolute_error = jnp.abs(predictions - targets)
    return reduce_loss(absolute_error, reduction, weights, axis)


def huber_loss(
    predictions: jax.Array,
    targets: jax.Array,
    delta: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Huber loss (smooth L1 loss).

    A loss function that is less sensitive to outliers than MSE:
    - For |predictions - targets| <= delta:
      0.5 * (predictions - targets)**2
    - For |predictions - targets| > delta:
      delta * (|predictions - targets| - 0.5 * delta)

    Args:
        predictions: Model output values
        targets: Ground truth values
        delta: Threshold where the loss changes from quadratic to linear
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        Loss value(s) after specified reduction

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> targ = jnp.array([0.0, 0.0, 0.0])
        >>> huber_loss(pred, targ, delta=1.5)
    """
    error = predictions - targets
    abs_error = jnp.abs(error)

    # Quadratic region: 0.5 * error^2
    quadratic = 0.5 * jnp.square(error)

    # Linear region: delta * (|error| - 0.5 * delta)
    linear = delta * (abs_error - 0.5 * delta)

    # Combine based on whether |error| <= delta
    loss = jnp.where(abs_error <= delta, quadratic, linear)

    return reduce_loss(loss, reduction, weights, axis)


def charbonnier_loss(
    predictions: jax.Array,
    targets: jax.Array,
    epsilon: float = 1e-3,
    alpha: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Charbonnier loss (generalized robust L1 loss).

    A differentiable variant of L1 loss, defined as:
    sqrt((predictions - targets)**2 + epsilon**2)**alpha

    Args:
        predictions: Model output values
        targets: Ground truth values
        epsilon: Small constant for numerical stability
        alpha: Exponent (usually 1.0)
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        Loss value(s) after specified reduction

    Example:
        >>> pred = jnp.array([1.0, 2.0, 3.0])
        >>> targ = jnp.array([0.0, 0.0, 0.0])
        >>> charbonnier_loss(pred, targ)
    """
    error = predictions - targets
    loss = jnp.power(jnp.sqrt(jnp.square(error) + epsilon**2), alpha)

    return reduce_loss(loss, reduction, weights, axis)


def psnr_loss(
    predictions: jax.Array,
    targets: jax.Array,
    max_value: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """
    Peak Signal-to-Noise Ratio (PSNR) expressed as a loss.

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
    # Calculate MSE
    mse = jnp.mean(jnp.square(predictions - targets), axis=axis)

    # Calculate PSNR and negate it (since we want to minimize)
    psnr = -20 * jnp.log10(max_value / jnp.sqrt(mse + 1e-8))

    return reduce_loss(psnr, reduction, weights, None)  # Axis already applied in MSE calculation
