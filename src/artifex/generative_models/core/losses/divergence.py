"""Divergence losses module.

This module provides loss functions based on divergence measures between
probability distributions, commonly used in generative models like VAEs and GANs.
"""

import jax
import jax.numpy as jnp
from calibrax.metrics.functional.divergence import (
    js_divergence as _calibrax_js_divergence,
    kl_divergence as _calibrax_kl_divergence,
    mmd as _calibrax_mmd,
    reverse_kl_divergence as _calibrax_reverse_kl_divergence,
    wasserstein_1d as _calibrax_wasserstein_1d,
)
from distrax import Distribution as DistraxDistribution

from artifex.generative_models.core.losses.base import reduce_loss


def kl_divergence(
    predictions: jax.Array | DistraxDistribution,
    targets: jax.Array | DistraxDistribution,
    log_predictions: jax.Array | None = None,
    log_targets: jax.Array | None = None,
    eps: float = 1e-8,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Kullback-Leibler divergence between two distributions.

    KL(P||Q) = sum(P(x) * log(P(x) / Q(x)))

    Args:
        predictions: Predicted distribution P(x) or samples
        targets: Target distribution Q(x) or samples
        log_predictions: Optional pre-computed log P(x)
        log_targets: Optional pre-computed log Q(x)
        eps: Small constant for numerical stability
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        KL divergence after specified reduction

    Example:
        >>> import distrax
        >>> p = distrax.Normal(loc=0.0, scale=1.0)
        >>> q = distrax.Normal(loc=1.0, scale=2.0)
        >>> kl_divergence(p, q)
    """
    # Handle distrax Distribution objects
    if isinstance(predictions, DistraxDistribution) and isinstance(targets, DistraxDistribution):
        kl = jnp.asarray(predictions.kl_divergence(targets))
        return reduce_loss(kl, reduction, weights, axis)

    predictions_array = jnp.asarray(predictions)
    targets_array = jnp.asarray(targets)

    # Handle array inputs
    if reduction == "mean" and weights is None and axis is None:
        return _calibrax_kl_divergence(predictions_array, targets_array)

    if log_predictions is None:
        predictions_array = jnp.clip(predictions_array, eps, 1.0)
        log_predictions = jnp.log(predictions_array)

    if log_targets is None:
        targets_array = jnp.clip(targets_array, eps, 1.0)
        log_targets = jnp.log(targets_array)

    kl_terms = predictions_array * (log_predictions - log_targets)
    divergence_axis = axis if axis is not None else -1
    kl = jnp.sum(kl_terms, axis=divergence_axis)
    return reduce_loss(kl, reduction, weights, axis=None)


def reverse_kl_divergence(
    predictions: jax.Array | DistraxDistribution,
    targets: jax.Array | DistraxDistribution,
    log_predictions: jax.Array | None = None,
    log_targets: jax.Array | None = None,
    eps: float = 1e-8,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Reverse Kullback-Leibler divergence between two distributions.

    KL(Q||P) = sum(Q(x) * log(Q(x) / P(x)))

    Args:
        predictions: Predicted distribution P(x) or samples
        targets: Target distribution Q(x) or samples
        log_predictions: Optional pre-computed log P(x)
        log_targets: Optional pre-computed log Q(x)
        eps: Small constant for numerical stability
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        Reverse KL divergence after specified reduction

    Example:
        >>> import distrax
        >>> p = distrax.Normal(loc=0.0, scale=1.0)
        >>> q = distrax.Normal(loc=1.0, scale=2.0)
        >>> reverse_kl_divergence(p, q)
    """
    # Handle distrax Distribution objects
    if isinstance(predictions, DistraxDistribution) and isinstance(targets, DistraxDistribution):
        kl = jnp.asarray(targets.kl_divergence(predictions))
        return reduce_loss(kl, reduction, weights, axis)

    predictions_array = jnp.asarray(predictions)
    targets_array = jnp.asarray(targets)

    # Handle array inputs
    if reduction == "mean" and weights is None and axis is None:
        return _calibrax_reverse_kl_divergence(predictions_array, targets_array)

    if log_predictions is None:
        predictions_array = jnp.clip(predictions_array, eps, 1.0)
        log_predictions = jnp.log(predictions_array)

    if log_targets is None:
        targets_array = jnp.clip(targets_array, eps, 1.0)
        log_targets = jnp.log(targets_array)

    kl_terms = targets_array * (log_targets - log_predictions)
    divergence_axis = axis if axis is not None else -1
    kl = jnp.sum(kl_terms, axis=divergence_axis)
    return reduce_loss(kl, reduction, weights, axis=None)


def js_divergence(
    predictions: jax.Array | DistraxDistribution,
    targets: jax.Array | DistraxDistribution,
    eps: float = 1e-8,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Jensen-Shannon divergence between two distributions.

    JS(P||Q) = 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5 * (P + Q)

    Args:
        predictions: Predicted distribution P(x) or samples
        targets: Target distribution Q(x) or samples
        eps: Small constant for numerical stability
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis or axes over which to reduce

    Returns:
        JS divergence after specified reduction

    Example:
        >>> p = jnp.array([0.2, 0.5, 0.3])
        >>> q = jnp.array([0.1, 0.7, 0.2])
        >>> js_divergence(p, q)
    """
    # Handle distrax Distribution objects
    if isinstance(predictions, DistraxDistribution) and isinstance(targets, DistraxDistribution):
        raise NotImplementedError("JS divergence not implemented for Distribution objects")

    predictions_array = jnp.asarray(predictions)
    targets_array = jnp.asarray(targets)

    if reduction == "mean" and weights is None and axis is None:
        return _calibrax_js_divergence(predictions_array, targets_array)

    # Clip inputs for numerical stability
    predictions_array = jnp.clip(predictions_array, eps, 1.0)
    targets_array = jnp.clip(targets_array, eps, 1.0)

    # Compute the mixture distribution M = 0.5 * (P + Q)
    mixture = 0.5 * (predictions_array + targets_array)
    mixture = jnp.clip(mixture, eps, 1.0)

    # Compute log probabilities
    log_p = jnp.log(predictions_array)
    log_q = jnp.log(targets_array)
    log_m = jnp.log(mixture)

    # Compute KL(P||M) and KL(Q||M)
    kl_p_m = predictions_array * (log_p - log_m)
    kl_q_m = targets_array * (log_q - log_m)

    # Sum over the specified axis for KL computations
    # Default to last axis if none specified (distribution axis)
    sum_axis = axis if axis is not None else -1
    kl_p_m = jnp.sum(kl_p_m, axis=sum_axis)
    kl_q_m = jnp.sum(kl_q_m, axis=sum_axis)

    # Compute JS = 0.5 * (KL(P||M) + KL(Q||M))
    js = 0.5 * (kl_p_m + kl_q_m)

    # Apply final reduction (no axis since already reduced above)
    return reduce_loss(js, reduction, weights, axis=None)


def wasserstein_distance(
    predictions: jax.Array | DistraxDistribution,
    targets: jax.Array | DistraxDistribution,
    p: int = 1,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """Wasserstein distance between empirical distributions (1D only).

    This implements a simple approximation based on sorting samples.
    For exact solution in higher dimensions, optimal transport methods are needed.

    Args:
        predictions: Predicted distribution samples
        targets: Target distribution samples
        p: Order of the Wasserstein distance (1 or 2)
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each element
        axis: Axis along which to compute distance (default: last axis)

    Returns:
        Wasserstein distance after specified reduction

    Example:
        >>> p = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> q = jnp.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
        >>> wasserstein_distance(p, q, p=1, axis=1)
    """
    # Handle distrax Distribution objects
    if isinstance(predictions, DistraxDistribution) or isinstance(targets, DistraxDistribution):
        raise NotImplementedError("Wasserstein distance not implemented for Distribution objects")

    predictions_array = jnp.asarray(predictions)
    targets_array = jnp.asarray(targets)

    if p == 1 and reduction == "mean" and weights is None and axis is None:
        return _calibrax_wasserstein_1d(predictions_array, targets_array)

    # Default to last axis if none specified
    sort_axis = axis if isinstance(axis, int) else (-1 if axis is None else axis[-1])

    # Sort both distributions along specified axis
    predictions_sorted = jnp.sort(predictions_array, axis=sort_axis)
    targets_sorted = jnp.sort(targets_array, axis=sort_axis)

    # Compute Lp distance between sorted samples
    if p == 1:
        # L1 Wasserstein distance (Earth Mover's Distance)
        distance = jnp.abs(predictions_sorted - targets_sorted)
    elif p == 2:
        # L2 Wasserstein distance
        distance = jnp.square(predictions_sorted - targets_sorted)
    else:
        # Lp Wasserstein distance
        distance = jnp.power(jnp.abs(predictions_sorted - targets_sorted), p)

    # Mean over the distribution axis
    distance = jnp.mean(distance, axis=sort_axis)

    # Apply p-root for p > 1
    if p > 1:
        distance = jnp.power(distance, 1.0 / p)

    return reduce_loss(distance, reduction, weights, axis=None)


def maximum_mean_discrepancy(
    predictions: jax.Array,
    targets: jax.Array,
    kernel_type: str = "rbf",
    kernel_bandwidth: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """Maximum Mean Discrepancy (MMD) between two distributions.

    MMD measures the distance between distributions using kernel methods.
    For the RBF kernel, delegates to calibrax's unbiased MMD estimator.
    Linear and polynomial kernels use a local biased estimator.

    Args:
        predictions: Predicted samples [batch, num_samples, features]
        targets: Target samples [batch, num_samples, features]
        kernel_type: Type of kernel ('rbf', 'linear', 'polynomial')
        kernel_bandwidth: Bandwidth parameter for RBF kernel
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for batch elements

    Returns:
        MMD distance after specified reduction

    Example:
        >>> pred = jax.random.normal(jax.random.key(0), (2, 100, 5))
        >>> target = jax.random.normal(jax.random.key(1), (2, 100, 5))
        >>> maximum_mean_discrepancy(pred, target)
    """
    if kernel_type == "rbf":
        # Delegate RBF kernel to calibrax's unbiased MMD estimator
        mmd_batch = jax.vmap(
            lambda p, t: _calibrax_mmd(p, t, kernel="rbf", bandwidth=kernel_bandwidth)
        )(predictions, targets)
        return reduce_loss(mmd_batch, reduction, weights)

    # Linear and polynomial kernels: local biased estimator
    def _compute_mmd_single(pred_b: jax.Array, target_b: jax.Array) -> jax.Array:
        """Compute biased MMD for a single batch element."""
        if kernel_type == "linear":
            k_pp = jnp.matmul(pred_b, pred_b.T)
            k_tt = jnp.matmul(target_b, target_b.T)
            k_pt = jnp.matmul(pred_b, target_b.T)
        elif kernel_type == "polynomial":
            k_pp = (jnp.matmul(pred_b, pred_b.T) + 1) ** 2
            k_tt = (jnp.matmul(target_b, target_b.T) + 1) ** 2
            k_pt = (jnp.matmul(pred_b, target_b.T) + 1) ** 2
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        mmd_squared = jnp.mean(k_pp) + jnp.mean(k_tt) - 2 * jnp.mean(k_pt)
        return jnp.sqrt(jnp.maximum(mmd_squared, 0.0))

    mmd_batch = jax.vmap(_compute_mmd_single)(predictions, targets)
    return reduce_loss(mmd_batch, reduction, weights)


def energy_distance(
    predictions: jax.Array,
    targets: jax.Array,
    beta: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """Energy distance between two distributions.

    Energy distance is a metric between probability distributions that
    generalizes the concept of Euclidean distance.

    Args:
        predictions: Predicted samples [batch, num_samples, features]
        targets: Target samples [batch, num_samples, features]
        beta: Power parameter (0 < beta <= 2)
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for batch elements

    Returns:
        Energy distance after specified reduction

    Example:
        >>> pred = jax.random.normal(jax.random.key(0), (2, 100, 5))
        >>> target = jax.random.normal(jax.random.key(1), (2, 100, 5))
        >>> energy_distance(pred, target)
    """
    batch_size = predictions.shape[0]
    energy_batch = []

    for b in range(batch_size):
        pred_b = predictions[b]  # [num_samples, features]
        target_b = targets[b]  # [num_samples, features]

        # Compute pairwise distances
        def pairwise_distance(x, y):
            squared_distance = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
            return jnp.power(squared_distance + 1e-12, beta / 2.0)

        # E[||X - Y||^beta]
        d_xy = jnp.mean(pairwise_distance(pred_b, target_b))

        # E[||X - X'||^beta]
        d_xx = jnp.mean(pairwise_distance(pred_b, pred_b))

        # E[||Y - Y'||^beta]
        d_yy = jnp.mean(pairwise_distance(target_b, target_b))

        # Energy distance: 2*E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        energy = 2 * d_xy - d_xx - d_yy
        energy_batch.append(energy)

    energy_batch = jnp.stack(energy_batch)
    return reduce_loss(energy_batch, reduction, weights)


def gaussian_kl_divergence(
    mean: jax.Array,
    logvar: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
    """KL divergence from N(mean, exp(logvar)) to N(0, 1).

    Closed-form KL for VAEs:
    KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar))

    This is the standard KL divergence used in VAE training, measuring how
    much the learned posterior q(z|x) = N(mean, exp(logvar)) diverges from
    the prior p(z) = N(0, 1).

    Args:
        mean: Latent mean, shape (batch, latent_dim)
        logvar: Latent log-variance, shape (batch, latent_dim)
        reduction: Reduction method ('none', 'mean', 'sum', 'batch_sum')
            - 'none': Returns per-sample KL, shape (batch,)
            - 'mean': Mean over all samples
            - 'sum': Sum over all samples
            - 'batch_sum': Sum over features, mean over batch (standard VAE)
        weights: Optional sample weights
        axis: Axes to reduce over (default: all except batch)

    Returns:
        KL divergence loss

    Example:
        >>> mean = jnp.zeros((8, 20))  # batch_size=8, latent_dim=20
        >>> logvar = jnp.zeros((8, 20))  # var = 1 (matches prior)
        >>> kl = gaussian_kl_divergence(mean, logvar)
        >>> assert kl == 0.0  # No divergence when posterior equals prior
    """
    # KL per dimension: -0.5 * (1 + logvar - mean^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mean**2 - jnp.exp(logvar))

    # Sum over latent dimensions (all except batch)
    if axis is None:
        axis = tuple(range(1, kl_per_dim.ndim))
    kl = jnp.sum(kl_per_dim, axis=axis)

    return reduce_loss(kl, reduction, weights)
