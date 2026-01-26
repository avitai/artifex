"""
Adversarial loss functions module.

This module provides loss functions commonly used in Generative Adversarial
Networks (GANs) and other adversarial training approaches.
"""

import jax
import jax.numpy as jnp

from artifex.generative_models.core.losses.base import reduce_loss


def vanilla_generator_loss(
    fake_scores: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Generate loss for vanilla GAN.

    Implements the standard generator loss which aims to maximize the
    probability of generated samples being classified as real:
    loss = -log(D(G(z)))

    Args:
        fake_scores: Discriminator outputs for fake samples
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Generator loss after specified reduction

    Example:
        >>> fake_out = jnp.array([0.3, 0.7, 0.1])
        >>> vanilla_generator_loss(fake_out)
    """
    # -log(D(G(z)))
    loss = -jnp.log(jnp.clip(fake_scores, 1e-7, 1.0))
    return reduce_loss(loss, reduction, weights)


def vanilla_discriminator_loss(
    real_scores: jax.Array,
    fake_scores: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Discriminator loss for vanilla GAN.

    Implements the standard discriminator loss which aims to maximize the
    probability of correctly classifying real and fake samples:
    loss = -log(D(x)) - log(1 - D(G(z)))

    Args:
        real_scores: Discriminator outputs for real samples
        fake_scores: Discriminator outputs for fake samples
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Discriminator loss after specified reduction

    Example:
        >>> real_out = jnp.array([0.9, 0.8, 0.7])
        >>> fake_out = jnp.array([0.3, 0.2, 0.1])
        >>> vanilla_discriminator_loss(real_out, fake_out)
    """
    # -log(D(x))
    real_loss = -jnp.log(jnp.clip(real_scores, 1e-7, 1.0))

    # -log(1 - D(G(z)))
    fake_loss = -jnp.log(jnp.clip(1.0 - fake_scores, 1e-7, 1.0))

    # Combine losses
    loss = real_loss + fake_loss

    return reduce_loss(loss, reduction, weights)


def least_squares_generator_loss(
    fake_scores: jax.Array,
    target_real: float = 1.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Generate loss for Least Squares GAN (LSGAN).

    Implements the LSGAN generator loss which uses least squares instead of
    the log loss from vanilla GAN:
    loss = 0.5 * (D(G(z)) - target_real)^2

    Args:
        fake_scores: Discriminator outputs for fake samples
        target_real: Target value for fake samples (usually 1.0)
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Generator loss after specified reduction

    Example:
        >>> fake_out = jnp.array([0.3, 0.7, 0.1])
        >>> least_squares_generator_loss(fake_out)
    """
    # 0.5 * (D(G(z)) - c)^2 where c is target_real
    loss = 0.5 * jnp.square(fake_scores - target_real)
    return reduce_loss(loss, reduction, weights)


def least_squares_discriminator_loss(
    real_scores: jax.Array,
    fake_scores: jax.Array,
    target_real: float = 1.0,
    target_fake: float = 0.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Discriminator loss for Least Squares GAN (LSGAN).

    Implements the LSGAN discriminator loss:
    loss = 0.5 * (D(x) - target_real)^2 + 0.5 * (D(G(z)) - target_fake)^2

    Args:
        real_scores: Discriminator outputs for real samples
        fake_scores: Discriminator outputs for fake samples
        target_real: Target value for real samples (usually 1.0)
        target_fake: Target value for fake samples (usually 0.0)
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Discriminator loss after specified reduction

    Example:
        >>> real_out = jnp.array([0.9, 0.8, 0.7])
        >>> fake_out = jnp.array([0.3, 0.2, 0.1])
        >>> least_squares_discriminator_loss(real_out, fake_out)
    """
    # 0.5 * (D(x) - target_real)^2
    real_loss = 0.5 * jnp.square(real_scores - target_real)

    # 0.5 * (D(G(z)) - target_fake)^2
    fake_loss = 0.5 * jnp.square(fake_scores - target_fake)

    # Combine losses
    loss = real_loss + fake_loss

    return reduce_loss(loss, reduction, weights)


def wasserstein_generator_loss(
    fake_scores: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Generate loss for Wasserstein GAN (WGAN).

    Implements the WGAN generator loss:
    loss = -D(G(z))

    Args:
        fake_scores: Discriminator/critic outputs for fake samples
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Generator loss after specified reduction

    Example:
        >>> fake_out = jnp.array([-0.5, 0.2, -0.3])
        >>> wasserstein_generator_loss(fake_out)
    """
    # -D(G(z))
    loss = -fake_scores
    return reduce_loss(loss, reduction, weights)


def wasserstein_discriminator_loss(
    real_scores: jax.Array,
    fake_scores: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Discriminator/critic loss for Wasserstein GAN (WGAN).

    Implements the WGAN discriminator/critic loss:
    loss = D(G(z)) - D(x)

    Args:
        real_scores: Discriminator/critic outputs for real samples
        fake_scores: Discriminator/critic outputs for fake samples
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Discriminator/critic loss after specified reduction

    Example:
        >>> real_out = jnp.array([1.0, 0.8, 0.6])
        >>> fake_out = jnp.array([-0.5, 0.2, -0.3])
        >>> wasserstein_discriminator_loss(real_out, fake_out)
    """
    # D(G(z)) - D(x)
    loss = fake_scores - real_scores
    return reduce_loss(loss, reduction, weights)


def hinge_generator_loss(
    fake_scores: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Generate loss for Hinge GAN.

    Implements the hinge loss for the generator:
    loss = -D(G(z))

    Args:
        fake_scores: Discriminator outputs for fake samples
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Generator loss after specified reduction

    Example:
        >>> fake_out = jnp.array([-0.5, 0.2, -0.3])
        >>> hinge_generator_loss(fake_out)
    """
    # -D(G(z))
    loss = -fake_scores
    return reduce_loss(loss, reduction, weights)


def hinge_discriminator_loss(
    real_scores: jax.Array,
    fake_scores: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """
    Discriminator loss for Hinge GAN.

    Implements the hinge loss for the discriminator:
    loss = max(0, 1 - D(x)) + max(0, 1 + D(G(z)))

    Args:
        real_scores: Discriminator outputs for real samples
        fake_scores: Discriminator outputs for fake samples
        reduction: Reduction method ('none', 'mean', 'sum')
        weights: Optional weights for each sample

    Returns:
        Discriminator loss after specified reduction

    Example:
        >>> real_out = jnp.array([0.9, 0.8, 0.7])
        >>> fake_out = jnp.array([-0.5, -0.2, -0.3])
        >>> hinge_discriminator_loss(real_out, fake_out)
    """
    # max(0, 1 - D(x))
    real_loss = jnp.maximum(0.0, 1.0 - real_scores)

    # max(0, 1 + D(G(z)))
    fake_loss = jnp.maximum(0.0, 1.0 + fake_scores)

    # Combine losses
    loss = real_loss + fake_loss

    return reduce_loss(loss, reduction, weights)


# =============================================================================
# Non-Saturating (Logit-Based) Loss Functions
# =============================================================================
# These functions work directly on raw logits (before sigmoid) and use softplus
# for numerical stability. Preferred for modern GAN training.


def ns_vanilla_generator_loss(
    fake_logits: jax.Array,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """Non-saturating vanilla GAN generator loss (logit-based).

    Uses softplus for numerical stability instead of log. Works on raw
    discriminator outputs (logits) before sigmoid activation.

    The non-saturating loss provides stronger gradients when the generator
    is performing poorly, leading to faster training.

    loss = softplus(-D(G(z))) = -log(sigmoid(D(G(z))))

    Args:
        fake_logits: Raw discriminator outputs for fake samples (before sigmoid).
        reduction: Reduction method ('none', 'mean', 'sum').
        weights: Optional weights for each sample.

    Returns:
        Generator loss after specified reduction.

    Example:
        >>> fake_logits = jnp.array([-1.0, 0.5, -0.3])
        >>> ns_vanilla_generator_loss(fake_logits)

    References:
        - Non-saturating GAN: https://arxiv.org/abs/1406.2661
    """
    # softplus(-x) = -log(sigmoid(x))
    loss = jax.nn.softplus(-fake_logits)
    return reduce_loss(loss, reduction, weights)


def ns_vanilla_discriminator_loss(
    real_logits: jax.Array,
    fake_logits: jax.Array,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    weights: jax.Array | None = None,
) -> jax.Array:
    """Non-saturating vanilla GAN discriminator loss (logit-based).

    Uses softplus for numerical stability. Works on raw discriminator
    outputs (logits) before sigmoid activation.

    loss = softplus(-D(x)) + softplus(D(G(z)))

    With label smoothing, the real target is scaled:
    loss = softplus(-D(x) * (1 - smoothing)) + softplus(D(G(z)))

    Args:
        real_logits: Raw discriminator outputs for real samples.
        fake_logits: Raw discriminator outputs for fake samples.
        label_smoothing: Smooth real labels to (1 - smoothing). Default 0.0.
        reduction: Reduction method ('none', 'mean', 'sum').
        weights: Optional weights for each sample.

    Returns:
        Discriminator loss after specified reduction.

    Example:
        >>> real_logits = jnp.array([2.0, 1.5, 1.0])
        >>> fake_logits = jnp.array([-1.0, -0.5, -0.3])
        >>> ns_vanilla_discriminator_loss(real_logits, fake_logits)

    References:
        - Label smoothing: https://arxiv.org/abs/1606.03498
    """
    # Apply label smoothing
    if label_smoothing > 0:
        real_target = 1.0 - label_smoothing
    else:
        real_target = 1.0

    # softplus(-x) = -log(sigmoid(x))
    real_loss = jax.nn.softplus(-real_logits * real_target)
    fake_loss = jax.nn.softplus(fake_logits)

    loss = real_loss + fake_loss
    return reduce_loss(loss, reduction, weights)
