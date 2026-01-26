"""Utility functions for RL training algorithms.

This module provides core functions that are shared across different RL trainers,
following the DRY principle to avoid code duplication.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_discounted_returns(
    rewards: jax.Array,
    gamma: float,
) -> jax.Array:
    """Compute discounted returns from rewards.

    For a sequence of rewards [r_0, r_1, ..., r_T], computes:
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Args:
        rewards: Array of rewards with shape (T,).
        gamma: Discount factor in [0, 1].

    Returns:
        Array of discounted returns with shape (T,).
    """

    def _scan_fn(carry: jax.Array, reward: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Scan function for backward pass."""
        future_return = carry
        current_return = reward + gamma * future_return
        return current_return, current_return

    # Scan backwards through rewards
    _, returns = jax.lax.scan(
        _scan_fn,
        jnp.array(0.0),
        rewards,
        reverse=True,
    )

    return returns


def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> jax.Array:
    """Compute Generalized Advantage Estimation (GAE).

    GAE provides a trade-off between bias and variance in advantage estimation.
    A_t = sum_{l=0}^{infty} (gamma * lambda)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Args:
        rewards: Array of rewards with shape (T,).
        values: Array of state values with shape (T+1,), including next value.
        dones: Array of done flags with shape (T,). True if episode ended.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter for bias-variance trade-off.

    Returns:
        Array of advantages with shape (T,).
    """
    # Convert dones to float mask (0 if done, 1 if not done)
    not_done_mask = 1.0 - dones.astype(jnp.float32)

    # Compute TD residuals (deltas)
    # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    deltas = rewards + gamma * values[1:] * not_done_mask - values[:-1]

    def _scan_fn(
        carry: jax.Array, inputs: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Scan function for GAE computation."""
        gae = carry
        delta, not_done = inputs
        gae = delta + gamma * gae_lambda * not_done * gae
        return gae, gae

    # Scan backwards through deltas
    _, advantages = jax.lax.scan(
        _scan_fn,
        jnp.array(0.0),
        (deltas, not_done_mask),
        reverse=True,
    )

    return advantages


def normalize_advantages(
    advantages: jax.Array,
    eps: float = 1e-8,
) -> jax.Array:
    """Normalize advantages to zero mean and unit variance.

    This is a variance reduction technique that stabilizes training.

    Args:
        advantages: Array of advantages to normalize.
        eps: Small constant for numerical stability.

    Returns:
        Normalized advantages.
    """
    mean = jnp.mean(advantages)
    std = jnp.std(advantages)
    return (advantages - mean) / (std + eps)


def compute_policy_entropy(log_probs: jax.Array) -> jax.Array:
    """Compute entropy of policy distribution.

    Entropy H = -sum(p * log(p)) = -sum(exp(log_p) * log_p)

    Args:
        log_probs: Log probabilities with shape (..., num_actions).
            Should be normalized (sum to 1 in probability space).

    Returns:
        Scalar entropy value (mean over batch if applicable).
    """
    # Convert log probs to probs
    probs = jnp.exp(log_probs)

    # Compute entropy: -sum(p * log(p))
    entropy = -jnp.sum(probs * log_probs, axis=-1)

    # Return mean entropy over batch
    return jnp.mean(entropy)


def compute_kl_divergence(
    policy_log_probs: jax.Array,
    ref_log_probs: jax.Array,
) -> jax.Array:
    """Compute KL divergence between policy and reference distributions.

    KL(p || q) = sum(p * (log(p) - log(q)))

    Args:
        policy_log_probs: Log probabilities from current policy.
        ref_log_probs: Log probabilities from reference policy.

    Returns:
        Mean KL divergence.
    """
    # KL = sum(exp(log_p) * (log_p - log_q))
    probs = jnp.exp(policy_log_probs)
    kl = jnp.sum(probs * (policy_log_probs - ref_log_probs), axis=-1)
    return jnp.mean(kl)


def compute_clipped_surrogate_loss(
    log_probs: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    clip_param: float,
) -> jax.Array:
    """Compute PPO clipped surrogate loss.

    L^CLIP = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]

    Args:
        log_probs: Log probabilities from current policy.
        old_log_probs: Log probabilities from old policy.
        advantages: Advantage estimates.
        clip_param: Clipping parameter epsilon.

    Returns:
        Clipped surrogate loss (negated for minimization).
    """
    # Compute probability ratios
    ratios = jnp.exp(log_probs - old_log_probs)

    # Unclipped objective
    surr1 = ratios * advantages

    # Clipped objective
    clipped_ratios = jnp.clip(ratios, 1.0 - clip_param, 1.0 + clip_param)
    surr2 = clipped_ratios * advantages

    # Take minimum (pessimistic bound)
    # Negate because we want to maximize the objective but minimize the loss
    return -jnp.mean(jnp.minimum(surr1, surr2))
