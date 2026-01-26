"""Reward functions for RL-based generative model training.

This module provides a protocol for reward functions and common implementations
for training generative models with reinforcement learning.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp


class RewardFunction(Protocol):
    """Protocol for reward functions used in RL training.

    Reward functions compute scalar rewards for generated samples,
    which are used to guide policy optimization.
    """

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute rewards for generated samples.

        Args:
            samples: Generated samples with shape (batch_size, ...).
            conditions: Optional conditioning inputs.
            **kwargs: Additional arguments for specific reward functions.

        Returns:
            Rewards with shape (batch_size,).
        """
        ...


class ConstantReward:
    """Constant reward function for testing.

    Returns a fixed reward value for all samples.
    """

    def __init__(self, value: float = 1.0) -> None:
        """Initialize constant reward.

        Args:
            value: The constant reward value to return.
        """
        self.value = value

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> jax.Array:
        """Return constant rewards.

        Args:
            samples: Generated samples.
            conditions: Ignored.
            **kwargs: Ignored.

        Returns:
            Array of constant rewards.
        """
        del conditions, kwargs  # Unused
        batch_size = samples.shape[0]
        return jnp.full((batch_size,), self.value)


class CompositeReward:
    """Composite reward that combines multiple reward functions.

    Computes weighted sum of multiple reward functions.
    """

    def __init__(
        self,
        reward_fns: list[RewardFunction],
        weights: list[float] | None = None,
    ) -> None:
        """Initialize composite reward.

        Args:
            reward_fns: List of reward functions to combine.
            weights: Optional weights for each function. Defaults to uniform.
        """
        self.reward_fns = reward_fns
        if weights is None:
            weights = [1.0 / len(reward_fns)] * len(reward_fns)
        self.weights = weights

        if len(self.weights) != len(self.reward_fns):
            msg = "Number of weights must match number of reward functions"
            raise ValueError(msg)

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute weighted sum of rewards.

        Args:
            samples: Generated samples.
            conditions: Optional conditioning inputs.
            **kwargs: Additional arguments passed to all functions.

        Returns:
            Combined rewards.
        """
        total_reward = jnp.zeros(samples.shape[0])

        for fn, weight in zip(self.reward_fns, self.weights):
            reward = fn(samples, conditions, **kwargs)
            total_reward = total_reward + weight * reward

        return total_reward


class ThresholdReward:
    """Reward based on thresholding a metric.

    Returns 1.0 if metric exceeds threshold, 0.0 otherwise.
    """

    def __init__(
        self,
        metric_fn: RewardFunction,
        threshold: float,
        above: bool = True,
    ) -> None:
        """Initialize threshold reward.

        Args:
            metric_fn: Function that computes the metric to threshold.
            threshold: Threshold value.
            above: If True, reward when metric > threshold. If False,
                reward when metric < threshold.
        """
        self.metric_fn = metric_fn
        self.threshold = threshold
        self.above = above

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute threshold-based rewards.

        Args:
            samples: Generated samples.
            conditions: Optional conditioning inputs.
            **kwargs: Additional arguments passed to metric function.

        Returns:
            Binary rewards based on threshold.
        """
        metrics = self.metric_fn(samples, conditions, **kwargs)

        if self.above:
            return (metrics > self.threshold).astype(jnp.float32)
        else:
            return (metrics < self.threshold).astype(jnp.float32)


class ScaledReward:
    """Reward function with scaling and offset.

    Applies linear transformation: scaled_reward = scale * reward + offset
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> None:
        """Initialize scaled reward.

        Args:
            reward_fn: Base reward function.
            scale: Multiplicative scale factor.
            offset: Additive offset.
        """
        self.reward_fn = reward_fn
        self.scale = scale
        self.offset = offset

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute scaled rewards.

        Args:
            samples: Generated samples.
            conditions: Optional conditioning inputs.
            **kwargs: Additional arguments passed to base function.

        Returns:
            Scaled rewards.
        """
        rewards = self.reward_fn(samples, conditions, **kwargs)
        return self.scale * rewards + self.offset


class ClippedReward:
    """Reward function with clipping.

    Clips rewards to [min_value, max_value] range.
    """

    def __init__(
        self,
        reward_fn: RewardFunction,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ) -> None:
        """Initialize clipped reward.

        Args:
            reward_fn: Base reward function.
            min_value: Minimum reward value.
            max_value: Maximum reward value.
        """
        self.reward_fn = reward_fn
        self.min_value = min_value
        self.max_value = max_value

    def __call__(
        self,
        samples: jax.Array,
        conditions: jax.Array | None = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Compute clipped rewards.

        Args:
            samples: Generated samples.
            conditions: Optional conditioning inputs.
            **kwargs: Additional arguments passed to base function.

        Returns:
            Clipped rewards.
        """
        rewards = self.reward_fn(samples, conditions, **kwargs)
        return jnp.clip(rewards, self.min_value, self.max_value)
