"""Proximal Policy Optimization (PPO) trainer.

PPO is a policy gradient algorithm that uses a clipped surrogate objective
to prevent large policy updates. This implementation includes:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate loss
- Value function loss
- Entropy bonus
- Gradient clipping
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.configs import PPOConfig
from artifex.generative_models.training.rl.utils import (
    compute_clipped_surrogate_loss,
    compute_gae_advantages,
    compute_policy_entropy,
    normalize_advantages,
)


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Implements PPO with:
    - Clipped surrogate objective for stable policy updates
    - GAE for advantage estimation
    - Value function fitting
    - Entropy bonus for exploration
    - Gradient clipping

    The model must be an Actor-Critic that returns (action_logits, value).

    Attributes:
        model: Actor-Critic network.
        optimizer: Flax NNX optimizer.
        config: PPO configuration.
    """

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        config: PPOConfig | None = None,
    ) -> None:
        """Initialize PPO trainer.

        Args:
            model: Actor-Critic network that returns (logits, value).
            optimizer: Flax NNX optimizer for the model.
            config: PPO configuration. Uses defaults if not provided.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config if config is not None else PPOConfig()

    def compute_gae(
        self,
        rewards: jax.Array,
        values: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards with shape (T,).
            values: Values with shape (T+1,), including next state value.
            dones: Done flags with shape (T,).

        Returns:
            Advantages with shape (T,).
        """
        return compute_gae_advantages(
            rewards,
            values,
            dones,
            self.config.gamma,
            self.config.gae_lambda,
        )

    def compute_clipped_loss(
        self,
        log_probs: jax.Array,
        old_log_probs: jax.Array,
        advantages: jax.Array,
    ) -> jax.Array:
        """Compute clipped surrogate policy loss.

        Args:
            log_probs: Current policy log probabilities.
            old_log_probs: Old policy log probabilities.
            advantages: Advantage estimates.

        Returns:
            Clipped surrogate loss.
        """
        return compute_clipped_surrogate_loss(
            log_probs,
            old_log_probs,
            advantages,
            self.config.clip_param,
        )

    def compute_value_loss(
        self,
        values: jax.Array,
        returns: jax.Array,
    ) -> jax.Array:
        """Compute value function loss (MSE).

        Args:
            values: Predicted values.
            returns: Target returns.

        Returns:
            Value function loss.
        """
        return jnp.mean((values - returns) ** 2)

    def compute_entropy(self, log_probs: jax.Array) -> jax.Array:
        """Compute policy entropy.

        Args:
            log_probs: Log probabilities with shape (..., num_actions).

        Returns:
            Mean entropy.
        """
        return compute_policy_entropy(log_probs)

    def train_step(
        self,
        batch: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single PPO training step.

        Args:
            batch: Dictionary containing:
                - "states": Batch of states.
                - "actions": Actions taken.
                - "old_log_probs": Log probs from old policy.
                - "returns": Target returns.
                - "advantages": Advantage estimates.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        # Normalize advantages
        advantages = normalize_advantages(advantages)

        def loss_fn(model: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            # Forward pass
            logits, values = model(states)
            values = values.squeeze(-1)

            # Compute log probabilities for actions taken
            log_probs_all = jax.nn.log_softmax(logits, axis=-1)
            log_probs = jnp.take_along_axis(
                log_probs_all,
                actions[:, None],
                axis=-1,
            ).squeeze(-1)

            # Policy loss (clipped surrogate)
            policy_loss = compute_clipped_surrogate_loss(
                log_probs,
                old_log_probs,
                advantages,
                self.config.clip_param,
            )

            # Value loss
            value_loss = jnp.mean((values - returns) ** 2)

            # Entropy bonus
            entropy = compute_policy_entropy(log_probs_all)

            # Total loss
            total_loss = (
                policy_loss
                + self.config.vf_coeff * value_loss
                - self.config.entropy_coeff * entropy
            )

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": total_loss,
            }

            return total_loss, metrics

        # Compute gradients
        grads, metrics = nnx.grad(loss_fn, has_aux=True)(self.model)

        # Update model
        self.optimizer.update(self.model, grads)

        return metrics["total_loss"], metrics
