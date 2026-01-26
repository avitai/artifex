"""REINFORCE policy gradient trainer.

REINFORCE is the simplest policy gradient algorithm that uses Monte Carlo
sampling to estimate the policy gradient. This implementation includes:
- Discounted returns computation
- Optional return normalization for variance reduction
- Entropy bonus for exploration
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.configs import REINFORCEConfig
from artifex.generative_models.training.rl.utils import (
    compute_discounted_returns,
    compute_policy_entropy,
    normalize_advantages,
)


class REINFORCETrainer:
    """REINFORCE policy gradient trainer.

    Implements the REINFORCE algorithm with variance reduction techniques:
    - Discounted returns for credit assignment
    - Return normalization to stabilize gradients
    - Entropy bonus to encourage exploration

    Attributes:
        model: Policy network (must output action logits).
        optimizer: Flax NNX optimizer.
        config: REINFORCE configuration.
    """

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        config: REINFORCEConfig | None = None,
    ) -> None:
        """Initialize REINFORCE trainer.

        Args:
            model: Policy network that outputs action logits.
            optimizer: Flax NNX optimizer for the model.
            config: REINFORCE configuration. Uses defaults if not provided.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config if config is not None else REINFORCEConfig()

    def compute_returns(self, rewards: jax.Array) -> jax.Array:
        """Compute discounted returns from rewards.

        Args:
            rewards: Array of rewards with shape (T,).

        Returns:
            Array of discounted returns with shape (T,).
        """
        return compute_discounted_returns(rewards, self.config.gamma)

    def normalize_returns(self, returns: jax.Array) -> jax.Array:
        """Normalize returns to zero mean and unit variance.

        Args:
            returns: Array of returns to normalize.

        Returns:
            Normalized returns.
        """
        return normalize_advantages(returns)

    def compute_loss(
        self,
        states: jax.Array,
        actions: jax.Array,
        returns: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute REINFORCE loss.

        Loss = -E[log(pi(a|s)) * R] - entropy_coeff * H(pi)

        Args:
            states: Batch of states with shape (batch_size, ...).
            actions: Batch of actions taken with shape (batch_size,).
            returns: Discounted returns with shape (batch_size,).

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Normalize returns if configured
        if self.config.normalize_returns:
            returns = self.normalize_returns(returns)

        # Forward pass to get action logits
        logits = self.model(states)

        # Compute log probabilities for actions taken
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        action_log_probs = jnp.take_along_axis(
            log_probs,
            actions[:, None],
            axis=-1,
        ).squeeze(-1)

        # Policy gradient loss: -E[log(pi(a|s)) * R]
        policy_loss = -jnp.mean(action_log_probs * returns)

        # Entropy bonus for exploration
        entropy = compute_policy_entropy(log_probs)

        # Total loss: policy loss - entropy bonus
        total_loss = policy_loss - self.config.entropy_coeff * entropy

        metrics = {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }

        return total_loss, metrics

    def train_step(
        self,
        trajectory: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single training step.

        Args:
            trajectory: Dictionary containing:
                - "states": Batch of states.
                - "actions": Actions taken.
                - "rewards": Rewards received.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        states = trajectory["states"]
        actions = trajectory["actions"]
        rewards = trajectory["rewards"]

        # Compute discounted returns
        returns = self.compute_returns(rewards)

        # Define loss function for gradient computation
        def loss_fn(model: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            # Temporarily swap model for gradient computation
            original_model = self.model
            self.model = model
            loss, metrics = self.compute_loss(states, actions, returns)
            self.model = original_model
            return loss, metrics

        # Compute gradients and update
        grads, metrics = nnx.grad(loss_fn, has_aux=True)(self.model)
        self.optimizer.update(self.model, grads)

        return metrics["total_loss"], metrics
