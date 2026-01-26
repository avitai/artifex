"""Group Relative Policy Optimization (GRPO) trainer.

GRPO is a critic-free RL algorithm from DeepSeek-R1 that:
- Generates multiple completions per prompt
- Normalizes advantages within each group
- Uses PPO-style clipping
- Saves ~50% memory by eliminating the value network

Reference: https://arxiv.org/abs/2402.03300 (DeepSeek-R1)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.configs import GRPOConfig
from artifex.generative_models.training.rl.utils import (
    compute_clipped_surrogate_loss,
    compute_kl_divergence,
    compute_policy_entropy,
)


class GRPOTrainer:
    """Group Relative Policy Optimization trainer.

    GRPO is a critic-free algorithm that:
    1. Generates G completions per prompt
    2. Computes rewards for each completion
    3. Normalizes rewards within each group: (r - mean) / std
    4. Uses normalized rewards as advantages
    5. Applies PPO-style clipped objective

    This eliminates the need for a value network, saving ~50% memory.

    Attributes:
        model: Policy model to train.
        optimizer: Flax NNX optimizer.
        config: GRPO configuration.
        reference_model: Optional frozen reference for KL penalty.
    """

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        config: GRPOConfig | None = None,
        reference_model: nnx.Module | None = None,
    ) -> None:
        """Initialize GRPO trainer.

        Args:
            model: Policy model to train.
            optimizer: Flax NNX optimizer for the model.
            config: GRPO configuration. Uses defaults if not provided.
            reference_model: Optional frozen reference model for KL penalty.
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config if config is not None else GRPOConfig()
        self.reference_model = reference_model

    def normalize_group_rewards(
        self,
        rewards: jax.Array,
        group_size: int,
        eps: float = 1e-8,
    ) -> jax.Array:
        """Normalize rewards within each group.

        For GRPO, we have G generations per prompt. We normalize
        rewards within each group to zero mean, unit variance.

        Args:
            rewards: Rewards with shape (batch_size,) where batch_size
                is num_prompts * group_size.
            group_size: Number of generations per prompt (G).
            eps: Small constant for numerical stability.

        Returns:
            Group-normalized advantages with same shape.
        """
        # Reshape to (num_prompts, group_size)
        num_prompts = rewards.shape[0] // group_size
        grouped = rewards.reshape(num_prompts, group_size)

        # Compute per-group statistics
        group_mean = jnp.mean(grouped, axis=1, keepdims=True)
        group_std = jnp.std(grouped, axis=1, keepdims=True)

        # Normalize within groups
        normalized = (grouped - group_mean) / (group_std + eps)

        # Flatten back
        return normalized.reshape(-1)

    def compute_loss(
        self,
        states: jax.Array,
        actions: jax.Array,
        old_log_probs: jax.Array,
        rewards: jax.Array,
        group_size: int | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute GRPO loss.

        Args:
            states: Input states with shape (batch_size, ...).
            actions: Actions taken with shape (batch_size,).
            old_log_probs: Log probs from old policy with shape (batch_size,).
            rewards: Rewards with shape (batch_size,).
            group_size: Number of generations per prompt. If None, uses
                config.num_generations.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        if group_size is None:
            group_size = self.config.num_generations

        # Normalize rewards within groups to get advantages
        advantages = self.normalize_group_rewards(rewards, group_size)

        # Forward pass to get current log probs
        logits = self.model(states)
        log_probs_all = jax.nn.log_softmax(logits, axis=-1)
        log_probs = jnp.take_along_axis(
            log_probs_all,
            actions[:, None],
            axis=-1,
        ).squeeze(-1)

        # Clipped surrogate loss
        policy_loss = compute_clipped_surrogate_loss(
            log_probs,
            old_log_probs,
            advantages,
            self.config.clip_param,
        )

        # Entropy bonus
        entropy = compute_policy_entropy(log_probs_all)

        # KL penalty if reference model exists
        kl_penalty = jnp.array(0.0)
        if self.reference_model is not None:
            ref_logits = self.reference_model(states)
            ref_log_probs = jax.nn.log_softmax(ref_logits, axis=-1)
            kl_penalty = compute_kl_divergence(log_probs_all, ref_log_probs)

        # Total loss
        total_loss = (
            policy_loss + self.config.beta * kl_penalty - self.config.entropy_coeff * entropy
        )

        metrics = {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "kl_penalty": kl_penalty,
            "total_loss": total_loss,
            "advantages_mean": jnp.mean(advantages),
            "advantages_std": jnp.std(advantages),
        }

        return total_loss, metrics

    def train_step(
        self,
        batch: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single GRPO training step.

        Args:
            batch: Dictionary containing:
                - "states": Input states.
                - "actions": Actions taken.
                - "old_log_probs": Log probs from old policy.
                - "rewards": Rewards for each completion.
                - "group_size": Optional, number of generations per prompt.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        rewards = batch["rewards"]
        group_size_value = batch.get("group_size")
        group_size: int = (
            int(group_size_value) if group_size_value is not None else self.config.num_generations
        )

        def loss_fn(model: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            original_model = self.model
            self.model = model
            loss, metrics = self.compute_loss(states, actions, old_log_probs, rewards, group_size)
            self.model = original_model
            return loss, metrics

        # Compute gradients
        grads, metrics = nnx.grad(loss_fn, has_aux=True)(self.model)

        # Update model
        self.optimizer.update(self.model, grads)

        return metrics["total_loss"], metrics
