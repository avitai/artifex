"""REINFORCE trainer over typed autoregressive rollout batches."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.adapters import SequencePolicyAdapter
from artifex.generative_models.training.rl.configs import REINFORCEConfig
from artifex.generative_models.training.rl.protocols import SequenceRolloutPolicyAdapter
from artifex.generative_models.training.rl.types import SequenceRolloutBatch
from artifex.generative_models.training.rl.utils import (
    compute_discounted_returns,
    compute_masked_policy_entropy,
    masked_mean,
    masked_normalize,
    normalize_advantages,
)


class REINFORCETrainer:
    """REINFORCE policy gradient trainer for sequence rollouts."""

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        config: REINFORCEConfig | None = None,
        *,
        policy_adapter: SequenceRolloutPolicyAdapter | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config if config is not None else REINFORCEConfig()
        self.policy_adapter = policy_adapter or SequencePolicyAdapter(model)

    def _resolve_policy_adapter(
        self,
        model: nnx.Module | None = None,
    ) -> SequenceRolloutPolicyAdapter:
        """Return the active policy adapter, rebound without mutating shared trainer state."""
        adapter = self.policy_adapter
        if model is None or getattr(adapter, "model", None) is model:
            return adapter

        bind = getattr(adapter, "bind", None)
        if callable(bind):
            return bind(model)

        if hasattr(adapter, "model"):
            msg = (
                "Transform-compatible REINFORCE loss functions require policy adapters "
                "to implement bind(model) when they capture a model instance."
            )
            raise TypeError(msg)
        return adapter

    def compute_returns(self, rewards: jax.Array) -> jax.Array:
        """Compute discounted returns for 1D or batched 2D reward tensors."""
        if rewards.ndim == 1:
            return compute_discounted_returns(rewards, self.config.gamma)
        if rewards.ndim == 2:
            return jax.vmap(compute_discounted_returns, in_axes=(0, None))(
                rewards,
                self.config.gamma,
            )
        msg = "rewards must be rank-1 or rank-2 for discounted-return computation"
        raise ValueError(msg)

    def normalize_returns(self, returns: jax.Array) -> jax.Array:
        """Normalize returns to zero mean and unit variance."""
        return normalize_advantages(returns)

    def _terminal_sequence_rewards_to_token_rewards(
        self,
        batch: SequenceRolloutBatch,
    ) -> jax.Array:
        """Project sequence-level rewards onto the last active response token."""
        rewards = batch.sequence_rewards
        if rewards is None:
            msg = "SequenceRolloutBatch must provide token_rewards, returns, or sequence rewards"
            raise ValueError(msg)

        action_rewards = jnp.zeros(batch.action_shape, dtype=rewards.dtype)
        action_mask = batch.action_mask
        has_actions = jnp.sum(action_mask, axis=-1) > 0
        last_indices = jnp.maximum(jnp.sum(action_mask, axis=-1).astype(jnp.int32) - 1, 0)
        batch_indices = jnp.arange(batch.batch_size)
        return action_rewards.at[batch_indices, last_indices].set(
            rewards * has_actions.astype(rewards.dtype)
        )

    def _resolve_returns(self, batch: SequenceRolloutBatch) -> jax.Array:
        """Resolve rollout returns from explicit returns or reward tensors."""
        if batch.returns is not None:
            return batch.returns

        rewards = batch.token_rewards
        if rewards is None:
            rewards = self._terminal_sequence_rewards_to_token_rewards(batch)
        return self.compute_returns(rewards)

    def _compute_loss_with_adapter(
        self,
        batch: SequenceRolloutBatch,
        policy_adapter: SequenceRolloutPolicyAdapter,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute REINFORCE loss using the provided policy adapter."""
        action_mask = batch.action_mask
        returns = self._resolve_returns(batch)
        if self.config.normalize_returns:
            returns = masked_normalize(returns, action_mask)

        action_log_probs = policy_adapter.action_log_probs(batch)
        policy_loss = -masked_mean(action_log_probs * returns, action_mask)
        entropy = compute_masked_policy_entropy(
            policy_adapter.log_prob_distributions(batch),
            action_mask,
        )
        total_loss = policy_loss - self.config.entropy_coeff * entropy

        metrics = {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }
        return total_loss, metrics

    def compute_loss(
        self,
        batch: SequenceRolloutBatch,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute REINFORCE loss for a typed sequence rollout batch."""
        return self._compute_loss_with_adapter(batch, self._resolve_policy_adapter())

    def create_loss_fn(self):
        """Create a transform-friendly REINFORCE loss function with explicit model input."""

        def loss_fn(
            model: nnx.Module,
            batch: SequenceRolloutBatch,
        ) -> tuple[jax.Array, dict[str, Any]]:
            return self._compute_loss_with_adapter(
                batch,
                self._resolve_policy_adapter(model),
            )

        return loss_fn

    def train_step(
        self,
        trajectory: SequenceRolloutBatch,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single REINFORCE update step."""
        loss_fn = self.create_loss_fn()
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model, trajectory)
        self.optimizer.update(self.model, grads)
        return loss, metrics
