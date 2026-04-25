"""GRPO trainer over grouped autoregressive rollout batches."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.adapters import SequencePolicyAdapter
from artifex.generative_models.training.rl.configs import GRPOConfig
from artifex.generative_models.training.rl.protocols import SequenceRolloutPolicyAdapter
from artifex.generative_models.training.rl.types import GroupRolloutBatch, SequenceRolloutBatch
from artifex.generative_models.training.rl.utils import (
    compute_masked_clipped_surrogate_loss,
    compute_masked_kl_divergence,
    compute_masked_policy_entropy,
)


class GRPOTrainer:
    """Group Relative Policy Optimization trainer for sequence rollouts."""

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        config: GRPOConfig | None = None,
        reference_model: nnx.Module | None = None,
        *,
        policy_adapter: SequenceRolloutPolicyAdapter | None = None,
        reference_adapter: SequenceRolloutPolicyAdapter | None = None,
    ) -> None:
        """Initialize the GRPO trainer."""
        self.model = model
        self.optimizer = optimizer
        self.config = config if config is not None else GRPOConfig()
        self.reference_model = reference_model
        self.policy_adapter = policy_adapter or SequencePolicyAdapter(model)
        self.reference_adapter = (
            reference_adapter
            if reference_adapter is not None
            else (SequencePolicyAdapter(reference_model) if reference_model is not None else None)
        )

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
            return cast(SequenceRolloutPolicyAdapter, bind(model))

        if hasattr(adapter, "model"):
            msg = (
                "Transform-compatible GRPO loss functions require policy adapters "
                "to implement bind(model) when they capture a model instance."
            )
            raise TypeError(msg)
        return adapter

    def _resolve_reference_adapter(self) -> SequenceRolloutPolicyAdapter | None:
        """Return the active reference adapter, rebound without mutating trainer state."""
        adapter = self.reference_adapter
        if adapter is None:
            return None
        if self.reference_model is None or getattr(adapter, "model", None) is self.reference_model:
            return adapter

        bind = getattr(adapter, "bind", None)
        if callable(bind):
            return cast(SequenceRolloutPolicyAdapter, bind(self.reference_model))

        if hasattr(adapter, "model"):
            msg = (
                "Transform-compatible GRPO loss functions require reference adapters "
                "to implement bind(model) when they capture a model instance."
            )
            raise TypeError(msg)
        return adapter

    def normalize_group_rewards(
        self,
        rewards: jax.Array,
        group_size: int,
        eps: float = 1e-8,
    ) -> jax.Array:
        """Normalize rewards within each prompt group."""
        num_prompts = rewards.shape[0] // group_size
        grouped = rewards.reshape(num_prompts, group_size)
        group_mean = jnp.mean(grouped, axis=1, keepdims=True)
        group_std = jnp.std(grouped, axis=1, keepdims=True)
        normalized = (grouped - group_mean) / (group_std + eps)
        return normalized.reshape(-1)

    def _compute_loss_with_adapters(
        self,
        batch: GroupRolloutBatch[SequenceRolloutBatch],
        policy_adapter: SequenceRolloutPolicyAdapter,
        reference_adapter: SequenceRolloutPolicyAdapter | None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute GRPO loss over a grouped sequence rollout batch."""
        rollout = batch.rollout
        if rollout.old_log_probs is None or rollout.sequence_rewards is None:
            msg = (
                "GRPOTrainer requires old_log_probs and sequence rewards on "
                "GroupRolloutBatch.rollout"
            )
            raise ValueError(msg)

        sequence_advantages = self.normalize_group_rewards(
            rollout.sequence_rewards,
            batch.group_size,
        )
        action_mask = rollout.action_mask
        broadcast_advantages = sequence_advantages[:, None] * action_mask

        action_log_probs = policy_adapter.action_log_probs(rollout)
        policy_loss = compute_masked_clipped_surrogate_loss(
            action_log_probs,
            rollout.old_log_probs,
            broadcast_advantages,
            action_mask,
            self.config.clip_param,
        )

        policy_log_probs = policy_adapter.log_prob_distributions(rollout)
        entropy = compute_masked_policy_entropy(policy_log_probs, action_mask)

        kl_penalty = jnp.array(0.0)
        if reference_adapter is not None:
            reference_log_probs = reference_adapter.log_prob_distributions(rollout)
            kl_penalty = compute_masked_kl_divergence(
                policy_log_probs,
                reference_log_probs,
                action_mask,
            )

        total_loss = (
            policy_loss + self.config.beta * kl_penalty - self.config.entropy_coeff * entropy
        )
        metrics = {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "kl_penalty": kl_penalty,
            "total_loss": total_loss,
            "advantages_mean": jnp.mean(sequence_advantages),
            "advantages_std": jnp.std(sequence_advantages),
        }
        return total_loss, metrics

    def compute_loss(
        self,
        batch: GroupRolloutBatch[SequenceRolloutBatch],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute GRPO loss for a grouped sequence rollout batch."""
        return self._compute_loss_with_adapters(
            batch,
            self._resolve_policy_adapter(),
            self._resolve_reference_adapter(),
        )

    def create_loss_fn(self):
        """Create a transform-friendly GRPO loss function with explicit model input."""

        def loss_fn(
            model: nnx.Module,
            batch: GroupRolloutBatch[SequenceRolloutBatch],
        ) -> tuple[jax.Array, dict[str, Any]]:
            return self._compute_loss_with_adapters(
                batch,
                self._resolve_policy_adapter(model),
                self._resolve_reference_adapter(),
            )

        return loss_fn

    def train_step(
        self,
        batch: GroupRolloutBatch[SequenceRolloutBatch],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single GRPO update step."""
        loss_fn = self.create_loss_fn()
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model, batch)
        self.optimizer.update(self.model, grads)
        return loss, metrics
