"""PPO trainer over typed autoregressive rollout batches."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from artifex.generative_models.training.rl.adapters import (
    SequencePolicyAdapter,
    SequenceValueHeadAdapter,
)
from artifex.generative_models.training.rl.configs import PPOConfig
from artifex.generative_models.training.rl.protocols import (
    SequenceRolloutPolicyAdapter,
    SequenceRolloutValueAdapter,
)
from artifex.generative_models.training.rl.types import SequenceRolloutBatch
from artifex.generative_models.training.rl.utils import (
    compute_clipped_surrogate_loss,
    compute_gae_advantages,
    compute_masked_clipped_surrogate_loss,
    compute_masked_policy_entropy,
    compute_policy_entropy,
    masked_mean,
    masked_normalize,
)


class PPOTrainer:
    """Proximal Policy Optimization trainer for sequence rollouts."""

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        config: PPOConfig | None = None,
        *,
        policy_adapter: SequenceRolloutPolicyAdapter | None = None,
        value_adapter: SequenceRolloutValueAdapter | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config if config is not None else PPOConfig()
        self.policy_adapter = policy_adapter or SequencePolicyAdapter(model)
        self.value_adapter = value_adapter or SequenceValueHeadAdapter(model)

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
                "Transform-compatible PPO loss functions require policy adapters "
                "to implement bind(model) when they capture a model instance."
            )
            raise TypeError(msg)
        return adapter

    def _resolve_value_adapter(
        self,
        model: nnx.Module | None = None,
    ) -> SequenceRolloutValueAdapter:
        """Return the active value adapter, rebound without mutating shared trainer state."""
        adapter = self.value_adapter
        if model is None or getattr(adapter, "model", None) is model:
            return adapter

        bind = getattr(adapter, "bind", None)
        if callable(bind):
            return bind(model)

        if hasattr(adapter, "model"):
            msg = (
                "Transform-compatible PPO loss functions require value adapters "
                "to implement bind(model) when they capture a model instance."
            )
            raise TypeError(msg)
        return adapter

    def compute_gae(
        self,
        rewards: jax.Array,
        values: jax.Array,
        dones: jax.Array,
    ) -> jax.Array:
        """Compute GAE for 1D or batched 2D rollout tensors."""
        if rewards.ndim == 1:
            return compute_gae_advantages(
                rewards,
                values,
                dones,
                self.config.gamma,
                self.config.gae_lambda,
            )
        if rewards.ndim == 2:
            return jax.vmap(compute_gae_advantages, in_axes=(0, 0, 0, None, None))(
                rewards,
                values,
                dones,
                self.config.gamma,
                self.config.gae_lambda,
            )
        msg = "rewards must be rank-1 or rank-2 for GAE computation"
        raise ValueError(msg)

    def compute_clipped_loss(
        self,
        log_probs: jax.Array,
        old_log_probs: jax.Array,
        advantages: jax.Array,
    ) -> jax.Array:
        """Compute the standard unmasked PPO clipped loss."""
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
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute value-function loss with optional rollout masking."""
        errors = (values - returns) ** 2
        if mask is None:
            return jnp.mean(errors)
        return masked_mean(errors, mask)

    def compute_entropy(
        self,
        log_probs: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute policy entropy with optional rollout masking."""
        if mask is None:
            return compute_policy_entropy(log_probs)
        return compute_masked_policy_entropy(log_probs, mask)

    def _clip_gradients(self, grads):
        """Clip gradients by global norm to match the public PPO config."""
        grad_norm = optax.global_norm(grads)
        clip_scale = jnp.minimum(1.0, self.config.max_grad_norm / (grad_norm + 1e-8))
        return jax.tree.map(lambda grad: grad * clip_scale, grads)

    def _compute_loss_with_adapters(
        self,
        batch: SequenceRolloutBatch,
        policy_adapter: SequenceRolloutPolicyAdapter,
        value_adapter: SequenceRolloutValueAdapter,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute PPO loss against the shared rollout adapter contracts."""
        if batch.old_log_probs is None or batch.returns is None or batch.advantages is None:
            msg = (
                "PPOTrainer requires old_log_probs, returns, and advantages on SequenceRolloutBatch"
            )
            raise ValueError(msg)

        action_mask = batch.action_mask
        advantages = masked_normalize(batch.advantages, action_mask)
        action_log_probs = policy_adapter.action_log_probs(batch)
        policy_loss = compute_masked_clipped_surrogate_loss(
            action_log_probs,
            batch.old_log_probs,
            advantages,
            action_mask,
            self.config.clip_param,
        )

        values = value_adapter.action_values(batch)
        value_loss = self.compute_value_loss(values, batch.returns, action_mask)
        entropy = self.compute_entropy(policy_adapter.log_prob_distributions(batch), action_mask)

        total_loss = (
            policy_loss + self.config.vf_coeff * value_loss - self.config.entropy_coeff * entropy
        )
        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }
        return total_loss, metrics

    def compute_loss(
        self,
        batch: SequenceRolloutBatch,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute PPO loss for a typed sequence rollout batch."""
        return self._compute_loss_with_adapters(
            batch,
            self._resolve_policy_adapter(),
            self._resolve_value_adapter(),
        )

    def create_loss_fn(self):
        """Create a transform-friendly PPO loss function with explicit model input."""

        def loss_fn(
            model: nnx.Module,
            batch: SequenceRolloutBatch,
        ) -> tuple[jax.Array, dict[str, Any]]:
            return self._compute_loss_with_adapters(
                batch,
                self._resolve_policy_adapter(model),
                self._resolve_value_adapter(model),
            )

        return loss_fn

    def train_step(
        self,
        batch: SequenceRolloutBatch,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single PPO update over a typed sequence rollout batch."""
        loss_fn = self.create_loss_fn()
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model, batch)
        grads = self._clip_gradients(grads)
        self.optimizer.update(self.model, grads)
        return loss, metrics
