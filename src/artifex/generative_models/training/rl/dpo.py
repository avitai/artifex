"""Direct Preference Optimization (DPO) trainer.

DPO enables preference learning without an explicit reward model by directly
optimizing the policy to prefer chosen over rejected responses. This
implementation includes:
- Standard DPO with reference model
- SimPO mode (reference-free)
- Label smoothing
- Reward accuracy tracking
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.configs import DPOConfig


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Implements DPO for preference learning:
    - Learns from preference pairs (chosen, rejected)
    - Uses log-ratio between policy and reference model
    - SimPO mode eliminates need for reference model

    Attributes:
        model: Policy model to train.
        reference_model: Frozen reference model (None in SimPO mode).
        optimizer: Flax NNX optimizer.
        config: DPO configuration.
    """

    def __init__(
        self,
        model: nnx.Module,
        reference_model: nnx.Module | None,
        optimizer: nnx.Optimizer,
        config: DPOConfig | None = None,
    ) -> None:
        """Initialize DPO trainer.

        Args:
            model: Policy model to train.
            reference_model: Frozen reference model. Can be None if
                config.reference_free=True (SimPO mode).
            optimizer: Flax NNX optimizer for the model.
            config: DPO configuration. Uses defaults if not provided.
        """
        self.model = model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.config = config if config is not None else DPOConfig()

    def compute_log_probs(
        self,
        model: nnx.Module,
        sequences: jax.Array,
    ) -> jax.Array:
        """Compute per-sequence log probabilities (autoregressive).

        Uses the standard autoregressive formulation: for each position t,
        gather the log probability of the actual next token, then average
        over the sequence dimension to get a per-sequence score.

        Args:
            model: Model to compute log probs with.
            sequences: Input sequences with shape (batch_size, seq_len).

        Returns:
            Log probabilities with shape (batch_size,).
        """
        # Forward pass to get logits: (batch_size, seq_len, vocab_size)
        logits = model(sequences)

        # Convert to log probabilities over vocab: (batch_size, seq_len, vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # Gather log-prob of the actual next token at each position
        # Input tokens shifted by 1: predict sequences[:, 1:] from logits[:, :-1]
        target_tokens = sequences[:, 1:]  # (batch_size, seq_len - 1)
        # Gather per-token log probs: (batch_size, seq_len - 1)
        per_token_log_probs = jnp.take_along_axis(
            log_probs[:, :-1, :], target_tokens[:, :, None], axis=-1
        ).squeeze(-1)

        # Average over sequence dimension to get per-sequence score
        batch_log_probs = jnp.mean(per_token_log_probs, axis=-1)  # (batch_size,)

        return batch_log_probs

    def compute_log_ratios(
        self,
        sequences: jax.Array,
    ) -> jax.Array:
        """Compute log ratios between policy and reference.

        log_ratio = log(pi(y|x)) - log(ref(y|x))

        Args:
            sequences: Input sequences with shape (batch_size, seq_len).

        Returns:
            Log ratios with shape (batch_size,).
        """
        policy_log_probs = self.compute_log_probs(self.model, sequences)

        if self.config.reference_free or self.reference_model is None:
            # SimPO mode: no reference model
            return policy_log_probs
        else:
            ref_log_probs = self.compute_log_probs(self.reference_model, sequences)
            return policy_log_probs - ref_log_probs

    def compute_loss(
        self,
        batch: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute DPO loss.

        DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

        Args:
            batch: Dictionary containing:
                - "chosen": Chosen sequences.
                - "rejected": Rejected sequences.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        chosen = batch["chosen"]
        rejected = batch["rejected"]

        # Compute log ratios for chosen and rejected
        chosen_log_ratios = self.compute_log_ratios(chosen)
        rejected_log_ratios = self.compute_log_ratios(rejected)

        # Compute preference margin
        margin = chosen_log_ratios - rejected_log_ratios

        # Scale by beta
        scaled_margin = self.config.beta * margin

        # DPO loss with optional label smoothing
        if self.config.label_smoothing > 0:
            # Smooth labels: (1 - eps) for chosen, eps for rejected
            eps = self.config.label_smoothing
            loss = -(1 - eps) * jax.nn.log_sigmoid(scaled_margin) - eps * jax.nn.log_sigmoid(
                -scaled_margin
            )
        else:
            # Standard DPO loss
            loss = -jax.nn.log_sigmoid(scaled_margin)

        loss = jnp.mean(loss)

        # Compute reward accuracy (how often chosen is preferred)
        reward_accuracy = jnp.mean((margin > 0).astype(jnp.float32))

        metrics = {
            "dpo_loss": loss,
            "reward_accuracy": reward_accuracy,
            "chosen_log_ratios": jnp.mean(chosen_log_ratios),
            "rejected_log_ratios": jnp.mean(rejected_log_ratios),
            "margin": jnp.mean(margin),
        }

        return loss, metrics

    def train_step(
        self,
        batch: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single DPO training step.

        Args:
            batch: Dictionary containing:
                - "chosen": Chosen sequences.
                - "rejected": Rejected sequences.

        Returns:
            Tuple of (loss, metrics_dict).
        """

        def loss_fn(model: nnx.Module) -> tuple[jax.Array, dict[str, Any]]:
            # Temporarily swap model for gradient computation
            original_model = self.model
            self.model = model
            loss, metrics = self.compute_loss(batch)
            self.model = original_model
            return loss, metrics

        # Compute gradients
        grads, metrics = nnx.grad(loss_fn, has_aux=True)(self.model)

        # Update model
        self.optimizer.update(self.model, grads)

        return metrics["dpo_loss"], metrics
