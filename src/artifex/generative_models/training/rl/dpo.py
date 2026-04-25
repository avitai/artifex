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

from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.adapters import SequencePolicyAdapter
from artifex.generative_models.training.rl.configs import DPOConfig
from artifex.generative_models.training.rl.protocols import LogProbScorer
from artifex.generative_models.training.rl.types import (
    canonicalize_response_mask,
    GeneratedSequenceBatch,
    PreferenceBatch,
)


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
        *,
        policy_adapter: LogProbScorer | None = None,
        reference_adapter: LogProbScorer | None = None,
    ) -> None:
        """Initialize DPO trainer.

        Args:
            model: Policy model to train.
            reference_model: Frozen reference model. Can be None if
                config.reference_free=True (SimPO mode).
            optimizer: Flax NNX optimizer for the model.
            config: DPO configuration. Uses defaults if not provided.
            policy_adapter: Optional scorer for policy sequence log-probs.
            reference_adapter: Optional scorer for reference sequence log-probs.
        """
        self.model = model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.config = config if config is not None else DPOConfig()
        self.policy_adapter = policy_adapter or SequencePolicyAdapter(model)
        self.reference_adapter = (
            reference_adapter
            if reference_adapter is not None
            else (SequencePolicyAdapter(reference_model) if reference_model is not None else None)
        )

    @staticmethod
    def _build_sequence_batch(
        sequences: jax.Array,
        loss_mask: jax.Array | None = None,
    ) -> GeneratedSequenceBatch:
        """Construct a typed sequence batch from legacy scorer inputs."""
        return GeneratedSequenceBatch.from_sequences(
            sequences,
            response_mask=canonicalize_response_mask(sequences, loss_mask),
        )

    @classmethod
    def _coerce_preference_batch(
        cls,
        batch: PreferenceBatch[GeneratedSequenceBatch] | dict[str, jax.Array],
    ) -> PreferenceBatch[GeneratedSequenceBatch]:
        """Normalize legacy dict batches into the typed preference contract."""
        if isinstance(batch, PreferenceBatch):
            if not isinstance(batch.chosen, GeneratedSequenceBatch) or not isinstance(
                batch.rejected,
                GeneratedSequenceBatch,
            ):
                msg = "DPOTrainer requires PreferenceBatch[GeneratedSequenceBatch]"
                raise TypeError(msg)
            return batch

        chosen_mask = batch.get("chosen_loss_mask", batch.get("chosen_attention_mask"))
        rejected_mask = batch.get("rejected_loss_mask", batch.get("rejected_attention_mask"))

        return PreferenceBatch(
            chosen=cls._build_sequence_batch(batch["chosen"], chosen_mask),
            rejected=cls._build_sequence_batch(batch["rejected"], rejected_mask),
        )

    def _resolve_policy_adapter(self, model: nnx.Module | None = None) -> LogProbScorer:
        """Return the scorer currently responsible for policy log-probs."""
        scorer = self.policy_adapter
        if model is None or getattr(scorer, "model", None) is model:
            return scorer

        bind = getattr(scorer, "bind", None)
        if callable(bind):
            return cast(LogProbScorer, bind(model))

        if hasattr(scorer, "model"):
            msg = (
                "Transform-compatible DPO loss functions require policy scorers "
                "to implement bind(model) when they capture a model instance."
            )
            raise TypeError(msg)
        return scorer

    def _resolve_reference_adapter(self) -> LogProbScorer | None:
        """Return the scorer currently responsible for reference log-probs."""
        scorer = self.reference_adapter
        if scorer is None:
            return None
        if self.reference_model is None or getattr(scorer, "model", None) is self.reference_model:
            return scorer

        bind = getattr(scorer, "bind", None)
        if callable(bind):
            return cast(LogProbScorer, bind(self.reference_model))

        if hasattr(scorer, "model"):
            msg = (
                "Transform-compatible DPO loss functions require reference scorers "
                "to implement bind(model) when they capture a model instance."
            )
            raise TypeError(msg)
        return scorer

    def compute_log_probs(
        self,
        model: nnx.Module | LogProbScorer,
        sequences: jax.Array,
        loss_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute per-sequence log probabilities (autoregressive).

        Uses the standard autoregressive formulation: for each position t,
        gather the log probability of the actual next token, then average over
        the selected target tokens to get a per-sequence score.

        Args:
            model: Model to compute log probs with.
            sequences: Input sequences with shape (batch_size, seq_len).
            loss_mask: Optional token mask selecting which next-token positions
                contribute to the sequence score. This can be used to exclude
                prompt tokens and padding when scoring prompt-conditioned
                preference data.

        Returns:
            Log probabilities with shape (batch_size,).
        """
        scorer = model if isinstance(model, LogProbScorer) else SequencePolicyAdapter(model)
        return scorer.sequence_log_probs(self._build_sequence_batch(sequences, loss_mask))

    def _compute_sequence_log_ratios_with_scorers(
        self,
        batch: GeneratedSequenceBatch,
        policy_scorer: LogProbScorer,
        reference_scorer: LogProbScorer | None,
    ) -> jax.Array:
        """Compute sequence log-ratios with explicit policy/reference scorers."""
        policy_log_probs = policy_scorer.sequence_log_probs(batch)

        if self.config.reference_free or self.reference_model is None:
            return policy_log_probs

        if reference_scorer is None:
            msg = "reference_adapter must be provided when reference_free=False"
            raise ValueError(msg)

        ref_log_probs = reference_scorer.sequence_log_probs(batch)
        return policy_log_probs - ref_log_probs

    def compute_sequence_log_ratios(
        self,
        batch: GeneratedSequenceBatch,
    ) -> jax.Array:
        """Compute log ratios between policy and reference for a typed sequence batch."""
        return self._compute_sequence_log_ratios_with_scorers(
            batch,
            self._resolve_policy_adapter(),
            self._resolve_reference_adapter(),
        )

    def compute_log_ratios(
        self,
        sequences: jax.Array,
        loss_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute log ratios between policy and reference.

        log_ratio = log(pi(y|x)) - log(ref(y|x))

        Args:
            sequences: Input sequences with shape (batch_size, seq_len).
            loss_mask: Optional token mask selecting the scored target tokens.

        Returns:
            Log ratios with shape (batch_size,).
        """
        return self.compute_sequence_log_ratios(
            self._build_sequence_batch(sequences, loss_mask),
        )

    def _compute_loss_with_scorers(
        self,
        preference_batch: PreferenceBatch[GeneratedSequenceBatch],
        policy_scorer: LogProbScorer,
        reference_scorer: LogProbScorer | None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute DPO loss from explicit scorers over a typed preference batch."""
        chosen_log_ratios = self._compute_sequence_log_ratios_with_scorers(
            preference_batch.chosen,
            policy_scorer,
            reference_scorer,
        )
        rejected_log_ratios = self._compute_sequence_log_ratios_with_scorers(
            preference_batch.rejected,
            policy_scorer,
            reference_scorer,
        )

        margin = chosen_log_ratios - rejected_log_ratios
        scaled_margin = self.config.beta * margin

        if self.config.label_smoothing > 0:
            eps = self.config.label_smoothing
            loss = -(1 - eps) * jax.nn.log_sigmoid(scaled_margin) - eps * jax.nn.log_sigmoid(
                -scaled_margin
            )
        else:
            loss = -jax.nn.log_sigmoid(scaled_margin)

        loss = jnp.mean(loss)
        reward_accuracy = jnp.mean((margin > 0).astype(jnp.float32))

        metrics = {
            "dpo_loss": loss,
            "reward_accuracy": reward_accuracy,
            "chosen_log_ratios": jnp.mean(chosen_log_ratios),
            "rejected_log_ratios": jnp.mean(rejected_log_ratios),
            "margin": jnp.mean(margin),
        }

        return loss, metrics

    def compute_loss(
        self,
        batch: PreferenceBatch[GeneratedSequenceBatch] | dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute DPO loss.

        DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

        Args:
            batch: Preference batch or legacy dict containing chosen/rejected
                sequences and optional response masks.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        preference_batch = self._coerce_preference_batch(batch)

        return self._compute_loss_with_scorers(
            preference_batch,
            self._resolve_policy_adapter(),
            self._resolve_reference_adapter(),
        )

    def create_loss_fn(self):
        """Create a transform-friendly DPO loss function with explicit model input."""

        def loss_fn(
            model: nnx.Module,
            batch: PreferenceBatch[GeneratedSequenceBatch] | dict[str, jax.Array],
        ) -> tuple[jax.Array, dict[str, Any]]:
            preference_batch = self._coerce_preference_batch(batch)
            return self._compute_loss_with_scorers(
                preference_batch,
                self._resolve_policy_adapter(model),
                self._resolve_reference_adapter(),
            )

        return loss_fn

    def train_step(
        self,
        batch: PreferenceBatch[GeneratedSequenceBatch] | dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Perform a single DPO training step.

        Args:
            batch: Preference batch or legacy dict containing chosen/rejected
                sequences and optional response masks.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        loss_fn = self.create_loss_fn()
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model, batch)
        self.optimizer.update(self.model, grads)
        return loss, metrics
