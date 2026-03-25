"""Concrete adapters for the typed RL contract layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.training.rl.types import (
    GeneratedSequenceBatch,
    prepare_autoregressive_token_mask,
    SequenceRolloutBatch,
)


SequenceBatchLike = GeneratedSequenceBatch | SequenceRolloutBatch


def extract_logits_from_output(model_output: Any) -> jax.Array:
    """Extract logits from common sequence-model output shapes."""
    if isinstance(model_output, jax.Array):
        return model_output

    if isinstance(model_output, tuple | list) and model_output:
        return extract_logits_from_output(model_output[0])

    if isinstance(model_output, Mapping):
        logits = model_output.get("logits")
        if logits is not None:
            return logits

    logits_attr = getattr(model_output, "logits", None)
    if logits_attr is not None:
        return logits_attr

    msg = "sequence model output must be a logits array or expose a 'logits' field"
    raise TypeError(msg)


def extract_values_from_output(model_output: Any) -> jax.Array:
    """Extract value predictions from common actor-critic output shapes."""
    if isinstance(model_output, nnx.Variable):
        return model_output[...]

    if isinstance(model_output, tuple | list) and len(model_output) >= 2:
        values = model_output[1]
        return values if isinstance(values, jax.Array) else extract_values_from_output(values)

    if isinstance(model_output, Mapping):
        values = model_output.get("values", model_output.get("value"))
        if values is not None:
            return values

    values_attr = getattr(model_output, "values", None)
    if values_attr is not None:
        return values_attr

    value_attr = getattr(model_output, "value", None)
    if value_attr is not None:
        return value_attr

    msg = "actor-critic output must expose a 'values' or 'value' field"
    raise TypeError(msg)


class SequencePolicyAdapter:
    """Sequence log-prob scorer for autoregressive generative models."""

    def __init__(
        self,
        model: nnx.Module,
        *,
        model_kwargs_factory: Callable[[SequenceBatchLike], dict[str, Any]] | None = None,
        logits_extractor: Callable[[Any], jax.Array] | None = None,
    ) -> None:
        self.model = model
        self.model_kwargs_factory = model_kwargs_factory
        self.logits_extractor = logits_extractor or extract_logits_from_output

    def bind(self, model: nnx.Module) -> SequencePolicyAdapter:
        """Return an equivalent adapter bound to a different model instance."""
        return SequencePolicyAdapter(
            model,
            model_kwargs_factory=self.model_kwargs_factory,
            logits_extractor=self.logits_extractor,
        )

    def _run_model(self, batch: SequenceBatchLike) -> jax.Array:
        """Execute the wrapped model and return batch-aligned logits."""
        sequence_batch = batch.sequence_batch if isinstance(batch, SequenceRolloutBatch) else batch
        if sequence_batch.sequences.ndim != 2:
            msg = "SequencePolicyAdapter expects 2D token sequences with shape (batch, seq_len)"
            raise ValueError(msg)

        model_kwargs = (
            self.model_kwargs_factory(batch) if self.model_kwargs_factory is not None else {}
        )
        model_output = self.model(sequence_batch.sequences, **model_kwargs)
        logits = self.logits_extractor(model_output)

        if logits.shape[:2] != sequence_batch.sequences.shape:
            msg = (
                "sequence model logits must align with input sequences on batch/length: "
                f"{logits.shape[:2]} != {sequence_batch.sequences.shape}"
            )
            raise ValueError(msg)
        return logits

    def log_prob_distributions(self, batch: SequenceRolloutBatch) -> jax.Array:
        """Return per-token next-token log-prob distributions."""
        logits = self._run_model(batch)
        return nnx.log_softmax(logits[:, :-1, :], axis=-1)

    def action_log_probs(self, batch: SequenceRolloutBatch) -> jax.Array:
        """Return log-probabilities for the sampled autoregressive actions."""
        log_probs = self.log_prob_distributions(batch)
        return jnp.take_along_axis(
            log_probs,
            batch.sequences[:, 1:, None],
            axis=-1,
        ).squeeze(-1)

    def sequence_log_probs(self, batch: GeneratedSequenceBatch) -> jax.Array:
        """Return one average autoregressive log-prob score per sequence."""
        logits = self._run_model(batch)
        log_probs = nnx.log_softmax(logits[:, :-1, :], axis=-1)
        target_tokens = batch.sequences[:, 1:]
        per_token_log_probs = jnp.take_along_axis(
            log_probs,
            target_tokens[:, :, None],
            axis=-1,
        ).squeeze(-1)

        token_mask = prepare_autoregressive_token_mask(batch.sequences, batch.response_mask)
        masked_log_probs = per_token_log_probs * token_mask
        normalizer = jnp.maximum(jnp.sum(token_mask, axis=-1), 1.0)
        return jnp.sum(masked_log_probs, axis=-1) / normalizer


class SequenceValueHeadAdapter:
    """Sequence value-head adapter for autoregressive actor-critic models."""

    def __init__(
        self,
        model: nnx.Module,
        *,
        model_kwargs_factory: Callable[[SequenceRolloutBatch], dict[str, Any]] | None = None,
        value_extractor: Callable[[Any], jax.Array] | None = None,
    ) -> None:
        self.model = model
        self.model_kwargs_factory = model_kwargs_factory
        self.value_extractor = value_extractor or extract_values_from_output

    def bind(self, model: nnx.Module) -> SequenceValueHeadAdapter:
        """Return an equivalent adapter bound to a different model instance."""
        return SequenceValueHeadAdapter(
            model,
            model_kwargs_factory=self.model_kwargs_factory,
            value_extractor=self.value_extractor,
        )

    def action_values(self, batch: SequenceRolloutBatch) -> jax.Array:
        """Return value estimates aligned with autoregressive next-token actions."""
        model_kwargs = (
            self.model_kwargs_factory(batch) if self.model_kwargs_factory is not None else {}
        )
        model_output = self.model(batch.sequences, **model_kwargs)
        values = self.value_extractor(model_output)

        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)

        if values.shape == batch.sequences.shape:
            return values[:, :-1]

        if values.shape == batch.action_shape:
            return values

        msg = (
            "sequence value predictions must align with sequences or action tokens: "
            f"{values.shape} != {batch.sequences.shape} and {values.shape} != {batch.action_shape}"
        )
        raise ValueError(msg)
