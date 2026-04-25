"""Concrete generation and rollout backends for typed RL contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

from artifex.generative_models.training.rl.protocols import (
    LogProbScorer,
    RewardModel,
    SequenceGeneratingModule,
    SequenceGenerationBackend,
    SequenceRolloutPolicyAdapter,
)
from artifex.generative_models.training.rl.types import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    GroupRolloutBatch,
    SequenceGenerationRequest,
    SequenceRolloutBatch,
)


SequenceGenerateFn = Callable[[SequenceGeneratingModule, SequenceGenerationRequest], jax.Array]


def _repeat_batch_aligned_pytree(value: Any | None, repeats: int) -> Any | None:
    """Repeat each batch-aligned leaf ``repeats`` times along axis zero."""
    if value is None:
        return None

    leaves = tree_util.tree_leaves(value)
    if not leaves:
        return value

    def repeat_leaf(leaf: Any) -> Any:
        if not hasattr(leaf, "shape"):
            msg = "conditioning and generation metadata leaves must expose shape for repetition"
            raise TypeError(msg)
        return jnp.repeat(leaf, repeats, axis=0)

    return tree_util.tree_map(repeat_leaf, value)


class LocalSequenceGenerationBackend:
    """In-process sequence generation backend built around ``model.generate``."""

    def __init__(
        self,
        model: SequenceGeneratingModule,
        *,
        generate_fn: SequenceGenerateFn | None = None,
        prompt_kwarg: str = "prompt_tokens",
        conditioning_kwarg: str | None = None,
    ) -> None:
        """Initialize the sequence generation backend."""
        self.model = model
        self.generate_fn = generate_fn
        self.prompt_kwarg = prompt_kwarg
        self.conditioning_kwarg = conditioning_kwarg

    def bind(self, model: SequenceGeneratingModule) -> LocalSequenceGenerationBackend:
        """Return an equivalent backend bound to a different model instance."""
        return LocalSequenceGenerationBackend(
            model,
            generate_fn=self.generate_fn,
            prompt_kwarg=self.prompt_kwarg,
            conditioning_kwarg=self.conditioning_kwarg,
        )

    def _default_generate(self, request: SequenceGenerationRequest) -> jax.Array:
        """Call ``model.generate`` using the typed request fields."""
        kwargs: dict[str, Any] = {
            "temperature": request.temperature,
        }
        if request.top_k is not None:
            kwargs["top_k"] = request.top_k
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        if request.prompts is not None:
            kwargs[self.prompt_kwarg] = jnp.repeat(
                request.prompts,
                request.num_generations,
                axis=0,
            )
            kwargs["max_new_tokens"] = request.max_new_tokens
        else:
            kwargs["max_length"] = request.max_new_tokens

        if self.conditioning_kwarg is not None and request.conditioning is not None:
            kwargs[self.conditioning_kwarg] = _repeat_batch_aligned_pytree(
                request.conditioning,
                request.num_generations,
            )

        return self.model.generate(request.total_samples, **kwargs)

    def generate_sequences(self, request: SequenceGenerationRequest) -> GeneratedSequenceBatch:
        """Generate prompt-conditioned sequences and attach aligned masks."""
        sequences = (
            self.generate_fn(self.model, request)
            if self.generate_fn is not None
            else self._default_generate(request)
        )

        if sequences.ndim != 2:
            msg = (
                "sequence generation backends must return token arrays with shape (batch, seq_len)"
            )
            raise ValueError(msg)
        if sequences.shape[0] != request.total_samples:
            msg = (
                "generated sequence batch size must match request.total_samples: "
                f"{sequences.shape[0]} != {request.total_samples}"
            )
            raise ValueError(msg)

        conditioning = _repeat_batch_aligned_pytree(request.conditioning, request.num_generations)
        generation_metadata = _repeat_batch_aligned_pytree(
            request.generation_metadata,
            request.num_generations,
        )

        prompt_mask = None
        response_mask = None
        if request.prompts is not None:
            repeated_prompts = jnp.repeat(request.prompts, request.num_generations, axis=0)
            prompt_length = repeated_prompts.shape[1]
            if sequences.shape[1] < prompt_length:
                msg = (
                    "generated sequences must include the full prompt prefix: "
                    f"{sequences.shape[1]} < {prompt_length}"
                )
                raise ValueError(msg)
            if not jnp.array_equal(sequences[:, :prompt_length], repeated_prompts):
                msg = "generated sequences must preserve the repeated prompt prefix"
                raise ValueError(msg)

            repeated_prompt_mask = jnp.repeat(
                request.prompt_mask
                if request.prompt_mask is not None
                else jnp.ones(request.prompts.shape, dtype=jnp.float32),
                request.num_generations,
                axis=0,
            ).astype(jnp.float32)
            response_length = sequences.shape[1] - prompt_length
            prompt_mask = jnp.concatenate(
                [
                    repeated_prompt_mask,
                    jnp.zeros((request.total_samples, response_length), dtype=jnp.float32),
                ],
                axis=1,
            )
            response_mask = jnp.concatenate(
                [
                    jnp.zeros((request.total_samples, prompt_length), dtype=jnp.float32),
                    jnp.ones((request.total_samples, response_length), dtype=jnp.float32),
                ],
                axis=1,
            )

        return GeneratedSequenceBatch(
            generation=GeneratedBatch(
                outputs=sequences,
                conditioning=conditioning,
                generation_metadata=generation_metadata,
            ),
            prompt_mask=prompt_mask,
            response_mask=response_mask,
        )


class LocalSequenceRolloutBackend:
    """In-process rollout backend that assembles typed sequence rollout batches."""

    def __init__(
        self,
        generation_backend: SequenceGenerationBackend,
        policy_adapter: SequenceRolloutPolicyAdapter,
        *,
        reward_model: RewardModel | None = None,
        reference_scorer: LogProbScorer | None = None,
    ) -> None:
        """Initialize the rollout backend."""
        self.generation_backend = generation_backend
        self.policy_adapter = policy_adapter
        self.reward_model = reward_model
        self.reference_scorer = reference_scorer

    def generate_rollout(self, request: SequenceGenerationRequest) -> SequenceRolloutBatch:
        """Generate a rollout batch with policy log-probs and optional rewards."""
        sequence_batch = self.generation_backend.generate_sequences(request)
        rollout = SequenceRolloutBatch(sequence_batch=sequence_batch)
        old_log_probs = self.policy_adapter.action_log_probs(rollout)

        rewards = None
        if self.reward_model is not None:
            rewards = self.reward_model.score_generations(sequence_batch.generation)

        reference_log_probs = None
        if self.reference_scorer is not None:
            reference_log_probs = self.reference_scorer.sequence_log_probs(sequence_batch)

        if rewards is not None or reference_log_probs is not None:
            sequence_batch = GeneratedSequenceBatch(
                generation=GeneratedBatch(
                    outputs=sequence_batch.sequences,
                    conditioning=sequence_batch.conditioning,
                    generation_metadata=sequence_batch.generation_metadata,
                    reference_log_probs=reference_log_probs,
                    rewards=rewards,
                ),
                prompt_mask=sequence_batch.prompt_mask,
                response_mask=sequence_batch.response_mask,
            )

        return SequenceRolloutBatch(
            sequence_batch=sequence_batch,
            old_log_probs=old_log_probs,
        )

    def generate_grouped_rollout(
        self,
        request: SequenceGenerationRequest,
    ) -> GroupRolloutBatch[SequenceRolloutBatch]:
        """Generate a prompt-grouped rollout batch for GRPO-style trainers."""
        return GroupRolloutBatch(
            rollout=self.generate_rollout(request),
            group_size=request.num_generations,
        )
