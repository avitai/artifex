"""Tests for RL generation and rollout backends."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


VOCAB_SIZE = 32


class SimpleSequenceGenerator(nnx.Module):
    """Small sequence model exposing both forward scoring and generation."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.embedding = nnx.Embed(VOCAB_SIZE, 8, rngs=rngs)
        self.hidden = nnx.Linear(8, 8, rngs=rngs)
        self.output = nnx.Linear(8, VOCAB_SIZE, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Return per-token logits for the provided token sequences."""
        embeddings = self.embedding(tokens)
        hidden = nnx.relu(self.hidden(embeddings))
        return self.output(hidden)

    def generate(
        self,
        n_samples: int = 1,
        *,
        prompt_tokens: jax.Array | None = None,
        max_new_tokens: int = 2,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> jax.Array:
        """Append deterministic continuations to the provided prompts."""
        del temperature, top_k, top_p

        if prompt_tokens is None:
            prompt_tokens = jnp.zeros((n_samples, 1), dtype=jnp.int32)

        if prompt_tokens.shape[0] != n_samples:
            msg = "prompt_tokens must match n_samples"
            raise ValueError(msg)

        continuation = (
            prompt_tokens[:, -1:] + jnp.arange(1, max_new_tokens + 1, dtype=jnp.int32)
        ) % VOCAB_SIZE
        return jnp.concatenate([prompt_tokens, continuation], axis=1)


class LastTokenReward:
    """Reward model that scores sequences by their last generated token."""

    def score_generations(self, batch) -> jax.Array:
        return batch.outputs[:, -1].astype(jnp.float32)


@pytest.fixture
def generator_model() -> SimpleSequenceGenerator:
    """Create a sequence generator for backend tests."""
    return SimpleSequenceGenerator(rngs=nnx.Rngs(0))


class TestSequenceGenerationBackend:
    """Tests for typed sequence generation backends."""

    def test_local_sequence_generation_backend_repeats_prompts_and_conditioning(
        self,
        generator_model: SimpleSequenceGenerator,
    ) -> None:
        """The local generation backend should expand prompt batches by group count."""
        from artifex.generative_models.training.rl import (
            GenerationRequest,
            LocalSequenceGenerationBackend,
            SequenceGenerationBackend,
            SequenceGenerationRequest,
        )

        backend = LocalSequenceGenerationBackend(generator_model)
        request = SequenceGenerationRequest(
            request=GenerationRequest(
                num_samples=2,
                conditioning={"prompt_embedding": jnp.arange(8, dtype=jnp.float32).reshape(2, 4)},
            ),
            prompts=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
            prompt_mask=jnp.ones((2, 3), dtype=jnp.float32),
            num_generations=2,
            max_new_tokens=2,
        )

        generated = backend.generate_sequences(request)

        assert isinstance(backend, SequenceGenerationBackend)
        assert generated.sequences.shape == (4, 5)
        assert generated.prompt_mask is not None
        assert generated.response_mask is not None
        assert jnp.array_equal(
            generated.prompt_mask,
            jnp.array(
                [
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                ],
                dtype=jnp.float32,
            ),
        )
        assert jnp.array_equal(
            generated.response_mask,
            jnp.array(
                [
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1],
                ],
                dtype=jnp.float32,
            ),
        )
        assert generated.conditioning is not None
        assert generated.conditioning["prompt_embedding"].shape == (4, 4)


class TestSequenceRolloutBackend:
    """Tests for typed sequence rollout backends."""

    def test_local_sequence_rollout_backend_attaches_old_log_probs_and_rewards(
        self,
        generator_model: SimpleSequenceGenerator,
    ) -> None:
        """The rollout backend should assemble the typed rollout batch used by trainers."""
        from artifex.generative_models.training.rl import (
            GenerationRequest,
            LocalSequenceGenerationBackend,
            LocalSequenceRolloutBackend,
            SequenceGenerationRequest,
            SequencePolicyAdapter,
            SequenceRolloutBackend,
        )

        generation_backend = LocalSequenceGenerationBackend(generator_model)
        rollout_backend = LocalSequenceRolloutBackend(
            generation_backend,
            SequencePolicyAdapter(generator_model),
            reward_model=LastTokenReward(),
            reference_scorer=SequencePolicyAdapter(SimpleSequenceGenerator(rngs=nnx.Rngs(1))),
        )
        request = SequenceGenerationRequest(
            request=GenerationRequest(num_samples=2),
            prompts=jnp.array([[1, 2, 3], [7, 8, 9]], dtype=jnp.int32),
            prompt_mask=jnp.ones((2, 3), dtype=jnp.float32),
            num_generations=2,
            max_new_tokens=2,
        )

        rollout = rollout_backend.generate_rollout(request)

        assert isinstance(rollout_backend, SequenceRolloutBackend)
        assert rollout.sequences.shape == (4, 5)
        assert rollout.old_log_probs is not None
        assert rollout.old_log_probs.shape == (4, 4)
        assert rollout.sequence_rewards is not None
        assert jnp.array_equal(
            rollout.sequence_rewards, rollout.sequences[:, -1].astype(jnp.float32)
        )
        assert rollout.sequence_batch.reference_log_probs is not None
        assert rollout.sequence_batch.reference_log_probs.shape == (4,)

    def test_local_sequence_rollout_backend_builds_grouped_rollouts(
        self,
        generator_model: SimpleSequenceGenerator,
    ) -> None:
        """The rollout backend should expose prompt-grouped rollouts for GRPO-style trainers."""
        from artifex.generative_models.training.rl import (
            GenerationRequest,
            LocalSequenceGenerationBackend,
            LocalSequenceRolloutBackend,
            SequenceGenerationRequest,
            SequencePolicyAdapter,
        )

        rollout_backend = LocalSequenceRolloutBackend(
            LocalSequenceGenerationBackend(generator_model),
            SequencePolicyAdapter(generator_model),
            reward_model=LastTokenReward(),
        )
        request = SequenceGenerationRequest(
            request=GenerationRequest(num_samples=2),
            prompts=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
            prompt_mask=jnp.ones((2, 3), dtype=jnp.float32),
            num_generations=3,
            max_new_tokens=2,
        )

        grouped = rollout_backend.generate_grouped_rollout(request)

        assert grouped.group_size == 3
        assert grouped.num_groups == 2
        assert grouped.rollout.batch_size == 6
