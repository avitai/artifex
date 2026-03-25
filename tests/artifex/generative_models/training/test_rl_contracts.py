"""Tests for typed RL core contracts.

These tests define the stable RL batch and role contracts before the next
adapter/objective refactor wave builds on them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


class TestTrajectoryBatch:
    """Tests for the shared trajectory batch contract."""

    def test_trajectory_batch_tracks_batch_size(self) -> None:
        """TrajectoryBatch should expose a stable batch size."""
        from artifex.generative_models.training.rl import TrajectoryBatch

        batch = TrajectoryBatch(
            states=jnp.ones((3, 4)),
            actions=jnp.array([0, 1, 2]),
            rewards=jnp.array([1.0, 0.0, -1.0]),
            old_log_probs=jnp.array([-0.1, -0.2, -0.3]),
        )

        assert batch.batch_size == 3

    def test_trajectory_batch_rejects_misaligned_optional_fields(self) -> None:
        """TrajectoryBatch should reject optional fields with the wrong batch size."""
        from artifex.generative_models.training.rl import TrajectoryBatch

        with pytest.raises(ValueError, match="old_log_probs"):
            TrajectoryBatch(
                states=jnp.ones((3, 4)),
                actions=jnp.array([0, 1, 2]),
                old_log_probs=jnp.array([-0.1, -0.2]),
            )


class TestGeneratedBatch:
    """Tests for the generic generation/scoring batch."""

    def test_generated_batch_accepts_non_sequence_metadata(self) -> None:
        """GeneratedBatch should carry generic outputs and aligned metadata."""
        from artifex.generative_models.training.rl import GeneratedBatch

        batch = GeneratedBatch(
            outputs=jnp.ones((2, 8, 8, 3)),
            conditioning={"text_embeddings": jnp.ones((2, 77, 16))},
            output_mask=jnp.ones((2, 8, 8, 3)),
            generation_metadata={
                "timesteps": jnp.tile(jnp.arange(4), (2, 1)),
                "latent_trace": jnp.ones((2, 4, 8)),
            },
            rewards=jnp.array([1.0, 0.5]),
        )

        assert batch.batch_size == 2
        assert batch.generation_metadata is not None

    def test_generated_batch_rejects_misaligned_metadata(self) -> None:
        """GeneratedBatch should reject misaligned metadata leaves."""
        from artifex.generative_models.training.rl import GeneratedBatch

        with pytest.raises(ValueError, match="generation_metadata"):
            GeneratedBatch(
                outputs=jnp.ones((2, 8, 8, 3)),
                generation_metadata={"timesteps": jnp.ones((3, 4))},
            )


class TestGenerationRequest:
    """Tests for typed generation request contracts."""

    def test_generation_request_accepts_batch_aligned_conditioning(self) -> None:
        """GenerationRequest should preserve aligned conditioning payloads."""
        from artifex.generative_models.training.rl import GenerationRequest

        request = GenerationRequest(
            num_samples=2,
            conditioning={"text_embeddings": jnp.ones((2, 77, 16))},
            generation_metadata={"temperature_schedule": jnp.ones((2, 4))},
        )

        assert request.batch_size == 2
        assert request.conditioning is not None

    def test_generation_request_rejects_misaligned_conditioning(self) -> None:
        """GenerationRequest should reject conditioning with the wrong batch size."""
        from artifex.generative_models.training.rl import GenerationRequest

        with pytest.raises(ValueError, match="conditioning"):
            GenerationRequest(
                num_samples=2,
                conditioning={"text_embeddings": jnp.ones((3, 77, 16))},
            )


class TestSequenceGenerationRequest:
    """Tests for typed sequence generation requests."""

    def test_sequence_generation_request_tracks_total_samples(self) -> None:
        """SequenceGenerationRequest should expand prompt batches by num_generations."""
        from artifex.generative_models.training.rl import (
            GenerationRequest,
            SequenceGenerationRequest,
        )

        request = SequenceGenerationRequest(
            request=GenerationRequest(
                num_samples=2,
                conditioning={"prompt_embeddings": jnp.ones((2, 8))},
            ),
            prompts=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
            prompt_mask=jnp.array([[1, 1, 1], [1, 1, 1]], dtype=jnp.float32),
            num_generations=3,
            max_new_tokens=4,
        )

        assert request.batch_size == 2
        assert request.total_samples == 6
        assert request.prompt_length == 3

    def test_sequence_generation_request_rejects_bad_prompt_mask_shape(self) -> None:
        """SequenceGenerationRequest should require prompt-aligned prompt masks."""
        from artifex.generative_models.training.rl import (
            GenerationRequest,
            SequenceGenerationRequest,
        )

        with pytest.raises(ValueError, match="prompt_mask"):
            SequenceGenerationRequest(
                request=GenerationRequest(num_samples=2),
                prompts=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
                prompt_mask=jnp.array([[1, 1], [1, 1]], dtype=jnp.float32),
                num_generations=2,
                max_new_tokens=2,
            )


class TestIterativeGenerationRequest:
    """Tests for typed iterative generation requests."""

    def test_iterative_generation_request_tracks_num_steps(self) -> None:
        """IterativeGenerationRequest should expose batch size and step count."""
        from artifex.generative_models.training.rl import (
            GenerationRequest,
            IterativeGenerationRequest,
        )

        request = IterativeGenerationRequest(
            request=GenerationRequest(
                num_samples=2,
                conditioning={"latents": jnp.ones((2, 8, 8, 4))},
            ),
            step_indices=jnp.tile(jnp.arange(4), (2, 1)),
            trajectory_mask=jnp.ones((2, 4), dtype=jnp.float32),
        )

        assert request.batch_size == 2
        assert request.num_steps == 4

    def test_iterative_generation_request_rejects_bad_step_shape(self) -> None:
        """IterativeGenerationRequest should reject misaligned trajectory masks."""
        from artifex.generative_models.training.rl import (
            GenerationRequest,
            IterativeGenerationRequest,
        )

        with pytest.raises(ValueError, match="trajectory_mask"):
            IterativeGenerationRequest(
                request=GenerationRequest(num_samples=2),
                step_indices=jnp.tile(jnp.arange(4), (2, 1)),
                trajectory_mask=jnp.ones((2, 3), dtype=jnp.float32),
            )


class TestGeneratedSequenceBatch:
    """Tests for sequence-specific generation/scoring batches."""

    def test_generated_sequence_batch_accepts_response_mask(self) -> None:
        """GeneratedSequenceBatch should preserve response masks."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
        )

        batch = GeneratedSequenceBatch(
            generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3], [4, 5, 6]])),
            response_mask=jnp.array([[0, 1, 1], [0, 1, 1]]),
            prompt_mask=jnp.array([[1, 0, 0], [1, 0, 0]]),
        )

        assert batch.batch_size == 2
        assert batch.response_mask is not None
        assert batch.sequences.shape == (2, 3)

    def test_generated_sequence_batch_rejects_bad_mask_shape(self) -> None:
        """GeneratedSequenceBatch masks should match the sequence shape."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
        )

        with pytest.raises(ValueError, match="response_mask"):
            GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3], [4, 5, 6]])),
                response_mask=jnp.array([[1, 1], [1, 1]]),
            )


class TestIterativeGenerationBatch:
    """Tests for non-sequence iterative generation contracts."""

    def test_iterative_generation_batch_tracks_num_steps(self) -> None:
        """IterativeGenerationBatch should expose batch size and step count."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            IterativeGenerationBatch,
        )

        batch = IterativeGenerationBatch(
            generation=GeneratedBatch(outputs=jnp.ones((2, 8, 8, 3))),
            step_indices=jnp.tile(jnp.arange(4), (2, 1)),
            transition_log_probs=jnp.zeros((2, 4)),
            trajectory_mask=jnp.ones((2, 4)),
        )

        assert batch.batch_size == 2
        assert batch.num_steps == 4

    def test_iterative_generation_batch_rejects_bad_step_shape(self) -> None:
        """IterativeGenerationBatch should reject step-aligned shape drift."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            IterativeGenerationBatch,
        )

        with pytest.raises(ValueError, match="transition_log_probs"):
            IterativeGenerationBatch(
                generation=GeneratedBatch(outputs=jnp.ones((2, 8, 8, 3))),
                step_indices=jnp.tile(jnp.arange(4), (2, 1)),
                transition_log_probs=jnp.zeros((2, 3)),
            )


class TestSequenceRolloutBatch:
    """Tests for typed sequence rollout batches used by policy-gradient trainers."""

    def test_sequence_rollout_batch_tracks_action_shape(self) -> None:
        """SequenceRolloutBatch should expose batch size and action-token count."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            SequenceRolloutBatch,
        )

        batch = SequenceRolloutBatch(
            sequence_batch=GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])),
                response_mask=jnp.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=jnp.float32),
            ),
            old_log_probs=jnp.zeros((2, 3)),
            returns=jnp.ones((2, 3)),
            advantages=jnp.ones((2, 3)),
        )

        assert batch.batch_size == 2
        assert batch.num_action_tokens == 3
        assert batch.action_mask.shape == (2, 3)

    def test_sequence_rollout_batch_rejects_misaligned_action_fields(self) -> None:
        """SequenceRolloutBatch should reject action-aligned shape drift."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            SequenceRolloutBatch,
        )

        with pytest.raises(ValueError, match="old_log_probs"):
            SequenceRolloutBatch(
                sequence_batch=GeneratedSequenceBatch(
                    generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])),
                ),
                old_log_probs=jnp.zeros((2, 2)),
            )

    def test_sequence_rollout_batch_is_jax_pytree(self) -> None:
        """SequenceRolloutBatch should flatten into JAX leaves instead of one opaque object."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            SequenceRolloutBatch,
        )

        batch = SequenceRolloutBatch(
            sequence_batch=GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])),
                response_mask=jnp.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=jnp.float32),
            ),
            old_log_probs=jnp.zeros((2, 3)),
            returns=jnp.ones((2, 3)),
            advantages=jnp.ones((2, 3)),
        )

        leaves = jax.tree_util.tree_leaves(batch)

        assert leaves
        assert all(not isinstance(leaf, SequenceRolloutBatch) for leaf in leaves)
        assert any(getattr(leaf, "shape", None) == (2, 4) for leaf in leaves)


class TestPreferenceBatch:
    """Tests for chosen/rejected preference batches."""

    def test_preference_batch_requires_matching_batch_size(self) -> None:
        """PreferenceBatch should keep chosen/rejected batches aligned."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            PreferenceBatch,
        )

        chosen = GeneratedSequenceBatch(
            generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3], [4, 5, 6]]))
        )
        rejected = GeneratedSequenceBatch(generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3]])))

        with pytest.raises(ValueError, match="batch size"):
            PreferenceBatch(chosen=chosen, rejected=rejected)

    def test_preference_batch_supports_non_sequence_generation_batches(self) -> None:
        """PreferenceBatch should not be sequence-only."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            PreferenceBatch,
        )

        batch = PreferenceBatch(
            chosen=GeneratedBatch(outputs=jnp.ones((2, 8, 8, 3))),
            rejected=GeneratedBatch(outputs=jnp.zeros((2, 8, 8, 3))),
        )

        assert batch.batch_size == 2


class TestGroupRolloutBatch:
    """Tests for grouped rollout batches."""

    def test_group_rollout_batch_requires_divisible_batch_size(self) -> None:
        """GroupRolloutBatch should require batch size divisibility by group size."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            GroupRolloutBatch,
            SequenceRolloutBatch,
        )

        rollout = SequenceRolloutBatch(
            sequence_batch=GeneratedSequenceBatch(
                generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
            ),
            old_log_probs=jnp.zeros((3, 2)),
        )

        with pytest.raises(ValueError, match="divisible"):
            GroupRolloutBatch(rollout=rollout, group_size=2)

    def test_group_rollout_batch_supports_jitted_group_reshaping(self) -> None:
        """GroupRolloutBatch should keep group_size static enough for JIT-compiled grouping."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            GroupRolloutBatch,
            SequenceRolloutBatch,
        )

        grouped = GroupRolloutBatch(
            rollout=SequenceRolloutBatch(
                sequence_batch=GeneratedSequenceBatch(
                    generation=GeneratedBatch(outputs=jnp.array([[1, 2, 3], [4, 5, 6]])),
                    response_mask=jnp.array([[0, 1, 1], [0, 1, 1]], dtype=jnp.float32),
                ),
                old_log_probs=jnp.zeros((2, 2)),
            ),
            group_size=2,
        )

        @jax.jit
        def grouped_action_sums(batch: GroupRolloutBatch) -> jax.Array:
            grouped_mask = batch.rollout.action_mask.reshape(
                batch.num_groups,
                batch.group_size,
                batch.rollout.num_action_tokens,
            )
            return jnp.sum(grouped_mask, axis=-1)

        action_sums = grouped_action_sums(grouped)

        assert action_sums.shape == (1, 2)
        assert jnp.allclose(action_sums, jnp.array([[2.0, 2.0]]))


class TestRLProtocols:
    """Tests for the shared RL role protocols."""

    def test_runtime_checkable_protocols_are_structural(self) -> None:
        """RL protocols should work with structural typing."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            GroupRolloutBatch,
            IterativeGenerationBackend,
            IterativeGenerationBatch,
            IterativeGenerationRequest,
            IterativePolicyAdapter,
            LogProbScorer,
            PolicyAdapter,
            RewardModel,
            SequenceGenerationBackend,
            SequenceGenerationRequest,
            SequenceRolloutBackend,
            SequenceRolloutBatch,
            SequenceRolloutPolicyAdapter,
            SequenceRolloutValueAdapter,
            TrajectoryBatch,
            ValueAdapter,
        )

        class DummyPolicy:
            def action_log_probs(self, batch: TrajectoryBatch):
                return jnp.zeros((batch.batch_size,))

        class DummyScorer:
            def sequence_log_probs(self, batch: GeneratedSequenceBatch):
                return jnp.zeros((batch.batch_size,))

        class DummyIterativePolicy:
            def transition_log_probs(self, batch: IterativeGenerationBatch):
                return jnp.zeros(batch.step_indices.shape)

        class DummyValue:
            def state_values(self, batch: TrajectoryBatch):
                return jnp.zeros((batch.batch_size,))

        class DummySequenceGenerationBackend:
            def generate_sequences(self, request: SequenceGenerationRequest):
                return GeneratedSequenceBatch(
                    generation=GeneratedBatch(outputs=jnp.zeros((request.total_samples, 2))),
                )

        class DummyIterativeGenerationBackend:
            def generate_iterations(self, request: IterativeGenerationRequest):
                return IterativeGenerationBatch(
                    generation=GeneratedBatch(outputs=jnp.zeros((request.batch_size, 4, 4, 3))),
                    step_indices=request.step_indices,
                )

        class DummySequenceRolloutPolicy:
            def action_log_probs(self, batch: SequenceRolloutBatch):
                return jnp.zeros(batch.action_mask.shape)

            def log_prob_distributions(self, batch: SequenceRolloutBatch):
                return jnp.zeros((*batch.action_mask.shape, 4))

        class DummySequenceRolloutBackend:
            def generate_rollout(self, request: SequenceGenerationRequest):
                sequence_batch = GeneratedSequenceBatch(
                    generation=GeneratedBatch(outputs=jnp.zeros((request.total_samples, 2))),
                )
                return SequenceRolloutBatch(sequence_batch=sequence_batch)

            def generate_grouped_rollout(self, request: SequenceGenerationRequest):
                return GroupRolloutBatch(
                    rollout=self.generate_rollout(request),
                    group_size=request.num_generations,
                )

        class DummySequenceRolloutValue:
            def action_values(self, batch: SequenceRolloutBatch):
                return jnp.zeros(batch.action_mask.shape)

        class DummyReward:
            def score_generations(self, batch: GeneratedBatch):
                return jnp.zeros((batch.batch_size,))

        assert isinstance(DummyPolicy(), PolicyAdapter)
        assert isinstance(DummyScorer(), LogProbScorer)
        assert isinstance(DummyIterativePolicy(), IterativePolicyAdapter)
        assert isinstance(DummyValue(), ValueAdapter)
        assert isinstance(DummySequenceGenerationBackend(), SequenceGenerationBackend)
        assert isinstance(DummyIterativeGenerationBackend(), IterativeGenerationBackend)
        assert isinstance(DummySequenceRolloutPolicy(), SequenceRolloutPolicyAdapter)
        assert isinstance(DummySequenceRolloutBackend(), SequenceRolloutBackend)
        assert isinstance(DummySequenceRolloutValue(), SequenceRolloutValueAdapter)
        assert isinstance(DummyReward(), RewardModel)


class TestRLCoreExports:
    """Tests for exports of the RL core contracts."""

    def test_rl_module_exports_typed_contracts(self) -> None:
        """The RL package should export typed batches and protocols."""
        from artifex.generative_models.training.rl import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            GenerationRequest,
            GroupRolloutBatch,
            IterativeGenerationBackend,
            IterativeGenerationBatch,
            IterativeGenerationRequest,
            IterativePolicyAdapter,
            LocalSequenceGenerationBackend,
            LocalSequenceRolloutBackend,
            LogProbScorer,
            PolicyAdapter,
            PreferenceBatch,
            RewardModel,
            SequenceGenerationBackend,
            SequenceGenerationRequest,
            SequenceRolloutBackend,
            SequenceRolloutBatch,
            SequenceRolloutPolicyAdapter,
            SequenceRolloutValueAdapter,
            TrajectoryBatch,
            ValueAdapter,
        )

        assert TrajectoryBatch is not None
        assert GenerationRequest is not None
        assert GeneratedBatch is not None
        assert GeneratedSequenceBatch is not None
        assert SequenceRolloutBatch is not None
        assert PreferenceBatch is not None
        assert GroupRolloutBatch is not None
        assert SequenceGenerationRequest is not None
        assert IterativeGenerationRequest is not None
        assert IterativeGenerationBatch is not None
        assert PolicyAdapter is not None
        assert IterativePolicyAdapter is not None
        assert LogProbScorer is not None
        assert ValueAdapter is not None
        assert SequenceGenerationBackend is not None
        assert IterativeGenerationBackend is not None
        assert SequenceRolloutPolicyAdapter is not None
        assert SequenceRolloutBackend is not None
        assert SequenceRolloutValueAdapter is not None
        assert RewardModel is not None
        assert LocalSequenceGenerationBackend is not None
        assert LocalSequenceRolloutBackend is not None

    def test_main_training_module_exports_typed_contracts(self) -> None:
        """The main training package should re-export RL core contracts."""
        from artifex.generative_models.training import (
            GeneratedBatch,
            GeneratedSequenceBatch,
            GenerationRequest,
            GroupRolloutBatch,
            IterativeGenerationBackend,
            IterativeGenerationBatch,
            IterativeGenerationRequest,
            IterativePolicyAdapter,
            LocalSequenceGenerationBackend,
            LocalSequenceRolloutBackend,
            LogProbScorer,
            PolicyAdapter,
            PreferenceBatch,
            RewardModel,
            SequenceGenerationBackend,
            SequenceGenerationRequest,
            SequenceRolloutBackend,
            SequenceRolloutBatch,
            SequenceRolloutPolicyAdapter,
            SequenceRolloutValueAdapter,
            TrajectoryBatch,
            ValueAdapter,
        )

        assert TrajectoryBatch is not None
        assert GenerationRequest is not None
        assert GeneratedBatch is not None
        assert GeneratedSequenceBatch is not None
        assert SequenceRolloutBatch is not None
        assert PreferenceBatch is not None
        assert GroupRolloutBatch is not None
        assert SequenceGenerationRequest is not None
        assert IterativeGenerationRequest is not None
        assert IterativeGenerationBatch is not None
        assert PolicyAdapter is not None
        assert IterativePolicyAdapter is not None
        assert LogProbScorer is not None
        assert ValueAdapter is not None
        assert SequenceGenerationBackend is not None
        assert IterativeGenerationBackend is not None
        assert SequenceRolloutPolicyAdapter is not None
        assert SequenceRolloutBackend is not None
        assert SequenceRolloutValueAdapter is not None
        assert RewardModel is not None
        assert LocalSequenceGenerationBackend is not None
        assert LocalSequenceRolloutBackend is not None
