"""Protocol definitions for typed RL core roles."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax

from artifex.generative_models.training.rl.types import (
    GeneratedBatch,
    GeneratedSequenceBatch,
    GroupRolloutBatch,
    IterativeGenerationBatch,
    IterativeGenerationRequest,
    SequenceGenerationRequest,
    SequenceRolloutBatch,
    TrajectoryBatch,
)


@runtime_checkable
class PolicyAdapter(Protocol):
    """Protocol for policy adapters that score trajectory actions."""

    def action_log_probs(self, batch: TrajectoryBatch) -> jax.Array:
        """Return log probabilities aligned with the trajectory batch."""
        ...


@runtime_checkable
class LogProbScorer(Protocol):
    """Protocol for components that score prompt-conditioned sequences."""

    def sequence_log_probs(self, batch: GeneratedSequenceBatch) -> jax.Array:
        """Return one sequence score per batch element."""
        ...


@runtime_checkable
class SequenceGenerationBackend(Protocol):
    """Protocol for prompt-conditioned local or remote sequence generation."""

    def generate_sequences(self, request: SequenceGenerationRequest) -> GeneratedSequenceBatch:
        """Return generated sequences plus prompt/response masks."""
        ...


@runtime_checkable
class SequenceGeneratingModule(Protocol):
    """Protocol for local modules that generate token sequences."""

    def generate(self, total_samples: int, **kwargs: Any) -> jax.Array:
        """Generate ``total_samples`` sequences from keyword generation controls."""
        ...


@runtime_checkable
class IterativeGenerationBackend(Protocol):
    """Protocol for non-sequence iterative generation backends."""

    def generate_iterations(self, request: IterativeGenerationRequest) -> IterativeGenerationBatch:
        """Return iterative trajectories aligned with the provided step schedule."""
        ...


@runtime_checkable
class IterativePolicyAdapter(Protocol):
    """Protocol for iterative generators such as diffusion alignment policies."""

    def transition_log_probs(self, batch: IterativeGenerationBatch) -> jax.Array:
        """Return transition log-probabilities aligned with the iterative batch."""
        ...


@runtime_checkable
class SequenceRolloutPolicyAdapter(Protocol):
    """Protocol for sequence-policy adapters used by policy-gradient trainers."""

    def action_log_probs(self, batch: SequenceRolloutBatch) -> jax.Array:
        """Return action-aligned log-probabilities for sampled tokens."""
        ...

    def log_prob_distributions(self, batch: SequenceRolloutBatch) -> jax.Array:
        """Return action-aligned token distributions for entropy or KL terms."""
        ...


@runtime_checkable
class SequenceRolloutBackend(Protocol):
    """Protocol for backends that assemble typed sequence rollouts."""

    def generate_rollout(self, request: SequenceGenerationRequest) -> SequenceRolloutBatch:
        """Return a rollout batch ready for sequence policy-gradient trainers."""
        ...

    def generate_grouped_rollout(
        self,
        request: SequenceGenerationRequest,
    ) -> GroupRolloutBatch[SequenceRolloutBatch]:
        """Return a rollout batch grouped by prompt for GRPO-style algorithms."""
        ...


@runtime_checkable
class ValueAdapter(Protocol):
    """Protocol for value-function adapters used by actor-critic objectives."""

    def state_values(self, batch: TrajectoryBatch) -> jax.Array:
        """Return one value prediction per trajectory element."""
        ...


@runtime_checkable
class SequenceRolloutValueAdapter(Protocol):
    """Protocol for sequence value-head adapters used by actor-critic losses."""

    def action_values(self, batch: SequenceRolloutBatch) -> jax.Array:
        """Return action-aligned value estimates for rollout tokens."""
        ...


@runtime_checkable
class RewardModel(Protocol):
    """Protocol for reward models that score generic generated batches."""

    def score_generations(self, batch: GeneratedBatch) -> jax.Array:
        """Return one reward per generated sample."""
        ...
