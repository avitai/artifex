"""Typed request and batch contracts for reinforcement-learning training.

These dataclasses provide a stable, explicit contract for RL requests and
batches so Artifex does not keep extending trainer-local ``dict[str, Array]``
schemas.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
from jax import tree_util


def _validate_batch_aligned(name: str, value: jax.Array | None, batch_size: int) -> None:
    """Require optional batch-aligned tensors to match the leading batch size."""
    if value is None:
        return
    if value.shape[0] != batch_size:
        msg = f"{name} must match batch size {batch_size}, got {value.shape[0]}"
        raise ValueError(msg)


def _validate_exact_shape(
    name: str,
    value: jax.Array | None,
    expected_shape: tuple[int, ...],
) -> None:
    """Require optional tensors to match an expected shape exactly."""
    if value is None:
        return
    if value.shape != expected_shape:
        msg = f"{name} must match shape {expected_shape}, got {value.shape}"
        raise ValueError(msg)


def _validate_pytree_batch_aligned(name: str, value: Any | None, batch_size: int) -> None:
    """Require pytree leaves to share the leading batch dimension."""
    if value is None:
        return

    leaves = tree_util.tree_leaves(value)
    if not leaves:
        return

    for index, leaf in enumerate(leaves):
        if not hasattr(leaf, "shape"):
            msg = f"{name} leaf {index} must expose a shape for batch alignment"
            raise TypeError(msg)
        if leaf.shape[0] != batch_size:
            msg = f"{name} leaf {index} must match batch size {batch_size}, got {leaf.shape[0]}"
            raise ValueError(msg)


def _resolve_batch_size(name: str, value: Any) -> int:
    """Read ``batch_size`` from batch-like contracts used in generic wrappers."""
    batch_size = getattr(value, "batch_size", None)
    if batch_size is None:
        msg = f"{name} must expose a batch_size property"
        raise TypeError(msg)
    return int(batch_size)


def canonicalize_response_mask(
    sequences: jax.Array,
    response_mask: jax.Array | None,
) -> jax.Array | None:
    """Convert supported response-mask shapes into sequence-aligned masks."""
    if response_mask is None:
        return None

    if response_mask.shape == sequences.shape:
        return response_mask.astype(jnp.float32)

    action_shape = sequences[:, 1:].shape
    if response_mask.shape == action_shape:
        prefix = jnp.zeros((response_mask.shape[0], 1), dtype=response_mask.dtype)
        return jnp.concatenate([prefix, response_mask], axis=1).astype(jnp.float32)

    msg = (
        "response_mask must have shape (batch, seq_len) or "
        "(batch, seq_len - 1) to match the provided sequences"
    )
    raise ValueError(msg)


def prepare_autoregressive_token_mask(
    sequences: jax.Array,
    response_mask: jax.Array | None,
) -> jax.Array:
    """Return an action-token mask aligned with ``sequences[:, 1:]``."""
    if response_mask is None:
        return jnp.ones(sequences[:, 1:].shape, dtype=jnp.float32)

    sequence_mask = canonicalize_response_mask(sequences, response_mask)
    if sequence_mask is None:
        msg = "canonicalize_response_mask returned None for a non-null response_mask"
        raise TypeError(msg)
    return sequence_mask[:, 1:].astype(jnp.float32)


@nnx.dataclass
class TrajectoryBatch(nnx.Pytree):
    """Batch contract for stepwise policy-gradient trajectories."""

    states: jax.Array = nnx.data()
    actions: jax.Array = nnx.data()
    rewards: jax.Array | None = nnx.data(default=None)
    old_log_probs: jax.Array | None = nnx.data(default=None)
    returns: jax.Array | None = nnx.data(default=None)
    advantages: jax.Array | None = nnx.data(default=None)
    dones: jax.Array | None = nnx.data(default=None)

    def __post_init__(self) -> None:
        """Validate transition batch invariants."""
        batch_size = self.batch_size
        if self.actions.shape[0] != batch_size:
            msg = (
                "actions must match the leading batch dimension of states: "
                f"{self.actions.shape[0]} != {batch_size}"
            )
            raise ValueError(msg)

        _validate_batch_aligned("rewards", self.rewards, batch_size)
        _validate_batch_aligned("old_log_probs", self.old_log_probs, batch_size)
        _validate_batch_aligned("returns", self.returns, batch_size)
        _validate_batch_aligned("advantages", self.advantages, batch_size)
        _validate_batch_aligned("dones", self.dones, batch_size)

    @property
    def batch_size(self) -> int:
        """Leading batch size shared by the trajectory tensors."""
        return int(self.states.shape[0])


@nnx.dataclass
class GenerationRequest(nnx.Pytree):
    """Generic generation request for backend-owned rollout or sampling."""

    num_samples: int = nnx.static(default=1)
    conditioning: Any | None = nnx.data(default=None)
    generation_metadata: Any | None = nnx.data(default=None)

    def __post_init__(self) -> None:
        """Validate generation request invariants."""
        if self.num_samples <= 0:
            msg = "num_samples must be positive"
            raise ValueError(msg)

        _validate_pytree_batch_aligned("conditioning", self.conditioning, self.num_samples)
        _validate_pytree_batch_aligned(
            "generation_metadata",
            self.generation_metadata,
            self.num_samples,
        )

    @property
    def batch_size(self) -> int:
        """Number of requested samples before backend-level expansion."""
        return self.num_samples


@nnx.dataclass
class SequenceGenerationRequest(nnx.Pytree):
    """Sequence-specific generation request for prompt-conditioned rollouts."""

    request: GenerationRequest = nnx.data()
    prompts: jax.Array | None = nnx.data(default=None)
    prompt_mask: jax.Array | None = nnx.data(default=None)
    num_generations: int = nnx.static(default=1)
    max_new_tokens: int = nnx.static(default=1)
    temperature: float = nnx.static(default=1.0)
    top_k: int | None = nnx.static(default=None)
    top_p: float | None = nnx.static(default=None)

    def __post_init__(self) -> None:
        """Validate grouped generation request invariants."""
        if self.num_generations <= 0:
            msg = "num_generations must be positive"
            raise ValueError(msg)
        if self.max_new_tokens <= 0:
            msg = "max_new_tokens must be positive"
            raise ValueError(msg)

        if self.prompts is None:
            if self.prompt_mask is not None:
                msg = "prompt_mask requires prompts to be present"
                raise ValueError(msg)
            return

        if self.prompts.ndim != 2:
            msg = "prompts must have shape (batch, prompt_len)"
            raise ValueError(msg)
        if self.prompts.shape[0] != self.request.batch_size:
            msg = (
                "prompts must match request.num_samples on the leading batch dimension: "
                f"{self.prompts.shape[0]} != {self.request.batch_size}"
            )
            raise ValueError(msg)

        _validate_exact_shape("prompt_mask", self.prompt_mask, self.prompts.shape)

    @property
    def batch_size(self) -> int:
        """Number of prompts or unconditional samples in the request."""
        return int(self.prompts.shape[0]) if self.prompts is not None else self.request.batch_size

    @property
    def total_samples(self) -> int:
        """Total samples after expanding each prompt by ``num_generations``."""
        return self.batch_size * self.num_generations

    @property
    def prompt_length(self) -> int:
        """Length of the provided prompts, or zero for unconditional generation."""
        return int(self.prompts.shape[1]) if self.prompts is not None else 0

    @property
    def conditioning(self) -> Any | None:
        """Optional conditioning payload delegated from the generic request."""
        return self.request.conditioning

    @property
    def generation_metadata(self) -> Any | None:
        """Optional generation metadata delegated from the generic request."""
        return self.request.generation_metadata


@nnx.dataclass
class IterativeGenerationRequest(nnx.Pytree):
    """Iterative generation request for diffusion-like rollout backends."""

    request: GenerationRequest = nnx.data()
    step_indices: jax.Array = nnx.data()
    trajectory_mask: jax.Array | None = nnx.data(default=None)

    def __post_init__(self) -> None:
        """Validate trajectory batch invariants."""
        if self.step_indices.ndim != 2:
            msg = "step_indices must have shape (batch, num_steps)"
            raise ValueError(msg)
        if self.step_indices.shape[0] != self.request.batch_size:
            msg = (
                "step_indices must match request.num_samples on the leading batch "
                f"dimension: {self.step_indices.shape[0]} != {self.request.batch_size}"
            )
            raise ValueError(msg)

        _validate_exact_shape("trajectory_mask", self.trajectory_mask, self.step_indices.shape)

    @property
    def batch_size(self) -> int:
        """Leading batch size for iterative generation requests."""
        return self.request.batch_size

    @property
    def num_steps(self) -> int:
        """Number of requested iterative-generation steps."""
        return int(self.step_indices.shape[1])

    @property
    def conditioning(self) -> Any | None:
        """Optional conditioning payload delegated from the generic request."""
        return self.request.conditioning

    @property
    def generation_metadata(self) -> Any | None:
        """Optional generation metadata delegated from the generic request."""
        return self.request.generation_metadata


@nnx.dataclass
class GeneratedBatch(nnx.Pytree):
    """Generic batch contract for generated artifacts across model families."""

    outputs: jax.Array = nnx.data()
    conditioning: Any | None = nnx.data(default=None)
    output_mask: jax.Array | None = nnx.data(default=None)
    generation_metadata: Any | None = nnx.data(default=None)
    old_log_probs: jax.Array | None = nnx.data(default=None)
    reference_log_probs: jax.Array | None = nnx.data(default=None)
    rewards: jax.Array | None = nnx.data(default=None)

    def __post_init__(self) -> None:
        """Validate generated sample batch invariants."""
        if self.outputs.ndim < 1:
            msg = "outputs must have at least a leading batch dimension"
            raise ValueError(msg)

        batch_size = self.batch_size

        _validate_exact_shape("output_mask", self.output_mask, self.outputs.shape)
        _validate_pytree_batch_aligned("conditioning", self.conditioning, batch_size)
        _validate_pytree_batch_aligned(
            "generation_metadata",
            self.generation_metadata,
            batch_size,
        )
        _validate_batch_aligned("old_log_probs", self.old_log_probs, batch_size)
        _validate_batch_aligned("reference_log_probs", self.reference_log_probs, batch_size)
        _validate_batch_aligned("rewards", self.rewards, batch_size)

    @property
    def batch_size(self) -> int:
        """Leading batch size shared by the generated outputs."""
        return int(self.outputs.shape[0])


@nnx.dataclass
class GeneratedSequenceBatch(nnx.Pytree):
    """Sequence-specific wrapper over the generic generated batch contract."""

    generation: GeneratedBatch = nnx.data()
    prompt_mask: jax.Array | None = nnx.data(default=None)
    response_mask: jax.Array | None = nnx.data(default=None)

    def __post_init__(self) -> None:
        """Validate generated sequence batch invariants."""
        if self.sequences.ndim < 2:
            msg = "sequences must be at least 2D with shape (batch, seq_len, ...)"
            raise ValueError(msg)

        sequence_shape = self.sequences.shape

        _validate_exact_shape("prompt_mask", self.prompt_mask, sequence_shape)
        _validate_exact_shape("response_mask", self.response_mask, sequence_shape)

    @classmethod
    def from_sequences(
        cls,
        sequences: jax.Array,
        *,
        conditioning: Any | None = None,
        generation_metadata: Any | None = None,
        old_log_probs: jax.Array | None = None,
        reference_log_probs: jax.Array | None = None,
        rewards: jax.Array | None = None,
        prompt_mask: jax.Array | None = None,
        response_mask: jax.Array | None = None,
    ) -> GeneratedSequenceBatch:
        """Convenience constructor for sequence-native call sites."""
        return cls(
            generation=GeneratedBatch(
                outputs=sequences,
                conditioning=conditioning,
                generation_metadata=generation_metadata,
                old_log_probs=old_log_probs,
                reference_log_probs=reference_log_probs,
                rewards=rewards,
            ),
            prompt_mask=prompt_mask,
            response_mask=response_mask,
        )

    @property
    def batch_size(self) -> int:
        """Leading batch size shared by the generated sequences."""
        return self.generation.batch_size

    @property
    def sequences(self) -> jax.Array:
        """Sequence outputs aligned with the generic generation contract."""
        return self.generation.outputs

    @property
    def conditioning(self) -> Any | None:
        """Optional conditioning payload from the generic generation contract."""
        return self.generation.conditioning

    @property
    def generation_metadata(self) -> Any | None:
        """Optional generation metadata from the generic generation contract."""
        return self.generation.generation_metadata

    @property
    def old_log_probs(self) -> jax.Array | None:
        """Optional old-log-prob tracking from the generic generation contract."""
        return self.generation.old_log_probs

    @property
    def reference_log_probs(self) -> jax.Array | None:
        """Optional reference log-prob tracking from the generic generation contract."""
        return self.generation.reference_log_probs

    @property
    def rewards(self) -> jax.Array | None:
        """Optional rewards from the generic generation contract."""
        return self.generation.rewards


@nnx.dataclass
class SequenceRolloutBatch(nnx.Pytree):
    """Typed autoregressive rollout contract for sequence policy-gradient trainers."""

    sequence_batch: GeneratedSequenceBatch = nnx.data()
    old_log_probs: jax.Array | None = nnx.data(default=None)
    token_rewards: jax.Array | None = nnx.data(default=None)
    returns: jax.Array | None = nnx.data(default=None)
    advantages: jax.Array | None = nnx.data(default=None)
    dones: jax.Array | None = nnx.data(default=None)

    def __post_init__(self) -> None:
        """Validate sequence rollout batch invariants."""
        if self.sequences.ndim != 2:
            msg = "SequenceRolloutBatch expects 2D token sequences with shape (batch, seq_len)"
            raise ValueError(msg)

        action_shape = self.action_shape
        _validate_exact_shape("old_log_probs", self.old_log_probs, action_shape)
        _validate_exact_shape("token_rewards", self.token_rewards, action_shape)
        _validate_exact_shape("returns", self.returns, action_shape)
        _validate_exact_shape("advantages", self.advantages, action_shape)
        _validate_exact_shape("dones", self.dones, action_shape)

    @property
    def batch_size(self) -> int:
        """Leading batch size shared by the rollout batch."""
        return self.sequence_batch.batch_size

    @property
    def sequences(self) -> jax.Array:
        """Generated token sequences used for autoregressive rollout scoring."""
        return self.sequence_batch.sequences

    @property
    def prompt_mask(self) -> jax.Array | None:
        """Optional prompt-token mask."""
        return self.sequence_batch.prompt_mask

    @property
    def response_mask(self) -> jax.Array | None:
        """Optional response-token mask."""
        return self.sequence_batch.response_mask

    @property
    def sequence_rewards(self) -> jax.Array | None:
        """Optional per-sequence rewards stored on the generated batch."""
        return self.sequence_batch.rewards

    @property
    def action_shape(self) -> tuple[int, int]:
        """Shape aligned with autoregressive next-token actions."""
        shape = self.sequences[:, 1:].shape
        return int(shape[0]), int(shape[1])

    @property
    def num_action_tokens(self) -> int:
        """Number of autoregressive action slots per sequence."""
        return int(self.action_shape[1])

    @property
    def action_mask(self) -> jax.Array:
        """Action-token mask aligned with next-token log-prob tensors."""
        return prepare_autoregressive_token_mask(self.sequences, self.response_mask)


@nnx.dataclass
class IterativeGenerationBatch(nnx.Pytree):
    """Generic contract for iterative generators such as diffusion models."""

    generation: GeneratedBatch = nnx.data()
    step_indices: jax.Array = nnx.data()
    transition_log_probs: jax.Array | None = nnx.data(default=None)
    reference_transition_log_probs: jax.Array | None = nnx.data(default=None)
    trajectory_mask: jax.Array | None = nnx.data(default=None)

    def __post_init__(self) -> None:
        """Validate grouped rollout batch invariants."""
        if self.step_indices.ndim != 2:
            msg = "step_indices must have shape (batch, num_steps)"
            raise ValueError(msg)
        if self.step_indices.shape[0] != self.batch_size:
            msg = (
                "step_indices must match the generation batch size: "
                f"{self.step_indices.shape[0]} != {self.batch_size}"
            )
            raise ValueError(msg)

        step_shape = self.step_indices.shape
        _validate_exact_shape("transition_log_probs", self.transition_log_probs, step_shape)
        _validate_exact_shape(
            "reference_transition_log_probs",
            self.reference_transition_log_probs,
            step_shape,
        )
        _validate_exact_shape("trajectory_mask", self.trajectory_mask, step_shape)

    @property
    def batch_size(self) -> int:
        """Leading batch size shared by the iterative generation batch."""
        return self.generation.batch_size

    @property
    def num_steps(self) -> int:
        """Number of iterative-generation steps tracked per sample."""
        return int(self.step_indices.shape[1])


GenerationBatchT = TypeVar("GenerationBatchT")
RolloutBatchT = TypeVar("RolloutBatchT")


@nnx.dataclass
class PreferenceBatch(nnx.Pytree, Generic[GenerationBatchT]):
    """Chosen/rejected generation batch used by preference-learning objectives."""

    chosen: GenerationBatchT = nnx.data()
    rejected: GenerationBatchT = nnx.data()

    def __post_init__(self) -> None:
        """Validate preference batch invariants."""
        chosen_batch_size = _resolve_batch_size("chosen", self.chosen)
        rejected_batch_size = _resolve_batch_size("rejected", self.rejected)
        if chosen_batch_size != rejected_batch_size:
            msg = (
                "chosen and rejected generation batches must have the same batch size: "
                f"{chosen_batch_size} != {rejected_batch_size}"
            )
            raise ValueError(msg)

    @property
    def batch_size(self) -> int:
        """Shared batch size for chosen/rejected preference batches."""
        return _resolve_batch_size("chosen", self.chosen)


@nnx.dataclass
class GroupRolloutBatch(nnx.Pytree, Generic[RolloutBatchT]):
    """Rollout batch grouped by prompt or conditioning context."""

    rollout: RolloutBatchT = nnx.data()
    group_size: int = nnx.static()

    def __post_init__(self) -> None:
        """Validate grouped preference batch invariants."""
        if self.group_size <= 0:
            msg = "group_size must be positive"
            raise ValueError(msg)
        if self.batch_size % self.group_size != 0:
            msg = (
                "rollout batch size must be divisible by group_size: "
                f"{self.batch_size} % {self.group_size} != 0"
            )
            raise ValueError(msg)

    @property
    def batch_size(self) -> int:
        """Number of samples in the grouped rollout batch."""
        return _resolve_batch_size("rollout", self.rollout)

    @property
    def num_groups(self) -> int:
        """Number of prompt-level groups in the rollout batch."""
        return self.batch_size // self.group_size
