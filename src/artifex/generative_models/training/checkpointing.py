"""Checkpoints a training run resumes from exactly.

One save writes the model, the complete step (including its optimizers), the run's base key,
data position and extra state through one substrax checkpoint. A restore
into objects built the same way continues where the saved run stood: the parameters, the
optimizer moments, the next batch, the keys drawn and the callbacks' bookkeeping all match the
uninterrupted run. A caller's check sees the record before anything is applied.

Retention keeps the newest ``keep`` checkpoints and, with ``best``, the one whose recorded
metric is best. The best is read from the store's records, not from memory, so a resumed run
keeps the best model the run before it saved.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.checkpoint import CheckpointMetadata, OrbaxCheckpointStore, Producer
from substrax.typing import Checkpointable, JsonValue


PRODUCER = Producer(name="artifex", version=version("avitai-artifex"))
"""The record's producer: this package and its installed version."""

type Extension = Checkpointable | nnx.Module
"""State a run checkpoints besides the model and optimizer: through ``get_state``/``set_state``
when it has them (callbacks, a data pipeline), else its ``nnx`` state (an EMA, a step's state)."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointPolicy:
    """Where a run checkpoints, how often, and which checkpoints it keeps.

    Attributes:
        directory: The checkpoint directory.
        every_steps: Save every this many steps; ``None`` saves only when asked.
        keep: Keep the newest this many checkpoints; ``None`` keeps all.
        best: ``(metric, mode)``: also keep the checkpoint whose recorded ``metric`` is the
            lowest (``"min"``) or highest (``"max"``).
    """

    directory: Path
    every_steps: int | None = None
    keep: int | None = None
    best: tuple[str, Literal["min", "max"]] | None = None

    def __post_init__(self) -> None:
        """Refuse a cadence or a retention that keeps nothing.

        Raises:
            ValueError: If ``every_steps`` or ``keep`` is below 1, or ``best``'s mode is not
                ``"min"`` or ``"max"``.
        """
        if self.every_steps is not None and self.every_steps < 1:
            raise ValueError(f"every_steps must be >= 1, got {self.every_steps}")
        if self.keep is not None and self.keep < 1:
            raise ValueError(f"keep must be >= 1, got {self.keep}")
        if self.best is not None and self.best[1] not in ("min", "max"):
            raise ValueError(f"best mode must be 'min' or 'max', got {self.best[1]!r}")

    def is_due(self, step: int) -> bool:
        """Whether the cadence saves at ``step`` (never at step 0).

        Args:
            step: The global step just completed.

        Returns:
            Whether a save is due.
        """
        return self.every_steps is not None and step > 0 and step % self.every_steps == 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RunState:
    """The objects a run checkpoints.

    The step owns all of its optimizers and update state. It is captured automatically;
    callers register only additional host or NNX state through ``extra_state``.

    Attributes:
        model: The model.
        step: The native step, including its optimizers and custom update state.
        base_key: The run's base key; every step's key derives from it and the step.
        epoch: The epoch the run is in.
        data: The data position (a datarax ``Pipeline``), or ``None`` for a fixed batch.
        extra_state: Further state by name, including callbacks or EMA when present.
    """

    model: nnx.Module
    step: nnx.Module
    base_key: jax.Array
    epoch: int
    data: Checkpointable | None = None
    extra_state: Mapping[str, Extension] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RestoredRun:
    """Where a restored run stands.

    Attributes:
        step: The step the checkpoint was saved at.
        epoch: The epoch it was saved in.
        base_key: The run's base key.
    """

    step: int
    epoch: int
    base_key: jax.Array


class Checkpointer:
    """Saves and restores runs under a :class:`CheckpointPolicy`."""

    def __init__(self, policy: CheckpointPolicy) -> None:
        """Checkpoint under ``policy``.

        Args:
            policy: Where, how often and what to keep.
        """
        self.policy = policy

    def _store(self) -> OrbaxCheckpointStore:
        # This policy keeps newest AND best; the store's independent newest-only
        # retention would discard candidates before this owner can preserve them.
        return OrbaxCheckpointStore(self.policy.directory, max_to_keep=None)

    def save(  # noqa: DOC502 -- numerical admission raises through the shared validator.
        self,
        step: int,
        run: RunState,
        *,
        metrics: Mapping[str, float] | None = None,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> Path:
        """Write the run at ``step``, then apply the retention.

        All floating and complex state arrays must be finite before writing. Typed keys,
        integer counters and plain Python metadata are not numerical arrays. In particular,
        callbacks may retain an infinite initial best-score sentinel in their metadata.

        Args:
            step: The global step.
            run: The run's model, step, base key, epoch, data position and extra state.
            metrics: Scalars recorded for the best checkpoint.
            extra: The caller's own record values (an architecture version, identities).

        Returns:
            The checkpoint's directory.

        Raises:
            FloatingPointError: If a state array is non-finite. Nothing is written or pruned.
        """
        items = _items(run)
        _require_finite_arrays(items)
        with self._store() as store:
            path = store.save(
                step,
                items,
                epoch=run.epoch,
                metrics=metrics,
                producer=PRODUCER,
                extra=extra,
            )
            self._retain(store)
        return path

    def restore(  # noqa: DOC503 -- numerical admission raises through the shared validator.
        self,
        run: RunState,
        *,
        step: int | None = None,
        validate: Callable[[CheckpointMetadata], None] | None = None,
    ) -> RestoredRun:
        """Restore the checkpoint at ``step`` (the newest by default) into these objects.

        The objects must be built the way the saved ones were: their current state is the
        restore template, which fixes each array's structure, dtype and placement. ``validate``
        sees the record first and may raise, in which case nothing is applied.
        Numerical arrays are admitted before applying any state. A saved data position
        requires a data object. Application to arbitrary extensions is sequential: a failing
        ``set_state`` may leave a partially applied run, which the caller must reconstruct.

        Args:
            run: The objects to restore into; its key and epoch are replaced by the saved
                ones in the result.
            step: The step to restore; the newest when ``None``.
            validate: A check of the record before anything is applied.

        Returns:
            The restored step, epoch and base key.

        Raises:
            FileNotFoundError: If the directory holds no checkpoint.
            ValueError: If the checkpoint contains a data position but no data was supplied.
            FloatingPointError: If a restored numerical array is non-finite.
        """
        with self._store() as store:
            if step is None:
                step = store.latest_step()
                if step is None:
                    raise FileNotFoundError(f"no checkpoint under {self.policy.directory}")
            metadata = store.read_metadata(step)
            if validate is not None:
                validate(metadata)
            if "data_iterator" in metadata.items and run.data is None:
                raise ValueError("checkpoint contains a data position; supply a data object")
            items = store.restore(step, templates=_items(run)).items
        _require_finite_arrays(items)
        nnx.update(run.model, items["model"])
        nnx.update(run.step, items["extensions"]["step"])
        if run.data is not None:
            run.data.set_state(items["data_iterator"])
        for name, extension in run.extra_state.items():
            _set_extension_state(extension, items["extensions"]["extra_state"][name])
        return RestoredRun(step=step, epoch=metadata.epoch or 0, base_key=items["rng"])

    def steps(self) -> list[int]:
        """Every step held, ascending.

        Returns:
            The steps.
        """
        with self._store() as store:
            return store.list_steps()

    def best_step(self) -> int | None:
        """The step whose recorded ``best`` metric is best, or ``None``.

        Returns:
            The step, or ``None`` without a ``best`` policy or a record of the metric.
        """
        if self.policy.best is None:
            return None
        metric, mode = self.policy.best
        with self._store() as store:
            return store.best_step(metric, mode=mode)

    def _retain(self, store: OrbaxCheckpointStore) -> None:
        """Delete every checkpoint neither among the newest ``keep`` nor the best."""
        if self.policy.keep is None:
            return
        steps = store.list_steps()
        kept = set(steps[-self.policy.keep :])
        if self.policy.best is not None:
            metric, mode = self.policy.best
            best = store.best_step(metric, mode=mode)
            if best is not None:
                kept.add(best)
        for step in steps:
            if step not in kept:
                store.delete(step)


def _require_finite_arrays(items: Mapping[str, Any]) -> None:
    """Admit numerical arrays before publication or live-state application."""
    arrays = tuple(
        leaf
        for leaf in jax.tree.leaves(items)
        if (dtype := getattr(leaf, "dtype", None)) is not None
        and jnp.issubdtype(dtype, jnp.inexact)
    )
    # Preserve saved float64/complex128 values during validation even when training
    # uses x64=False. This scoped setting does not change the run or its arrays.
    with jax.enable_x64():
        finite = bool(_arrays_are_finite(arrays))
    if not finite:
        raise FloatingPointError("checkpoint contains non-finite state arrays")


@jax.jit
def _arrays_are_finite(arrays: tuple[jax.Array, ...]) -> jax.Array:
    """Reduce state arrays on device without transferring their contents to the host."""
    return jnp.all(jnp.asarray([jnp.all(jnp.isfinite(leaf)) for leaf in arrays]))


def _items(run: RunState) -> dict[str, Any]:
    """The run's state and restore templates in Substrax's admitted item namespace.

    The complete step and additional state are named subtrees of ``extensions``, the
    store's native container for arbitrary training state. No separate optimizer item or
    application serializer is needed, including for steps with multiple optimizers.
    """
    items: dict[str, Any] = {
        "model": nnx.state(run.model),
        "rng": run.base_key,
        "extensions": {
            "step": nnx.state(run.step),
            "extra_state": {name: _extension_state(ext) for name, ext in run.extra_state.items()},
        },
    }
    if run.data is not None:
        items["data_iterator"] = run.data.get_state()
    return items


def _extension_state(extension: Extension) -> Any:
    """An extension's state: its ``get_state()`` when it has one, else its ``nnx`` state."""
    if isinstance(extension, Checkpointable):
        return extension.get_state()
    return nnx.state(extension)


def _set_extension_state(extension: Extension, state: Any) -> None:
    """Restore what :func:`_extension_state` saved."""
    if isinstance(extension, Checkpointable):
        extension.set_state(state)
    else:
        nnx.update(extension, state)
