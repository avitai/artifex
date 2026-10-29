"""Model checkpoint callback for training.

Monitors a metric and saves the native run through the shared Checkpointer on a
fixed epoch cadence, each checkpoint at the trainer's global step with the metric
in the record's ``metrics`` and the epoch in its ``epoch``. Because a checkpoint
is written only when the monitored metric improves, the store's most recent
``save_top_k`` checkpoints are the ``save_top_k`` best; retention and best-step
lookup are the store's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from substrax.callbacks import BaseCallback, TrainerLike

from artifex.generative_models.training.checkpointing import (
    Checkpointer,
    CheckpointPolicy,
    RunState,
)


@runtime_checkable
class CheckpointingTrainer(TrainerLike, Protocol):
    """The complete native run and its accumulated numerical admission."""

    @property
    def step(self) -> int:
        """The global training step, the address a checkpoint is saved under."""
        ...

    def run_state(self) -> RunState:
        """Return the complete native step and model, not model-only weights."""
        ...

    def check_finite(self) -> None:
        """Refuse any failed update before checkpoint capture."""
        ...


@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for model checkpointing.

    Attributes:
        dirpath: Directory to save checkpoints; named by the caller, so the callback never
            writes into the working directory on its own.
        monitor: Metric name to monitor (e.g., "val_loss", "accuracy").
        mode: "min" if lower is better, "max" if higher is better.
        save_top_k: Number of checkpoints to keep (-1 = all, 0 = none).
        every_n_epochs: Save checkpoint every n epochs (1 = every epoch).
    """

    dirpath: str | Path
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    every_n_epochs: int = 1


class ModelCheckpoint(BaseCallback):
    """Save model checkpoints based on monitored metrics.

    Uses the numerically admitted native ``Checkpointer``. Minimal overhead
    when not saving.
    """

    __slots__ = (
        "config",
        "best_score",
        "best_checkpoint_step",
        "saved_checkpoint_steps",
        "_store",
        "_dirpath",
    )

    def __init__(self, config: CheckpointConfig) -> None:
        """Initialize model checkpoint callback.

        Args:
            config: Checkpoint configuration.
        """
        self.config = config
        self.best_score: float | None = None
        self.best_checkpoint_step: int | None = None
        self.saved_checkpoint_steps: list[int] = []
        self._store: Checkpointer | None = None
        # The store creates the directory on its first save.
        self._dirpath: Path = Path(config.dirpath)

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement over best.

        Args:
            current: Current metric value.

        Returns:
            True if current is better than best.
        """
        if self.best_score is None:
            return True
        if self.config.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _require_store(self) -> Checkpointer:
        """Open the native store once and recover selection from committed records."""
        if self._store is None:
            max_to_keep = None if self.config.save_top_k < 0 else self.config.save_top_k
            self._store = Checkpointer(
                CheckpointPolicy(
                    directory=self._dirpath,
                    keep=max_to_keep,
                    best=(self.config.monitor, self.config.mode),
                )
            )
            self._refresh_committed_state()
        return self._store

    def _refresh_committed_state(self) -> None:
        """Cache native committed selection for reporting and cheap epoch checks."""
        if self._store is not None:
            self.saved_checkpoint_steps = self._store.steps()
            self.best_checkpoint_step = self._store.best_step()
            self.best_score = self._store.best_score()

    def _save_checkpoint(self, trainer: TrainerLike, score: float) -> None:
        """Save the native run at the trainer's step, with its monitored metric.

        Args:
            trainer: The trainer instance; it must expose the global ``step``.
            score: Current metric score.

        Raises:
            TypeError: If the trainer has no integer ``step`` to save the checkpoint under.
        """
        # Read the step here, on the save path only: a Protocol isinstance on every epoch end
        # costs microseconds the callback must not spend when nothing is saved.
        step = getattr(trainer, "step", None)
        if not isinstance(step, int):
            raise TypeError(
                f"{type(trainer).__name__} has no step; ModelCheckpoint saves at the global step"
            )
        if not isinstance(trainer, CheckpointingTrainer):
            raise TypeError("ModelCheckpoint requires a native checkpointing Trainer")
        trainer.check_finite()
        self._require_store().save(
            step,
            trainer.run_state(),
            metrics={self.config.monitor: score},
        )
        self._refresh_committed_state()

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Check if checkpoint should be saved.

        Args:
            trainer: The trainer instance; it must expose the global ``step`` when a save
                is due (:class:`CheckpointingTrainer`).
            epoch: Current epoch number.
            logs: Dictionary of metrics from this epoch.
        """
        # Check if this is an epoch we should consider
        if self.config.every_n_epochs > 1 and epoch % self.config.every_n_epochs != 0:
            return

        # Get current value
        current = logs.get(self.config.monitor)
        if current is None:
            return

        # Convert JAX arrays to Python floats
        current = float(current)

        # Check if save_top_k is 0 (no saving)
        if self.config.save_top_k == 0:
            return

        self._require_store()
        # Check for improvement
        is_improvement = self._is_improvement(current)
        # Keep the "save all" mode explicit, but avoid unnecessary checkpoint
        # work when only the tracked best checkpoints matter.
        if self.config.save_top_k > 0 and not is_improvement:
            return

        self._save_checkpoint(trainer, current)

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Release the store's resources at the end of training.

        Args:
            _trainer: The trainer instance (unused).
        """
        if self._store is not None:
            self._store.close()
            self._store = None
