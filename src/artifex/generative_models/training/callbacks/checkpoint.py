"""Model checkpoint callback for training.

Monitors a metric and saves the model through substrax's Orbax-backed store on a
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

from flax import nnx
from substrax.callbacks import BaseCallback, TrainerLike
from substrax.checkpoint import OrbaxCheckpointStore


@runtime_checkable
class CheckpointingTrainer(TrainerLike, Protocol):
    """What the checkpoint callback asks of its trainer: the model and the global step."""

    @property
    def step(self) -> int:
        """The global training step, the address a checkpoint is saved under."""
        ...


@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for model checkpointing.

    Attributes:
        dirpath: Directory to save checkpoints.
        monitor: Metric name to monitor (e.g., "val_loss", "accuracy").
        mode: "min" if lower is better, "max" if higher is better.
        save_top_k: Number of checkpoints to keep (-1 = all, 0 = none).
        every_n_epochs: Save checkpoint every n epochs (1 = every epoch).
    """

    dirpath: str | Path = "checkpoints"
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    every_n_epochs: int = 1


class ModelCheckpoint(BaseCallback):
    """Save model checkpoints based on monitored metrics.

    Uses substrax's ``OrbaxCheckpointStore`` under the hood. Minimal overhead
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
        self._store: OrbaxCheckpointStore | None = None
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

    def _save_checkpoint(self, trainer: TrainerLike, epoch: int, score: float) -> None:
        """Save the model item at the trainer's step, the metric in the record's metrics.

        Args:
            trainer: The trainer instance; it must expose the global ``step``.
            epoch: Current epoch number.
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
        if self._store is None:
            max_to_keep = None if self.config.save_top_k < 0 else self.config.save_top_k
            self._store = OrbaxCheckpointStore(self._dirpath, max_to_keep=max_to_keep)

        self._store.save(
            step,
            {"model": nnx.state(trainer.model)},
            epoch=epoch,
            metrics={self.config.monitor: score},
        )
        self.saved_checkpoint_steps = self._store.list_steps()
        self.best_checkpoint_step = self._store.best_step(
            self.config.monitor, mode=self.config.mode
        )

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

        # Check for improvement
        is_improvement = self._is_improvement(current)
        if is_improvement:
            self.best_score = current

        # Keep the "save all" mode explicit, but avoid unnecessary checkpoint
        # work when only the tracked best checkpoints matter.
        if self.config.save_top_k > 0 and not is_improvement:
            return

        self._save_checkpoint(trainer, epoch, current)

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Release the store's resources at the end of training.

        Args:
            _trainer: The trainer instance (unused).
        """
        if self._store is not None:
            self._store.close()
            self._store = None
