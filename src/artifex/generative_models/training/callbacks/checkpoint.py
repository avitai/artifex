"""Model checkpoint callback for training.

Monitors a metric and saves Orbax-managed checkpoints on a fixed epoch cadence.
Retention and best-step tracking are delegated to Orbax instead of reimplemented
locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from artifex.generative_models.core.checkpointing import (
    save_checkpoint,
    setup_checkpoint_manager,
)
from artifex.generative_models.training.callbacks.base import BaseCallback, TrainerLike


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

    Uses Orbax checkpointing under the hood. Minimal overhead when not saving.
    """

    __slots__ = (
        "config",
        "best_score",
        "best_checkpoint_step",
        "saved_checkpoint_steps",
        "_checkpoint_manager",
        "_dirpath",
    )

    def __init__(self, config: CheckpointConfig):
        """Initialize model checkpoint callback.

        Args:
            config: Checkpoint configuration.
        """
        self.config = config
        self.best_score: float | None = None
        self.best_checkpoint_step: int | None = None
        self.saved_checkpoint_steps: list[int] = []
        self._checkpoint_manager = None
        self._dirpath: Path = Path(config.dirpath)

        # Create checkpoint directory
        self._dirpath.mkdir(parents=True, exist_ok=True)

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
        else:  # max mode
            return current > self.best_score

    def _save_checkpoint(self, trainer: TrainerLike, epoch: int, score: float) -> None:
        """Save a checkpoint through Orbax.

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number.
            score: Current metric score.
        """
        # Setup checkpoint manager if not already done
        if self._checkpoint_manager is None:
            max_to_keep = None if self.config.save_top_k < 0 else self.config.save_top_k
            self._checkpoint_manager, _ = setup_checkpoint_manager(
                str(self._dirpath),
                max_to_keep=max_to_keep,
                best_fn=lambda metrics: float(metrics[self.config.monitor]),
                best_mode=self.config.mode,
            )

        save_checkpoint(
            self._checkpoint_manager,
            trainer.model,
            epoch,
            metrics={self.config.monitor: score},
        )
        self.saved_checkpoint_steps = list(self._checkpoint_manager.all_steps())
        self.best_checkpoint_step = self._checkpoint_manager.best_step()

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Check if checkpoint should be saved.

        Args:
            trainer: The trainer instance.
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
