"""Model checkpoint callback for training.

Monitors a metric and saves checkpoints when it improves.
Uses the existing Orbax-based checkpointing infrastructure.
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
        filename: Checkpoint filename template with {epoch} and metric placeholders.
        monitor: Metric name to monitor (e.g., "val_loss", "accuracy").
        mode: "min" if lower is better, "max" if higher is better.
        save_top_k: Number of best checkpoints to keep (-1 = all, 0 = none).
        save_last: Whether to save checkpoint on every epoch (as "last").
        every_n_epochs: Save checkpoint every n epochs (1 = every epoch).
        save_weights_only: If True, only save model weights (not optimizer state).
    """

    dirpath: str | Path = "checkpoints"
    filename: str = "model-{epoch:02d}-{val_loss:.4f}"
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    save_last: bool = True
    every_n_epochs: int = 1
    save_weights_only: bool = False


class ModelCheckpoint(BaseCallback):
    """Save model checkpoints based on monitored metrics.

    Uses Orbax checkpointing under the hood. Minimal overhead when not saving.
    """

    __slots__ = (
        "config",
        "best_score",
        "best_checkpoint_path",
        "saved_checkpoints",
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
        self.best_checkpoint_path: Path | None = None
        self.saved_checkpoints: list[tuple[float, Path]] = []
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

    def _format_filename(self, epoch: int, logs: dict[str, Any]) -> str:
        """Format checkpoint filename with epoch and metric values.

        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics.

        Returns:
            Formatted filename string.
        """
        # Build format kwargs from epoch and logs
        format_kwargs: dict[str, Any] = {"epoch": epoch}

        # Convert JAX/numpy arrays to Python floats for formatting
        # Keep Python int/float as-is
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                format_kwargs[k] = v
            elif hasattr(v, "item"):
                # JAX/numpy arrays have .item() method
                format_kwargs[k] = float(v.item())
            elif hasattr(v, "__float__"):
                format_kwargs[k] = float(v)
            else:
                format_kwargs[k] = v

        try:
            return self.config.filename.format(**format_kwargs)
        except KeyError:
            # Fallback if metric not in logs
            return f"model-{epoch:02d}"

    def _save_checkpoint(
        self, trainer: TrainerLike, epoch: int, _score: float, logs: dict[str, Any]
    ) -> Path:
        """Save a checkpoint.

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number.
            score: Current metric score.
            logs: Dictionary of metrics.

        Returns:
            Path to saved checkpoint.
        """
        # Setup checkpoint manager if not already done
        if self._checkpoint_manager is None:
            self._checkpoint_manager, _ = setup_checkpoint_manager(str(self._dirpath))

        # Save using existing infrastructure
        save_checkpoint(self._checkpoint_manager, trainer.model, epoch)

        # Construct the checkpoint path
        checkpoint_path = self._dirpath / self._format_filename(epoch, logs)

        return checkpoint_path

    def _remove_old_checkpoints(self) -> None:
        """Remove checkpoints beyond save_top_k."""
        if self.config.save_top_k <= 0:
            return

        while len(self.saved_checkpoints) > self.config.save_top_k:
            # Remove the worst checkpoint (last in sorted list for min mode)
            if self.config.mode == "min":
                # For min mode, larger scores are worse
                self.saved_checkpoints.sort(key=lambda x: x[0])
                worst = self.saved_checkpoints.pop()
            else:
                # For max mode, smaller scores are worse
                self.saved_checkpoints.sort(key=lambda x: x[0], reverse=True)
                worst = self.saved_checkpoints.pop()

            # Try to remove the file
            _, path = worst
            try:
                if path.exists():
                    if path.is_dir():
                        import shutil

                        shutil.rmtree(path)
                    else:
                        path.unlink()
            except OSError:
                pass  # Ignore removal errors

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

        if is_improvement or self.config.save_top_k == -1:
            # Update best score
            if is_improvement:
                self.best_score = current

            # Save checkpoint
            checkpoint_path = self._save_checkpoint(trainer, epoch, current, logs)
            self.saved_checkpoints.append((current, checkpoint_path))

            # Update best path
            if is_improvement:
                self.best_checkpoint_path = checkpoint_path

            # Clean up old checkpoints
            self._remove_old_checkpoints()
