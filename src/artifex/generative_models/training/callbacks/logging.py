"""Logging callbacks for training.

Provides callbacks that integrate with the training loop to log metrics,
images, and other training artifacts using various logging backends.

These callbacks follow DRY by wrapping existing Logger implementations
from `artifex.generative_models.utils.logging`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from artifex.generative_models.training.callbacks.base import BaseCallback, TrainerLike
from artifex.generative_models.utils.logging.logger import Logger


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass(slots=True)
class LoggerCallbackConfig:
    """Configuration for base logger callback.

    Attributes:
        log_every_n_steps: Log metrics every N training steps.
        log_on_epoch_end: Whether to log metrics at end of each epoch.
        prefix: Prefix to add to metric names.
    """

    log_every_n_steps: int = 1
    log_on_epoch_end: bool = True
    prefix: str = ""


@dataclass(slots=True)
class WandbLoggerConfig:
    """Configuration for W&B logging callback.

    Attributes:
        project: W&B project name (required).
        entity: W&B entity (username or team name).
        name: Run name. If None, W&B auto-generates.
        tags: List of tags for the run.
        notes: Notes about the run.
        config: Dictionary of hyperparameters to log.
        mode: W&B mode: "online", "offline", or "disabled".
        resume: Whether to resume a previous run.
        log_every_n_steps: Log metrics every N training steps.
        log_on_epoch_end: Whether to log metrics at end of each epoch.
        log_dir: Local directory for W&B files.
    """

    project: str
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    mode: Literal["online", "offline", "disabled"] = "online"
    resume: Literal["allow", "never", "must", "auto"] | bool | None = None
    log_every_n_steps: int = 1
    log_on_epoch_end: bool = True
    log_dir: Optional[str] = None


@dataclass(slots=True)
class TensorBoardLoggerConfig:
    """Configuration for TensorBoard logging callback.

    Attributes:
        log_dir: Directory for TensorBoard logs.
        flush_secs: How often to flush to disk (seconds).
        max_queue: Max queue size for pending events.
        log_every_n_steps: Log metrics every N training steps.
        log_on_epoch_end: Whether to log metrics at end of each epoch.
        log_graph: Whether to log model graph.
    """

    log_dir: str = "logs/tensorboard"
    flush_secs: int = 120
    max_queue: int = 10
    log_every_n_steps: int = 1
    log_on_epoch_end: bool = True
    log_graph: bool = False


@dataclass(slots=True)
class ProgressBarConfig:
    """Configuration for progress bar callback.

    Attributes:
        refresh_rate: How often to refresh the progress bar (steps).
        show_eta: Whether to show estimated time of arrival.
        show_metrics: Whether to display metrics in progress bar.
        leave: Whether to leave progress bar after completion.
        disable: Whether to disable progress bar entirely.
    """

    refresh_rate: int = 10
    show_eta: bool = True
    show_metrics: bool = True
    leave: bool = True
    disable: bool = False


# =============================================================================
# Base Logger Callback
# =============================================================================


class LoggerCallback(BaseCallback):
    """Base callback that wraps any Logger instance for training integration.

    This callback delegates actual logging to an existing Logger implementation,
    following the DRY principle by reusing the existing logging infrastructure.

    Example:
        from artifex.generative_models.utils.logging import ConsoleLogger

        logger = ConsoleLogger(name="training")
        callback = LoggerCallback(logger=logger)
        trainer.fit(callbacks=[callback])
    """

    __slots__ = ("logger", "config", "_step_count")

    def __init__(
        self,
        logger: Logger,
        config: Optional[LoggerCallbackConfig] = None,
    ):
        """Initialize the logger callback.

        Args:
            logger: Logger instance to use for logging.
            config: Logging configuration. Uses defaults if not provided.
        """
        self.logger = logger
        self.config = config or LoggerCallbackConfig()
        self._step_count: int = 0

    def on_batch_end(
        self,
        _trainer: TrainerLike,
        batch: int,
        logs: dict[str, Any],
    ) -> None:
        """Log metrics at the end of a training batch.

        Args:
            _trainer: The trainer instance (unused).
            batch: Current batch number.
            logs: Dictionary of metrics from this batch.
        """
        self._step_count += 1

        # Check if we should log at this step
        if batch % self.config.log_every_n_steps != 0:
            return

        # Skip if no metrics
        if not logs:
            return

        # Convert JAX arrays to Python floats and add prefix
        metrics = self._prepare_metrics(logs)
        if metrics:
            self.logger.log_scalars(metrics, step=batch)

    def on_epoch_end(
        self,
        _trainer: TrainerLike,
        epoch: int,
        logs: dict[str, Any],
    ) -> None:
        """Log metrics at the end of an epoch.

        Args:
            _trainer: The trainer instance (unused).
            epoch: Current epoch number.
            logs: Dictionary of metrics from this epoch.
        """
        if not self.config.log_on_epoch_end:
            return

        # Convert JAX arrays to Python floats and add prefix
        metrics = self._prepare_metrics(logs)
        if metrics:
            self.logger.log_scalars(metrics, step=epoch)

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Close the logger at the end of training.

        Args:
            _trainer: The trainer instance (unused).
        """
        self.logger.close()

    def _prepare_metrics(self, logs: dict[str, Any]) -> dict[str, float]:
        """Prepare metrics for logging.

        Converts JAX arrays to floats and adds prefix.

        Args:
            logs: Raw metrics dictionary.

        Returns:
            Prepared metrics dictionary.
        """
        metrics = {}
        for name, value in logs.items():
            # Convert JAX arrays to Python floats
            if hasattr(value, "item"):
                value = float(value.item())
            elif hasattr(value, "__float__"):
                value = float(value)

            # Only log numeric values
            if isinstance(value, (int, float)):
                key = f"{self.config.prefix}{name}" if self.config.prefix else name
                metrics[key] = value

        return metrics


# =============================================================================
# W&B Logger Callback
# =============================================================================


class WandbLoggerCallback(BaseCallback):
    """Weights & Biases experiment tracking callback.

    Wraps the existing WandbLogger for training loop integration.

    Features:
        - Automatic metric logging
        - Hyperparameter tracking
        - Run resumption support
        - Multiple run modes (online, offline, disabled)

    Example:
        config = WandbLoggerConfig(
            project="my-project",
            name="experiment-1",
            tags=["baseline", "vae"],
        )
        callback = WandbLoggerCallback(config=config)
        trainer.fit(callbacks=[callback])
    """

    __slots__ = ("config", "_wandb", "_run", "_initialized")

    def __init__(self, config: WandbLoggerConfig):
        """Initialize the W&B callback.

        Args:
            config: W&B configuration.
        """
        self.config = config
        self._wandb = None
        self._run = None
        self._initialized = False

        # Import and initialize W&B eagerly to fail fast
        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=config.project,
                entity=config.entity,
                name=config.name,
                tags=config.tags,
                notes=config.notes,
                config=config.config,
                mode=config.mode,
                resume=config.resume,
                dir=config.log_dir,
            )
            self._initialized = True
        except ImportError as err:
            raise ImportError(
                "Weights & Biases is required for WandbLoggerCallback. "
                "Install with `pip install wandb`."
            ) from err

    def on_train_begin(self, _trainer: TrainerLike) -> None:
        """Log hyperparameters at the start of training.

        Args:
            _trainer: The trainer instance.
        """
        if not self._initialized or self._run is None:
            return

        # Log any hyperparams from config
        if self.config.config:
            for key, value in self.config.config.items():
                self._run.config[key] = value

    def on_batch_end(
        self,
        _trainer: TrainerLike,
        batch: int,
        logs: dict[str, Any],
    ) -> None:
        """Log metrics at the end of a training batch.

        Args:
            _trainer: The trainer instance (unused).
            batch: Current batch number.
            logs: Dictionary of metrics from this batch.
        """
        if not self._initialized or self._wandb is None:
            return

        # Check if we should log at this step
        if batch % self.config.log_every_n_steps != 0:
            return

        # Convert and log metrics
        metrics = self._prepare_metrics(logs)
        if metrics:
            self._wandb.log(metrics, step=batch)

    def on_epoch_end(
        self,
        _trainer: TrainerLike,
        epoch: int,
        logs: dict[str, Any],
    ) -> None:
        """Log metrics at the end of an epoch.

        Args:
            _trainer: The trainer instance (unused).
            epoch: Current epoch number.
            logs: Dictionary of metrics from this epoch.
        """
        if not self._initialized or self._wandb is None:
            return

        if not self.config.log_on_epoch_end:
            return

        metrics = self._prepare_metrics(logs)
        if metrics:
            # Add epoch prefix for clarity
            epoch_metrics = {f"epoch/{k}": v for k, v in metrics.items()}
            self._wandb.log(epoch_metrics, step=epoch)

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Finish the W&B run at the end of training.

        Args:
            _trainer: The trainer instance (unused).
        """
        if self._initialized and self._wandb is not None:
            self._wandb.finish()

    def _prepare_metrics(self, logs: dict[str, Any]) -> dict[str, float]:
        """Prepare metrics for logging.

        Args:
            logs: Raw metrics dictionary.

        Returns:
            Prepared metrics dictionary with only numeric values.
        """
        metrics = {}
        for name, value in logs.items():
            # Convert JAX arrays to Python floats
            if hasattr(value, "item"):
                value = float(value.item())
            elif hasattr(value, "__float__"):
                value = float(value)

            # Only log numeric values
            if isinstance(value, (int, float)):
                metrics[name] = value

        return metrics


# =============================================================================
# TensorBoard Logger Callback
# =============================================================================


class TensorBoardLoggerCallback(BaseCallback):
    """TensorBoard logging callback.

    Features:
        - Scalar metrics logging
        - Configurable flush interval
        - Per-step and per-epoch logging

    Example:
        config = TensorBoardLoggerConfig(
            log_dir="logs/experiment-1",
            flush_secs=60,
        )
        callback = TensorBoardLoggerCallback(config=config)
        trainer.fit(callbacks=[callback])
    """

    __slots__ = ("config", "_writer", "_initialized")

    def __init__(self, config: TensorBoardLoggerConfig):
        """Initialize the TensorBoard callback.

        Args:
            config: TensorBoard configuration.
        """
        self.config = config
        self._writer = None
        self._initialized = False

    def on_train_begin(self, _trainer: TrainerLike) -> None:
        """Initialize TensorBoard writer at the start of training.

        Args:
            _trainer: The trainer instance (unused).
        """
        try:
            import os

            from torch.utils.tensorboard import SummaryWriter

            # Create log directory
            os.makedirs(self.config.log_dir, exist_ok=True)

            self._writer = SummaryWriter(
                log_dir=self.config.log_dir,
                flush_secs=self.config.flush_secs,
                max_queue=self.config.max_queue,
            )
            self._initialized = True
        except ImportError as err:
            raise ImportError(
                "TensorBoard is required for TensorBoardLoggerCallback. "
                "Install with `pip install tensorboard`."
            ) from err

    def on_batch_end(
        self,
        _trainer: TrainerLike,
        batch: int,
        logs: dict[str, Any],
    ) -> None:
        """Log metrics at the end of a training batch.

        Args:
            _trainer: The trainer instance (unused).
            batch: Current batch number.
            logs: Dictionary of metrics from this batch.
        """
        if not self._initialized or self._writer is None:
            return

        # Check if we should log at this step
        if batch % self.config.log_every_n_steps != 0:
            return

        # Log each metric
        for name, value in logs.items():
            # Convert JAX arrays to Python floats
            if hasattr(value, "item"):
                value = float(value.item())
            elif hasattr(value, "__float__"):
                value = float(value)

            if isinstance(value, (int, float)):
                self._writer.add_scalar(name, value, batch)

    def on_epoch_end(
        self,
        _trainer: TrainerLike,
        epoch: int,
        logs: dict[str, Any],
    ) -> None:
        """Log metrics at the end of an epoch.

        Args:
            _trainer: The trainer instance (unused).
            epoch: Current epoch number.
            logs: Dictionary of metrics from this epoch.
        """
        if not self._initialized or self._writer is None:
            return

        if not self.config.log_on_epoch_end:
            return

        # Log each metric with epoch prefix
        for name, value in logs.items():
            # Convert JAX arrays to Python floats
            if hasattr(value, "item"):
                value = float(value.item())
            elif hasattr(value, "__float__"):
                value = float(value)

            if isinstance(value, (int, float)):
                self._writer.add_scalar(f"epoch/{name}", value, epoch)

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Close the TensorBoard writer at the end of training.

        Args:
            _trainer: The trainer instance (unused).
        """
        if self._initialized and self._writer is not None:
            self._writer.close()


# =============================================================================
# Progress Bar Callback
# =============================================================================


class ProgressBarCallback(BaseCallback):
    """Rich console progress bar callback.

    Features:
        - Real-time training progress
        - Metric display
        - ETA estimation
        - Nested progress for epochs/steps

    Example:
        config = ProgressBarConfig(
            refresh_rate=10,
            show_eta=True,
            show_metrics=True,
        )
        callback = ProgressBarCallback(config=config)
        trainer.fit(callbacks=[callback])
    """

    __slots__ = ("config", "_progress", "_task_id", "_epoch_task_id")

    def __init__(self, config: Optional[ProgressBarConfig] = None):
        """Initialize the progress bar callback.

        Args:
            config: Progress bar configuration. Uses defaults if not provided.
        """
        self.config = config or ProgressBarConfig()
        self._progress = None
        self._task_id = None
        self._epoch_task_id = None

    def on_train_begin(self, _trainer: TrainerLike) -> None:
        """Create and start the progress bar.

        Args:
            _trainer: The trainer instance.
        """
        if self.config.disable:
            return

        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeRemainingColumn,
            )

            columns = [
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
            ]

            if self.config.show_eta:
                columns.append(TimeRemainingColumn())

            self._progress = Progress(*columns)
            self._progress.start()

        except ImportError as err:
            raise ImportError(
                "Rich is required for ProgressBarCallback. Install with `pip install rich`."
            ) from err

    def on_epoch_begin(
        self,
        trainer: TrainerLike,
        epoch: int,
    ) -> None:
        """Add a task for the current epoch.

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number.
        """
        if self.config.disable or self._progress is None:
            return

        # Get total steps from trainer if available
        total = getattr(trainer, "steps_per_epoch", 100)

        self._task_id = self._progress.add_task(
            f"Epoch {epoch}",
            total=total,
        )

    def on_batch_end(
        self,
        _trainer: TrainerLike,
        batch: int,  # noqa: ARG002
        logs: dict[str, Any],
    ) -> None:
        """Update progress bar after each batch.

        Args:
            _trainer: The trainer instance (unused).
            batch: Current batch number (unused, progress tracked internally).
            logs: Dictionary of metrics from this batch.
        """
        del batch  # Unused, progress tracked by advance()
        if self.config.disable or self._progress is None:
            return

        if self._task_id is None:
            return

        # Advance the progress bar
        self._progress.advance(self._task_id)

        # Update description with metrics if enabled
        if self.config.show_metrics and logs:
            # Format a few key metrics
            metric_strs = []
            for name, value in list(logs.items())[:3]:  # Limit to 3 metrics
                if hasattr(value, "item"):
                    value = float(value.item())
                elif hasattr(value, "__float__"):
                    value = float(value)

                if isinstance(value, float):
                    metric_strs.append(f"{name}: {value:.4f}")

            if metric_strs:
                description = " | ".join(metric_strs)
                self._progress.update(self._task_id, description=description)

    def on_epoch_end(
        self,
        _trainer: TrainerLike,
        epoch: int,  # noqa: ARG002
        logs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Complete the epoch task.

        Args:
            _trainer: The trainer instance (unused).
            epoch: Current epoch number (unused).
            logs: Dictionary of metrics from this epoch (unused).
        """
        del epoch, logs  # Unused in progress bar
        if self.config.disable or self._progress is None:
            return

        if self._task_id is not None:
            # Mark task as complete
            self._progress.update(self._task_id, completed=True)

            if not self.config.leave:
                self._progress.remove_task(self._task_id)

            self._task_id = None

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Stop the progress bar.

        Args:
            _trainer: The trainer instance (unused).
        """
        if self.config.disable:
            return

        if self._progress is not None:
            self._progress.stop()
