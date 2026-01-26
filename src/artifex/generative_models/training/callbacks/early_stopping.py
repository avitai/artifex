"""Early stopping callback for training.

Monitors a metric and stops training when it stops improving.
Designed for minimal overhead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from artifex.generative_models.training.callbacks.base import BaseCallback, TrainerLike


@dataclass(slots=True)
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        monitor: Metric name to monitor (e.g., "val_loss", "accuracy").
        min_delta: Minimum change to qualify as an improvement.
        patience: Number of epochs with no improvement before stopping.
        mode: "min" if lower is better, "max" if higher is better.
        check_finite: If True, stop when metric becomes NaN or Inf.
        stopping_threshold: Stop immediately when metric reaches this value.
        divergence_threshold: Stop if metric exceeds this value (min mode only).
    """

    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 10
    mode: Literal["min", "max"] = "min"
    check_finite: bool = True
    stopping_threshold: float | None = None
    divergence_threshold: float | None = None


class EarlyStopping(BaseCallback):
    """Stop training when a monitored metric stops improving.

    Minimal overhead implementation using simple comparisons.
    """

    __slots__ = (
        "config",
        "wait_count",
        "best_score",
        "stopped_epoch",
        "_should_stop",
    )

    def __init__(self, config: EarlyStoppingConfig):
        """Initialize early stopping callback.

        Args:
            config: Early stopping configuration.
        """
        self.config = config
        self.wait_count: int = 0
        self.best_score: float | None = None
        self.stopped_epoch: int | None = None
        self._should_stop: bool = False

    @property
    def should_stop(self) -> bool:
        """Whether training should stop."""
        return self._should_stop

    def on_epoch_end(self, _trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Check if training should stop.

        Args:
            _trainer: The trainer instance (unused).
            epoch: Current epoch number.
            logs: Dictionary of metrics from this epoch.
        """
        # Get current value
        current = logs.get(self.config.monitor)
        if current is None:
            return

        # Convert JAX arrays to Python floats
        current = float(current)

        # Check for non-finite values
        if self.config.check_finite and not math.isfinite(current):
            self._should_stop = True
            self.stopped_epoch = epoch
            return

        # Check stopping threshold (goal reached)
        if self.config.stopping_threshold is not None:
            if self._meets_threshold(current):
                self._should_stop = True
                self.stopped_epoch = epoch
                return

        # Check divergence threshold
        if self.config.divergence_threshold is not None:
            if self.config.mode == "min" and current > self.config.divergence_threshold:
                self._should_stop = True
                self.stopped_epoch = epoch
                return

        # First epoch - establish baseline
        if self.best_score is None:
            self.best_score = current
            return

        # Check for improvement
        if self._is_improvement(current):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.config.patience:
                self._should_stop = True
                self.stopped_epoch = epoch

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement over best.

        Args:
            current: Current metric value.

        Returns:
            True if current is better than best by at least min_delta.
        """
        if self.best_score is None:
            return True

        delta = self.config.min_delta
        if self.config.mode == "min":
            return current < self.best_score - delta
        else:  # max mode
            return current > self.best_score + delta

    def _meets_threshold(self, current: float) -> bool:
        """Check if current value meets the stopping threshold.

        Args:
            current: Current metric value.

        Returns:
            True if threshold is met (training goal achieved).
        """
        threshold = self.config.stopping_threshold
        if threshold is None:
            return False

        if self.config.mode == "min":
            return current <= threshold
        else:  # max mode
            return current >= threshold
