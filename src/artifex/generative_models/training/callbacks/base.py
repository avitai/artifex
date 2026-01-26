"""Base callback protocol and container for training lifecycle hooks.

Provides a lightweight, Protocol-based callback system for training loops.
Designed for minimal overhead - callbacks are simple function calls with no
additional abstraction layers.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from flax import nnx


@runtime_checkable
class TrainerLike(Protocol):
    """Minimal protocol for what callbacks need from a trainer.

    This decouples callbacks from the concrete Trainer implementation,
    following the Interface Segregation Principle. Callbacks should only
    depend on this protocol, not the full Trainer class.

    The actual Trainer class satisfies this protocol.
    """

    @property
    def model(self) -> nnx.Module:
        """The NNX model being trained."""
        ...


@runtime_checkable
class TrainingCallback(Protocol):
    """Protocol defining training lifecycle hooks.

    Implement any subset of these methods to respond to training events.
    Methods not implemented will use no-op defaults via BaseCallback.

    This is a Protocol, not an ABC - implementations don't need to inherit,
    they just need to implement the required methods.
    """

    def on_train_begin(self, trainer: TrainerLike) -> None:
        """Called at the start of training."""
        ...

    def on_train_end(self, trainer: TrainerLike) -> None:
        """Called at the end of training."""
        ...

    def on_epoch_begin(self, trainer: TrainerLike, epoch: int) -> None:
        """Called at the start of each epoch."""
        ...

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        ...

    def on_batch_begin(self, trainer: TrainerLike, batch: int) -> None:
        """Called at the start of each batch."""
        ...

    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any]) -> None:
        """Called at the end of each batch."""
        ...

    def on_validation_begin(self, trainer: TrainerLike) -> None:
        """Called at the start of validation."""
        ...

    def on_validation_end(self, trainer: TrainerLike, logs: dict[str, Any]) -> None:
        """Called at the end of validation."""
        ...


class BaseCallback:
    """Base class providing no-op implementations for all callback methods.

    Inherit from this class and override only the methods you need.
    This avoids boilerplate while ensuring all callbacks have a consistent interface.
    """

    __slots__ = ()  # No instance attributes for minimal memory

    def on_train_begin(self, _trainer: TrainerLike) -> None:
        """Called at the start of training."""

    def on_train_end(self, _trainer: TrainerLike) -> None:
        """Called at the end of training."""

    def on_epoch_begin(self, _trainer: TrainerLike, _epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, _trainer: TrainerLike, _epoch: int, _logs: dict[str, Any]) -> None:
        """Called at the end of each epoch."""

    def on_batch_begin(self, _trainer: TrainerLike, _batch: int) -> None:
        """Called at the start of each batch."""

    def on_batch_end(self, _trainer: TrainerLike, _batch: int, _logs: dict[str, Any]) -> None:
        """Called at the end of each batch."""

    def on_validation_begin(self, _trainer: TrainerLike) -> None:
        """Called at the start of validation."""

    def on_validation_end(self, _trainer: TrainerLike, _logs: dict[str, Any]) -> None:
        """Called at the end of validation."""


class CallbackList:
    """Container for managing multiple callbacks with minimal overhead.

    Dispatches events to all registered callbacks in order.
    Uses simple iteration with no additional abstraction.
    """

    __slots__ = ("callbacks",)  # Minimal memory footprint

    def __init__(self, callbacks: list[TrainingCallback] | None = None):
        """Initialize with optional list of callbacks.

        Args:
            callbacks: List of callback instances. Defaults to empty list.
        """
        self.callbacks: list[TrainingCallback] = callbacks or []

    def add(self, callback: TrainingCallback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def remove(self, callback: TrainingCallback) -> None:
        """Remove a callback from the list."""
        self.callbacks.remove(callback)

    def __len__(self) -> int:
        """Return number of callbacks."""
        return len(self.callbacks)

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)

    # Event dispatch methods - inline for minimal overhead
    # Use positional args for maximum performance (no dict creation)

    def on_train_begin(self, trainer: TrainerLike) -> None:
        """Dispatch on_train_begin to all callbacks."""
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: TrainerLike) -> None:
        """Dispatch on_train_end to all callbacks."""
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: TrainerLike, epoch: int) -> None:
        """Dispatch on_epoch_begin to all callbacks."""
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None:
        """Dispatch on_epoch_end to all callbacks."""
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer: TrainerLike, batch: int) -> None:
        """Dispatch on_batch_begin to all callbacks."""
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch)

    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any]) -> None:
        """Dispatch on_batch_end to all callbacks."""
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch, logs)

    def on_validation_begin(self, trainer: TrainerLike) -> None:
        """Dispatch on_validation_begin to all callbacks."""
        for cb in self.callbacks:
            cb.on_validation_begin(trainer)

    def on_validation_end(self, trainer: TrainerLike, logs: dict[str, Any]) -> None:
        """Dispatch on_validation_end to all callbacks."""
        for cb in self.callbacks:
            cb.on_validation_end(trainer, logs)
