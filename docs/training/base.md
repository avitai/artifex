# Callbacks Base

**Module:** `generative_models.training.callbacks.base`

**Source:** `generative_models/training/callbacks/base.py`

## Overview

Base classes and protocols for the training callback system. Provides a lightweight, Protocol-based callback system for training loops that avoids circular imports through the `TrainerLike` protocol.

## Protocols

### TrainerLike

```python
class TrainerLike(Protocol):
    """Protocol for trainer-like objects that callbacks can interact with.

    This protocol defines the minimal interface that a trainer must implement
    to work with callbacks. Using a protocol avoids circular imports between
    callbacks and trainer modules.
    """

    @property
    def model(self) -> nnx.Module:
        """The NNX model being trained."""
        ...
```

Defines the minimal interface for trainers to work with callbacks without causing circular imports. The actual Trainer class satisfies this protocol.

---

### TrainingCallback

```python
class TrainingCallback(Protocol):
    """Protocol defining the callback interface."""

    def on_train_begin(self, trainer: TrainerLike) -> None: ...
    def on_train_end(self, trainer: TrainerLike) -> None: ...
    def on_epoch_begin(self, trainer: TrainerLike, epoch: int) -> None: ...
    def on_epoch_end(self, trainer: TrainerLike, epoch: int, logs: dict[str, Any]) -> None: ...
    def on_batch_begin(self, trainer: TrainerLike, batch: int) -> None: ...
    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict[str, Any]) -> None: ...
    def on_validation_begin(self, trainer: TrainerLike) -> None: ...
    def on_validation_end(self, trainer: TrainerLike, logs: dict[str, Any]) -> None: ...
```

Defines the 8 lifecycle hooks that callbacks can implement.

## Classes

### BaseCallback

```python
class BaseCallback:
    """Base callback class with no-op implementations of all hooks."""
```

Base class providing no-op implementations of all callback hooks. Extend this class and override only the methods you need.

**Example:**

```python
from artifex.generative_models.training.callbacks import BaseCallback

class MyCallback(BaseCallback):
    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch} completed with loss: {metrics.get('loss', 'N/A')}")
```

---

### CallbackList

```python
class CallbackList:
    """Container for multiple callbacks that dispatches to all."""

    def __init__(self, callbacks: list[TrainingCallback] | None = None): ...
    def append(self, callback: TrainingCallback) -> None: ...
```

Container that holds multiple callbacks and dispatches lifecycle events to all of them.

**Example:**

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
)

callbacks = CallbackList([
    EarlyStopping(EarlyStoppingConfig(patience=10)),
    ModelCheckpoint(CheckpointConfig(dirpath="./checkpoints")),
])

# All callbacks receive events
callbacks.on_epoch_end(trainer, epoch=5, metrics={"loss": 0.5})
```

## Usage

```python
from artifex.generative_models.training.callbacks import (
    TrainerLike,
    TrainingCallback,
    BaseCallback,
    CallbackList,
)

# Implement custom callback
class LoggingCallback(BaseCallback):
    def on_batch_end(self, trainer: TrainerLike, batch: int, logs: dict):
        if batch % 100 == 0:
            print(f"Batch {batch}: {logs}")

# Use with trainer
callbacks = CallbackList([LoggingCallback()])
```

## Module Statistics

- **Classes:** 2 (BaseCallback, CallbackList)
- **Protocols:** 2 (TrainerLike, TrainingCallback)
- **Imports:** 4
