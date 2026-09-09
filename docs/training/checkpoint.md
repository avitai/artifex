# Checkpointing Callbacks

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.callbacks.checkpoint`

**Source:** `src/artifex/generative_models/training/callbacks/checkpoint.py`

## Overview

Model checkpointing callback that saves Orbax-managed checkpoints on the
configured epoch cadence. Retention and best-checkpoint selection are handled
by Orbax using the monitored metric.

## Classes

### CheckpointConfig

```python
@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for model checkpointing."""

    dirpath: str | Path = "checkpoints"
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    every_n_epochs: int = 1
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dirpath` | `str \| Path` | `"checkpoints"` | Directory to save checkpoints |
| `monitor` | `str` | `"val_loss"` | Metric name to monitor |
| `mode` | `Literal["min", "max"]` | `"min"` | Whether lower or higher is better |
| `save_top_k` | `int` | `3` | Number of best checkpoints to keep (-1 = all, 0 = none) |
| `every_n_epochs` | `int` | `1` | Save checkpoint every n epochs |

---

### ModelCheckpoint

```python
class ModelCheckpoint(BaseCallback):
    """Save model checkpoints based on monitored metrics."""

    def __init__(self, config: CheckpointConfig): ...
```

Callback that saves model checkpoints when monitored metrics improve. Uses Orbax checkpointing infrastructure with automatic cleanup of old checkpoints.
Callback that saves eligible checkpoints and delegates best-step tracking and
retention to Orbax.

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `best_score` | `float \| None` | Best metric value seen so far |
| `best_checkpoint_step` | `int \| None` | Step index for the best retained checkpoint |
| `saved_checkpoint_steps` | `list[int]` | Retained checkpoint steps managed by Orbax |

---

## Usage

### Basic Checkpointing

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.callbacks import (
    CallbackList,
    ModelCheckpoint,
    CheckpointConfig,
)

# Save best 3 checkpoints based on validation loss
checkpoint = ModelCheckpoint(CheckpointConfig(
    dirpath="./checkpoints",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
))

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=loss_fn,
    callbacks=CallbackList([checkpoint]),
)
trainer.train(train_data=train_data, num_epochs=10, batch_size=64, val_data=val_data)

# Access best checkpoint metadata after training
print(f"Best checkpoint step: {checkpoint.best_checkpoint_step}")
print(f"Best score: {checkpoint.best_score}")
```

### Monitor Accuracy (Higher is Better)

```python
checkpoint = ModelCheckpoint(CheckpointConfig(
    dirpath="./checkpoints",
    monitor="val_accuracy",
    mode="max",  # Higher accuracy is better
    save_top_k=1,  # Keep only the best
))
```

### Save All Checkpoints

```python
checkpoint = ModelCheckpoint(CheckpointConfig(
    dirpath="./checkpoints",
    save_top_k=-1,  # Keep all checkpoints
    every_n_epochs=5,  # Save every 5 epochs
))
```

### Combined with Other Callbacks

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    ModelCheckpoint,
    CheckpointConfig,
    EarlyStoppingCallback,
    EarlyStoppingConfig,
    ProgressBarCallback,
    ProgressBarConfig,
)

callbacks = CallbackList([
    ModelCheckpoint(CheckpointConfig(
        dirpath="./checkpoints",
        monitor="val_loss",
        save_top_k=3,
    )),
    EarlyStoppingCallback(EarlyStoppingConfig(
        monitor="val_loss",
        patience=10,
    )),
    ProgressBarCallback(ProgressBarConfig()),
])

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=loss_fn,
    callbacks=callbacks,
)
trainer.train(train_data=train_data, num_epochs=10, batch_size=64, val_data=val_data)
```

---

## How It Works

1. **Metric Monitoring**: Tracks the specified metric (`monitor`) at the end of each epoch
2. **Orbax Save**: Saves the model state through substrax's `OrbaxCheckpointStore`
3. **Retention Policy**: the store keeps the newest `save_top_k` checkpoints, which are
   the `save_top_k` best because a checkpoint is written only on improvement
4. **Best Tracking**: `best_checkpoint_step` is the store's `best_step` over the monitored metric

---

## Integration with Orbax

ModelCheckpoint writes through substrax's `OrbaxCheckpointStore`:

```python
from substrax.checkpoint import OrbaxCheckpointStore

# Checkpoints are stored under step-numbered Orbax directories
with OrbaxCheckpointStore("./checkpoints") as store:
    model, metadata = store.restore(model, step=10)
```

See [Checkpointing Guide](../user-guide/advanced/checkpointing.md) for the store's
full surface, including trainer checkpoints that carry optimizer state.

---

## Module Statistics

- **Classes:** 2 (CheckpointConfig, ModelCheckpoint)
- **Dependencies:** substrax's Orbax checkpoint store
- **Slots:** Uses `__slots__` for memory efficiency
