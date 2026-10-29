# Checkpointing Callbacks

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.callbacks.checkpoint`

**Source:** `src/artifex/generative_models/training/callbacks/checkpoint.py`

## Overview

Checkpointing callback that saves the complete native numerical run through the
shared Checkpointer and Substrax store on the configured epoch cadence. Each checkpoint is written
at the trainer's global step, with the monitored metric in the record's
`metrics` and the epoch in its `epoch`. Retention and best-checkpoint selection
are the store's, over the monitored metric.

## Classes

### CheckpointConfig

```python
@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for model checkpointing."""

    dirpath: str | Path
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    every_n_epochs: int = 1
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dirpath` | `str \| Path` | required | Directory to save checkpoints; the callback never picks one under the working directory |
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

The callback saves model, native step (optimizer and objective state), base key
and registered state when its metric improves. The `CheckpointingTrainer`
protocol supplies the global `step`, `run_state()` and `check_finite()`.
Incomplete trainers are refused before saving. Failed numerical admission or
persistence does not advance the callback's recorded best score.

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `best_score` | `float \| None` | Best metric value seen so far |
| `best_checkpoint_step` | `int \| None` | Global step of the best retained checkpoint |
| `saved_checkpoint_steps` | `list[int]` | Retained checkpoint steps, as the store lists them |

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
2. **Native Save**: Checks pending numerical failure and saves complete native
   state through Checkpointer, at the trainer's global step, with the metric and
   actual run epoch in the record
3. **Retention Policy**: the store keeps the newest `save_top_k` checkpoints, which are
   the `save_top_k` best because a checkpoint is written only on improvement
4. **Best Tracking**: `best_checkpoint_step` is the store's `best_step` over the monitored metric

---

## Integration with Orbax

ModelCheckpoint writes through substrax's `OrbaxCheckpointStore`:

```python
from substrax.checkpoint import OrbaxCheckpointStore

# Rebuild a Trainer with the same model, step and registered-state structure.
with OrbaxCheckpointStore("./checkpoints") as store:
    best = store.best_step("val_loss", mode="min")
    checkpoint = store.restore(best, templates=trainer.checkpoint_state())
trainer.apply_checkpoint_state(checkpoint.items, step=checkpoint.step)
print(checkpoint.metadata.metrics["val_loss"], checkpoint.metadata.epoch)
```

See [Checkpointing Guide](../user-guide/advanced/checkpointing.md) for the store's
full surface, including trainer checkpoints that carry optimizer state.

---

## Module Statistics

- **Classes:** 2 (CheckpointConfig, ModelCheckpoint)
- **Dependencies:** substrax's Orbax checkpoint store
- **Slots:** Uses `__slots__` for memory efficiency
