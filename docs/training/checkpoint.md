# Checkpointing Callbacks

**Module:** `generative_models.training.callbacks.checkpoint`

**Source:** `generative_models/training/callbacks/checkpoint.py`

## Overview

Model checkpointing callback that monitors metrics and saves checkpoints when they improve. Uses Orbax checkpointing under the hood with minimal overhead when not saving.

## Classes

### CheckpointConfig

```python
@dataclass(slots=True)
class CheckpointConfig:
    """Configuration for model checkpointing."""

    dirpath: str | Path = "checkpoints"
    filename: str = "model-{epoch:02d}-{val_loss:.4f}"
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    save_top_k: int = 3
    save_last: bool = True
    every_n_epochs: int = 1
    save_weights_only: bool = False
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dirpath` | `str \| Path` | `"checkpoints"` | Directory to save checkpoints |
| `filename` | `str` | `"model-{epoch:02d}-{val_loss:.4f}"` | Filename template with `{epoch}` and metric placeholders |
| `monitor` | `str` | `"val_loss"` | Metric name to monitor |
| `mode` | `Literal["min", "max"]` | `"min"` | Whether lower or higher is better |
| `save_top_k` | `int` | `3` | Number of best checkpoints to keep (-1 = all, 0 = none) |
| `save_last` | `bool` | `True` | Whether to save checkpoint on every epoch |
| `every_n_epochs` | `int` | `1` | Save checkpoint every n epochs |
| `save_weights_only` | `bool` | `False` | Only save model weights (not optimizer state) |

---

### ModelCheckpoint

```python
class ModelCheckpoint(BaseCallback):
    """Save model checkpoints based on monitored metrics."""

    def __init__(self, config: CheckpointConfig): ...
```

Callback that saves model checkpoints when monitored metrics improve. Uses Orbax checkpointing infrastructure with automatic cleanup of old checkpoints.

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `best_score` | `float \| None` | Best metric value seen so far |
| `best_checkpoint_path` | `Path \| None` | Path to the best checkpoint |
| `saved_checkpoints` | `list[tuple[float, Path]]` | List of (score, path) for saved checkpoints |

---

## Usage

### Basic Checkpointing

```python
from artifex.generative_models.training.callbacks import (
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

trainer.fit(callbacks=[checkpoint])

# Access best checkpoint after training
print(f"Best checkpoint: {checkpoint.best_checkpoint_path}")
print(f"Best score: {checkpoint.best_score}")
```

### Monitor Accuracy (Higher is Better)

```python
checkpoint = ModelCheckpoint(CheckpointConfig(
    dirpath="./checkpoints",
    monitor="val_accuracy",
    mode="max",  # Higher accuracy is better
    save_top_k=1,  # Keep only the best
    filename="best-model-{epoch:02d}-{val_accuracy:.4f}",
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

### Custom Filename Template

```python
# Filename template supports {epoch} and any metric in logs
checkpoint = ModelCheckpoint(CheckpointConfig(
    filename="model-epoch{epoch:03d}-loss{val_loss:.6f}-acc{val_accuracy:.4f}",
    monitor="val_loss",
))
```

### Combined with Other Callbacks

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    ModelCheckpoint,
    CheckpointConfig,
    EarlyStopping,
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
    EarlyStopping(EarlyStoppingConfig(
        monitor="val_loss",
        patience=10,
    )),
    ProgressBarCallback(ProgressBarConfig()),
])

trainer.fit(callbacks=callbacks)
```

---

## How It Works

1. **Metric Monitoring**: Tracks the specified metric (`monitor`) at the end of each epoch
2. **Improvement Check**: Compares current value against best using `mode` (min/max)
3. **Checkpoint Saving**: Uses Orbax infrastructure to save model state
4. **Automatic Cleanup**: Removes old checkpoints beyond `save_top_k` limit
5. **Best Tracking**: Maintains reference to the best checkpoint path

### Checkpoint Cleanup Strategy

When `save_top_k > 0`, checkpoints are sorted by their metric value:

- **Min mode**: Keeps checkpoints with lowest scores, removes highest
- **Max mode**: Keeps checkpoints with highest scores, removes lowest

---

## Integration with Orbax

ModelCheckpoint uses the existing Orbax-based checkpointing infrastructure:

```python
from artifex.generative_models.core.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    setup_checkpoint_manager,
)

# Checkpoints are saved using Orbax under the hood
# You can load them manually:
checkpoint_manager, _ = setup_checkpoint_manager("./checkpoints")
model = load_checkpoint(checkpoint_manager, model, step=10)
```

See [Checkpointing Guide](../user-guide/advanced/checkpointing.md) for advanced checkpointing features including optimizer state and corruption recovery.

---

## Module Statistics

- **Classes:** 2 (CheckpointConfig, ModelCheckpoint)
- **Dependencies:** Orbax checkpointing infrastructure
- **Slots:** Uses `__slots__` for memory efficiency
