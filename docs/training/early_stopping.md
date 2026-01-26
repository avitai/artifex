# Early Stopping Callbacks

**Module:** `generative_models.training.callbacks.early_stopping`

**Source:** `generative_models/training/callbacks/early_stopping.py`

## Overview

Early stopping callback that monitors a metric and stops training when it stops improving. Designed for minimal overhead with simple comparisons and no external dependencies.

## Classes

### EarlyStoppingConfig

```python
@dataclass(slots=True)
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    monitor: str = "val_loss"
    min_delta: float = 0.0
    patience: int = 10
    mode: Literal["min", "max"] = "min"
    check_finite: bool = True
    stopping_threshold: float | None = None
    divergence_threshold: float | None = None
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitor` | `str` | `"val_loss"` | Metric name to monitor |
| `min_delta` | `float` | `0.0` | Minimum change to qualify as improvement |
| `patience` | `int` | `10` | Epochs without improvement before stopping |
| `mode` | `Literal["min", "max"]` | `"min"` | Whether lower or higher is better |
| `check_finite` | `bool` | `True` | Stop when metric becomes NaN or Inf |
| `stopping_threshold` | `float \| None` | `None` | Stop immediately when metric reaches this value |
| `divergence_threshold` | `float \| None` | `None` | Stop if metric exceeds this (min mode only) |

---

### EarlyStopping

```python
class EarlyStopping(BaseCallback):
    """Stop training when a monitored metric stops improving."""

    def __init__(self, config: EarlyStoppingConfig): ...
```

Callback that tracks a metric and signals when training should stop due to lack of improvement or other conditions.

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `should_stop` | `bool` | Whether training should stop |
| `best_score` | `float \| None` | Best metric value seen |
| `wait_count` | `int` | Number of epochs without improvement |
| `stopped_epoch` | `int \| None` | Epoch when stopping was triggered |

---

## Usage

### Basic Early Stopping

```python
from artifex.generative_models.training.callbacks import (
    EarlyStopping,
    EarlyStoppingConfig,
)

# Stop if validation loss doesn't improve for 10 epochs
early_stop = EarlyStopping(EarlyStoppingConfig(
    monitor="val_loss",
    patience=10,
    mode="min",
))

trainer.fit(callbacks=[early_stop])

# Check if training stopped early
if early_stop.should_stop:
    print(f"Stopped at epoch {early_stop.stopped_epoch}")
```

### Monitor Accuracy (Higher is Better)

```python
early_stop = EarlyStopping(EarlyStoppingConfig(
    monitor="val_accuracy",
    patience=15,
    mode="max",  # Higher accuracy is better
    min_delta=0.001,  # Require at least 0.1% improvement
))
```

### Stop at Target Performance

```python
# Stop when validation loss reaches 0.01 (goal achieved)
early_stop = EarlyStopping(EarlyStoppingConfig(
    monitor="val_loss",
    stopping_threshold=0.01,
    mode="min",
))
```

### Detect Training Divergence

```python
# Stop if loss explodes (divergence detection)
early_stop = EarlyStopping(EarlyStoppingConfig(
    monitor="val_loss",
    patience=10,
    mode="min",
    divergence_threshold=10.0,  # Stop if loss > 10
    check_finite=True,  # Also stop on NaN/Inf
))
```

### Combined with Checkpointing

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
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
        min_delta=1e-4,
    )),
])

trainer.fit(callbacks=callbacks)
```

---

## How It Works

1. **Metric Tracking**: At each epoch end, reads the monitored metric from logs
2. **Finite Check**: Optionally stops immediately if metric is NaN or Inf
3. **Threshold Check**: Optionally stops if target performance reached
4. **Divergence Check**: Optionally stops if metric exceeds threshold (min mode)
5. **Improvement Check**: Compares current value to best with `min_delta` tolerance
6. **Patience Tracking**: Counts epochs without improvement, stops when patience exceeded

### Improvement Criteria

For a value to be considered an improvement:

- **Min mode**: `current < best - min_delta`
- **Max mode**: `current > best + min_delta`

### Stopping Conditions

Training stops when ANY of these conditions are met:

1. **Non-finite value**: `check_finite=True` and metric is NaN or Inf
2. **Goal reached**: `stopping_threshold` is set and metric reaches target
3. **Divergence**: `divergence_threshold` is set and metric exceeds it (min mode)
4. **No improvement**: `patience` epochs pass without improvement

---

## Integration with Training Loop

The trainer checks `callback.should_stop` after each epoch:

```python
# Inside trainer
for epoch in range(num_epochs):
    # Training step...
    logs = {"val_loss": val_loss, "val_accuracy": val_acc}

    # Call callbacks
    for callback in callbacks:
        callback.on_epoch_end(self, epoch, logs)

        # Check for early stopping
        if hasattr(callback, "should_stop") and callback.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            return
```

---

## Module Statistics

- **Classes:** 2 (EarlyStoppingConfig, EarlyStopping)
- **Dependencies:** None (uses only Python standard library)
- **Slots:** Uses `__slots__` for memory efficiency
