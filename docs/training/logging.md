# Logging Callbacks

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.callbacks.logging`

**Source:** `src/artifex/generative_models/training/callbacks/logging.py`

Logging callbacks wrap the shared logging backends and progress display helpers
used by the low-level `Trainer`.

## Overview

Artifex ships four logging-facing callback families:

- `LoggerCallback` for any repo-owned `Logger` implementation
- `WandbLoggerCallback` for Weights & Biases tracking
- `TensorBoardLoggerCallback` for scalar/event logging
- `ProgressBarCallback` for Rich-based terminal progress output

All of them plug into `CallbackList` and run through `trainer.train(...)`.

## LoggerCallbackConfig

```python
from artifex.generative_models.training.callbacks import LoggerCallbackConfig

config = LoggerCallbackConfig(
    log_every_n_steps=10,
    log_on_epoch_end=True,
    prefix="train/",
)
```

`LoggerCallbackConfig` controls scalar logging frequency and metric prefixing.

## TensorBoardLoggerConfig

```python
from artifex.generative_models.training.callbacks import TensorBoardLoggerConfig

config = TensorBoardLoggerConfig(
    log_dir="logs/tensorboard",
    flush_secs=60,
    max_queue=20,
    log_every_n_steps=10,
    log_on_epoch_end=True,
)
```

`TensorBoardLoggerConfig` only documents the runtime-active settings:
`log_dir`, `flush_secs`, `max_queue`, `log_every_n_steps`, and
`log_on_epoch_end`.

## ProgressBarConfig

```python
from artifex.generative_models.training.callbacks import ProgressBarConfig

config = ProgressBarConfig(
    show_eta=True,
    show_metrics=True,
    leave=False,
)
```

`ProgressBarConfig` controls whether the callback shows ETA, whether metrics are
rendered inline, whether finished tasks remain visible, and whether the bar is
disabled entirely.

## Shared Trainer Integration

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.callbacks import (
    CallbackList,
    ProgressBarCallback,
    ProgressBarConfig,
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
)

callbacks = CallbackList(
    [
        TensorBoardLoggerCallback(
            TensorBoardLoggerConfig(log_dir="logs/tensorboard", log_every_n_steps=10)
        ),
        ProgressBarCallback(ProgressBarConfig(show_metrics=True, show_eta=True)),
    ]
)

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=loss_fn,
    callbacks=callbacks,
)

trainer.train(train_data=train_data, num_epochs=10, batch_size=64, val_data=val_data)
```

## WandB Usage

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    WandbLoggerCallback,
    WandbLoggerConfig,
)

callbacks = CallbackList(
    [
        WandbLoggerCallback(
            WandbLoggerConfig(
                project="artifex-experiments",
                name="baseline-run",
                tags=["gan", "baseline"],
                config={"learning_rate": 1e-4},
            )
        )
    ]
)
```

Use `mode="offline"` when you want local logging without a live W&B session.

## Related Documentation

- [Training Systems](index.md)
- [Profiling Callbacks](profiling.md)
- [User Guide: Logging](../user-guide/training/logging.md)
