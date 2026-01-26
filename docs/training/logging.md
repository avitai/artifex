# Logging Callbacks

**Module:** `generative_models.training.callbacks.logging`

**Source:** `generative_models/training/callbacks/logging.py`

## Overview

Logging callbacks for integrating experiment tracking and progress monitoring into training loops. These callbacks wrap existing logging infrastructure for seamless integration with the callback system.

## Classes

### LoggerCallbackConfig

```python
@dataclass(slots=True)
class LoggerCallbackConfig:
    log_every_n_steps: int = 1
    log_on_epoch_end: bool = True
    prefix: str = ""
```

Configuration for base logger callback.

**Attributes:**

- `log_every_n_steps`: Log metrics every N training steps
- `log_on_epoch_end`: Whether to log metrics at end of each epoch
- `prefix`: Prefix to add to metric names

---

### LoggerCallback

```python
class LoggerCallback(BaseCallback):
    def __init__(
        self,
        logger: Logger,
        config: Optional[LoggerCallbackConfig] = None,
    )
```

Base callback that wraps any `Logger` instance for training integration.

**Parameters:**

- `logger`: Logger instance to use for logging (from `utils.logging`)
- `config`: Logging configuration. Uses defaults if not provided

**Example:**

```python
from artifex.generative_models.utils.logging import ConsoleLogger
from artifex.generative_models.training.callbacks import LoggerCallback

logger = ConsoleLogger(name="training")
callback = LoggerCallback(logger=logger)
trainer.fit(callbacks=[callback])
```

---

### WandbLoggerConfig

```python
@dataclass(slots=True)
class WandbLoggerConfig:
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
```

Configuration for W&B logging callback.

**Attributes:**

- `project`: W&B project name (required)
- `entity`: W&B entity (username or team name)
- `name`: Run name. If None, W&B auto-generates
- `tags`: List of tags for the run
- `notes`: Notes about the run
- `config`: Dictionary of hyperparameters to log
- `mode`: W&B mode: "online", "offline", or "disabled"
- `resume`: Whether to resume a previous run
- `log_every_n_steps`: Log metrics every N training steps
- `log_on_epoch_end`: Whether to log metrics at end of each epoch
- `log_dir`: Local directory for W&B files

---

### WandbLoggerCallback

```python
class WandbLoggerCallback(BaseCallback):
    def __init__(self, config: WandbLoggerConfig)
```

Weights & Biases experiment tracking callback.

**Features:**

- Automatic metric logging
- Hyperparameter tracking
- Run resumption support
- Multiple run modes (online, offline, disabled)

**Example:**

```python
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
)

config = WandbLoggerConfig(
    project="my-project",
    name="experiment-1",
    tags=["baseline", "vae"],
    config={"learning_rate": 1e-3, "batch_size": 32},
)
callback = WandbLoggerCallback(config=config)
trainer.fit(callbacks=[callback])
```

---

### TensorBoardLoggerConfig

```python
@dataclass(slots=True)
class TensorBoardLoggerConfig:
    log_dir: str = "logs/tensorboard"
    flush_secs: int = 120
    max_queue: int = 10
    log_every_n_steps: int = 1
    log_on_epoch_end: bool = True
    log_graph: bool = False
```

Configuration for TensorBoard logging callback.

**Attributes:**

- `log_dir`: Directory for TensorBoard logs
- `flush_secs`: How often to flush to disk (seconds)
- `max_queue`: Max queue size for pending events
- `log_every_n_steps`: Log metrics every N training steps
- `log_on_epoch_end`: Whether to log metrics at end of each epoch
- `log_graph`: Whether to log model graph

---

### TensorBoardLoggerCallback

```python
class TensorBoardLoggerCallback(BaseCallback):
    def __init__(self, config: TensorBoardLoggerConfig)
```

TensorBoard logging callback.

**Features:**

- Scalar metrics logging
- Configurable flush interval
- Per-step and per-epoch logging

**Requirements:** Requires `tensorboard` package (`pip install tensorboard`)

**Example:**

```python
from artifex.generative_models.training.callbacks import (
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
)

config = TensorBoardLoggerConfig(
    log_dir="logs/experiment-1",
    flush_secs=60,
)
callback = TensorBoardLoggerCallback(config=config)
trainer.fit(callbacks=[callback])
```

---

### ProgressBarConfig

```python
@dataclass(slots=True)
class ProgressBarConfig:
    refresh_rate: int = 10
    show_eta: bool = True
    show_metrics: bool = True
    leave: bool = True
    disable: bool = False
```

Configuration for progress bar callback.

**Attributes:**

- `refresh_rate`: How often to refresh the progress bar (steps)
- `show_eta`: Whether to show estimated time of arrival
- `show_metrics`: Whether to display metrics in progress bar
- `leave`: Whether to leave progress bar after completion
- `disable`: Whether to disable progress bar entirely

---

### ProgressBarCallback

```python
class ProgressBarCallback(BaseCallback):
    def __init__(self, config: Optional[ProgressBarConfig] = None)
```

Rich console progress bar callback.

**Features:**

- Real-time training progress
- Metric display
- ETA estimation
- Nested progress for epochs/steps

**Requirements:** Requires `rich` package (`pip install rich`)

**Example:**

```python
from artifex.generative_models.training.callbacks import (
    ProgressBarCallback,
    ProgressBarConfig,
)

config = ProgressBarConfig(
    refresh_rate=10,
    show_eta=True,
    show_metrics=True,
)
callback = ProgressBarCallback(config=config)
trainer.fit(callbacks=[callback])
```

## Usage with Multiple Callbacks

Logging callbacks can be combined with other callbacks:

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
    WandbLoggerCallback,
    WandbLoggerConfig,
    ProgressBarCallback,
)

# Configure callbacks
callbacks = CallbackList([
    # Logging
    WandbLoggerCallback(WandbLoggerConfig(
        project="my-project",
        name="experiment-1",
    )),
    ProgressBarCallback(),

    # Training control
    EarlyStopping(EarlyStoppingConfig(
        monitor="val_loss",
        patience=10,
    )),

    # Checkpointing
    ModelCheckpoint(CheckpointConfig(
        dirpath="checkpoints",
        monitor="val_loss",
    )),
])

trainer.fit(callbacks=callbacks)
```

## Module Statistics

- **Classes:** 8
- **Functions:** 0
- **Imports:** 6
