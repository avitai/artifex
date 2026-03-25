# Training Logging

Artifex logging lives at the shared `Trainer` layer. Model-family trainers own
objectives; callback-driven logging, checkpointing, and progress reporting are
attached through `CallbackList`.

## Core Pattern

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.callbacks import CallbackList

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=loss_fn,
    callbacks=CallbackList([]),
)

trainer.train(train_data=train_data, num_epochs=10, batch_size=64, val_data=val_data)
```

## TensorBoard

Install TensorBoard with:

```bash
pip install tensorboard
```

Then configure the callback:

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
)

callbacks = CallbackList(
    [
        TensorBoardLoggerCallback(
            TensorBoardLoggerConfig(
                log_dir="logs/tensorboard",
                flush_secs=60,
                max_queue=20,
                log_every_n_steps=10,
                log_on_epoch_end=True,
            )
        )
    ]
)
```

`TensorBoardLoggerConfig` supports only the live runtime knobs above.

## Progress Bars

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    ProgressBarCallback,
    ProgressBarConfig,
)

callbacks = CallbackList(
    [
        ProgressBarCallback(
            ProgressBarConfig(
                show_eta=True,
                show_metrics=True,
                leave=False,
            )
        )
    ]
)
```

`ProgressBarConfig` is intentionally small:

| Parameter | Description |
|-----------|-------------|
| `show_eta` | Render estimated time remaining |
| `show_metrics` | Render up to a few metrics inline |
| `leave` | Keep finished bars visible |
| `disable` | Turn the callback off entirely |

## WandB

Install W&B with:

```bash
pip install wandb
```

Example:

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
                project="artifex",
                name="experiment-01",
                tags=["baseline"],
                config={"learning_rate": 1e-4, "batch_size": 64},
            )
        )
    ]
)
```

## Combining Logging Callbacks

```python
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
```

## Related Documentation

- [Training Systems](../../training/index.md)
- [Profiling](profiling.md)
- [Training Guide](training-guide.md)
