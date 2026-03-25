# TensorBoard Integration

Artifex currently supports TensorBoard through training callbacks. It does not
ship a separate fit-style integration layer or a custom TensorBoard-specific
trainer class.

## Supported Owners

- `TensorBoardLoggerCallback`
- `TensorBoardLoggerConfig`
- `CallbackList`
- `Trainer.train(...)`
- `JAXProfiler`
- `ProfilingConfig`

## Wire The Built-In Callback

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    JAXProfiler,
    ProfilingConfig,
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
)
from artifex.generative_models.training.trainer import Trainer

callbacks = CallbackList(
    [
        TensorBoardLoggerCallback(
            TensorBoardLoggerConfig(
                log_dir="logs/experiment-1",
                flush_secs=60,
                log_every_n_steps=10,
            )
        ),
        JAXProfiler(
            ProfilingConfig(
                log_dir="logs/profiles",
                start_step=10,
                end_step=20,
            )
        ),
    ]
)

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=loss_fn,
    callbacks=callbacks,
)

trainer.train(
    train_data=train_data,
    num_epochs=10,
    batch_size=32,
    val_data=val_data,
)
```

## What Gets Logged

- batch metrics emitted through `on_batch_end`
- epoch summaries emitted through `on_epoch_end`
- validation summaries emitted through `on_validation_end`
- profiler traces captured by `JAXProfiler`

## Extending The Integration

If you need extra TensorBoard behavior, implement a `BaseCallback` or reuse
`LoggerCallback` and add it to `CallbackList`. Keep custom integration code on
the supported callback hooks that the live `Trainer` actually invokes.
