# Training Profiling

Profiling is configured at the shared `Trainer` layer through `CallbackList`.
Use the trace profiler for short focused windows and the memory profiler for
longer-running visibility into device usage.

## JAX Trace Profiling

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.callbacks import (
    CallbackList,
    JAXProfiler,
    ProfilingConfig,
)

callbacks = CallbackList(
    [
        JAXProfiler(
            ProfilingConfig(
                log_dir="logs/profiles",
                start_step=10,
                end_step=20,
            )
        )
    ]
)

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=loss_fn,
    callbacks=callbacks,
)

trainer.train(train_data=train_data, num_epochs=10, batch_size=64)
```

`ProfilingConfig` only exposes:

| Parameter | Description |
|-----------|-------------|
| `log_dir` | Output directory for trace files |
| `start_step` | First profiled training step |
| `end_step` | Final profiled training step |

## Memory Profiling

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    MemoryProfiler,
    MemoryProfileConfig,
)

callbacks = CallbackList(
    [
        MemoryProfiler(
            MemoryProfileConfig(
                log_dir="logs/memory",
                profile_every_n_steps=100,
                log_device_memory=True,
            )
        )
    ]
)
```

## Recommendations

- Put `start_step` after initial compilation warmup.
- Keep the trace window small.
- Use memory snapshots for long jobs where full tracing would be too expensive.
- Combine profiling with logging callbacks only when you need both signals in
  the same run.

## Related Documentation

- [Logging](logging.md)
- [Training Guide](training-guide.md)
- [Training Profiling Reference](../../training/profiling.md)
