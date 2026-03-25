# Profiling Callbacks

**Status:** `Supported runtime training surface`

**Module:** `artifex.generative_models.training.callbacks.profiling`

**Source:** `src/artifex/generative_models/training/callbacks/profiling.py`

Profiling callbacks run through the shared `Trainer` callback system. Artifex
currently exposes a narrow JAX trace profiler and a separate memory profiler.

## ProfilingConfig

```python
from artifex.generative_models.training.callbacks import ProfilingConfig

config = ProfilingConfig(
    log_dir="logs/profiles",
    start_step=10,
    end_step=20,
)
```

`ProfilingConfig` only controls where traces are written and which training
steps define the capture window.

## MemoryProfileConfig

```python
from artifex.generative_models.training.callbacks import MemoryProfileConfig

config = MemoryProfileConfig(
    log_dir="logs/memory",
    profile_every_n_steps=100,
    log_device_memory=True,
)
```

`MemoryProfileConfig` controls the output directory and how often device memory
snapshots are collected.

## Shared Trainer Integration

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.training.callbacks import (
    CallbackList,
    JAXProfiler,
    MemoryProfiler,
    MemoryProfileConfig,
    ProfilingConfig,
)

callbacks = CallbackList(
    [
        JAXProfiler(ProfilingConfig(log_dir="logs/profiles", start_step=10, end_step=20)),
        MemoryProfiler(
            MemoryProfileConfig(log_dir="logs/memory", profile_every_n_steps=100)
        ),
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

## Practical Guidance

- Set `start_step` after the first JIT warmup steps.
- Keep the trace window short.
- Use `MemoryProfiler` when you need long-running memory visibility.
- Treat `JAXProfiler` and `MemoryProfiler` as separate tools; they solve
  different problems.

## Related Documentation

- [Training Systems](index.md)
- [Logging Callbacks](logging.md)
- [User Guide: Profiling](../user-guide/training/profiling.md)
