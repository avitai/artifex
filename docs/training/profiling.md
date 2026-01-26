# Profiling Callbacks

**Module:** `generative_models.training.callbacks.profiling`

**Source:** `generative_models/training/callbacks/profiling.py`

## Overview

Profiling callbacks for integrating JAX-native performance analysis into training loops. These callbacks provide trace-based profiling for TensorBoard visualization and memory usage tracking with minimal overhead.

## Classes

### ProfilingConfig

```python
@dataclass(slots=True)
class ProfilingConfig:
    log_dir: str = "logs/profiles"
    start_step: int = 10
    end_step: int = 20
    trace_memory: bool = True
    trace_python: bool = False
```

Configuration for JAX trace profiling.

**Attributes:**

- `log_dir`: Directory to save profiling traces
- `start_step`: Step at which to start profiling (skip warmup)
- `end_step`: Step at which to stop profiling
- `trace_memory`: Whether to include memory usage in traces
- `trace_python`: Whether to trace Python execution (slower but more detail)

---

### JAXProfiler

```python
class JAXProfiler(BaseCallback):
    def __init__(self, config: ProfilingConfig)
```

JAX profiler callback for performance analysis.

Integrates with JAX's built-in profiler to capture traces that can be viewed in TensorBoard or Perfetto. Automatically skips warmup steps to get more representative profiling data.

**Features:**

- Integration with JAX's built-in profiler
- TensorBoard trace visualization
- Configurable profiling window (start/end steps)
- Automatic cleanup on training end
- No interference with JIT compilation
- Minimal overhead outside profiling window

**Example:**

```python
from artifex.generative_models.training.callbacks import (
    JAXProfiler,
    ProfilingConfig,
)

config = ProfilingConfig(
    log_dir="logs/profiles",
    start_step=10,  # Skip JIT warmup
    end_step=20,    # Profile 10 steps
)
profiler = JAXProfiler(config)
trainer.fit(callbacks=[profiler])

# View in TensorBoard:
# tensorboard --logdir logs/profiles
```

**Best Practices:**

- Set `start_step` after JIT warmup (typically 5-10 steps)
- Keep profiling window small (10-20 steps) to minimize impact
- Use `trace_python=True` only when debugging Python bottlenecks
- Traces can be viewed in TensorBoard or [Perfetto](https://ui.perfetto.dev/)

---

### MemoryProfileConfig

```python
@dataclass(slots=True)
class MemoryProfileConfig:
    log_dir: str = "logs/memory"
    profile_every_n_steps: int = 100
    log_device_memory: bool = True
```

Configuration for memory profiling.

**Attributes:**

- `log_dir`: Directory to save memory profile data
- `profile_every_n_steps`: Collect memory info every N steps
- `log_device_memory`: Whether to log device (GPU/TPU) memory stats

---

### MemoryProfiler

```python
class MemoryProfiler(BaseCallback):
    def __init__(self, config: MemoryProfileConfig)
```

Memory usage profiling callback.

Tracks memory usage during training and saves a timeline to JSON. Useful for identifying memory leaks and understanding memory patterns.

**Features:**

- Track JAX device memory usage (GPU/TPU)
- Log peak memory per step
- Export memory timeline to JSON
- Configurable profiling interval
- Minimal overhead between collection intervals

**Example:**

```python
from artifex.generative_models.training.callbacks import (
    MemoryProfiler,
    MemoryProfileConfig,
)

config = MemoryProfileConfig(
    log_dir="logs/memory",
    profile_every_n_steps=50,
)
profiler = MemoryProfiler(config)
trainer.fit(callbacks=[profiler])

# Memory profile saved to logs/memory/memory_profile.json
```

**Output Format:**

The memory profile is saved as JSON with the following structure:

```json
[
  {
    "step": 0,
    "memory": {
      "cuda:0": {
        "bytes_in_use": 1073741824,
        "peak_bytes_in_use": 2147483648
      }
    }
  },
  {
    "step": 100,
    "memory": {
      "cuda:0": {
        "bytes_in_use": 1073741824,
        "peak_bytes_in_use": 2147483648
      }
    }
  }
]
```

**Note:** Not all devices support `memory_stats()`. CPU devices typically return `None`, in which case those devices are skipped.

---

## Usage with Multiple Callbacks

Profiling callbacks can be combined with other callbacks:

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
    JAXProfiler,
    ProfilingConfig,
    MemoryProfiler,
    MemoryProfileConfig,
    ProgressBarCallback,
)

# Configure callbacks
callbacks = CallbackList([
    # Profiling (runs first to capture full training)
    JAXProfiler(ProfilingConfig(
        log_dir="logs/profiles",
        start_step=10,
        end_step=20,
    )),
    MemoryProfiler(MemoryProfileConfig(
        log_dir="logs/memory",
        profile_every_n_steps=100,
    )),

    # Progress display
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

## Performance Considerations

Both profiling callbacks are designed for minimal overhead:

- **JAXProfiler**: Zero overhead outside the profiling window (start_step to end_step)
- **MemoryProfiler**: Only collects data at configured intervals; no overhead between intervals

The callbacks do not interfere with JAX's JIT compilation. JIT-compiled functions produce identical results before, during, and after profiling.

## Viewing Traces

### TensorBoard

```bash
tensorboard --logdir logs/profiles
```

Navigate to the "Profile" tab to view:

- XLA compilation times
- Device execution times
- Memory allocation patterns
- Kernel execution traces

### Perfetto

1. Go to [Perfetto UI](https://ui.perfetto.dev/)
2. Click "Open trace file"
3. Select the `.trace` file from your log directory

Perfetto provides more detailed trace analysis capabilities including:

- Timeline visualization
- Flame graphs
- Memory analysis

## Module Statistics

- **Classes:** 4
- **Functions:** 0
- **Imports:** 4
