# Performance Profiling

This guide covers profiling tools in Artifex for analyzing and optimizing training performance, including JAX trace profiling and memory tracking.

## Overview

Artifex provides two profiling callbacks:

- **JAXProfiler**: Captures JAX execution traces for visualization in TensorBoard/Perfetto
- **MemoryProfiler**: Tracks GPU/TPU memory usage over time

These tools help identify performance bottlenecks, optimize memory usage, and understand training dynamics.

## JAX Trace Profiling

### Quick Start

```python
from artifex.generative_models.training.callbacks import (
    JAXProfiler,
    ProfilingConfig,
)

# Create profiler callback
profiler = JAXProfiler(ProfilingConfig(
    log_dir="logs/profiles",
    start_step=10,  # Skip warmup/JIT compilation
    end_step=20,    # Profile 10 steps
))

# Add to trainer callbacks
trainer = Trainer(
    model=model,
    training_config=training_config,
    callbacks=[profiler],
)
```

### Configuration

```python
from artifex.generative_models.training.callbacks import (
    JAXProfiler,
    ProfilingConfig,
)

config = ProfilingConfig(
    log_dir="logs/profiles",    # Directory for trace files
    start_step=10,              # Step to start profiling
    end_step=20,                # Step to stop profiling
    trace_memory=True,          # Include memory in traces
    trace_python=False,         # Trace Python execution (slower)
)

profiler = JAXProfiler(config)
```

### ProfilingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | `str` | `"logs/profiles"` | Directory for trace files |
| `start_step` | `int` | `10` | Step to start profiling |
| `end_step` | `int` | `20` | Step to stop profiling |
| `trace_memory` | `bool` | `True` | Include memory in traces |
| `trace_python` | `bool` | `False` | Trace Python (slower, more detail) |

### Viewing Traces

#### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/profiles

# Navigate to "Profile" tab in browser
```

#### Perfetto

1. Open [ui.perfetto.dev](https://ui.perfetto.dev) in your browser
2. Click "Open trace file"
3. Select the `.perfetto-trace` file from your log directory

### Understanding Traces

The trace shows:

- **XLA Compilation**: Time spent compiling JAX programs
- **Kernel Execution**: Time spent running operations on GPU/TPU
- **Memory Allocation**: When and how much memory is allocated
- **Data Transfer**: Host-to-device and device-to-host transfers

#### Common Patterns to Look For

**Good patterns:**

- Most time in kernel execution
- Minimal data transfers
- Steady memory usage

**Potential issues:**

- Repeated XLA compilation (missing JIT)
- Frequent host-device transfers
- Memory spikes indicating inefficient allocation

### Profiling Best Practices

#### 1. Skip Warmup

```python
config = ProfilingConfig(
    start_step=10,  # Skip first 10 steps (JIT compilation)
    end_step=20,
)
```

The first few steps include JIT compilation overhead. Skip them to get representative performance data.

#### 2. Profile Short Windows

```python
config = ProfilingConfig(
    start_step=100,
    end_step=110,  # Only 10 steps
)
```

Profiling is expensive. Keep the window short (10-20 steps) for manageable trace files.

#### 3. Profile Representative Workloads

```python
# Profile at different batch sizes
for batch_size in [32, 64, 128]:
    config = ProfilingConfig(
        log_dir=f"logs/profiles/batch_{batch_size}",
        start_step=10,
        end_step=20,
    )
    # Run training...
```

## Memory Profiling

### Quick Start

```python
from artifex.generative_models.training.callbacks import (
    MemoryProfiler,
    MemoryProfileConfig,
)

# Create memory profiler
profiler = MemoryProfiler(MemoryProfileConfig(
    log_dir="logs/memory",
    profile_every_n_steps=100,
))

# Add to trainer callbacks
trainer = Trainer(
    model=model,
    training_config=training_config,
    callbacks=[profiler],
)
```

### Configuration

```python
from artifex.generative_models.training.callbacks import (
    MemoryProfiler,
    MemoryProfileConfig,
)

config = MemoryProfileConfig(
    log_dir="logs/memory",           # Directory for memory profile
    profile_every_n_steps=100,       # Frequency of memory checks
    log_device_memory=True,          # Log GPU/TPU memory stats
)

profiler = MemoryProfiler(config)
```

### MemoryProfileConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | `str` | `"logs/memory"` | Directory for profile output |
| `profile_every_n_steps` | `int` | `100` | Profiling frequency |
| `log_device_memory` | `bool` | `True` | Track device memory |

### Output Format

Memory profiles are saved as JSON:

```json
[
  {
    "step": 100,
    "memory": {
      "cuda:0": {
        "bytes_in_use": 1073741824,
        "peak_bytes_in_use": 1610612736
      }
    }
  },
  {
    "step": 200,
    "memory": {
      "cuda:0": {
        "bytes_in_use": 1073741824,
        "peak_bytes_in_use": 1610612736
      }
    }
  }
]
```

### Analyzing Memory Profiles

```python
import json
import matplotlib.pyplot as plt

# Load profile
with open("logs/memory/memory_profile.json") as f:
    profile = json.load(f)

# Extract data
steps = [p["step"] for p in profile]
memory = [p["memory"]["cuda:0"]["bytes_in_use"] / 1e9 for p in profile]  # GB
peak_memory = [p["memory"]["cuda:0"]["peak_bytes_in_use"] / 1e9 for p in profile]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, memory, label="Current")
plt.plot(steps, peak_memory, label="Peak")
plt.xlabel("Step")
plt.ylabel("Memory (GB)")
plt.legend()
plt.title("GPU Memory Usage")
plt.savefig("memory_plot.png")
```

## Complete Profiling Example

```python
from artifex.generative_models.training import Trainer, TrainingConfig
from artifex.generative_models.training.callbacks import (
    JAXProfiler,
    ProfilingConfig,
    MemoryProfiler,
    MemoryProfileConfig,
    ProgressBarCallback,
)


def profile_training(model, train_data, num_steps=1000):
    """Profile a training run."""

    callbacks = [
        # JAX trace profiling (steps 100-110)
        JAXProfiler(ProfilingConfig(
            log_dir="logs/profiles/trace",
            start_step=100,
            end_step=110,
        )),

        # Memory profiling (every 50 steps)
        MemoryProfiler(MemoryProfileConfig(
            log_dir="logs/profiles/memory",
            profile_every_n_steps=50,
        )),

        # Progress bar for feedback
        ProgressBarCallback(),
    ]

    trainer = Trainer(
        model=model,
        training_config=TrainingConfig(num_epochs=1),
        callbacks=callbacks,
    )

    trainer.train(train_data)

    print("Profiling complete!")
    print("- Trace: logs/profiles/trace/")
    print("- Memory: logs/profiles/memory/memory_profile.json")
```

## Common Performance Issues

### 1. Excessive JIT Compilation

**Symptom**: Slow first few steps, traces show XLA compilation

**Solution**:

```python
# Ensure functions are JIT-compiled once
@jax.jit
def train_step(model, batch, key):
    ...
    return loss, metrics

# Don't use Python control flow that changes trace
# Bad: changes trace every step
if step % 10 == 0:
    # Different computation path

# Good: use jax.lax.cond for conditional logic
```

### 2. Memory Leaks

**Symptom**: Memory usage increases over time

**Solution**:

```python
# Check for accumulating state
# Bad: accumulating in Python list
all_losses = []
for step in range(num_steps):
    loss = train_step(...)
    all_losses.append(loss)  # Memory leak!

# Good: aggregate in-place or periodically
running_loss = 0.0
for step in range(num_steps):
    loss = train_step(...)
    running_loss += float(loss)  # Convert to Python float
```

### 3. Data Transfer Bottlenecks

**Symptom**: Long gaps between kernel executions in traces

**Solution**:

```python
# Pre-transfer data to device
data = jax.device_put(data)

# Use asynchronous data loading
# Prefetch next batch while current batch processes
```

### 4. Inefficient Batch Size

**Symptom**: Low GPU utilization

**Solution**:

```python
# Profile with different batch sizes
for batch_size in [32, 64, 128, 256]:
    # Compare throughput and memory usage
    ...

# Use gradient accumulation for effective larger batches
from artifex.generative_models.training import GradientAccumulator
```

## Manual Profiling

For custom profiling beyond the callbacks:

```python
import jax

# Manual trace profiling
with jax.profiler.trace("logs/manual_profile"):
    for step in range(10):
        loss = train_step(model, batch, key)
        jax.block_until_ready(loss)  # Ensure completion

# Check device memory
for device in jax.devices():
    stats = device.memory_stats()
    if stats:
        print(f"{device}: {stats['bytes_in_use'] / 1e9:.2f} GB")

# Profile specific operations
with jax.profiler.StepTraceAnnotation("forward_pass"):
    logits = model(inputs)

with jax.profiler.StepTraceAnnotation("backward_pass"):
    grads = jax.grad(loss_fn)(model)
```

## Integration with Other Callbacks

Profiling callbacks work alongside other callbacks:

```python
from artifex.generative_models.training.callbacks import (
    JAXProfiler,
    ProfilingConfig,
    MemoryProfiler,
    MemoryProfileConfig,
    WandbLoggerCallback,
    WandbLoggerConfig,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
)

callbacks = [
    # Profiling
    JAXProfiler(ProfilingConfig(start_step=100, end_step=110)),
    MemoryProfiler(MemoryProfileConfig(profile_every_n_steps=100)),

    # Logging
    WandbLoggerCallback(WandbLoggerConfig(project="my-project")),

    # Training control
    EarlyStopping(EarlyStoppingConfig(patience=10)),
    ModelCheckpoint(CheckpointConfig(dirpath="checkpoints")),
]
```

## Related Documentation

- [Logging](logging.md) - Experiment tracking and visualization
- [Training Guide](training-guide.md) - Core training patterns
- [Advanced Features](advanced-features.md) - Gradient accumulation, mixed precision
- [Distributed Training](../advanced/distributed.md) - Multi-device training
