# Inference Optimization

Advanced techniques for optimizing generative model inference: JIT compilation, quantization, memory management, and hardware-specific optimizations.

## Overview

Inference optimization is critical for deploying generative models in production. This guide covers techniques to maximize throughput, minimize latency, and reduce memory usage while maintaining generation quality.

!!! tip "Performance Gains"
    - **JIT Compilation**: 10-100x speedup on first-time compilation
    - **Mixed Precision**: 2-4x throughput improvement
    - **Quantization**: 2-4x memory reduction
    - **Batching**: Near-linear scaling with batch size

<div class="grid cards" markdown>

- :material-flash:{ .lg .middle } **JIT Compilation**

    ---

    Compile JAX functions for dramatic speedups

    [:octicons-arrow-right-24: JIT Guide](#jit-compilation)

- :material-memory:{ .lg .middle } **Memory Optimization**

    ---

    Reduce memory footprint for larger models

    [:octicons-arrow-right-24: Memory Guide](#memory-optimization)

- :material-numeric:{ .lg .middle } **Quantization**

    ---

    Use lower precision for faster inference

    [:octicons-arrow-right-24: Quantization Guide](#quantization)

- :material-server:{ .lg .middle } **Hardware Tuning**

    ---

    Optimize for GPUs, TPUs, and multi-device setups

    [:octicons-arrow-right-24: Hardware Guide](#hardware-specific-optimization)

</div>

---

## Prerequisites

```python
from flax import nnx
import jax
import jax.numpy as jnp
from artifex.generative_models.core import DeviceManager
```

---

## JIT Compilation

### Basic JIT Usage

JAX's Just-In-Time compilation dramatically improves performance.

```python
class JITOptimizedInference:
    """Inference with JIT compilation."""

    def __init__(self, model):
        self.model = model

        # Compile inference function
        self.generate_jit = jax.jit(self._generate_impl)

        # Warmup compilation
        self._warmup()

    def _generate_impl(self, z: jax.Array) -> jax.Array:
        """Implementation to be JIT-compiled."""
        return self.model.decode(z)

    def _warmup(self):
        """Warmup JIT compilation with dummy input."""
        dummy_z = jnp.zeros((1, self.model.latent_dim))
        _ = self.generate_jit(dummy_z)
        print("JIT compilation complete")

    def generate(self, z: jax.Array) -> jax.Array:
        """Generate samples (uses compiled function)."""
        return self.generate_jit(z)
```

### Static vs Dynamic Shapes

JIT works best with static shapes. Handle dynamic shapes carefully.

```python
def create_shape_specific_functions(model, batch_sizes: list[int]):
    """Compile separate functions for each batch size.

    Args:
        model: Generative model
        batch_sizes: List of expected batch sizes

    Returns:
        Dictionary of compiled functions
    """
    compiled_fns = {}

    for batch_size in batch_sizes:
        @jax.jit
        def generate_fn(z):
            return model.decode(z)

        # Warmup with specific shape
        dummy_z = jnp.zeros((batch_size, model.latent_dim))
        _ = generate_fn(dummy_z)

        compiled_fns[batch_size] = generate_fn

    return compiled_fns


# Usage
compiled_fns = create_shape_specific_functions(model, [1, 4, 16, 32])

# Use appropriate function
batch_size = 16
z = jax.random.normal(key, (batch_size, latent_dim))
samples = compiled_fns[batch_size](z)
```

### Compilation Cache Management

```python
import os
from jax.experimental.compilation_cache import compilation_cache

class CachedInference:
    """Inference with persistent compilation cache."""

    def __init__(self, model, cache_dir: str = "/tmp/jax_cache"):
        self.model = model

        # Enable compilation caching
        os.makedirs(cache_dir, exist_ok=True)
        compilation_cache.set_cache_dir(cache_dir)

        # Compile function (cache persists across runs)
        self.generate = jax.jit(model.decode)

        # First run compiles or loads from cache
        self._warmup()

    def _warmup(self):
        dummy_z = jnp.zeros((1, self.model.latent_dim))
        _ = self.generate(dummy_z)
```

---

## Quantization

### Mixed Precision Inference

Use FP16 for faster inference on modern GPUs.

```python
class MixedPrecisionModel(nnx.Module):
    """Model with mixed precision inference."""

    def __init__(
        self,
        base_model,
        *,
        compute_dtype: jnp.dtype = jnp.float16,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.base_model = base_model
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with mixed precision."""
        # Convert input to compute dtype
        x = x.astype(self.compute_dtype)

        # Run model (parameters stay in param_dtype)
        output = self.base_model(x)

        # Convert back to float32 for output
        return output.astype(jnp.float32)


# Create mixed precision model
mixed_model = MixedPrecisionModel(
    model,
    compute_dtype=jnp.float16,
    param_dtype=jnp.float32,
)
```

### Dynamic Quantization

Quantize activations dynamically at inference time.

```python
def quantize_inference(
    model,
    x: jax.Array,
    num_bits: int = 8,
) -> jax.Array:
    """Run inference with dynamic quantization.

    Args:
        model: Model to run
        x: Input tensor
        num_bits: Quantization bits (8 for INT8)

    Returns:
        Model output
    """
    # Quantize input
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (2**num_bits - 1)
    x_quant = jnp.round((x - x_min) / scale).astype(jnp.int8)

    # Dequantize for computation
    x_dequant = x_quant.astype(jnp.float32) * scale + x_min

    # Run model
    output = model(x_dequant)

    return output
```

### INT8 Quantization

Full INT8 quantization for maximum efficiency.

```python
class QuantizedLinear(nnx.Module):
    """INT8 quantized linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        # Initialize weights in FP32
        kernel = jax.random.normal(
            rngs.params(),
            (in_features, out_features)
        ) * jnp.sqrt(2.0 / in_features)

        # Quantize to INT8
        self.scale = jnp.max(jnp.abs(kernel))
        self.kernel_quant = jnp.round(
            kernel / self.scale * 127
        ).astype(jnp.int8)

        # Bias stays in FP32
        self.bias = nnx.Param(jnp.zeros(out_features))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Quantized forward pass."""
        # Dequantize weights
        kernel = self.kernel_quant.astype(jnp.float32) * self.scale / 127

        # Compute
        output = jnp.dot(x, kernel) + self.bias.value

        return output
```

---

## Memory Optimization

### Gradient Checkpointing for Large Models

Save memory by recomputing activations during backward pass.

```python
from jax.experimental import checkify

def memory_efficient_inference(model, x: jax.Array) -> jax.Array:
    """Inference with gradient checkpointing (saves memory).

    Args:
        model: Large generative model
        x: Input

    Returns:
        Model output with reduced memory usage
    """
    # Use checkpoint to reduce memory
    @jax.checkpoint
    def checkpointed_forward(x_inner):
        return model(x_inner)

    return checkpointed_forward(x)
```

### Batch Size Tuning

Find optimal batch size for your hardware.

```python
def find_optimal_batch_size(
    model,
    input_shape: tuple,
    max_batch_size: int = 128,
) -> int:
    """Find largest batch size that fits in memory.

    Args:
        model: Model to test
        input_shape: Shape of single input
        max_batch_size: Maximum batch size to try

    Returns:
        Optimal batch size
    """
    device_manager = DeviceManager()
    device = device_manager.get_device()

    batch_size = 1
    while batch_size <= max_batch_size:
        try:
            # Try inference with this batch size
            x = jax.random.normal(
                jax.random.key(0),
                (batch_size, *input_shape)
            )

            # Move to device
            x = jax.device_put(x, device)

            # Run inference
            _ = model(x)

            # Success - try larger batch
            batch_size *= 2

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM - return previous batch size
                return batch_size // 2
            else:
                raise

    return max_batch_size
```

### Activation Compression

Reduce memory by compressing intermediate activations.

```python
class CompressedModel(nnx.Module):
    """Model with activation compression."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with activation compression."""

        # Process in chunks to reduce peak memory
        chunk_size = 16
        batch_size = x.shape[0]

        outputs = []
        for i in range(0, batch_size, chunk_size):
            chunk = x[i:i + chunk_size]
            chunk_output = self.base_model(chunk)
            outputs.append(chunk_output)

        return jnp.concatenate(outputs, axis=0)
```

---

## Batching Strategies

### Dynamic Batching

Accumulate requests into batches for efficiency.

```python
import asyncio
from collections import deque

class DynamicBatcher:
    """Dynamic batching for inference requests."""

    def __init__(
        self,
        model,
        max_batch_size: int = 32,
        timeout_ms: int = 100,
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms

        # Request queue
        self.queue = deque()
        self.processing = False

    async def infer_async(self, x: jax.Array) -> jax.Array:
        """Submit inference request.

        Args:
            x: Input tensor (single sample)

        Returns:
            Model output
        """
        # Create future for result
        future = asyncio.Future()
        self.queue.append((x, future))

        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self):
        """Process batched requests."""
        self.processing = True

        # Wait for timeout or max batch size
        await asyncio.sleep(self.timeout_ms / 1000)

        if not self.queue:
            self.processing = False
            return

        # Collect batch
        batch_inputs = []
        futures = []

        while self.queue and len(batch_inputs) < self.max_batch_size:
            x, future = self.queue.popleft()
            batch_inputs.append(x)
            futures.append(future)

        # Run batched inference
        batch_x = jnp.stack(batch_inputs)
        batch_output = self.model(batch_x)

        # Return results
        for i, future in enumerate(futures):
            future.set_result(batch_output[i])

        self.processing = False

        # Process remaining requests
        if self.queue:
            asyncio.create_task(self._process_batch())
```

### Pipeline Parallelism

Split large models across multiple devices.

```python
class PipelineParallelModel(nnx.Module):
    """Model with pipeline parallelism."""

    def __init__(
        self,
        encoder,
        decoder,
        devices: list,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device_encoder = devices[0]
        self.device_decoder = devices[1]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with pipeline parallelism."""

        # Encoder on device 0
        x = jax.device_put(x, self.device_encoder)
        latent = self.encoder(x)

        # Transfer to device 1
        latent = jax.device_put(latent, self.device_decoder)
        output = self.decoder(latent)

        return output


# Create pipeline parallel model
device_manager = DeviceManager()
devices = device_manager.get_devices()

if len(devices) >= 2:
    pipeline_model = PipelineParallelModel(
        encoder=model.encoder,
        decoder=model.decoder,
        devices=devices[:2],
    )
```

---

## Hardware-Specific Optimization

### GPU Optimization

Optimize for NVIDIA GPUs.

```python
def optimize_for_gpu(model):
    """Apply GPU-specific optimizations.

    Args:
        model: Model to optimize

    Returns:
        Optimized model
    """
    # Use XLA optimizations
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_fast_min_max=true '
        '--xla_gpu_enable_triton_gemm=true '
        '--xla_gpu_triton_gemm_any=true'
    )

    # Enable tensor cores for mixed precision
    jax.config.update('jax_enable_x64', False)

    # JIT compile with aggressive optimization
    @jax.jit
    def optimized_forward(x):
        return model(x)

    return optimized_forward
```

### TPU Optimization

Optimize for Google TPUs.

```python
def optimize_for_tpu(model, num_devices: int = 8):
    """Apply TPU-specific optimizations.

    Args:
        model: Model to optimize
        num_devices: Number of TPU cores

    Returns:
        Optimized model with data parallelism
    """
    # Replicate model across TPU cores
    @jax.pmap
    def pmap_forward(x):
        return model(x)

    # Shard inputs across devices
    def inference(x):
        # Reshape to (num_devices, batch_per_device, ...)
        batch_size = x.shape[0]
        batch_per_device = batch_size // num_devices

        x_sharded = x.reshape(
            (num_devices, batch_per_device) + x.shape[1:]
        )

        # Run on all TPU cores
        output_sharded = pmap_forward(x_sharded)

        # Concatenate results
        return output_sharded.reshape((batch_size,) + output_sharded.shape[2:])

    return inference
```

### Multi-Device Inference

Distribute inference across multiple devices.

```python
class MultiDeviceInference:
    """Inference distributed across multiple devices."""

    def __init__(self, model):
        self.model = model
        self.device_manager = DeviceManager()
        self.devices = self.device_manager.get_devices()
        self.num_devices = len(self.devices)

        # Replicate model on all devices
        self.replicated_model = jax.pmap(
            lambda x: model(x)
        )

    def infer_distributed(self, x: jax.Array) -> jax.Array:
        """Run inference across all devices.

        Args:
            x: Input batch (must be divisible by num_devices)

        Returns:
            Model output
        """
        batch_size = x.shape[0]
        assert batch_size % self.num_devices == 0

        # Reshape for pmap
        batch_per_device = batch_size // self.num_devices
        x_sharded = x.reshape(
            (self.num_devices, batch_per_device) + x.shape[1:]
        )

        # Run on all devices
        output_sharded = self.replicated_model(x_sharded)

        # Merge results
        return output_sharded.reshape((batch_size,) + output_sharded.shape[2:])
```

---

## Performance Benchmarking

### Throughput Measurement

```python
import time

def benchmark_throughput(
    model,
    input_shape: tuple,
    batch_sizes: list[int],
    num_iterations: int = 100,
):
    """Benchmark model throughput.

    Args:
        model: Model to benchmark
        input_shape: Shape of single input
        batch_sizes: Batch sizes to test
        num_iterations: Number of iterations per test

    Returns:
        Dictionary with throughput results
    """
    results = {}

    for batch_size in batch_sizes:
        # Create dummy input
        x = jax.random.normal(
            jax.random.key(0),
            (batch_size, *input_shape)
        )

        # Warmup
        for _ in range(10):
            _ = model(x)

        # Benchmark
        jax.block_until_ready(model(x))  # Ensure GPU work completes

        start = time.time()
        for _ in range(num_iterations):
            output = model(x)
            jax.block_until_ready(output)
        elapsed = time.time() - start

        # Compute metrics
        samples_per_sec = (batch_size * num_iterations) / elapsed
        latency_ms = (elapsed / num_iterations) * 1000

        results[batch_size] = {
            'throughput': samples_per_sec,
            'latency_ms': latency_ms,
            'batch_size': batch_size,
        }

    return results
```

---

## Best Practices

### DO

!!! success "Recommended Optimizations"
    ✅ **Always JIT compile** production inference functions

    ✅ **Use mixed precision (FP16)** on modern GPUs for 2-4x speedup

    ✅ **Batch requests** to maximize GPU utilization

    ✅ **Cache compiled functions** across server restarts

    ✅ **Profile memory usage** to find optimal batch size

    ✅ **Use pmap for multi-device** inference when available

### DON'T

!!! danger "Avoid These Mistakes"
    ❌ **Don't recompile** on every request (cache JIT functions)

    ❌ **Don't use FP64** unless absolutely necessary (2x slower)

    ❌ **Don't ignore batch size** (single samples waste resources)

    ❌ **Don't mix device types** (CPU/GPU transfers are slow)

    ❌ **Don't skip warmup** (first call is always slow)

    ❌ **Don't quantize without testing** quality impact

---

## Troubleshooting

### Common Performance Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Slow first inference** | 10-60s delay on first call | Warmup JIT compilation during startup |
| **OOM errors** | Out of memory during inference | Reduce batch size or use gradient checkpointing |
| **Low GPU utilization** | GPU usage < 50% | Increase batch size or use pipelining |
| **Recompilation on every call** | Consistent slow performance | Use static shapes or cache per-shape functions |
| **Slow multi-device** | Linear speedup not achieved | Check data transfer overhead, use pmap |
| **High latency** | Individual requests take too long | Use dynamic batching or reduce model size |

---

## Summary

Effective optimization can provide 10-100x speedups:

- **JIT Compilation**: Essential for production (10-100x faster)
- **Mixed Precision**: 2-4x throughput on modern GPUs
- **Quantization**: Reduce memory 2-4x with minimal quality loss
- **Batching**: Near-linear scaling with batch size
- **Multi-Device**: Distribute large workloads across GPUs/TPUs

Choose optimizations based on your deployment constraints and quality requirements.

---

## Next Steps

<div class="grid cards" markdown>

- :material-flask:{ .lg .middle } **Sampling Methods**

    ---

    Learn advanced sampling techniques for quality generation

    [:octicons-arrow-right-24: Sampling Guide](sampling.md)

- :material-cloud-upload:{ .lg .middle } **Deployment**

    ---

    Deploy optimized models to production

    [:octicons-arrow-right-24: Deployment Guide](../integrations/deployment.md)

- :material-chart-line:{ .lg .middle } **Benchmarking**

    ---

    Measure and compare model performance

    [:octicons-arrow-right-24: Evaluation Framework](../../benchmarks/index.md)

- :material-gpu:{ .lg .middle } **Distributed Training**

    ---

    Scale training across multiple devices

    [:octicons-arrow-right-24: Distributed Guide](../advanced/distributed.md)

</div>
