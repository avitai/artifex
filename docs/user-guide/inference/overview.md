# Inference Overview

This guide provides a complete overview of performing inference with Artifex models, covering model loading, batch processing, streaming inference, and performance optimization.

## Overview

<div class="grid cards" markdown>

- :material-download:{ .lg .middle } **Model Loading**

    ---

    Load trained models for inference efficiently

    [:octicons-arrow-right-24: Model Loading](#model-loading)

- :material-file-multiple:{ .lg .middle } **Batch Inference**

    ---

    Process multiple inputs efficiently with batching

    [:octicons-arrow-right-24: Batch Inference](#batch-inference)

- :material-access-point:{ .lg .middle } **Streaming Inference**

    ---

    Real-time generation with streaming

    [:octicons-arrow-right-24: Streaming](#streaming-inference)

- :material-speedometer:{ .lg .middle } **Performance**

    ---

    Optimize inference speed and memory

    [:octicons-arrow-right-24: Performance](#performance-optimization)

</div>

## Prerequisites

```bash
uv pip install "artifex[cuda]"  # With GPU support
```

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.core import DeviceManager
from artifex.generative_models.core.checkpointing import load_checkpoint, setup_checkpoint_manager
```

## Model Loading

### Loading from Checkpoint

```python
from artifex.generative_models.core.checkpointing import (
    load_checkpoint,
    setup_checkpoint_manager,
)
from artifex.generative_models.models.vae import VAE

def load_model_for_inference(
    checkpoint_dir: str,
    config: dict,
) -> VAE:
    """
    Load a trained model from checkpoint for inference.

    Args:
        checkpoint_dir: Path to checkpoint directory
        config: Model configuration dictionary

    Returns:
        Loaded model ready for inference
    """
    # Initialize device manager
    device_manager = DeviceManager()

    # Create model template with same architecture
    rngs = nnx.Rngs(0)  # Seed doesn't matter for loading
    model_template = VAE(
        input_shape=config["input_shape"],
        latent_dim=config["latent_dim"],
        encoder_features=config.get("encoder_features", [32, 64, 128]),
        decoder_features=config.get("decoder_features", [128, 64, 32]),
        rngs=rngs,
    )

    # Setup checkpoint manager
    checkpoint_manager, _ = setup_checkpoint_manager(checkpoint_dir)

    # Load weights
    model, step = load_checkpoint(checkpoint_manager, model_template)

    print(f"Loaded model from step {step}")

    return model


# Example usage
config = {
    "input_shape": (28, 28, 1),
    "latent_dim": 64,
    "encoder_features": [32, 64, 128],
    "decoder_features": [128, 64, 32],
}

model = load_model_for_inference("./checkpoints/vae", config)
```

### Loading from Export Directory

```python
import json
from pathlib import Path

def load_exported_model(export_dir: str):
    """
    Load model from export directory with metadata.

    Args:
        export_dir: Path to export directory

    Returns:
        Tuple of (model, config, metadata)
    """
    export_path = Path(export_dir)

    # Load metadata
    with open(export_path / "metadata.json") as f:
        metadata = json.load(f)

    # Load config
    with open(export_path / "config.json") as f:
        config = json.load(f)

    # Load model
    model = load_model_for_inference(str(export_path / "checkpoints"), config)

    print(f"Loaded model: {metadata['model_type']}")
    print(f"Trained for {metadata['training_steps']} steps")
    print(f"Final loss: {metadata['final_loss']:.4f}")

    return model, config, metadata


# Example
model, config, metadata = load_exported_model("./exports/vae_mnist_final")
```

### JIT Compilation for Inference

```python
@jax.jit
def generate_samples_jit(model: VAE, z: jax.Array) -> jax.Array:
    """JIT-compiled sample generation."""
    return model.decode(z)


@jax.jit
def encode_images_jit(model: VAE, images: jax.Array) -> jax.Array:
    """JIT-compiled encoding."""
    latent_params = model.encode(images)
    # Return mean for deterministic encoding
    mean = latent_params["mean"]
    return mean


def inference_with_jit(model: VAE, num_samples: int = 16):
    """Perform inference with JIT compilation."""

    # First call compiles (may be slow)
    print("Compiling...")
    rngs = nnx.Rngs(42)
    z = jax.random.normal(rngs.sample(), (num_samples, model.latent_dim))

    samples = generate_samples_jit(model, z)
    print(f"Compiled! Generated {samples.shape}")

    # Subsequent calls are fast
    print("Running inference...")
    import time
    start = time.time()

    for _ in range(10):
        z = jax.random.normal(rngs.sample(), (num_samples, model.latent_dim))
        samples = generate_samples_jit(model, z)

    elapsed = time.time() - start
    print(f"10 batches in {elapsed:.3f}s ({elapsed / 10 * 1000:.1f}ms per batch)")

    return samples
```

## Batch Inference

### Processing Multiple Inputs

```python
def batch_encode_images(
    model: VAE,
    images: jnp.ndarray,
    batch_size: int = 32,
) -> jnp.ndarray:
    """
    Encode images in batches.

    Args:
        model: Trained VAE model
        images: Input images [N, H, W, C]
        batch_size: Batch size for processing

    Returns:
        Encoded latents [N, latent_dim]
    """
    num_samples = len(images)
    latents = []

    for i in range(0, num_samples, batch_size):
        batch = images[i:i + batch_size]

        # Encode batch
        latent_params = model.encode(batch)
        batch_latents = latent_params["mean"]  # Use mean for deterministic

        latents.append(batch_latents)

    return jnp.concatenate(latents, axis=0)


def batch_generate_samples(
    model: VAE,
    num_samples: int,
    batch_size: int = 32,
    *,
    rngs: nnx.Rngs,
) -> jnp.ndarray:
    """
    Generate samples in batches.

    Args:
        model: Trained VAE model
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        rngs: Random number generators

    Returns:
        Generated samples [num_samples, H, W, C]
    """
    samples = []

    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)

        # Sample latent codes
        z = jax.random.normal(rngs.sample(), (current_batch_size, model.latent_dim))

        # Generate
        batch_samples = model.decode(z)
        samples.append(batch_samples)

    return jnp.concatenate(samples, axis=0)
```

### Parallel Batch Processing

```python
def parallel_batch_inference(
    model: VAE,
    images: jnp.ndarray,
    num_devices: int = None,
) -> jnp.ndarray:
    """
    Process batches in parallel across devices.

    Args:
        model: Trained model
        images: Input images
        num_devices: Number of devices to use (None = all available)

    Returns:
        Processed outputs
    """
    devices = jax.devices()[:num_devices] if num_devices else jax.devices()
    num_devices = len(devices)

    print(f"Using {num_devices} devices for parallel inference")

    # Split data across devices
    batch_size_per_device = len(images) // num_devices
    image_shards = [
        images[i * batch_size_per_device:(i + 1) * batch_size_per_device]
        for i in range(num_devices)
    ]

    # Process in parallel
    @jax.jit
    def process_shard(shard):
        latent_params = model.encode(shard)
        return latent_params["mean"]

    # Map over shards
    latents_shards = jax.tree_map(process_shard, image_shards)

    # Concatenate results
    return jnp.concatenate(latents_shards, axis=0)
```

## Streaming Inference

### Real-Time Generation

```python
class StreamingGenerator:
    """Stream samples one at a time for real-time applications."""

    def __init__(
        self,
        model: VAE,
        seed: int = 42,
    ):
        self.model = model
        self.rngs = nnx.Rngs(seed)

        # Pre-compile generation function
        self._generate_fn = jax.jit(self._generate_single)

    def _generate_single(self, z: jax.Array) -> jax.Array:
        """Generate single sample (JIT-compiled)."""
        return self.model.decode(z)

    def __iter__(self):
        """Iterator interface for streaming."""
        return self

    def __next__(self) -> jax.Array:
        """Generate next sample."""
        z = jax.random.normal(self.rngs.sample(), (1, self.model.latent_dim))
        sample = self._generate_fn(z)
        return sample[0]  # Remove batch dimension


# Usage
def stream_samples(model: VAE, num_samples: int = 100):
    """Stream samples in real-time."""
    import time

    generator = StreamingGenerator(model, seed=42)

    print("Streaming samples...")
    for i, sample in enumerate(generator):
        if i >= num_samples:
            break

        # Process sample (e.g., display, save, send over network)
        print(f"Sample {i + 1}: shape {sample.shape}")

        # Simulate processing time
        time.sleep(0.01)
```

### Asynchronous Inference

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncInferenceServer:
    """Asynchronous inference server for concurrent requests."""

    def __init__(
        self,
        model: VAE,
        max_workers: int = 4,
    ):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Pre-compile functions
        self.encode_fn = jax.jit(lambda x: model.encode(x)["mean"])
        self.decode_fn = jax.jit(model.decode)

    async def encode_async(self, images: jnp.ndarray) -> jnp.ndarray:
        """Asynchronously encode images."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.encode_fn, images)

    async def decode_async(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Asynchronously decode latents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.decode_fn, latents)

    async def generate_async(self, num_samples: int, rngs: nnx.Rngs) -> jnp.ndarray:
        """Asynchronously generate samples."""
        z = jax.random.normal(rngs.sample(), (num_samples, self.model.latent_dim))
        return await self.decode_async(z)

    def shutdown(self):
        """Shutdown executor."""
        self.executor.shutdown(wait=True)


# Usage
async def process_concurrent_requests():
    """Process multiple inference requests concurrently."""
    model = load_model_for_inference("./checkpoints/vae", config)
    server = AsyncInferenceServer(model, max_workers=4)

    # Simulate concurrent requests
    tasks = [
        server.generate_async(16, nnx.Rngs(i))
        for i in range(10)
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    print(f"Processed {len(results)} concurrent requests")

    server.shutdown()
    return results


# Run async inference
# results = asyncio.run(process_concurrent_requests())
```

## Performance Optimization

### Memory Management

```python
def memory_efficient_inference(
    model: VAE,
    images: jnp.ndarray,
    batch_size: int = 16,
):
    """
    Memory-efficient inference with explicit cleanup.

    Args:
        model: Trained model
        images: Input images
        batch_size: Small batch size to reduce memory usage

    Returns:
        Processed outputs
    """
    device_manager = DeviceManager()
    results = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]

        # Process batch
        latent_params = model.encode(batch)
        latents = latent_params["mean"]

        # Convert to numpy to free device memory
        results.append(jnp.array(latents))

        # Explicit cleanup every N batches
        if (i // batch_size) % 10 == 0:
            device_manager.cleanup()

    return jnp.concatenate(results, axis=0)
```

### Inference Benchmarking

```python
import time
from typing import Callable

def benchmark_inference(
    inference_fn: Callable,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict:
    """
    Benchmark inference performance.

    Args:
        inference_fn: Function to benchmark
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs (not counted)

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup_runs):
        _ = inference_fn()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = inference_fn()

        # Force synchronization (important for JAX)
        result.block_until_ready() if hasattr(result, 'block_until_ready') else None

        elapsed = time.time() - start
        times.append(elapsed)

    times = jnp.array(times)

    return {
        "mean_ms": float(jnp.mean(times) * 1000),
        "std_ms": float(jnp.std(times) * 1000),
        "min_ms": float(jnp.min(times) * 1000),
        "max_ms": float(jnp.max(times) * 1000),
        "median_ms": float(jnp.median(times) * 1000),
    }


# Example usage
def benchmark_vae_inference(model: VAE):
    """Benchmark VAE inference."""

    rngs = nnx.Rngs(42)

    # Benchmark encoding
    test_images = jax.random.normal(rngs.sample(), (32, 28, 28, 1))

    def encode_fn():
        return model.encode(test_images)["mean"]

    encode_stats = benchmark_inference(encode_fn, num_runs=100)
    print("Encoding performance:")
    print(f"  Mean: {encode_stats['mean_ms']:.2f}ms ± {encode_stats['std_ms']:.2f}ms")
    print(f"  Median: {encode_stats['median_ms']:.2f}ms")

    # Benchmark decoding
    z = jax.random.normal(rngs.sample(), (32, model.latent_dim))

    def decode_fn():
        return model.decode(z)

    decode_stats = benchmark_inference(decode_fn, num_runs=100)
    print("\nDecoding performance:")
    print(f"  Mean: {decode_stats['mean_ms']:.2f}ms ± {decode_stats['std_ms']:.2f}ms")
    print(f"  Median: {decode_stats['median_ms']:.2f}ms")

    return encode_stats, decode_stats
```

### Throughput Optimization

```python
def optimize_batch_size(
    model: VAE,
    max_batch_size: int = 256,
    step: int = 16,
) -> int:
    """
    Find optimal batch size for throughput.

    Args:
        model: Trained model
        max_batch_size: Maximum batch size to try
        step: Batch size increment

    Returns:
        Optimal batch size
    """
    rngs = nnx.Rngs(42)
    best_batch_size = step
    best_throughput = 0.0

    print("Testing batch sizes for optimal throughput...")

    for batch_size in range(step, max_batch_size + 1, step):
        try:
            # Generate test batch
            z = jax.random.normal(rngs.sample(), (batch_size, model.latent_dim))

            # Benchmark
            def inference_fn():
                return model.decode(z)

            stats = benchmark_inference(inference_fn, num_runs=20, warmup_runs=5)

            # Calculate throughput (samples per second)
            throughput = (batch_size / stats["mean_ms"]) * 1000

            print(f"  Batch size {batch_size}: {throughput:.1f} samples/sec")

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch_size = batch_size

        except Exception as e:
            print(f"  Batch size {batch_size}: Failed ({e})")
            break

    print(f"\nOptimal batch size: {best_batch_size}")
    print(f"Peak throughput: {best_throughput:.1f} samples/sec")

    return best_batch_size
```

## Common Patterns

### Inference Pipeline

```python
class InferencePipeline:
    """Complete inference pipeline with preprocessing and postprocessing."""

    def __init__(
        self,
        model: VAE,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
    ):
        self.model = model
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.postprocess_fn = postprocess_fn or (lambda x: x)

        # JIT-compile core functions
        self.encode_fn = jax.jit(lambda x: model.encode(x)["mean"])
        self.decode_fn = jax.jit(model.decode)

    def encode(self, images: jnp.ndarray) -> jnp.ndarray:
        """Full encoding pipeline."""
        # Preprocess
        preprocessed = self.preprocess_fn(images)

        # Encode
        latents = self.encode_fn(preprocessed)

        return latents

    def decode(self, latents: jnp.ndarray) -> jnp.ndarray:
        """Full decoding pipeline."""
        # Decode
        samples = self.decode_fn(latents)

        # Postprocess
        postprocessed = self.postprocess_fn(samples)

        return postprocessed

    def reconstruct(self, images: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct images through encode-decode."""
        latents = self.encode(images)
        return self.decode(latents)

    def generate(self, num_samples: int, *, rngs: nnx.Rngs) -> jnp.ndarray:
        """Generate new samples."""
        z = jax.random.normal(rngs.sample(), (num_samples, self.model.latent_dim))
        return self.decode(z)


# Example with preprocessing/postprocessing
def normalize_images(images):
    """Normalize to [-1, 1]."""
    return (images / 255.0) * 2.0 - 1.0


def denormalize_images(images):
    """Denormalize to [0, 255]."""
    return ((images + 1.0) / 2.0 * 255.0).astype(jnp.uint8)


pipeline = InferencePipeline(
    model=model,
    preprocess_fn=normalize_images,
    postprocess_fn=denormalize_images,
)

# Use pipeline
samples = pipeline.generate(16, rngs=nnx.Rngs(42))
```

## Best Practices

!!! success "DO"
    - Always JIT-compile inference functions
    - Use appropriate batch sizes for your hardware
    - Pre-compile during initialization (warmup)
    - Monitor memory usage with large batches
    - Use asynchronous inference for servers
    - Benchmark different batch sizes

!!! danger "DON'T"
    - Don't skip JIT compilation for production
    - Don't use batch size 1 unless necessary
    - Don't forget to call `block_until_ready()` in benchmarks
    - Don't load models in hot paths
    - Don't ignore device memory limits
    - Don't use floating point for final outputs

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow first inference | JIT compilation | Pre-compile with warmup runs |
| Out of memory | Batch too large | Reduce batch size, use memory-efficient mode |
| Inconsistent timing | Async execution | Use `block_until_ready()` for accurate timing |
| Low throughput | Small batches | Increase batch size, optimize for hardware |
| High latency | Not using JIT | JIT-compile all inference functions |

## Summary

This guide covered:

1. **Model Loading**: Load from checkpoints or exports
2. **Batch Inference**: Process multiple inputs efficiently
3. **Streaming**: Real-time generation for interactive applications
4. **Performance**: Optimize speed, memory, and throughput

**Key Takeaways**:

- JIT compilation is essential for performance
- Batch size significantly impacts throughput
- Memory management matters for large-scale inference
- Benchmarking helps find optimal configurations

## Next Steps

<div class="grid cards" markdown>

- :material-flask:{ .lg .middle } **Sampling Methods**

    ---

    Advanced sampling techniques

    [:octicons-arrow-right-24: Sampling Guide](sampling.md)

- :material-speedometer:{ .lg .middle } **Optimization**

    ---

    Deep dive into inference optimization

    [:octicons-arrow-right-24: Optimization Guide](optimization.md)

- :material-server:{ .lg .middle } **Deployment**

    ---

    Deploy models to production

    [:octicons-arrow-right-24: Deployment Guide](../integrations/deployment.md)

- :material-chart-box:{ .lg .middle } **Benchmarks**

    ---

    Evaluate model performance

    [:octicons-arrow-right-24: Benchmarks](../training/overview.md#evaluation)

</div>
