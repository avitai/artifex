# Inference Pipeline

Production-ready inference infrastructure for generative models, including optimized generators, sampling strategies, batching, model optimization, and serving endpoints.

## Overview

<div class="grid cards" markdown>

- :material-play:{ .lg .middle } **Generators**

    ---

    Model-specific generators for VAE, GAN, Diffusion, Flow, EBM, and Autoregressive

- :material-speedometer:{ .lg .middle } **Optimization**

    ---

    Quantization, pruning, compilation, and caching for faster inference

- :material-server:{ .lg .middle } **Serving**

    ---

    REST, gRPC, and streaming endpoints for production deployment

- :material-tune:{ .lg .middle } **Sampling**

    ---

    Temperature, top-k, nucleus, beam search, and classifier-free guidance

</div>

## Quick Start

### Basic Generation

```python
from artifex.inference import InferencePipeline

# Create inference pipeline
pipeline = InferencePipeline.from_pretrained("model_checkpoint")

# Generate samples
samples = pipeline.generate(
    num_samples=16,
    seed=42,
)
```

### Optimized Inference

```python
from artifex.inference import InferencePipeline
from artifex.inference.optimization import quantize, compile_model

# Load and optimize model
pipeline = InferencePipeline.from_pretrained("model_checkpoint")
pipeline.model = quantize(pipeline.model, dtype="int8")
pipeline.model = compile_model(pipeline.model)

# Fast inference
samples = pipeline.generate(num_samples=64)
```

## Generators

Model-specific generators optimized for each architecture.

### VAE Generator

```python
from artifex.inference.generators import VAEGenerator

generator = VAEGenerator(
    model=vae_model,
    latent_dim=64,
)

# Generate from random latents
samples = generator.generate(num_samples=16)

# Generate from specific latent
samples = generator.decode(latent_vectors)

# Interpolate between samples
interpolated = generator.interpolate(z1, z2, steps=10)
```

[:octicons-arrow-right-24: VAE Generator](vae_generator.md)

### GAN Generator

```python
from artifex.inference.generators import GANGenerator

generator = GANGenerator(
    model=gan_model,
    latent_dim=128,
)

# Generate samples
samples = generator.generate(num_samples=16)

# Conditional generation
samples = generator.generate(num_samples=16, labels=class_labels)

# Truncation trick for quality
samples = generator.generate(num_samples=16, truncation=0.7)
```

[:octicons-arrow-right-24: GAN Generator](gan_generator.md)

### Diffusion Generator

```python
from artifex.inference.generators import DiffusionGenerator

generator = DiffusionGenerator(
    model=diffusion_model,
    num_steps=50,           # Sampling steps
    sampler="ddim",         # DDIM, DDPM, DPM++
)

# Generate samples
samples = generator.generate(num_samples=16)

# With classifier-free guidance
samples = generator.generate(
    num_samples=16,
    prompt_embeds=text_embeddings,
    guidance_scale=7.5,
)
```

[:octicons-arrow-right-24: Diffusion Generator](diffusion_generator.md)

### Flow Generator

```python
from artifex.inference.generators import FlowGenerator

generator = FlowGenerator(
    model=flow_model,
    num_steps=100,
)

# Generate samples
samples = generator.generate(num_samples=16)

# Compute log-likelihood
log_prob = generator.log_prob(data)
```

[:octicons-arrow-right-24: Flow Generator](flow_generator.md)

### Energy Generator

```python
from artifex.inference.generators import EnergyGenerator

generator = EnergyGenerator(
    model=ebm_model,
    sampler="langevin",
    num_steps=100,
    step_size=0.01,
)

# Generate via MCMC
samples = generator.generate(num_samples=16)
```

[:octicons-arrow-right-24: Energy Generator](energy_generator.md)

### Autoregressive Generator

```python
from artifex.inference.generators import AutoregressiveGenerator

generator = AutoregressiveGenerator(
    model=ar_model,
    max_length=512,
    temperature=0.8,
)

# Generate sequences
samples = generator.generate(
    prompts=input_tokens,
    num_samples=4,
)
```

[:octicons-arrow-right-24: Autoregressive Generator](autoregressive_generator.md)

## Sampling Strategies

### Temperature Scaling

```python
from artifex.inference.sampling import TemperatureSampler

sampler = TemperatureSampler(temperature=0.7)
samples = sampler.sample(logits)
```

[:octicons-arrow-right-24: Temperature](temperature.md)

### Top-K Sampling

```python
from artifex.inference.sampling import TopKSampler

sampler = TopKSampler(k=50)
samples = sampler.sample(logits)
```

[:octicons-arrow-right-24: Top-K](top_k.md)

### Nucleus (Top-P) Sampling

```python
from artifex.inference.sampling import NucleusSampler

sampler = NucleusSampler(p=0.9)
samples = sampler.sample(logits)
```

[:octicons-arrow-right-24: Nucleus](nucleus.md)

### Beam Search

```python
from artifex.inference.sampling import BeamSearchSampler

sampler = BeamSearchSampler(
    beam_width=5,
    length_penalty=0.6,
)
samples = sampler.sample(model, input_ids)
```

[:octicons-arrow-right-24: Beam Search](beam_search.md)

### Ancestral Sampling

```python
from artifex.inference.sampling import AncestralSampler

sampler = AncestralSampler()
samples = sampler.sample(diffusion_model, noise)
```

[:octicons-arrow-right-24: Ancestral](ancestral.md)

### Classifier-Free Guidance

```python
from artifex.inference.sampling import ClassifierFreeGuidance

cfg = ClassifierFreeGuidance(guidance_scale=7.5)
samples = cfg.sample(
    model=diffusion_model,
    cond_embeddings=text_embeddings,
    uncond_embeddings=null_embeddings,
)
```

[:octicons-arrow-right-24: Classifier-Free](classifier_free.md)

## Optimization

### Quantization

```python
from artifex.inference.optimization import quantize

# INT8 quantization
quantized_model = quantize(
    model,
    dtype="int8",
    calibration_data=calibration_set,
)

# Dynamic quantization
quantized_model = quantize(model, dtype="int8", dynamic=True)
```

[:octicons-arrow-right-24: Quantization](quantization.md)

### Pruning

```python
from artifex.inference.optimization import prune

# Magnitude-based pruning
pruned_model = prune(
    model,
    sparsity=0.5,
    method="magnitude",
)

# Structured pruning
pruned_model = prune(
    model,
    sparsity=0.3,
    method="structured",
    granularity="channel",
)
```

[:octicons-arrow-right-24: Pruning](pruning.md)

### Compilation

```python
from artifex.inference.optimization import compile_model

# JIT compilation
compiled_model = compile_model(model, backend="jax")

# XLA compilation
compiled_model = compile_model(model, backend="xla")
```

[:octicons-arrow-right-24: Compilation](compilation.md)

### Caching

```python
from artifex.inference.optimization import KVCache

# Key-value caching for transformers
cache = KVCache(
    max_length=2048,
    num_layers=24,
    num_heads=16,
)

output, new_cache = model(input, cache=cache)
```

[:octicons-arrow-right-24: Caching](caching.md)

### Knowledge Distillation

```python
from artifex.inference.optimization import distill

# Distill to smaller model
student_model = distill(
    teacher=large_model,
    student=small_model,
    train_data=train_data,
    temperature=4.0,
)
```

[:octicons-arrow-right-24: Distillation](distillation.md)

## Batching

### Dynamic Batching

```python
from artifex.inference.batching import DynamicBatcher

batcher = DynamicBatcher(
    max_batch_size=32,
    max_wait_time=0.1,  # seconds
)

async def handle_request(request):
    return await batcher.process(request)
```

[:octicons-arrow-right-24: Dynamic Batching](dynamic.md)

### Padding Strategies

```python
from artifex.inference.batching import PaddedBatcher

batcher = PaddedBatcher(
    pad_token_id=0,
    max_length=512,
    padding_side="right",
)
```

[:octicons-arrow-right-24: Padding](padding.md)

### Streaming

```python
from artifex.inference.batching import StreamingBatcher

batcher = StreamingBatcher(
    chunk_size=16,
    overlap=2,
)

async for chunk in batcher.stream(long_input):
    process(chunk)
```

[:octicons-arrow-right-24: Streaming](streaming.md)

## Model Conversion

### ONNX Export

```python
from artifex.inference.conversion import export_onnx

export_onnx(
    model,
    output_path="model.onnx",
    input_shape=(1, 3, 256, 256),
    opset_version=17,
)
```

[:octicons-arrow-right-24: ONNX](onnx.md)

### TensorRT

```python
from artifex.inference.conversion import export_tensorrt

export_tensorrt(
    model,
    output_path="model.trt",
    precision="fp16",
    max_batch_size=32,
)
```

[:octicons-arrow-right-24: TensorRT](tensorrt.md)

### TensorFlow.js

```python
from artifex.inference.conversion import export_tfjs

export_tfjs(
    model,
    output_dir="tfjs_model/",
    quantization="float16",
)
```

[:octicons-arrow-right-24: TensorFlow.js](tfjs.md)

## Serving

### REST API

```python
from artifex.inference.serving import RESTServer

server = RESTServer(
    model=model,
    host="0.0.0.0",
    port=8080,
)

server.run()
```

[:octicons-arrow-right-24: REST](rest.md)

### gRPC

```python
from artifex.inference.serving import GRPCServer

server = GRPCServer(
    model=model,
    port=50051,
    max_workers=10,
)

server.run()
```

[:octicons-arrow-right-24: gRPC](grpc.md)

### Middleware

```python
from artifex.inference.serving import (
    RateLimiter,
    AuthMiddleware,
    LoggingMiddleware,
)

server.add_middleware(RateLimiter(requests_per_minute=100))
server.add_middleware(AuthMiddleware(api_key_header="X-API-Key"))
server.add_middleware(LoggingMiddleware())
```

[:octicons-arrow-right-24: Middleware](middleware.md)

## Inference Metrics

### Latency Tracking

```python
from artifex.inference.metrics import LatencyTracker

tracker = LatencyTracker()

with tracker.measure("generation"):
    samples = generator.generate(num_samples=16)

print(f"P50 latency: {tracker.percentile(50):.2f}ms")
print(f"P99 latency: {tracker.percentile(99):.2f}ms")
```

[:octicons-arrow-right-24: Latency](latency.md)

### Memory Monitoring

```python
from artifex.inference.metrics import MemoryMonitor

monitor = MemoryMonitor()

with monitor.track():
    samples = generator.generate(num_samples=64)

print(f"Peak memory: {monitor.peak_memory_mb:.2f} MB")
```

[:octicons-arrow-right-24: Memory](memory.md)

### Throughput

```python
from artifex.inference.metrics import ThroughputTracker

tracker = ThroughputTracker()

tracker.record(batch_size=32, latency_ms=50)
print(f"Throughput: {tracker.samples_per_second:.2f} samples/s")
```

[:octicons-arrow-right-24: Throughput](throughput.md)

## Module Reference

| Category | Modules |
|----------|---------|
| **Inference** | [base](base.md), [pipeline](pipeline.md) |
| **Generators** | [autoregressive_generator](autoregressive_generator.md), [diffusion_generator](diffusion_generator.md), [energy_generator](energy_generator.md), [flow_generator](flow_generator.md), [gan_generator](gan_generator.md), [vae_generator](vae_generator.md) |
| **Sampling** | [ancestral](ancestral.md), [beam_search](beam_search.md), [classifier_free](classifier_free.md), [nucleus](nucleus.md), [temperature](temperature.md), [top_k](top_k.md) |
| **Optimization** | [caching](caching.md), [compilation](compilation.md), [distillation](distillation.md), [production](production.md), [pruning](pruning.md), [quantization](quantization.md) |
| **Batching** | [dynamic](dynamic.md), [padding](padding.md), [streaming](streaming.md) |
| **Conversion** | [onnx](onnx.md), [tensorrt](tensorrt.md), [tfjs](tfjs.md) |
| **Serving** | [grpc](grpc.md), [middleware](middleware.md), [rest](rest.md), [stateless](stateless.md) |
| **Metrics** | [latency](latency.md), [memory](memory.md), [throughput](throughput.md) |
| **Adaptation** | [adapter](adapter.md), [lora](lora.md), [prefix_tuning](prefix_tuning.md), [prompt_tuning](prompt_tuning.md) |

## Related Documentation

- [CLI Commands](../cli/index.md) - Command-line generation
- [Distributed Training](../user-guide/advanced/distributed.md) - Multi-device setup
- [Parallelism Guide](../user-guide/advanced/parallelism.md) - Parallel inference
- [Checkpointing](../user-guide/advanced/checkpointing.md) - Model checkpoints
