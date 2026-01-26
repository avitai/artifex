# Diffusion Transformer (DiT) Demo

**Level:** Advanced | **Runtime:** ~5 minutes (CPU) / ~1-2 minutes (GPU) | **Format:** Python + Jupyter

## Overview

This advanced example demonstrates Diffusion Transformers (DiT), which combines the power of Vision Transformers with diffusion models. DiT represents a significant advancement in diffusion model architectures, using transformer blocks instead of traditional U-Net architectures for the denoising process.

## What You'll Learn

- DiffusionTransformer backbone architecture
- DiT model sizes and scaling (S, B, L, XL configurations)
- Conditional generation with classifier-free guidance (CFG)
- Patch-based image processing
- Performance benchmarking across model sizes
- Advanced sampling techniques

## Files

- **Python Script**: [`examples/generative_models/diffusion/dit_demo.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/diffusion/dit_demo.py)
- **Jupyter Notebook**: [`examples/generative_models/diffusion/dit_demo.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/diffusion/dit_demo.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/diffusion/dit_demo.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/diffusion/dit_demo.ipynb
```

## Key Concepts

### Diffusion Transformer Architecture

DiT replaces the traditional U-Net backbone with a Vision Transformer:

- **Patch Embedding**: Images are divided into patches and linearly embedded
- **Positional Encoding**: Added to maintain spatial information
- **Transformer Blocks**: Self-attention and feed-forward layers
- **Adaptive Layer Normalization**: Conditioned on timestep and class labels

### Classifier-Free Guidance (CFG)

CFG enables stronger conditional generation by learning both conditional and unconditional models simultaneously:

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

Where:

- $c$ is the conditioning (e.g., class label)
- $\emptyset$ is the unconditional case
- $s$ is the guidance scale (higher = stronger conditioning)

### Model Scaling

DiT comes in different sizes, trading off quality and speed:

| Model | Hidden Dim | Depth | Heads | Parameters |
|-------|-----------|-------|-------|------------|
| DiT-S | 384       | 12    | 6     | ~33M       |
| DiT-B | 768       | 12    | 12    | ~130M      |
| DiT-L | 1024      | 24    | 16    | ~458M      |
| DiT-XL| 1152      | 28    | 16    | ~675M      |

### Patch-Based Processing

Images are processed as sequences of patches:

1. Divide image into non-overlapping patches (e.g., 16×16)
2. Flatten each patch into a vector
3. Apply linear projection
4. Add positional embeddings
5. Process through transformer blocks

## Code Structure

The example demonstrates 7 major sections:

1. **Import Dependencies**: Setting up the environment
2. **Test DiT Components**: Verifying backbone and full model
3. **Test Different Model Sizes**: Comparing S, B, L configurations
4. **Conditional Generation**: Using classifier-free guidance
5. **Visualization**: Displaying generated samples
6. **Performance Benchmark**: Measuring throughput across sizes
7. **Summary**: Key takeaways and next steps

## Example Code

### Creating DiT Model

```python
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.core.configuration import DiTConfig, NoiseScheduleConfig
from artifex.generative_models.models.diffusion.dit import DiTModel

# Initialize RNG
rngs = nnx.Rngs(42)

# Create noise schedule config
noise_schedule_config = NoiseScheduleConfig(
    name="dit_schedule",
    schedule_type="linear",
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

# Create DiT-S model config
config = DiTConfig(
    name="dit_s",
    noise_schedule=noise_schedule_config,
    input_shape=(3, 32, 32),  # C, H, W format
    patch_size=4,
    hidden_size=384,  # DiT-S
    depth=12,
    num_heads=6,
    num_classes=10,
)

# Create model
dit_model = DiTModel(config, rngs=rngs)

# Test forward pass
batch_size = 4
images = jnp.ones((batch_size, 32, 32, 3))
timesteps = jnp.array([100, 200, 300, 400])
labels = jnp.array([0, 1, 2, 3])

# Predict noise
noise_pred = dit_model(images, timesteps, labels, deterministic=True)
print(f"Noise prediction shape: {noise_pred.shape}")  # (4, 32, 32, 3)
```

### Testing Different Model Sizes

```python
# Create shared noise schedule config
noise_schedule = NoiseScheduleConfig(
    name="schedule",
    schedule_type="linear",
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

# DiT-S (Small) - Fast, good for prototyping
config_s = DiTConfig(
    name="dit_s",
    noise_schedule=noise_schedule,
    input_shape=(3, 32, 32),
    patch_size=4,
    hidden_size=384,
    depth=12,
    num_heads=6,
)
dit_s = DiTModel(config_s, rngs=rngs)

# DiT-B (Base) - Balanced quality/speed
config_b = DiTConfig(
    name="dit_b",
    noise_schedule=noise_schedule,
    input_shape=(3, 32, 32),
    patch_size=4,
    hidden_size=768,
    depth=12,
    num_heads=12,
)
dit_b = DiTModel(config_b, rngs=rngs)

# DiT-L (Large) - High quality, slower
config_l = DiTConfig(
    name="dit_l",
    noise_schedule=noise_schedule,
    input_shape=(3, 32, 32),
    patch_size=4,
    hidden_size=1024,
    depth=24,
    num_heads=16,
)
dit_l = DiTModel(config_l, rngs=rngs)
```

### Classifier-Free Guidance

```python
# Create conditional model with CFG support
config = DiTConfig(
    name="conditional_dit",
    noise_schedule=noise_schedule_config,
    input_shape=(3, 16, 16),  # Smaller for faster demo
    patch_size=2,
    hidden_size=384,  # DiT-S for speed
    depth=12,
    num_heads=6,
    num_classes=10,  # 10 class labels (like CIFAR-10)
    cfg_scale=3.0,   # Guidance scale
)

model = DiTModel(config, rngs=rngs)

# Test conditional forward pass
x = jnp.ones((2, 16, 16, 3))
t = jnp.array([5, 8])
y = jnp.array([2, 7])  # Class labels

output = model(x, t, y, deterministic=True, cfg_scale=3.0)

# Generate samples using the built-in generate method
samples = model.generate(
    n_samples=4,
    rngs=rngs,
    num_steps=10,
    y=jnp.array([0, 1, 2, 3]),  # One sample per class
    cfg_scale=3.0,
    img_size=16,
)
print(f"Generated samples shape: {samples.shape}")  # (4, 16, 16, 3)
```

### Performance Benchmarking

```python
import time

# Create benchmark model (DiT-B)
benchmark_schedule = NoiseScheduleConfig(
    name="benchmark_schedule",
    schedule_type="linear",
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

config = DiTConfig(
    name="benchmark_dit",
    noise_schedule=benchmark_schedule,
    input_shape=(3, 64, 64),
    patch_size=4,
    hidden_size=768,  # DiT-B
    depth=12,
    num_heads=12,
)

model = DiTModel(config, rngs=rngs)

# Prepare inputs
batch_size = 8
x = jnp.ones((batch_size, 64, 64, 3))
t = jnp.array([100] * batch_size)

# Warmup (JIT compilation)
for _ in range(3):
    _ = model(x, t, deterministic=True)

# Benchmark
num_iterations = 10
start = time.time()
for _ in range(num_iterations):
    output = model(x, t, deterministic=True)
    output.block_until_ready()
elapsed = time.time() - start

throughput = (batch_size * num_iterations) / elapsed
print(f"DiT-B Throughput: {throughput:.1f} samples/sec")
```

## Features Demonstrated

### DiffusionTransformer Backbone

- Vision Transformer architecture
- Adaptive layer normalization (adaLN)
- Position-wise feed-forward networks
- Multi-head self-attention

### DiT Model Sizes

- Small (S): Fast prototyping and testing
- Base (B): Production-ready performance
- Large (L): High-quality generation
- Configurable depth, hidden size, and heads

### Conditional Generation

- Class-conditional generation
- Classifier-free guidance
- Guidance scale tuning
- Null conditioning for unconditional mode

### Patch-Based Processing

- Efficient patch embeddings
- Positional encoding strategies
- Sequence-to-image reconstruction
- Variable patch sizes

### Performance Analysis

- Throughput benchmarking
- Memory profiling
- Quality vs. speed trade-offs
- Scaling behavior analysis

## Experiments to Try

1. **Vary patch size**: Try 2×2, 4×4, 8×8 patches and observe quality/speed trade-offs
2. **Modify model size**: Create custom configurations between S/B/L
3. **Tune guidance scale**: Experiment with CFG scales from 1.0 to 5.0
4. **Custom conditioning**: Add additional conditioning (text, attributes, etc.)
5. **Training from scratch**: Implement full training loop on your dataset
6. **Distillation**: Train a smaller model to match larger model's quality

## Next Steps

After understanding this example:

1. **Full Training**: Implement training loop with ImageNet or custom data
2. **Custom Conditioning**: Add text or multi-modal conditioning
3. **Faster Sampling**: Explore DDIM, DPM-Solver, or other fast samplers
4. **Latent DiT**: Apply DiT in latent space (like Stable Diffusion)
5. **Model Compression**: Distillation, pruning, quantization
6. **Evaluation**: FID, Inception Score, and other metrics

## Troubleshooting

### Out of Memory

- Reduce model size (use DiT-S instead of DiT-L)
- Decrease batch size
- Use smaller images or larger patch size
- Enable gradient checkpointing

### Slow Generation

- Use GPU acceleration
- Reduce number of denoising steps (try 50-100 instead of 1000)
- Use smaller model (DiT-S)
- Implement faster samplers (DDIM)

### Poor Sample Quality

- Increase model size (DiT-B or DiT-L)
- Tune classifier-free guidance scale
- Increase number of denoising steps
- Check training convergence

### Patch Size Issues

Ensure image size is divisible by patch size:

```python
assert image_size % patch_size == 0, "Image size must be divisible by patch size"
```

## Additional Resources

- **Paper**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **Paper**: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- **Artifex Diffusion Guide**: [Diffusion Models Guide](../../user-guide/models/diffusion-guide.md)
- **API Reference**: [DiffusionTransformer API](../../api/models/diffusion.md)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

## Related Examples

- [Simple Diffusion](simple-diffusion.md) - Diffusion basics
- [Simple EBM](../energy/simple-ebm.md) - Energy-based models
- [Advanced Diffusion](../../examples/advanced/advanced-diffusion.md) - More diffusion techniques

## Performance Comparison

Expected performance on a modern GPU (A100):

| Model | Samples/sec | Memory (GB) | FID (ImageNet) |
|-------|-------------|-------------|----------------|
| DiT-S | ~120        | ~4          | ~9.5           |
| DiT-B | ~50         | ~12         | ~5.3           |
| DiT-L | ~25         | ~24         | ~3.4           |
| DiT-XL| ~15         | ~32         | ~2.3           |

*Note: Actual performance depends on hardware, image size, and implementation details.*
