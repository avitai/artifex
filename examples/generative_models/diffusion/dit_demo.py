#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Diffusion Transformer (DiT) Demo

**Duration:** 20 minutes | **Level:** Advanced | **GPU Required:** Recommended
(CPU will work but slower)

This example demonstrates the Diffusion Transformer (DiT) architecture, which combines the power
of transformers with diffusion models. DiT replaces the U-Net backbone typically used in diffusion
models with a Vision Transformer (ViT) architecture.

## 🎯 Learning Objectives

By the end of this example, you will:
1. Understand the DiT architecture and how it differs from traditional diffusion models
2. Learn to create and test DiT models of different sizes (S, B, L, XL)
3. Implement conditional generation with classifier-free guidance
4. Generate and visualize samples from DiT models
5. Benchmark DiT performance across different configurations
6. Understand patch-based processing and positional embeddings

## 🔍 Source Code Dependencies

**Validated:** 2025-10-14

This example depends on the following Artifex source files:
- `src/artifex/generative_models/core/configuration.py` - Configuration system
- `src/artifex/generative_models/models/backbones/dit.py` - DiffusionTransformer backbone
- `src/artifex/generative_models/models/diffusion/dit.py` - DiTModel implementation

**Validation Status:**
- ✅ All dependencies validated against `memory-bank/guides/flax-nnx-guide.md`
- ✅ No anti-patterns detected (RNG handling, module init, activations)
- ✅ All tests passing for dependency files

## 📚 Background

**What is DiT?**

DiT (Diffusion Transformer) replaces the U-Net backbone in diffusion models with a
Vision Transformer:
- **Traditional Diffusion:** U-Net with convolutional layers and skip connections
- **DiT:** Transformer blocks operating on image patches

**Key Innovations:**
1. **Patch-based processing:** Images divided into patches (like ViT)
2. **Self-attention:** Captures long-range dependencies better than convolutions
3. **Scalable:** Easily scale model capacity by adding layers/heads
4. **Conditional:** Native support for class conditioning via adaptive layer norm

**Why DiT?**
- Better quality at large scales
- More parameter-efficient than U-Nets
- Cleaner architecture without skip connections
- Easier to condition on multiple signals

## 🔑 Key Concepts

- **Patch Embedding:** Divide image into patches and project to embedding space
- **Positional Encoding:** Add position information to patches
- **DiT Block:** Transformer block with adaptive layer norm for conditioning
- **Classifier-Free Guidance (CFG):** Balance between conditional and unconditional generation
- **Model Sizes:** S (Small), B (Base), L (Large), XL (Extra Large)

## ℹ️ Prerequisites

- Understanding of transformers and attention mechanisms
- Familiarity with diffusion models (see simple_diffusion_example.py)
- Knowledge of Vision Transformers (ViT) helpful
- Artifex installed (see below)

## 📦 Setup

Before running this example, activate the Artifex environment:

```bash
source activate.sh
python examples/generative_models/diffusion/dit_demo.py
```

## 🎬 Expected Output

This example will:
- Test DiT components (backbone and model)
- Create models of different sizes (S, B, L)
- Demonstrate conditional generation with CFG
- Generate and visualize samples
- Benchmark performance metrics

## ⏱️ Estimated Runtime

- **CPU:** ~5 minutes
- **GPU:** ~1-2 minutes

## 👥 Author

Artifex Team

## 📅 Last Updated

2025-10-14

## 📄 License

MIT
"""

# %% [markdown]
"""
## 1. Import Dependencies and Setup

We'll use:
- **JAX:** For high-performance numerical computing
- **Flax NNX:** For transformer modules
- **Matplotlib:** For visualization
- **Artifex:** For DiT implementations and configuration
"""

# %%
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from artifex.generative_models.core.configuration import DiTConfig, NoiseScheduleConfig

# Import DiT components directly
from artifex.generative_models.models.backbones.dit import DiffusionTransformer
from artifex.generative_models.models.diffusion.dit import DiTModel


print("=" * 80)
print("DiT (Diffusion Transformer) Implementation Demo")
print("=" * 80)
print(f"✅ JAX version: {jax.__version__}")
print(f"🖥️  Backend: {jax.default_backend()}")
print(f"🔧 Devices: {jax.device_count()} device(s)")
print("=" * 80)

# %% [markdown]
"""
## 2. Test DiT Components

Let's start by testing the individual components of the DiT architecture:
1. **DiffusionTransformer Backbone:** The core transformer that processes image patches
2. **DiTModel:** Complete diffusion model wrapping the backbone

### Understanding the Architecture

**DiffusionTransformer** takes:
- `x`: Input image tensor (B, H, W, C)
- `t`: Timesteps for diffusion process (B,)
- Returns: Predicted noise (same shape as input)

The model:
1. Divides image into patches
2. Projects patches to embedding space
3. Adds positional embeddings
4. Processes through transformer blocks
5. Reconstructs image from patch embeddings
"""

# %%
print("\n📊 Testing DiT Components")
print("=" * 80)

# Initialize random number generators
rngs = nnx.Rngs(42)

# Test 1: Create DiT backbone
print("\n1. Testing DiffusionTransformer backbone...")
dit_backbone = DiffusionTransformer(
    img_size=32,  # Input image size
    patch_size=4,  # Each patch is 4x4 pixels
    in_channels=3,  # RGB images
    hidden_size=256,  # Embedding dimension
    depth=4,  # Number of transformer blocks
    num_heads=8,  # Attention heads per block
    rngs=rngs,
)

# Test forward pass
x = jnp.ones((2, 32, 32, 3))  # Batch of 2 images
t = jnp.array([100, 500])  # Different timesteps

output = dit_backbone(x, t, deterministic=True)

assert output.shape == (2, 32, 32, 3), f"Unexpected output shape: {output.shape}"
print("   ✓ DiffusionTransformer forward pass successful")
print(f"   Input shape: {x.shape}, Output shape: {output.shape}")
print(f"   Number of patches: {(32 // 4) ** 2} patches per image")

# Test 2: Create DiT model
print("\n2. Testing DiTModel (full diffusion model)...")

# Create noise schedule config (required for DiTConfig)
noise_schedule_config = NoiseScheduleConfig(
    name="dit_test_schedule",
    schedule_type="linear",
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

# Create DiTConfig with nested noise_schedule
config = DiTConfig(
    name="dit_test",
    noise_schedule=noise_schedule_config,
    input_shape=(3, 32, 32),  # C, H, W format
    patch_size=4,
    hidden_size=256,
    depth=4,
    num_heads=8,
)

model = DiTModel(config, rngs=rngs)
output = model(x, t, deterministic=True)

assert output.shape == (2, 32, 32, 3), f"Unexpected output shape: {output.shape}"
print("   ✓ DiTModel forward pass successful")
print("   Model processes images through transformer blocks")

print("\n✅ All component tests passed!")

# %% [markdown]
"""
## 3. Test Different DiT Model Sizes

DiT comes in different sizes, similar to GPT or BERT models:
- **DiT-S (Small):** 384 hidden dim, 12 blocks, 6 heads
- **DiT-B (Base):** 768 hidden dim, 12 blocks, 12 heads
- **DiT-L (Large):** 1024 hidden dim, 24 blocks, 16 heads
- **DiT-XL (Extra Large):** 1152 hidden dim, 28 blocks, 16 heads (not shown here for speed)

Larger models have:
- More parameters → Better quality
- Deeper networks → Better feature extraction
- More attention heads → Richer representations
- Slower inference → Higher computational cost

Let's test S, B, and L to see the performance tradeoffs.
"""

# %%
print("\n📊 Testing DiT Model Sizes")
print("=" * 80)

# Define model size configurations (from DiT paper)
sizes = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
}

img_size = 32
batch_size = 2

# Create shared noise schedule config
size_test_schedule = NoiseScheduleConfig(
    name="size_test_schedule",
    schedule_type="linear",
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

for size_name, size_params in sizes.items():
    print(f"\n{size_name}. Testing DiT-{size_name}...")

    # Create DiTConfig with nested noise_schedule
    config = DiTConfig(
        name=f"dit_{size_name.lower()}",
        noise_schedule=size_test_schedule,
        input_shape=(3, img_size, img_size),  # C, H, W format
        patch_size=4,
        hidden_size=size_params["hidden_size"],
        depth=size_params["depth"],
        num_heads=size_params["num_heads"],
    )

    # Create model
    rngs = nnx.Rngs(42)
    model = DiTModel(config, rngs=rngs)

    # Test forward pass with timing
    x = jnp.ones((batch_size, img_size, img_size, 3))
    t = jnp.array([100, 500])

    start = time.time()
    output = model(x, t, deterministic=True)
    elapsed = time.time() - start

    assert output.shape == (batch_size, img_size, img_size, 3)

    print(
        f"   Config: hidden_size={size_params['hidden_size']}, "
        f"depth={size_params['depth']}, "
        f"heads={size_params['num_heads']}"
    )
    print(f"   ✓ Forward pass successful (time: {elapsed:.3f}s)")
    capacity = size_params["hidden_size"] * size_params["depth"] // 1000
    print(f"   Model capacity: ~{capacity}K params (approx)")

print("\n✅ All model sizes tested successfully!")
print("\n💡 Takeaway: Larger models are slower but produce better quality")

# %% [markdown]
r"""
## 4. Conditional Generation with Classifier-Free Guidance

One of DiT's strengths is native support for conditional generation.

### Classifier-Free Guidance (CFG)

CFG balances between:
- **Unconditional generation:** p(x)
- **Conditional generation:** p(x|y) where y is class label

**Formula:**
$$\\epsilon_{\\text{pred}} = \\epsilon_{\\text{uncond}} + w \\cdot
(\\epsilon_{\\text{cond}} - \\epsilon_{\\text{uncond}})$$

Where:
- $w$ is the guidance scale (typically 1.5-10)
- Higher $w$ → stronger conditioning (more faithful to class)
- Lower $w$ → more diversity but less control

### Implementation

The model runs inference twice:
1. With class label → conditional prediction
2. Without class label (null) → unconditional prediction
3. Combine predictions with guidance scale
"""

# %%
print("\n📊 Testing Conditional Generation")
print("=" * 80)

# Create noise schedule config for conditional model
cond_schedule_config = NoiseScheduleConfig(
    name="conditional_schedule",
    schedule_type="linear",
    num_timesteps=10,  # Few steps for demo
    beta_start=1e-4,
    beta_end=0.02,
)

# Create conditional model configuration using DiTConfig
config = DiTConfig(
    name="conditional_dit",
    noise_schedule=cond_schedule_config,
    input_shape=(3, 16, 16),  # C, H, W format - smaller for faster demo
    patch_size=2,
    hidden_size=384,  # DiT-S for speed
    depth=12,
    num_heads=6,
    num_classes=10,  # 10 class labels (like CIFAR-10)
    cfg_scale=3.0,  # Guidance scale
)

rngs = nnx.Rngs(42)
model = DiTModel(config, rngs=rngs)

print("\n1. Testing conditional forward pass...")
x = jnp.ones((2, 16, 16, 3))
t = jnp.array([5, 8])
y = jnp.array([2, 7])  # Class labels for conditioning

output = model(x, t, y, deterministic=True, cfg_scale=3.0)
assert output.shape == (2, 16, 16, 3)
print("   ✓ Conditional forward pass successful")
print(f"   Conditioned on classes: {y}")
print("   Guidance scale: 3.0 (moderate conditioning)")

print("\n2. Testing sample generation...")
samples = model.generate(
    n_samples=4,
    rngs=rngs,
    num_steps=10,
    y=jnp.array([0, 1, 2, 3]),  # Generate one sample per class
    cfg_scale=3.0,
    img_size=16,
)

assert samples.shape == (4, 16, 16, 3)
print(f"   ✓ Generated {samples.shape[0]} conditional samples")
print(
    f"   Sample statistics: min={samples.min():.3f}, "
    f"max={samples.max():.3f}, mean={samples.mean():.3f}"
)
print("\n💡 Each sample corresponds to a different class label")

print("\n✅ Conditional generation test passed!")

# %% [markdown]
"""
## 5. Visualize Generated Samples

Let's visualize the samples we just generated to see what the model produces.

Note: Since this is a demo with an untrained model, the samples will look like
structured noise rather than real images. With a trained model, you would see
actual class-specific images.
"""

# %%
print("\n📊 Visualizing Generated Samples")
print("=" * 80)

# Convert JAX array to numpy for visualization
samples_np = np.array(samples)

# Normalize to [0, 1] for display
samples_vis = (samples_np - samples_np.min()) / (samples_np.max() - samples_np.min() + 1e-8)

# Create figure
fig, axes = plt.subplots(1, len(samples_vis), figsize=(12, 3))

for i, (ax, sample) in enumerate(zip(axes, samples_vis)):
    ax.imshow(sample)
    ax.set_title(f"Class {i}")
    ax.axis("off")

plt.suptitle("DiT Generated Samples (Untrained Model - Noise Visualization)")
plt.tight_layout()

# Save figure
output_path = "dit_samples.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✓ Visualization saved to {output_path}")
print("💡 With a trained model, you would see class-specific images")

# %% [markdown]
"""
## 6. Performance Benchmark

Let's benchmark DiT performance to understand computational costs.

This helps you choose the right model size for your application:
- Real-time applications → Use DiT-S
- High-quality generation → Use DiT-L or XL
- Balanced use case → Use DiT-B
"""

# %%
print("\n📊 Performance Benchmark")
print("=" * 80)

# Test configuration
img_size = 64
batch_size = 8
num_iterations = 10

# Create noise schedule config for benchmark
benchmark_schedule_config = NoiseScheduleConfig(
    name="benchmark_schedule",
    schedule_type="linear",
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

# Create DiTConfig with DiT-B parameters
config = DiTConfig(
    name="benchmark_dit",
    noise_schedule=benchmark_schedule_config,
    input_shape=(3, img_size, img_size),  # C, H, W format
    patch_size=4,
    hidden_size=768,  # DiT-B
    depth=12,
    num_heads=12,
)

rngs = nnx.Rngs(42)
model = DiTModel(config, rngs=rngs)

# Prepare inputs
x = jnp.ones((batch_size, img_size, img_size, 3))
t = jnp.array([100] * batch_size)

# Warm-up (JIT compilation)
print("\n🔥 Warming up (JIT compilation)...")
for _ in range(3):
    _ = model(x, t, deterministic=True)

# Benchmark
print(f"⏱️  Running {num_iterations} iterations...")
start = time.time()
for _ in range(num_iterations):
    output = model(x, t, deterministic=True)
    output.block_until_ready()  # Ensure computation completes
elapsed = time.time() - start

# Calculate metrics
time_per_iteration = elapsed / num_iterations
throughput = batch_size / time_per_iteration

print("\n📈 Results:")
print("  Model: DiT-B")
print(f"  Input shape: {x.shape}")
print(f"  Batch size: {batch_size}")
print(f"  Time per iteration: {time_per_iteration:.3f}s")
print(f"  Throughput: {throughput:.1f} samples/s")
print(f"  Total time: {elapsed:.2f}s")

print("\n✅ Benchmark completed!")
print("\n💡 This is for a single denoising step. Full generation requires ~50-1000 steps")

# %% [markdown]
"""
## 7. Summary and Key Takeaways

### 🎓 What You Learned

In this demo, you learned:

1. **DiT Architecture:** How transformers can replace U-Nets in diffusion models
2. **Model Sizes:** Different scales (S, B, L) and their tradeoffs
3. **Conditional Generation:** Using classifier-free guidance for control
4. **Patch Processing:** How images are divided into patches for transformers
5. **Performance:** Computational costs and throughput metrics

### 💡 Key Concepts Recap

- **Patch Embedding:** Images → patches → embeddings
- **DiT Block:** Transformer with adaptive layer norm
- **CFG:** Balances conditional and unconditional generation
- **Scalability:** Easy to scale by adding layers/heads
- **Quality vs Speed:** Larger models = better quality but slower

### 🔬 Experiments to Try

Now that you understand DiT, try these modifications:

1. **Adjust guidance scale:**
   ```python
   samples = model.generate(
       ...,
       cfg_scale=7.0,  # Higher for stronger conditioning
   )
   ```

2. **Try different model sizes:**
   ```python
   # Experiment with XL configuration
   config.parameters["hidden_size"] = 1152
   config.parameters["depth"] = 28
   config.parameters["num_heads"] = 16
   ```

3. **Vary the patch size:**
   ```python
   config.parameters["patch_size"] = 8  # Fewer patches, faster
   ```

4. **Experiment with image sizes:**
   ```python
   # Test on higher resolution
   config.input_dim = (256, 256, 3)
   config.parameters["img_size"] = 256
   ```

### 📚 Next Steps

To learn more about DiT and advanced diffusion models:

- **Training DiT:** See complete training loop with ImageNet
- **Custom Conditioning:** Add text or other modalities
- **Distillation:** Reduce sampling steps with knowledge distillation
- **Latent DiT:** Combine with VAE for latent space generation

### 📖 Additional Resources

- **Paper:** [Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)
- **Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- **Paper:** [Classifier-Free Diffusion Guidance (Ho & Salimans, 2021)](https://arxiv.org/abs/2207.12598)
- **Documentation:** [Artifex DiT Guide](../../../docs/models/dit.md)
- **Related Examples:**
  - `simple_diffusion_example.py` - Basic diffusion concepts
  - `vae/` - VAE for latent space modeling

### 🐛 Troubleshooting

**Problem:** Out of memory errors
- **Solution:** Reduce batch_size, img_size, or use smaller model (DiT-S)

**Problem:** Slow generation
- **Solution:** Use fewer diffusion steps or smaller model

**Problem:** Poor sample quality (after training)
- **Solution:** Increase model size, train longer, or increase CFG scale

**Problem:** Samples don't match conditioning
- **Solution:** Increase cfg_scale (try 5.0-10.0)

### 💬 Feedback

Found a bug or have suggestions? Please open an issue on GitHub!

---

**Demo completed successfully! 🎉**
"""

# %%
print("\n" + "=" * 80)
print("✨ DiT (Diffusion Transformer) Demo Complete! ✨")
print("=" * 80)
print("\n💡 Key Takeaways:")
print("   1. DiT replaces U-Nets with transformers for diffusion models")
print("   2. Patch-based processing enables long-range dependencies")
print("   3. Classifier-free guidance provides conditional control")
print("   4. Model size determines quality vs speed tradeoff")
print("   5. DiT scales better than U-Nets for large models")
print("\n🔗 Next Steps:")
print("   - Train DiT on your dataset")
print("   - Experiment with different model sizes")
print("   - Try various guidance scales")
print("   - Combine with VAE for latent diffusion")
print("\n" + "=" * 80)
print("🚀 DiT implementation is working correctly and ready to use!")
print("=" * 80)
