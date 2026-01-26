#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
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
# Simple Diffusion Model Example

**Duration:** 10 minutes | **Level:** Beginner | **GPU Required:** No

This example demonstrates the fundamentals of diffusion models using JAX and Flax NNX.
You'll learn how diffusion models generate data by gradually denoising random noise.

## 🎯 Learning Objectives

By the end of this example, you will:
1. Understand the basic principles of diffusion models
2. Learn how to implement a simplified diffusion process
3. See how noise schedules control the generation process
4. Generate visual samples from random noise

## 🔍 Source Code Dependencies

**Validated:** 2025-10-14

This example depends on the following Artifex source files:
- `src/artifex/generative_models/core/base.py` - GenerativeModel base class

**Validation Status:**
- ✅ All dependencies validated against `memory-bank/guides/flax-nnx-guide.md`
- ✅ No anti-patterns detected (RNG handling, module init, activations)
- ✅ All tests passing for dependency files

## 📚 Background

Diffusion models are a powerful class of generative models that work by:
1. **Forward Process:** Gradually adding noise to data until it becomes pure noise
2. **Reverse Process:** Learning to remove noise step by step to generate new data

This simplified example focuses on the reverse process (generation) without requiring
a full neural network training pipeline.

## 🔑 Key Concepts

- **Beta Schedule:** Controls how much noise is added at each step
- **Alpha Values:** Complementary to betas (alpha = 1 - beta)
- **Cumulative Product:** Tracks the total noise level at each timestep
- **Denoising Steps:** The number of iterations to go from noise to data

## ℹ️ Prerequisites

- Basic understanding of generative models
- Familiarity with JAX and NumPy
- Artifex installed (see below)

## 📦 Setup

Before running this example, activate the Artifex environment:

```bash
source activate.sh
python examples/generative_models/diffusion/simple_diffusion_example.py
```

## 🎬 Expected Output

This example will:
- Create a simplified diffusion model
- Generate 4 sample images from random noise
- Save visualization to `examples_output/diffusion_samples.png`
- Show how noise transforms into structured patterns

## ⏱️ Estimated Runtime

- **CPU:** ~30 seconds
- **GPU:** ~10 seconds

## 👥 Author

Artifex Team

## 📅 Last Updated

2025-10-14

## 📄 License

MIT
"""

# %% [markdown]
"""
## 1. Import Dependencies

We'll use:
- **JAX:** For high-performance numerical computing
- **Flax NNX:** For neural network modules and RNG handling
- **Matplotlib:** For visualization
- **Artifex:** For the GenerativeModel base class
"""

# %%
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from artifex.generative_models.core.base import GenerativeModel


print("✅ All dependencies imported successfully!")
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## 2. Define the SimpleDiffusionModel

This class implements a minimal diffusion-like model that demonstrates the core concepts
without requiring extensive training or complex architectures.

### Key Features:
- Inherits from `GenerativeModel` for consistent interface
- Uses a linear beta schedule for noise
- Implements a simplified denoising process
- Generates structured patterns from random noise
"""


# %%
class SimpleDiffusionModel(GenerativeModel):
    """A simplified diffusion-like noise generator.

    This is a minimal implementation that generates simple patterns
    by gradually denoising random noise, simulating the diffusion process
    but without the complexity of a full diffusion model.

    The model demonstrates three key phases of generation:
    1. **Early steps:** Reduce noise magnitude
    2. **Middle steps:** Introduce spatial structure (centered patterns)
    3. **Late steps:** Refine and sharpen the output
    """

    def __init__(
        self,
        config,
        *,
        rngs: nnx.Rngs,
        precision: jax.lax.Precision | None = None,
    ):
        """Initialize the simple diffusion model.

        Args:
            config: Model configuration dictionary with keys:
                - input_dim: Shape of generated samples (H, W, C)
                - noise_steps: Number of denoising steps (default: 50)
                - beta_start: Starting noise level (default: 1e-4)
                - beta_end: Ending noise level (default: 0.02)
            rngs: Random number generators for reproducibility
            precision: Numerical precision for computations
        """
        # CRITICAL: Always call super().__init__() in NNX modules
        super().__init__(
            rngs=rngs,
            precision=precision,
        )

        self.config = config
        self.data_shape = config["input_dim"]
        self.noise_steps = config.get("noise_steps", 50)
        self.beta_start = config.get("beta_start", 1e-4)
        self.beta_end = config.get("beta_end", 0.02)

        # Create beta schedule (linear interpolation)
        # Beta controls the amount of noise at each timestep
        self.betas = jnp.linspace(self.beta_start, self.beta_end, self.noise_steps)

        # Alpha = 1 - beta (signal retention rate)
        self.alphas = 1.0 - self.betas

        # Cumulative product of alphas (total signal remaining)
        self.alpha_cumprod = jnp.cumprod(self.alphas)

    def sample(self, batch_size=1, num_steps=None, *, rngs=None):
        """Generate samples by denoising random noise.

        This method implements a simplified reverse diffusion process:
        1. Start with pure random noise
        2. Iteratively reduce noise and add structure
        3. Return clean generated samples

        Args:
            batch_size: Number of samples to generate
            num_steps: Number of denoising steps (uses noise_steps if None)
            rngs: Random number generators for sampling

        Returns:
            Generated samples of shape (batch_size, H, W, C) in range [-1, 1]
        """
        # Use provided RNGs or fall back to stored ones
        if rngs is None:
            rngs = self._rngs

        if num_steps is None:
            num_steps = self.noise_steps

        # Start from random noise
        # CORRECT RNG PATTERN: Check if key exists, provide fallback
        if rngs is not None and "dropout" in rngs:
            rng = rngs.dropout()
        elif rngs is not None and "sample" in rngs:
            rng = rngs.sample()
        else:
            rng = jax.random.key(0)

        # Generate initial noise from standard normal distribution
        x = jax.random.normal(rng, shape=(batch_size, *self.data_shape))

        # Simple denoising process with three phases
        for i in range(num_steps):
            # Current timestep (counting backward from noise_steps)
            t = self.noise_steps - i - 1
            alpha = self.alpha_cumprod[t]

            # Phase 1: Early steps (first third) - Reduce noise magnitude
            if i < num_steps // 3:
                x = 0.99 * x

            # Phase 2: Middle steps - Add spatial structure
            elif i < 2 * num_steps // 3:
                # Create a centered circular pattern
                center_h = self.data_shape[0] // 2
                center_w = self.data_shape[1] // 2

                # Calculate distance from center for each pixel
                h_dist = jnp.abs(jnp.arange(self.data_shape[0]) - center_h)
                w_dist = jnp.abs(jnp.arange(self.data_shape[1]) - center_w)
                h_dist = h_dist.reshape(-1, 1) / self.data_shape[0]
                w_dist = w_dist.reshape(1, -1) / self.data_shape[1]

                # Euclidean distance creates radial patterns
                dist = jnp.sqrt(h_dist**2 + w_dist**2)
                pattern = 1.0 - dist.reshape(1, *dist.shape, 1)
                pattern = jnp.broadcast_to(pattern, x.shape)

                # Mix noise with pattern based on alpha schedule
                noise_weight = jnp.sqrt(1 - alpha)
                pattern_weight = jnp.sqrt(alpha)
                x = noise_weight * x + pattern_weight * pattern

            # Phase 3: Later steps - Refine and sharpen
            else:
                noise_weight = jnp.sqrt(1 - alpha) * 0.5
                signal_weight = jnp.sqrt(alpha) * 1.2
                x = noise_weight * x + signal_weight * jnp.tanh(x)

        # Ensure output is in [-1, 1] range using tanh
        x = jnp.tanh(x)
        return x


print("✅ SimpleDiffusionModel class defined!")

# %% [markdown]
"""
## 3. Configure the Model

Now we'll set up the model parameters and create an instance.

### Configuration Parameters:
- **input_dim:** Shape of generated images (32x32 RGB)
- **noise_steps:** Number of denoising iterations (50)
- **beta_start:** Initial noise level (0.0001)
- **beta_end:** Final noise level (0.02)

### Beta Schedule Explained:
The beta values control how much noise is present at each timestep. A linear schedule
gradually increases noise from a small value to a larger one, creating a smooth
transition from data to noise.
"""

# %%
# Set random seed for reproducibility
seed = 42
key = jax.random.key(seed)

# Split key for different purposes (good practice)
key, params_key, dropout_key = jax.random.split(key, 3)

# Create RNG streams for the model
rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

# Define shape of generated samples
# Format: (height, width, channels)
data_shape = (32, 32, 3)  # 32x32 RGB images
batch_size = 4  # Generate 4 samples

# Configure diffusion model
config = {
    "name": "simple_diffusion",
    "input_dim": data_shape,
    "noise_steps": 50,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}

print("=" * 80)
print("Diffusion Model Configuration")
print("=" * 80)
print(f"🖼️  Image shape: {data_shape}")
print(f"📊 Batch size: {batch_size}")
print(f"🔢 Noise steps: {config['noise_steps']}")
print(f"📉 Beta range: [{config['beta_start']:.6f}, {config['beta_end']:.6f}]")
print("=" * 80)

# %% [markdown]
"""
## 4. Create and Inspect the Model

Let's instantiate the model and examine its properties.
"""

# %%
print("\n🏗️  Creating SimpleDiffusionModel...")
model = SimpleDiffusionModel(config, rngs=rngs)

print("✅ Model created successfully!")
print("\n📋 Model Properties:")
print(f"  - Data shape: {model.data_shape}")
print(f"  - Noise steps: {model.noise_steps}")
print(f"  - Beta start: {model.beta_start}")
print(f"  - Beta end: {model.beta_end}")
print(f"  - Alpha min: {model.alpha_cumprod.min():.6f}")
print(f"  - Alpha max: {model.alpha_cumprod.max():.6f}")

# %% [markdown]
"""
## 5. Generate Samples

Now comes the exciting part - generating images from random noise!

The model will:
1. Start with pure Gaussian noise
2. Apply the denoising process over 50 steps
3. Produce structured patterns that emerge from randomness
"""

# %%
print("\n🎨 Generating samples from noise...")

# Create new RNG for sampling
key, sample_key = jax.random.split(key)
sampling_rngs = nnx.Rngs(dropout=sample_key)

# Generate samples (this is where the magic happens!)
samples = model.sample(batch_size=batch_size, num_steps=50, rngs=sampling_rngs)

print(f"✅ Generated {batch_size} samples!")
print(f"   Output shape: {samples.shape}")
print(f"   Value range: [{samples.min():.3f}, {samples.max():.3f}]")

# %% [markdown]
"""
## 6. Visualize Results

Let's visualize the generated samples to see what the model created.
"""

# %%
print("\n📊 Creating visualization...")

# Create figure with subplots
fig, axes = plt.subplots(1, batch_size, figsize=(12, 3))

for i in range(batch_size):
    # Get sample
    sample = samples[i]

    # Display the image
    if batch_size > 1:
        ax = axes[i]
    else:
        ax = axes

    # Convert from [-1, 1] to [0, 1] for RGB display
    sample = (sample + 1.0) / 2.0

    ax.imshow(sample)
    ax.set_title(f"Sample {i + 1}")
    ax.axis("off")

plt.tight_layout()

# Save the figure
output_dir = "examples_output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "diffusion_samples.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"✅ Visualization saved to: {output_path}")

# %% [markdown]
"""
## 7. Summary and Key Takeaways

### 🎓 What You Learned

In this example, you learned:

1. **Diffusion Model Basics:** How diffusion models generate data through iterative denoising
2. **Beta Schedules:** How noise schedules control the generation process
3. **Three-Phase Generation:** How to structure the denoising process
   (reduce noise → add structure → refine)
4. **RNG Handling:** Proper patterns for random number generation in Flax NNX
5. **JAX Operations:** Using JAX for efficient array operations

### 💡 Key Concepts Recap

- **Beta (β):** Controls noise level at each step
- **Alpha (α):** Signal retention rate (α = 1 - β)
- **Cumulative Product:** Tracks total signal remaining over time
- **Reverse Process:** Transforms noise into structured data

### 🔬 Experiments to Try

Now that you understand the basics, try these modifications:

1. **Change the beta schedule:**
   ```python
   config["beta_start"] = 1e-3  # More aggressive noise
   config["beta_end"] = 0.05
   ```

2. **Adjust the number of steps:**
   ```python
   config["noise_steps"] = 100  # Finer-grained process
   ```

3. **Modify the image size:**
   ```python
   data_shape = (64, 64, 3)  # Larger images
   ```

4. **Experiment with different patterns:**
   - Modify the middle phase to create different structures
   - Try spiral patterns instead of radial
   - Add color gradients

### 📚 Next Steps

To learn more about diffusion models, explore:

- **Full DDPM Implementation:** See `ddpm_cifar10.py` for complete training pipeline
- **Conditional Generation:** Learn to control generation with conditions
- **Advanced Samplers:** Explore DDIM, DPM-Solver, and other fast samplers
- **Latent Diffusion:** Understand how Stable Diffusion works in latent space

### 📖 Additional Resources

- **Paper:** [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- **Documentation:** [Artifex Diffusion Models Guide](../../../docs/models/diffusion.md)
- **Related Examples:**
  - `dit_demo.py` - Diffusion Transformer architecture
  - `advanced_diffusion.py` - Advanced techniques

### 🐛 Troubleshooting

**Problem:** Output looks like pure noise
- **Solution:** Increase `noise_steps` or adjust beta schedule range

**Problem:** Patterns are too faint
- **Solution:** Modify the weights in the middle phase (increase pattern_weight)

**Problem:** GPU memory issues
- **Solution:** Reduce `batch_size` or `data_shape`

### 💬 Feedback

Found a bug or have suggestions? Please open an issue on GitHub!

---

**Example completed successfully! 🎉**
"""

# %%
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("✨ Simple Diffusion Example Complete! ✨")
    print("=" * 80)
    print(f"\n📁 Output saved to: {output_path}")
    print("\n💡 Key Takeaways:")
    print("   1. Diffusion models transform noise into structured data")
    print("   2. Beta schedules control the generation process")
    print("   3. Iterative denoising reveals patterns from randomness")
    print("\n🔗 Next Steps:")
    print("   - Try modifying the beta schedule")
    print("   - Experiment with different image sizes")
    print("   - Explore dit_demo.py for advanced diffusion models")
    print("\n" + "=" * 80)
