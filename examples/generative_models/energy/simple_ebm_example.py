#!/usr/bin/env python3
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
r"""
# Energy-Based Models (EBM) Example

**Duration:** 15 minutes | **Level:** Intermediate | **GPU Required:** No
(recommended for faster sampling)

This example demonstrates how to create, train, and sample from Energy-Based Models (EBMs)
using the Artifex framework. You'll learn how EBMs learn probability distributions through
energy functions and how to generate samples using MCMC.

## 🎯 Learning Objectives

By the end of this example, you will:
1. Understand how Energy-Based Models work
2. Learn to compute energy functions and gradients (scores)
3. Generate samples using MCMC (Markov Chain Monte Carlo)
4. Apply contrastive divergence for efficient training
5. Use persistent contrastive divergence with sample buffers
6. Create deep EBMs with residual connections

## 🔍 Source Code Dependencies

**Validated:** 2025-10-14

This example depends on the following Artifex source files:
- `src/artifex/generative_models/models/energy/base.py` - Base EBM classes
- `src/artifex/generative_models/models/energy/ebm.py` - EBM implementations
- `src/artifex/generative_models/models/energy/mcmc.py` - MCMC sampling utilities
- `src/artifex/generative_models/core/configuration.py` - Configuration system
- `src/artifex/generative_models/factory.py` - Model factory

**Validation Status:**
- ✅ All dependencies validated against `memory-bank/guides/flax-nnx-guide.md`
- ✅ No anti-patterns detected (RNG handling, module init, activations)
- ✅ All tests passing for dependency files (Week 0 fixes applied)

## 📚 Background

Energy-Based Models (EBMs) are a powerful class of generative models that learn a function
E(x) called the "energy" of input x. The probability distribution is defined as:

$$p(x) \\propto \\exp(-E(x))$$

**Key Intuition:**
- **Low energy** → **High probability** (data-like samples)
- **High energy** → **Low probability** (unlikely samples)

The model learns to assign low energy to real data and high energy to fake data.

## 🔑 Key Concepts

- **Energy Function E(x):** Maps data to scalar energy values
- **Score Function ∇E(x):** Gradient of energy w.r.t. input
- **MCMC Sampling:** Markov Chain Monte Carlo for generating samples
- **Contrastive Divergence:** Efficient training algorithm
- **Persistent CD:** Reuses chains across iterations for better mixing
- **Sample Buffer:** Stores MCMC samples to improve efficiency

## ℹ️ Prerequisites

- Understanding of generative models
- Familiarity with MCMC concepts (helpful but not required)
- Basic knowledge of JAX and neural networks
- Artifex installed (see below)

## 📦 Setup

Before running this example, activate the Artifex environment:

```bash
source activate.sh
python examples/generative_models/energy/simple_ebm_example.py
```

## 🎬 Expected Output

This example will:
- Create EBMs for MNIST-like data
- Compute energy values for test images
- Generate samples using MCMC
- Demonstrate contrastive divergence loss
- Show persistent CD with sample buffers
- Create deep EBM with residual connections

## ⏱️ Estimated Runtime

- **CPU:** ~2 minutes
- **GPU:** ~30 seconds

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
- **JAX:** For high-performance numerical computing and automatic differentiation
- **Flax NNX:** For neural network modules
- **Artifex:** For EBM implementations, MCMC utilities, and configuration
"""

# %%
import time

import jax
import jax.numpy as jnp
from flax import nnx


# Optional: Force CPU for testing
# import os
# os.environ["JAX_PLATFORMS"] = "cpu"

print("=" * 80)
print("Energy-Based Model (EBM) Example")
print("=" * 80)
print(f"✅ JAX version: {jax.__version__}")
print(f"🖥️  Backend: {jax.default_backend()}")
print(f"🔧 Devices: {jax.devices()}")
print("=" * 80)

# %% [markdown]
"""
## 2. Create a Simple EBM for MNIST

Let's start by creating a basic EBM designed for MNIST-like data (28×28 grayscale images).

The `create_mnist_ebm` factory function creates an EBM with:
- CNN architecture optimized for 28×28 images
- Energy network that outputs scalar values
- Built-in MCMC sampling capabilities
"""

# %%
from artifex.generative_models.models.energy import create_mnist_ebm


print("\n📊 Creating EBM for MNIST-like data...")

# Initialize random number generators
rngs = nnx.Rngs(0)

# Create the model
model = create_mnist_ebm(rngs=rngs)

print("✅ Created MNIST EBM")
print(f"   Model type: {type(model).__name__}")
print("   Input shape: (28, 28, 1)")
print("   Output: Scalar energy values")

# %% [markdown]
"""
## 3. Compute Energy Values

The core of an EBM is the energy function E(x). Let's compute energies for test images.

**What's happening:**
- Model takes images as input
- Outputs a dictionary with:
  - `energy`: Scalar values (lower = more data-like)
  - `score`: Gradient ∇E(x) used for MCMC sampling
"""

# %%
print("\n⚡ Testing energy computation...")

# Create test batch (all ones as a simple test)
batch_size = 4
test_images = jnp.ones((batch_size, 28, 28, 1))

# Forward pass: compute energy and score
output = model(test_images)

print("✅ Energy computation successful!")
print(f"   Energy values shape: {output['energy'].shape}")
print(f"   Energy values: {output['energy']}")
print(f"   Score (gradient) shape: {output['score'].shape}")
print("\n💡 Interpretation:")
print("   - Lower energy = model thinks it's more likely")
print("   - Score shows direction to move in MCMC sampling")

# %% [markdown]
"""
## 4. Generate Samples Using MCMC

EBMs generate samples using Markov Chain Monte Carlo (MCMC):
1. Start from random noise
2. Iteratively move toward lower energy regions
3. Use score (gradient) to guide the movement

This is called **Langevin dynamics** or **score-based sampling**.
"""

# %%
print("\n🎨 Generating samples using MCMC...")

start_time = time.time()

# Generate samples using built-in MCMC sampler
samples = model.generate(
    n_samples=4,
    shape=(28, 28, 1),
    rngs=rngs,
    n_steps=50,  # Number of MCMC steps (more = better quality, slower)
)

elapsed = time.time() - start_time

print(f"✅ Generated {samples.shape[0]} samples in {elapsed:.2f}s")
print(f"   Sample shape: {samples.shape}")
print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
print("\n💡 MCMC Process:")
print("   1. Start from random noise")
print("   2. For each step, move toward lower energy")
print("   3. Add noise to avoid getting stuck")
print("   4. Result: samples from the learned distribution")

# %% [markdown]
"""
## 5. Using the Configuration System

Artifex provides a flexible configuration system for creating models.
This allows you to:
- Define model architecture declaratively
- Easily experiment with different configurations
- Save and load model specifications
"""

# %%
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import DeepEBM


print("\n⚙️  Creating EBM with custom configuration...")

# Define nested configurations for the EBM components

# Energy network configuration (what architecture computes E(x))
energy_network_config = EnergyNetworkConfig(
    name="custom_energy_network",
    hidden_dims=(64, 128, 64),  # CNN channel progression (must be tuple)
    activation="silu",  # SiLU/Swish activation
    network_type="cnn",  # Use CNN architecture for images
    use_bias=True,
)

# MCMC sampling configuration (how to generate samples)
mcmc_config = MCMCConfig(
    name="custom_mcmc",
    n_steps=60,  # Number of MCMC steps for sampling
    step_size=0.01,  # Langevin dynamics step size
    noise_scale=0.005,  # Noise added during sampling
)

# Sample buffer configuration (for persistent contrastive divergence)
sample_buffer_config = SampleBufferConfig(
    name="custom_buffer",
    capacity=4096,
    reinit_prob=0.05,
)

# Create the main DeepEBMConfig with nested configs
config = DeepEBMConfig(
    name="custom_ebm",
    input_shape=(28, 28, 1),  # MNIST-like images
    energy_network=energy_network_config,
    mcmc=mcmc_config,
    sample_buffer=sample_buffer_config,
    alpha=0.01,  # Regularization coefficient
)

# Create model from configuration
custom_model = DeepEBM(config=config, rngs=rngs)

print("✅ Created custom EBM from configuration")
print(f"   Architecture: {config.energy_network.hidden_dims}")
print(f"   Activation: {config.energy_network.activation}")
print(f"   MCMC steps: {config.mcmc.n_steps}")

# %% [markdown]
r"""
## 6. Contrastive Divergence Loss

EBMs are trained using **Contrastive Divergence** (CD):
- Maximize probability of real data (lower their energy)
- Minimize probability of fake/generated data (raise their energy)

**Loss Formula:**
$$\\mathcal{L} = E_{\\text{real}}(x) - E_{\\text{fake}}(x) + \\alpha \\|E\\|^2$$

Where:
- First term: Energy of real data (want to minimize)
- Second term: Energy of fake data (want to maximize, hence negative)
- Third term: Regularization to prevent energy from going to -∞
"""

# %%
print("\n📉 Computing contrastive divergence loss...")

# Create real and fake data batches
real_data = jax.random.normal(rngs.sample(), (4, 28, 28, 1))
fake_data = jax.random.normal(rngs.sample(), (4, 28, 28, 1))

# Compute CD loss
loss_dict = custom_model.contrastive_divergence_loss(
    real_data=real_data,
    fake_data=fake_data,
    alpha=0.01,  # Regularization weight
)

print("✅ Loss computation successful!")
print(f"   Total loss: {loss_dict['loss']:.4f}")
print(f"   Real energy: {loss_dict['real_energy_mean']:.4f}")
print(f"   Fake energy: {loss_dict['fake_energy_mean']:.4f}")
print("\n💡 Training Goal:")
print("   - Push real energy DOWN")
print("   - Push fake energy UP")
print("   - Maximize energy difference between real and fake")

# %% [markdown]
"""
## 7. Persistent Contrastive Divergence

**Problem with standard CD:** Starting MCMC from random noise every iteration is slow.

**Solution: Persistent CD (PCD)**
- Maintain a buffer of samples across iterations
- Initialize new chains from buffer instead of noise
- Chains can explore the distribution more thoroughly
- Much more efficient for training

This is a key technique for practical EBM training.
"""

# %%
from artifex.generative_models.models.energy.mcmc import (
    persistent_contrastive_divergence,
    SampleBuffer,
)


print("\n🔄 Using persistent contrastive divergence...")

# Initialize sample buffer for efficient training
buffer = SampleBuffer(
    capacity=256,  # Maximum number of samples to store
    reinit_prob=0.05,  # 5% chance to reinitialize a sample
)

print(f"📦 Created sample buffer (capacity: {buffer.capacity})")

# Generate samples using persistent CD
real_samples = jax.random.normal(rngs.sample(), (8, 28, 28, 1))

real_processed, fake_samples = persistent_contrastive_divergence(
    energy_fn=custom_model.energy,
    real_samples=real_samples,
    sample_buffer=buffer,
    rng_key=rngs.sample(),
    n_mcmc_steps=30,  # Fewer steps needed with persistence
    step_size=0.01,
)

print("✅ Persistent CD completed!")
print(f"   Buffer size: {len(buffer.buffer)} / {buffer.capacity}")
print(f"   Generated fake samples: {fake_samples.shape}")
print(f"   Fake sample range: [{fake_samples.min():.3f}, {fake_samples.max():.3f}]")
print("\n💡 Why Persistent CD?")
print("   - Reuses MCMC chains across iterations")
print("   - Chains mix better over time")
print("   - Much faster than starting from scratch")
print("   - Essential for training on complex data")

# %% [markdown]
"""
## 8. Deep EBM with Residual Connections

For more complex data (e.g., CIFAR-10, ImageNet), we need deeper architectures.

The `DeepEBM` class provides:
- Multiple residual blocks for deep networks
- Spectral normalization for training stability
- Support for higher resolution images
- More expressive energy functions
"""

# %%
# DeepEBM is already imported above, we just need fresh rngs for a new model
deep_rngs = nnx.Rngs(42)

print("\n🏗️  Creating Deep EBM...")

# Configuration for Deep EBM (suitable for CIFAR-10 or similar)
# Create nested configs for the deep energy network
deep_energy_network_config = EnergyNetworkConfig(
    name="deep_energy_network",
    hidden_dims=(32, 64, 128),  # Channel progression (must be tuple)
    activation="silu",
    network_type="cnn",  # CNN for images
    use_bias=True,
    use_spectral_norm=True,  # Stabilizes training
    use_residual=True,  # Residual connections for deep networks
)

deep_mcmc_config = MCMCConfig(
    name="deep_mcmc",
    n_steps=100,  # More steps for complex data
    step_size=0.005,
    noise_scale=0.001,
)

deep_sample_buffer_config = SampleBufferConfig(
    name="deep_buffer",
    capacity=8192,  # Larger buffer for complex data
    reinit_prob=0.05,
)

deep_config = DeepEBMConfig(
    name="deep_ebm",
    input_shape=(32, 32, 3),  # RGB images (CIFAR-10 size)
    energy_network=deep_energy_network_config,
    mcmc=deep_mcmc_config,
    sample_buffer=deep_sample_buffer_config,
    alpha=0.001,  # Lower regularization for deep models
)

# Create Deep EBM
deep_ebm = DeepEBM(config=deep_config, rngs=deep_rngs)

print("✅ Created Deep EBM with residual connections")
print(f"   Input shape: {deep_config.input_shape}")
print(f"   Hidden dims: {deep_config.energy_network.hidden_dims}")
print(f"   Spectral norm: {deep_config.energy_network.use_spectral_norm}")
print(f"   Residual: {deep_config.energy_network.use_residual}")

# Test on a batch
test_batch = jnp.ones((2, 32, 32, 3))
deep_output = deep_ebm(test_batch)

print("\n🧪 Test inference:")
print(f"   Deep EBM energy shape: {deep_output['energy'].shape}")
print(f"   Energy values: {deep_output['energy']}")

# %% [markdown]
"""
## 9. Summary and Key Takeaways

### 🎓 What You Learned

In this example, you learned:

1. **EBM Fundamentals:** How energy functions define probability distributions
2. **Energy Computation:** Computing E(x) and score ∇E(x) for any input
3. **MCMC Sampling:** Generating samples using Langevin dynamics
4. **Contrastive Divergence:** The standard training algorithm for EBMs
5. **Persistent CD:** Efficient training with sample buffers
6. **Deep Architectures:** Residual connections and spectral normalization

### 💡 Key Concepts Recap

- **Energy Function:** E(x) where p(x) ∝ exp(-E(x))
- **Low Energy = High Probability:** Model assigns low energy to data
- **MCMC Sampling:** Iteratively move toward low energy regions
- **Contrastive Divergence:** Push down real energy, push up fake energy
- **Persistent CD:** Reuse MCMC chains for better mixing
- **Sample Buffer:** Stores samples across training iterations

### 🔬 Experiments to Try

Now that you understand the basics, try these modifications:

1. **Adjust MCMC parameters:**
   ```python
   samples = model.generate(
       n_samples=4,
       n_steps=100,  # More steps for better samples
       step_size=0.02,  # Larger steps (but less stable)
   )
   ```

2. **Change the architecture:**
   ```python
   config.hidden_dims = [128, 256, 128]  # Deeper network
   config.activation = "gelu"  # Different activation
   ```

3. **Modify CD parameters:**
   ```python
   loss_dict = model.contrastive_divergence_loss(
       real_data=real_data,
       fake_data=fake_data,
       alpha=0.1,  # Stronger regularization
   )
   ```

4. **Experiment with sample buffer:**
   ```python
   buffer = SampleBuffer(
       capacity=512,  # Larger buffer
       reinit_prob=0.1,  # More reinitialization
   )
   ```

### 📚 Next Steps

To learn more about Energy-Based Models:

- **Training EBMs:** See complete training loop with optimization
- **Conditional EBMs:** Learn to control generation with class labels
- **Score Matching:** Alternative training method to contrastive divergence
- **Denoising Score Matching:** Connection to diffusion models

### 📖 Additional Resources

- **Paper:** [Energy-Based Models (LeCun et al.)](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
- **Paper:** [Training with Contrastive Divergence](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)
- **Paper:** [Improved Contrastive Divergence Training](https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf)
- **Documentation:** [Artifex EBM Guide](../../../docs/models/energy.md)
- **Related Examples:**
  - `dit_demo.py` - Diffusion models (related to score-based models)
  - `simple_diffusion_example.py` - Understanding diffusion

### 🐛 Troubleshooting

**Problem:** MCMC samples look like noise
- **Solution:** Increase `n_steps` or decrease `step_size`

**Problem:** Training is too slow
- **Solution:** Use persistent CD with a sample buffer

**Problem:** Energy values explode or collapse
- **Solution:** Increase regularization `alpha` or use spectral normalization

**Problem:** Samples don't match data distribution
- **Solution:** Train longer, use more MCMC steps, or increase model capacity

### 💬 Feedback

Found a bug or have suggestions? Please open an issue on GitHub!

---

**Example completed successfully! 🎉**
"""

# %%
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("✨ Energy-Based Model Example Complete! ✨")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   1. EBMs learn energy functions E(x) where p(x) ∝ exp(-E(x))")
    print("   2. Lower energy = higher probability under the model")
    print("   3. Training uses contrastive divergence with MCMC sampling")
    print("   4. Persistent CD with sample buffers improves efficiency")
    print("   5. Deep EBMs with residual connections handle complex data")
    print("\n🔗 Next Steps:")
    print("   - Try different MCMC parameters (steps, step_size)")
    print("   - Experiment with model architectures")
    print("   - Explore persistent CD with different buffer sizes")
    print("   - Learn about score matching as an alternative to CD")
    print("\n" + "=" * 80)
