# Energy-Based Models User Guide

Complete guide to building, training, and using Energy-Based Models with Artifex.

## Overview

This guide covers practical usage of EBMs in Artifex, from basic setup to advanced techniques. You'll learn how to:

<div class="grid cards" markdown>

- :material-cog: **Configure EBMs**

    ---

    Set up energy functions and MCMC sampling parameters

- :material-play: **Train Models**

    ---

    Train with persistent contrastive divergence and monitor stability

- :material-creation: **Generate Samples**

    ---

    Sample using Langevin dynamics and MCMC methods

- :material-tune: **Tune & Debug**

    ---

    Optimize hyperparameters and troubleshoot common issues

</div>

---

## Quick Start

### Basic EBM Example

Artifex provides factory functions for common use cases:

```python
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.models.energy import create_mnist_ebm

# Initialize RNGs
rngs = nnx.Rngs(0)

# Create EBM optimized for MNIST-like data
model = create_mnist_ebm(rngs=rngs)

# Compute energy for a batch of images
batch = jnp.ones((4, 28, 28, 1))
output = model(batch)

print(f"Energy values: {output['energy']}")
print(f"Score shape: {output['score'].shape}")

# Generate samples using MCMC
model.eval()  # Set to evaluation mode
samples = model.generate(n_samples=4, shape=(28, 28, 1))
print(f"Generated samples shape: {samples.shape}")
```

### Configuration-Based Approach

For more control, use the config-based API:

```python
from artifex.generative_models.core.configuration.energy_config import (
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import EBM

# Configure energy network
energy_config = EnergyNetworkConfig(
    name="energy_net",
    hidden_dims=(128, 256, 128),
    activation="silu",
    network_type="mlp",
)

# Configure MCMC sampling
mcmc_config = MCMCConfig(
    name="langevin",
    n_steps=60,           # Number of MCMC steps
    step_size=0.01,
    noise_scale=0.005,
)

# Configure sample buffer (replay buffer)
buffer_config = SampleBufferConfig(
    name="buffer",
    capacity=8192,        # Maximum samples to store
    reinit_prob=0.05,     # Probability to reinitialize samples
)

# Create EBM config
config = EBMConfig(
    name="mnist_ebm",
    input_dim=784,  # Flattened MNIST
    energy_network=energy_config,
    mcmc=mcmc_config,
    sample_buffer=buffer_config,
    alpha=0.01,  # Regularization
)

# Create model
rngs = nnx.Rngs(params=0, noise=1, sample=2)
model = EBM(config, rngs=rngs)
```

---

## Creating EBM Models

### 1. Standard EBM (MLP Energy Function)

For tabular or low-dimensional data:

```python
from artifex.generative_models.core.configuration.energy_config import (
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import EBM

# MLP energy function configuration
energy_config = EnergyNetworkConfig(
    name="tabular_energy",
    hidden_dims=(256, 256, 128),
    activation="gelu",
    dropout_rate=0.1,
    network_type="mlp",
)

# MCMC sampling configuration
mcmc_config = MCMCConfig(
    name="mcmc",
    n_steps=60,
    step_size=0.01,
    noise_scale=0.005,
)

# Sample buffer configuration
buffer_config = SampleBufferConfig(
    name="buffer",
    capacity=4096,
)

# Create EBM config
config = EBMConfig(
    name="tabular_ebm",
    input_dim=784,  # Flattened input
    energy_network=energy_config,
    mcmc=mcmc_config,
    sample_buffer=buffer_config,
    alpha=0.01,
)

rngs = nnx.Rngs(params=0, noise=1, sample=2)
model = EBM(config, rngs=rngs)
```

**Key Parameters:**

| Config Class | Parameter | Default | Description |
|-------------|-----------|---------|-------------|
| `EnergyNetworkConfig` | `network_type` | "mlp" | Energy function architecture ("mlp" or "cnn") |
| `EnergyNetworkConfig` | `hidden_dims` | (128, 128) | Hidden layer dimensions |
| `EnergyNetworkConfig` | `activation` | "gelu" | Activation function name |
| `MCMCConfig` | `n_steps` | 60 | Number of Langevin dynamics steps |
| `MCMCConfig` | `step_size` | 0.01 | Step size for gradient descent |
| `MCMCConfig` | `noise_scale` | 0.005 | Noise scale for exploration |
| `SampleBufferConfig` | `capacity` | 8192 | Maximum samples in replay buffer |
| `SampleBufferConfig` | `reinit_prob` | 0.05 | Probability to reinitialize from scratch |
| `EBMConfig` | `alpha` | 0.01 | Regularization strength |

### 2. CNN Energy Function (for Images)

For image data:

```python
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import DeepEBM

# CNN energy network for images
energy_config = EnergyNetworkConfig(
    name="image_energy",
    hidden_dims=(64, 128, 256),
    activation="silu",
    network_type="cnn",
)

mcmc_config = MCMCConfig(
    name="mcmc",
    n_steps=100,
    step_size=0.005,
    noise_scale=0.001,
)

buffer_config = SampleBufferConfig(
    name="buffer",
    capacity=8192,
    reinit_prob=0.05,
)

config = DeepEBMConfig(
    name="image_ebm",
    input_shape=(32, 32, 3),  # CIFAR-10 dimensions (H, W, C)
    energy_network=energy_config,
    mcmc=mcmc_config,
    sample_buffer=buffer_config,
    alpha=0.001,
)

rngs = nnx.Rngs(params=0, noise=1, sample=2)
model = DeepEBM(config, rngs=rngs)
```

### 3. Deep EBM (Complex Data)

For complex datasets requiring deeper architectures:

```python
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import DeepEBM

energy_config = EnergyNetworkConfig(
    name="deep_energy",
    hidden_dims=(32, 64, 128, 256),
    activation="silu",
    network_type="cnn",
    use_residual=True,
    use_spectral_norm=True,
)

mcmc_config = MCMCConfig(
    name="mcmc",
    n_steps=100,
    step_size=0.005,
    noise_scale=0.001,
)

buffer_config = SampleBufferConfig(
    name="buffer",
    capacity=8192,
)

config = DeepEBMConfig(
    name="deep_ebm",
    input_shape=(32, 32, 3),
    energy_network=energy_config,
    mcmc=mcmc_config,
    sample_buffer=buffer_config,
    alpha=0.001,
)

rngs = nnx.Rngs(params=0, noise=1, sample=2)
model = DeepEBM(config, rngs=rngs)
```

**Deep EBM Features:**

- **Residual connections**: Enable deeper networks (10+ layers)
- **Spectral normalization**: Stabilizes training
- **GroupNorm**: Better than BatchNorm for MCMC sampling

---

## Training EBMs

### Basic Training Loop

```python
import optax
from flax import nnx
from artifex.generative_models.training.trainers import (
    EnergyTrainer,
    EnergyTrainingConfig,
)

# Training configuration
train_config = EnergyTrainingConfig(
    training_method="pcd",       # Persistent Contrastive Divergence
    mcmc_sampler="langevin",
    mcmc_steps=20,
    step_size=0.01,
    noise_scale=0.005,
    replay_buffer_size=10000,
    replay_buffer_init_prob=0.95,
)

# Create trainer, optimizer, and RNG key
trainer = EnergyTrainer(train_config)
optimizer = nnx.Optimizer(model, optax.adam(1e-4))
rng = jax.random.key(42)

# Training loop
for epoch in range(100):
    for batch in train_loader:
        rng, step_rng = jax.random.split(rng)
        loss, metrics = trainer.train_step(model, optimizer, batch, step_rng)
```

### Training with Monitoring

Monitor key metrics during training:

```python
def train_step_with_monitoring(model, batch):
    """Training step with detailed monitoring using model.train_step()."""
    loss_dict = model.train_step(batch)

    # Log metrics (keys from contrastive_divergence_loss)
    print(f"Step metrics:")
    print(f"  Loss: {loss_dict['loss']:.4f}")
    print(f"  Real energy: {loss_dict['real_energy_mean']:.4f}")
    print(f"  Fake energy: {loss_dict['fake_energy_mean']:.4f}")

    # Compute energy gap
    energy_gap = float(loss_dict['fake_energy_mean'] - loss_dict['real_energy_mean'])
    print(f"  Energy gap: {energy_gap:.4f}")

    # Check for issues
    if energy_gap < 0:
        print("WARNING: Negative energy gap - real data has higher energy!")

    if abs(float(loss_dict['real_energy_mean'])) > 100:
        print("WARNING: Energy explosion detected!")

    return loss_dict

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        loss_dict = train_step_with_monitoring(model, batch)
```

### Hyperparameter Guidelines

**MCMC Sampling:**

```python
# Quick sampling (less accurate)
quick_config = {
    "mcmc_steps": 20,
    "mcmc_step_size": 0.02,
    "mcmc_noise_scale": 0.01,
}

# Standard sampling (balanced)
standard_config = {
    "mcmc_steps": 60,
    "mcmc_step_size": 0.01,
    "mcmc_noise_scale": 0.005,
}

# High-quality sampling (slower)
quality_config = {
    "mcmc_steps": 200,
    "mcmc_step_size": 0.005,
    "mcmc_noise_scale": 0.001,
}
```

**Learning Rates:**

```python
# EBMs typically need lower learning rates than supervised models
learning_rates = {
    "small_model": 1e-4,
    "medium_model": 5e-5,
    "large_model": 1e-5,
}
```

---

## Generating Samples

### Sampling from the Model

```python
# Generate samples using Langevin dynamics MCMC
n_samples = 16
model.eval()  # Set to evaluation mode for sampling
samples = model.generate(
    n_samples=n_samples,
    shape=(784,),         # Shape of each sample (must match input_dim for MLP)
    n_steps=100,          # More steps = better quality
    step_size=0.01,
    noise_scale=0.005,
)

print(f"Generated samples shape: {samples.shape}")
```

### Sampling with Different Configurations

```python
# Quick sampling (fewer steps, larger step size)
quick_samples = model.generate(
    n_samples=16,
    shape=(784,),
    n_steps=30,
    step_size=0.02,
    noise_scale=0.01,
)

# High-quality sampling (more steps, smaller step size)
hq_samples = model.generate(
    n_samples=16,
    shape=(784,),
    n_steps=200,
    step_size=0.005,
    noise_scale=0.001,
)

# Sample from the replay buffer (returns buffered MCMC samples)
buffer_samples = model.sample_from_buffer(n_samples=16)
```

### Using the EnergyTrainer for Generation

The `EnergyTrainer` also provides a `generate_samples` method for MCMC-based generation:

```python
from artifex.generative_models.training.trainers import (
    EnergyTrainer,
    EnergyTrainingConfig,
)

trainer = EnergyTrainer(EnergyTrainingConfig(mcmc_steps=20, step_size=0.01))
rng = jax.random.key(0)

# Generate samples via the trainer's MCMC chain
samples = trainer.generate_samples(
    model=model,
    batch_size=16,
    key=rng,
    shape=(784,),       # Shape of each sample
    num_steps=200,      # Defaults to 10x config mcmc_steps if None
)
```

---

## Advanced Techniques

### 1. Sample Buffer Management

The sample buffer is critical for stable training:

```python
# Access buffer statistics
buffer_size = len(model.sample_buffer.buffer)
print(f"Buffer contains {buffer_size} samples")

# Manually populate buffer by running train steps
# The train_step method automatically updates the sample buffer
for batch in train_loader:
    loss_dict = model.train_step(batch)
    # Samples are automatically added to the buffer during training

# Clear buffer (for reinitialization)
model.sample_buffer.buffer = []
```

### 2. Energy Landscape Visualization

Visualize the energy landscape:

```python
import matplotlib.pyplot as plt

def visualize_energy_landscape(model, data_range=(-3, 3), resolution=100):
    """Visualize 2D energy landscape."""
    x = jnp.linspace(data_range[0], data_range[1], resolution)
    y = jnp.linspace(data_range[0], data_range[1], resolution)
    X, Y = jnp.meshgrid(x, y)

    # Compute energy for each point
    points = jnp.stack([X.ravel(), Y.ravel()], axis=1)
    energies = model.energy(points)
    energies = energies.reshape(resolution, resolution)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, energies, levels=50, cmap='viridis')
    plt.colorbar(label='Energy')
    plt.title('Energy Landscape')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# For 2D data
visualize_energy_landscape(model)
```

### 3. Annealed Importance Sampling

For better sampling quality:

```python
def annealed_sampling(model, n_samples, sample_shape, n_steps=1000, rng_key=None):
    """Annealed importance sampling for high-quality samples."""
    if rng_key is None:
        rng_key = jax.random.key(0)

    # Start with high temperature
    temperatures = jnp.linspace(10.0, 1.0, n_steps)

    # Initialize samples
    rng_key, init_key = jax.random.split(rng_key)
    samples = jax.random.normal(init_key, (n_samples, *sample_shape))

    for i, temp in enumerate(temperatures):
        rng_key, step_key = jax.random.split(rng_key)

        # Compute energy gradient
        energy_grad = jax.grad(lambda x: jnp.sum(model.energy(x)))(samples)

        # Langevin step with temperature
        step_size = 0.01 * temp
        noise_scale = jnp.sqrt(2 * step_size * temp)

        samples = samples - step_size * energy_grad
        samples = samples + noise_scale * jax.random.normal(
            step_key, samples.shape
        )

    return samples

# Use annealed sampling (pass sample shape explicitly)
high_quality_samples = annealed_sampling(
    model, n_samples=16, sample_shape=(784,), rng_key=jax.random.key(0)
)
```

---

## Troubleshooting

### Common Issues and Solutions

<div class="grid cards" markdown>

- :material-alert: **Energy Explosion**

    ---

    **Symptoms**: Energy values grow unbounded, NaN losses

    **Solutions**:
  - Reduce learning rate (try 1e-5)
  - Add/increase regularization (alpha=0.01 to 0.1)
  - Use spectral normalization
  - Clip gradients: `max_grad_norm=1.0`

    ```python
    # Use a higher alpha when creating the config
    config = EBMConfig(..., alpha=0.1)
    ```

- :material-alert: **Poor Sample Quality**

    ---

    **Symptoms**: Samples look like noise or blurry

    **Solutions**:
  - Increase MCMC steps (60 → 100+)
  - Better step size tuning
  - Larger buffer capacity
  - Deeper energy function

    ```python
    # Increase MCMC steps and buffer capacity in config
    mcmc_config = MCMCConfig(name="mcmc", n_steps=100, ...)
    buffer_config = SampleBufferConfig(name="buffer", capacity=16384, ...)
    ```

- :material-alert: **Mode Collapse**

    ---

    **Symptoms**: All samples look similar

    **Solutions**:
  - Increase buffer reinit probability
  - Use data augmentation
  - Longer MCMC chains
  - Larger buffer

    ```python
    # Increase reinit probability in the sample buffer config
    buffer_config = SampleBufferConfig(name="buffer", capacity=8192, reinit_prob=0.1)
    ```

- :material-alert: **Training Instability**

    ---

    **Symptoms**: Oscillating losses, sudden divergence

    **Solutions**:
  - Lower learning rate
  - Use persistent buffer
  - Add gradient clipping
  - Monitor energy gap

    ```python
    # Use PCD training method for persistent buffer
    train_config = EnergyTrainingConfig(training_method="pcd", ...)
    trainer = EnergyTrainer(train_config)
    ```

</div>

### Debugging Checklist

```python
def diagnose_ebm(model, batch, sample_shape):
    """Diagnostic checks for EBM training.

    Args:
        model: Trained EBM or DeepEBM instance
        batch: A batch dict with 'image' or 'data' key
        sample_shape: Shape of each sample (e.g. (784,) or (28, 28, 1))
    """
    # Extract data from batch
    real_data = batch.get('image', batch.get('data'))

    # 1. Check energy values
    real_energy = model.energy(real_data).mean()
    print(f"Real data energy: {real_energy:.3f}")

    # Generate samples
    model.eval()
    fake_samples = model.generate(n_samples=16, shape=sample_shape, n_steps=100)
    fake_energy = model.energy(fake_samples).mean()
    print(f"Generated samples energy: {fake_energy:.3f}")

    # Energy gap should be positive
    gap = fake_energy - real_energy
    print(f"Energy gap: {gap:.3f}")

    # 2. Check MCMC convergence
    rng_key = jax.random.key(0)
    init_samples = jax.random.uniform(rng_key, (16, *sample_shape), minval=-1.0, maxval=1.0)
    init_energy = model.energy(init_samples).mean()

    # Use generate() to run MCMC from random init
    final_samples = model.generate(
        n_samples=16,
        shape=sample_shape,
        n_steps=100,
        step_size=0.01,
        noise_scale=0.005,
    )
    final_energy = model.energy(final_samples).mean()

    energy_decrease = init_energy - final_energy
    print(f"MCMC energy decrease: {energy_decrease:.3f}")

    # 3. Check buffer health
    buffer_size = len(model.sample_buffer.buffer)
    print(f"Buffer size: {buffer_size}/{model.sample_buffer.capacity}")

    # 4. Check sample validity
    sample_min, sample_max = float(fake_samples.min()), float(fake_samples.max())
    print(f"Sample range: [{sample_min:.3f}, {sample_max:.3f}]")

    return {
        "real_energy": real_energy,
        "fake_energy": fake_energy,
        "energy_gap": gap,
        "mcmc_decrease": energy_decrease,
        "buffer_usage": buffer_size / model.sample_buffer.capacity,
    }

# Run diagnostics
diagnostics = diagnose_ebm(model, batch, sample_shape=(784,))
```

---

## Best Practices

### 1. Start Simple

```python
# Begin with a small model and simple data using the factory function
from artifex.generative_models.models.energy import create_simple_ebm

rngs = nnx.Rngs(params=0, noise=1, sample=2)
model = create_simple_ebm(
    input_dim=2,  # 2D for visualization
    rngs=rngs,
    hidden_dims=(64, 64),
    activation="relu",
    mcmc_steps=30,
    step_size=0.02,
    sample_buffer_capacity=1024,
)
```

### 2. Gradually Increase Complexity

```python
from artifex.generative_models.models.energy import create_mnist_ebm, create_cifar_ebm

# Once stable, increase capacity (MNIST-like data)
medium_model = create_mnist_ebm(
    rngs=rngs,
    hidden_dims=(128, 256),
    mcmc_steps=60,
    sample_buffer_capacity=4096,
)

# For complex data (CIFAR-like images with residual + spectral norm)
complex_model = create_cifar_ebm(
    rngs=rngs,
    hidden_dims=(64, 128, 256, 512),
    use_residual=True,
    use_spectral_norm=True,
    mcmc_steps=100,
    sample_buffer_capacity=8192,
)
```

### 3. Monitor Training Carefully

```python
# Log detailed metrics
def detailed_training_step(model, batch, step, sample_shape):
    loss_dict = model.train_step(batch)

    if step % 100 == 0:
        # Detailed logging (keys from contrastive_divergence_loss)
        print(f"\nStep {step}:")
        print(f"  Loss: {loss_dict['loss']:.4f}")
        print(f"  Real energy: {loss_dict['real_energy_mean']:.4f}")
        print(f"  Fake energy: {loss_dict['fake_energy_mean']:.4f}")
        gap = float(loss_dict['fake_energy_mean'] - loss_dict['real_energy_mean'])
        print(f"  Gap: {gap:.4f}")

        # Generate samples for visual inspection
        if step % 1000 == 0:
            model.eval()
            samples = model.generate(n_samples=64, shape=sample_shape)
            model.train()
            visualize_samples(samples, f"step_{step}.png")

    return loss_dict
```

### 4. Use Proper Preprocessing

```python
def preprocess_for_ebm(images):
    """Proper preprocessing for image EBMs."""
    # Normalize to [-1, 1]
    images = (images - 127.5) / 127.5

    # Add small noise during training
    if training:
        noise = jax.random.normal(rng_key, images.shape) * 0.005
        images = images + noise
        images = jnp.clip(images, -1.0, 1.0)

    return images
```

---

## Performance Optimization

### GPU Acceleration

```python
# EBMs benefit significantly from GPU
from artifex.generative_models.core.device_manager import DeviceManager

device_manager = DeviceManager()
device = device_manager.get_device()
print(f"Using device: {device}")

# Move data to GPU
batch_gpu = jax.device_put(batch, device)
```

### Batch Size Tuning

```python
# Larger batches = more stable gradients
# But: limited by GPU memory

batch_sizes = {
    "small_model": 256,
    "medium_model": 128,
    "large_model": 64,
}
```

### JIT Compilation

```python
# Use nnx.jit with the EnergyTrainer for JIT-compiled training
from artifex.generative_models.training.trainers import EnergyTrainer, EnergyTrainingConfig

trainer = EnergyTrainer(EnergyTrainingConfig(training_method="pcd"))
jit_step = nnx.jit(trainer.train_step)

# Much faster after first call
rng = jax.random.key(0)
rng, step_rng = jax.random.split(rng)
loss, metrics = jit_step(model, optimizer, batch, step_rng)
```

---

## Example: Complete MNIST Training

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.models.energy import create_mnist_ebm

# Create model using factory function
rngs = nnx.Rngs(params=0, noise=1, sample=2)
model = create_mnist_ebm(
    rngs=rngs,
    hidden_dims=(128, 256, 512),
    activation="silu",
    mcmc_steps=60,
    step_size=0.01,
    noise_scale=0.005,
    sample_buffer_capacity=8192,
    alpha=0.01,
)

# Preprocessing: normalize images to [-1, 1]
def preprocess(images):
    images = jnp.array(images, dtype=jnp.float32) / 255.0
    images = (images - 0.5) / 0.5  # Normalize to [-1, 1]
    return {"image": images}

# Training loop using model.train_step()
num_epochs = 50
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for step, batch in enumerate(train_loader):
        batch = preprocess(batch)
        loss_dict = model.train_step(batch)

        if step % 100 == 0:
            gap = float(loss_dict['fake_energy_mean'] - loss_dict['real_energy_mean'])
            print(f"  Step {step}: Loss={loss_dict['loss']:.4f}, "
                  f"Gap={gap:.4f}")

    # Generate samples
    if (epoch + 1) % 10 == 0:
        model.eval()
        samples = model.generate(n_samples=64, shape=(28, 28, 1))
        model.train()
        save_image_grid(samples, f"epoch_{epoch+1}.png")

print("Training complete!")
```

---

## Further Reading

- [EBM Explained](../concepts/ebm-explained.md) - Theoretical foundations
- [EBM API Reference](../../api/models/ebm.md) - Complete API documentation
- [Training Guide](../training/training-guide.md) - General training workflows
- [Examples](../../examples/energy/simple-ebm.md) - More EBM examples

---

## Summary

**Key Takeaways:**

- EBMs learn by assigning low energy to data, high energy to non-data
- Persistent Contrastive Divergence (PCD) with MCMC sampling is the standard training method
- Sample buffer management is critical for stable training
- Monitor energy gap: fake_energy should be > real_energy
- Start simple, increase complexity gradually
- Use spectral normalization and regularization for stability

**Recommended Workflow:**

1. Start with simple 2D data to verify training works
2. Use MLP energy for tabular, CNN for images
3. Monitor energy gap and buffer health
4. Tune MCMC steps and step size for your data
5. Use DeepEBM for complex distributions
6. Visualize samples frequently during training

For theoretical understanding, see the [EBM Explained guide](../concepts/ebm-explained.md).
