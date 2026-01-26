# Simple Energy-Based Model (EBM) Example

**Level:** Intermediate | **Runtime:** ~2 minutes (CPU) / ~30 seconds (GPU) | **Format:** Python + Jupyter

## Overview

This comprehensive example demonstrates Energy-Based Models (EBMs) with MCMC sampling. It covers the fundamentals of energy functions, Langevin dynamics sampling, and contrastive divergence training, including advanced techniques like persistent contrastive divergence and deep EBM architectures.

## What You'll Learn

- Energy function computation and interpretation
- Langevin dynamics for MCMC sampling
- Contrastive divergence (CD) training
- Persistent contrastive divergence with sample buffers
- Deep EBM architectures with residual connections
- Score function estimation

## Files

- **Python Script**: [`examples/generative_models/energy/simple_ebm_example.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/energy/simple_ebm_example.py)
- **Jupyter Notebook**: [`examples/generative_models/energy/simple_ebm_example.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/energy/simple_ebm_example.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/energy/simple_ebm_example.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/energy/simple_ebm_example.ipynb
```

## Key Concepts

### Energy-Based Models

EBMs define a probability distribution through an energy function:

$$p(x) = \frac{\exp(-E(x))}{Z}$$

Where:

- $E(x)$ is the energy function (lower energy = higher probability)
- $Z$ is the partition function (normalization constant)

### Langevin Dynamics

Sampling from the model using Langevin dynamics:

$$x_{t+1} = x_t - \frac{\epsilon}{2} \nabla_x E(x_t) + \sqrt{\epsilon} \cdot \text{noise}$$

Where $\epsilon$ is the step size and noise is Gaussian.

### Contrastive Divergence Loss

Training objective that contrasts real data with model samples:

$$\mathcal{L}_{CD} = E_{x \sim p_{data}}[E(x)] - E_{x \sim p_{model}}[E(x)]$$

The goal is to:

- **Lower energy** for real data samples
- **Raise energy** for generated samples

### Persistent Contrastive Divergence

An improved version of CD that maintains a buffer of persistent samples across training iterations, leading to better gradient estimates.

## Code Structure

The example demonstrates 9 major sections:

1. **Simple EBM Creation**: Basic energy-based model for MNIST
2. **Energy Computation**: Computing and interpreting energy values
3. **MCMC Sampling**: Generating samples using Langevin dynamics
4. **Configuration System**: Declarative model creation
5. **Contrastive Divergence**: Training with CD loss
6. **Persistent CD**: Advanced training with sample buffers
7. **Deep EBM**: Complex architectures with residual connections
8. **Visualization**: Analyzing samples and energy landscapes
9. **Summary**: Key takeaways and experiments

## Example Code

### Creating a Simple EBM

```python
from flax import nnx
import jax.numpy as jnp
from artifex.generative_models.models.energy import create_mnist_ebm

# Initialize RNG
rngs = nnx.Rngs(0)

# Create EBM using factory function (recommended for MNIST)
ebm = create_mnist_ebm(rngs=rngs)

# Compute energy for data
data = jnp.ones((32, 28, 28, 1))
energy = ebm.energy(data)
print(f"Energy shape: {energy.shape}")  # (32,)
```

### Langevin Dynamics Sampling

```python
import jax
from artifex.generative_models.models.energy import langevin_dynamics

# Initialize samples
key = jax.random.key(0)
init_samples = jax.random.normal(key, (16, 28, 28, 1))

# Generate samples using MCMC (Langevin dynamics)
samples = langevin_dynamics(
    energy_fn=ebm.energy,
    initial_samples=init_samples,
    n_steps=100,
    step_size=0.01,
    noise_scale=0.005,
)
```

### Contrastive Divergence Training

```python
# EBM uses built-in loss_fn for contrastive divergence training
# The loss is computed during the forward pass with MCMC samples

# Forward pass on real data
real_data = data_batch  # Your training data
outputs = ebm(real_data)

# Compute CD loss using model's loss_fn (handles MCMC internally)
loss_dict = ebm.loss_fn(x=real_data, outputs=outputs)

print(f"CD Loss: {loss_dict['loss']:.4f}")
print(f"Real Energy: {loss_dict['real_energy_mean']:.4f}")
print(f"Fake Energy: {loss_dict['fake_energy_mean']:.4f}")
print(f"Real Energy: {loss_dict['real_energy_mean']:.4f}")
print(f"Fake Energy: {loss_dict['fake_energy_mean']:.4f}")
```

### Persistent Contrastive Divergence

```python
# Persistent CD is handled through EBM configuration
# Configure with sample buffer when creating the model

from artifex.generative_models.core.configuration.energy_config import (
    EBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import EBM

# Configure sample buffer for persistent CD
buffer_config = SampleBufferConfig(
    name="buffer",
    capacity=10000,
    reinit_prob=0.05,  # Probability to reinitialize from noise
)

# Full EBM config with sample buffer
config = EBMConfig(
    name="ebm_pcd",
    input_dim=784,
    energy_network=EnergyNetworkConfig(
        name="energy_net",
        hidden_dims=(256, 128),
        activation="swish",
    ),
    mcmc=MCMCConfig(name="mcmc", n_steps=60, step_size=0.01),
    sample_buffer=buffer_config,  # Enable persistent CD
)

# Create model - buffer is managed internally
ebm_pcd = EBM(config, rngs=nnx.Rngs(0))

# Training uses persistent samples automatically
# The model's loss_fn handles buffer management internally
        init_samples=init_samples,
        step_size=0.01,
        n_steps=20  # Fewer steps with persistent buffer
    )

    # Update buffer
    buffer.add(samples)

    # Compute loss and update model
    loss = contrastive_divergence_loss(ebm, real_data, samples)
```

### Deep EBM Architecture

```python
from artifex.generative_models.core.configuration.energy_config import (
    DeepEBMConfig,
    EnergyNetworkConfig,
    MCMCConfig,
    SampleBufferConfig,
)
from artifex.generative_models.models.energy import DeepEBM

# Create energy network configuration
energy_network_config = EnergyNetworkConfig(
    name="deep_energy_network",
    hidden_dims=(32, 64, 128),  # Channel progression (tuple)
    activation="silu",
    network_type="cnn",
    use_spectral_norm=True,
    use_residual=True,
)

# Create MCMC sampling configuration
mcmc_config = MCMCConfig(
    name="langevin_mcmc",
    n_steps=100,
    step_size=0.005,
    noise_scale=0.001,
)

# Create sample buffer configuration
sample_buffer_config = SampleBufferConfig(
    name="replay_buffer",
    capacity=8192,
    reinit_prob=0.05,
)

# Create deep EBM configuration with nested configs
deep_config = DeepEBMConfig(
    name="deep_ebm",
    input_shape=(32, 32, 3),
    energy_network=energy_network_config,
    mcmc=mcmc_config,
    sample_buffer=sample_buffer_config,
    alpha=0.001,
)

# Create model
deep_ebm = DeepEBM(config=deep_config, rngs=rngs)
```

## Features Demonstrated

### Energy Function Design

- MLP-based energy functions
- CNN-based energy functions
- Deep architectures with residual connections
- Spectral normalization for stability

### MCMC Sampling

- Langevin dynamics implementation
- Step size tuning
- Temperature control
- Burn-in periods

### Training Techniques

- Standard contrastive divergence
- Persistent contrastive divergence
- Sample buffer management
- Gradient estimation

### Advanced Architectures

- Deep convolutional EBMs
- Residual connections
- Batch normalization
- Spectral normalization

## Experiments to Try

1. **Modify energy architecture**: Try different hidden dimensions or activation functions
2. **Tune MCMC parameters**: Experiment with step sizes, number of steps, temperature
3. **Compare CD vs Persistent CD**: Observe training stability and sample quality
4. **Add noise annealing**: Gradually reduce noise during sampling
5. **Conditional EBMs**: Extend to conditional generation with labels
6. **Hybrid models**: Combine EBMs with other generative models

## Next Steps

After understanding this example:

1. **Training Loop**: Implement full training on real datasets (MNIST, CIFAR-10)
2. **Score Matching**: Explore score matching as an alternative to CD
3. **Conditional Generation**: Add class or attribute conditioning
4. **Energy Landscape Analysis**: Visualize and analyze learned energy functions
5. **Compositional Generation**: Combine multiple EBMs for complex generation

## Troubleshooting

### MCMC Not Converging

- Increase number of sampling steps
- Reduce step size
- Add noise annealing schedule
- Check energy function gradients

### Training Instability

- Use spectral normalization
- Reduce learning rate
- Increase batch size
- Use persistent CD with larger buffer

### Slow Sampling

- Use GPU acceleration
- Reduce number of MCMC steps
- Use persistent buffers to start from better initializations
- Consider faster sampling methods (e.g., HMC)

### Memory Issues

- Reduce buffer size for persistent CD
- Use smaller batch sizes
- Reduce model size (fewer hidden dims)

## Additional Resources

- **Paper**: [A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
- **Paper**: [Training Products of Experts by Minimizing Contrastive Divergence](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)
- **Artifex EBM Guide**: (Coming soon)
- **API Reference**: (Coming soon)

## Related Examples

- [Simple Diffusion](../diffusion/simple-diffusion.md) - Diffusion models basics
- [BlackJAX Sampling](../../core/mcmc.md) - Advanced MCMC methods
- [DiT Demo](../diffusion/dit-demo.md) - Transformer-based generation
