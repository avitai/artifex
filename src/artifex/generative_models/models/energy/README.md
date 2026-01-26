# Energy-Based Models (EBMs)

This module provides a comprehensive implementation of Energy-Based Models using Flax NNX. Energy-Based Models learn data distributions by modeling an energy function E(x) where the probability distribution is given by p(x) ∝ exp(-E(x)).

## Overview

Energy-Based Models are a class of generative models that associate a scalar energy value to each input configuration. Lower energy values correspond to higher probability under the model. Training involves learning the energy function to assign low energies to real data and high energies to generated samples.

## Key Components

### Base Classes

- **`EnergyFunction`**: Abstract base class for energy functions
- **`EnergyBasedModel`**: Base class for EBM implementations, extending GenerativeModel
- **`MLPEnergyFunction`**: MLP-based energy function for tabular data
- **`CNNEnergyFunction`**: CNN-based energy function for image data

### Main Implementations

- **`EBM`**: Standard Energy-Based Model with configurable energy functions
- **`DeepEBM`**: Deep EBM with residual connections and normalization for complex datasets
- **`DeepCNNEnergyFunction`**: Advanced CNN energy function with residual blocks

### MCMC Sampling

- **`langevin_dynamics`**: Basic Langevin dynamics MCMC sampling
- **`improved_langevin_dynamics`**: Enhanced version with adaptive step size
- **`persistent_contrastive_divergence`**: Training algorithm with sample buffer
- **`SampleBuffer`**: Buffer for storing and reusing MCMC samples

## Usage Examples

### Basic Usage

```python
from flax import nnx
from artifex.generative_models.models.energy import EBM, create_mnist_ebm

# Create an EBM for MNIST-like data
rngs = nnx.Rngs(0)
model = create_mnist_ebm(rngs=rngs)

# Forward pass to get energy values
import jax.numpy as jnp
images = jnp.ones((2, 28, 28, 1))
output = model(images)
print("Energy values:", output["energy"])
print("Score function:", output["score"])

# Generate samples using MCMC
samples = model.generate(
    n_samples=4,
    shape=(28, 28, 1),
    rngs=rngs,
    n_steps=100
)
```

### Using Configuration System

```python
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import ModelConfiguration

# Create configuration
config = ModelConfiguration(
    name="my_ebm",
    model_class="artifex.generative_models.models.energy.EBM",
    input_dim=(28, 28, 1),
    hidden_dims=[32, 64, 128],
    output_dim=1,  # Energy models output scalar energy values
    activation="silu",
    parameters={
        "energy_type": "cnn",
        "mcmc_steps": 60,
        "step_size": 0.01,
    }
)

# Create model from configuration
model = create_model(config, rngs=rngs)
```

### Custom Energy Function

```python
from flax import nnx
from artifex.generative_models.models.energy import EnergyBasedModel, EnergyFunction

class CustomEnergyFunction(EnergyFunction):
    def __init__(self, *, rngs, **kwargs):
        super().__init__(**kwargs)
        # Define your custom architecture
        self.layers = [
            nnx.Linear(784, 256, rngs=rngs),
            nnx.Linear(256, 128, rngs=rngs),
            nnx.Linear(128, 1, rngs=rngs),
        ]

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)  # Flatten
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on last layer
                x = nnx.relu(x)
        return x.squeeze(-1)

# Create EBM with custom energy function
energy_fn = CustomEnergyFunction(rngs=rngs)
model = EnergyBasedModel(energy_fn=energy_fn)
```

## Training

EBMs are typically trained using contrastive divergence, where the model learns to:

1. Assign low energy to real data samples
2. Assign high energy to generated (negative) samples

```python
# Training step example
def training_step(model, real_batch, rngs):
    # Generate negative samples using MCMC
    negative_samples = model.generate(
        n_samples=real_batch.shape[0],
        shape=real_batch.shape[1:],
        rngs=rngs,
        n_steps=60
    )

    # Compute contrastive divergence loss
    loss_dict = model.contrastive_divergence_loss(
        real_data=real_batch,
        fake_data=negative_samples,
        alpha=0.01  # Regularization strength
    )

    return loss_dict["loss"]
```

For persistent contrastive divergence with sample buffer:

```python
from artifex.generative_models.models.energy.mcmc import persistent_contrastive_divergence, SampleBuffer

# Initialize sample buffer
buffer = SampleBuffer(capacity=8192, reinit_prob=0.05)

def persistent_training_step(model, real_batch, buffer, rngs):
    real_samples, fake_samples = persistent_contrastive_divergence(
        energy_fn=model.energy,
        real_samples=real_batch,
        sample_buffer=buffer,
        rng_key=rngs.sample(),
        n_mcmc_steps=60
    )

    return model.contrastive_divergence_loss(real_samples, fake_samples)
```

## MCMC Sampling Details

### Langevin Dynamics

The basic sampling algorithm follows the Langevin dynamics equation:

```
x_{t+1} = x_t - α * ∇_x E(x_t) + √(2α) * ε_t
```

Where:

- α is the step size
- ∇_x E(x_t) is the gradient of energy w.r.t. input
- ε_t is Gaussian noise

### Sample Buffer Strategy

For efficient training, the implementation uses a sample buffer strategy where:

- 95% of initial samples come from previous MCMC chains (stored in buffer)
- 5% are initialized from scratch
- This significantly reduces the number of MCMC steps needed

### Hyperparameter Guidelines

- **Step size**: Start with 0.01, reduce for more complex models
- **Noise scale**: Usually 0.005, adjust based on data scale
- **MCMC steps**: 60-100 for training, 200+ for high-quality sampling
- **Buffer capacity**: 8192 works well for most datasets
- **Regularization α**: 0.01 for simple models, 0.001 for deep models

## Available Configurations

The module provides several pre-configured model types:

- **`create_simple_ebm`**: MLP-based EBM for tabular data
- **`create_mnist_ebm`**: CNN EBM optimized for MNIST
- **`create_cifar_ebm`**: Deep EBM optimized for CIFAR
- **`EBMConfig`**: General EBM configuration
- **`DeepEBMConfig`**: Configuration for deep models
- **`CNNEBMConfig`**: CNN-specific configuration
- **`SimpleMLPEBMConfig`**: MLP-specific configuration

## Creating EBM Models

EBM models can be created using the factory system or helper functions:

```python
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import ModelConfiguration

# Using the factory
config = ModelConfiguration(
    name="my_ebm",
    model_class="artifex.generative_models.models.energy.EBM",
    input_dim=(28, 28, 1),
    hidden_dims=[32, 64, 128],
    output_dim=1,
)
model = create_model(config, rngs=rngs)

# Using helper functions
from artifex.generative_models.models.energy import (
    create_simple_ebm,
    create_mnist_ebm,
    create_cifar_ebm,
)
```

Available helper functions:

- `create_simple_ebm()`: MLP-based EBM for tabular data
- `create_mnist_ebm()`: CNN EBM optimized for MNIST
- `create_cifar_ebm()`: Deep EBM optimized for CIFAR

## Best Practices

1. **Data Preprocessing**: Normalize data to [-1, 1] range for best results
2. **Gradient Clipping**: Use small gradient clipping (0.03) for stability
3. **Regularization**: Add small L2 regularization on energy values
4. **Sample Buffer**: Use persistent sample buffer for efficient training
5. **Activation Functions**: Use smooth activations (SiLU/Swish) for better gradients
6. **Monitoring**: Track energy values and sample quality during training

## References

- [A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
- [Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One](https://arxiv.org/abs/1912.03263)
