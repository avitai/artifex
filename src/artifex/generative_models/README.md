# artifex generative models

A modular generative modeling library built on JAX and Flax NNX, focusing on research flexibility and clean abstractions.

## Key Features

- **Multiple Model Types**: Implementations of VAEs, GANs, Diffusion Models, Flow Models, Autoregressive Models, Energy-Based Models, and Geometric Models
- **Protocol-Based Design**: Clear type-safe interfaces using Python's Protocol system
- **Modular Architecture**: Composable building blocks for creating custom generative models
- **JAX Integration**: Leverage JAX's speed, automatic differentiation, and hardware acceleration
- **Flax NNX**: Modern neural network library with intuitive object-oriented API
- **Extension Mechanism**: Flexible system for adding domain-specific functionality
- **Multi-Modal Framework**: Support for different data modalities with specialized processors
- **Centralized Factory System**: Unified model creation API
- **Advanced MCMC Sampling**: Integration with BlackJAX for MCMC sampling algorithms

## Package Structure

```
generative_models/
├── core/              # Core abstractions and utilities
│   ├── protocols/     # Type-safe interfaces
│   ├── configuration/ # Unified configuration system
│   ├── losses/        # Modular loss functions
│   ├── sampling/      # Sampling strategies
│   └── evaluation/    # Metrics and benchmarks
├── factory/           # Centralized factory system
│   ├── core.py        # ModelFactory implementation
│   ├── registry.py    # ModelTypeRegistry
│   └── builders/      # Model-specific builders
├── models/            # Model implementations
│   ├── vae/           # VAE variants
│   ├── gan/           # GAN variants
│   ├── diffusion/     # Diffusion models
│   ├── flow/          # Normalizing flows
│   ├── energy/        # Energy-based models
│   ├── autoregressive/# Autoregressive models
│   └── geometric/     # Geometric models
├── modalities/        # Multi-modal support
├── extensions/        # Extension mechanism
├── training/          # Training loops
├── inference/         # Inference utilities
├── scaling/           # Scaling and sharding strategies
└── zoo/               # Pre-configured models
```

See the [factory README](factory/README.md) for details on the centralized model creation system.

## Available Models

- **VAE Models**: VAE, β-VAE, VQ-VAE, Conditional VAE
- **GAN Models**: DCGAN, WGAN, StyleGAN, CycleGAN, PatchGAN
- **Diffusion Models**: DDPM, DDIM, Score-based, DiT, Latent Diffusion
- **Flow Models**: RealNVP, Glow, MAF, IAF, Neural Spline Flows
- **Energy-Based Models**: EBMs with MCMC sampling and Langevin dynamics
- **Autoregressive Models**: PixelCNN, WaveNet, Transformer-based
- **Geometric Models**: Point clouds, meshes, voxel grids, protein structures

## Quick Start

### Basic Model Creation

```python
import jax
from flax import nnx
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import ModelConfiguration

# Create configuration
config = ModelConfiguration(
    name="my_vae",
    model_class="artifex.generative_models.models.vae.VAE",
    input_dim=(28, 28, 1),
    hidden_dims=[256, 128],
    output_dim=64,  # latent_dim
    parameters={
        "beta": 1.0,
        "kl_weight": 1.0,
    }
)

# Create model
rngs = nnx.Rngs(0)
model = create_model(config, rngs=rngs)

# Forward pass
batch = jax.random.normal(jax.random.key(0), (16, 28, 28, 1))
outputs = model(batch, rngs=rngs)

# Generate samples
samples = model.generate(rngs=rngs, n_samples=16)
```

### Using the Model Zoo

```python
from artifex.generative_models.zoo import zoo

# List available models
print(zoo.list_configs())

# Create pre-configured model
model = zoo.create_model("vae_mnist", rngs=rngs)
```

## Extension Mechanism

Add domain-specific functionality without modifying core code:

```python
from artifex.generative_models.extensions.protein import create_protein_extensions

# Create protein-specific extensions
extensions = create_protein_extensions(
    {
        "use_backbone_constraints": True,
        "bond_length_weight": 1.0,
        "bond_angle_weight": 0.5,
    },
    rngs=rngs
)

# Create model with extensions
model = create_model(config, extensions=extensions, rngs=rngs)
```

See the [extensions README](extensions/README.md) for more details.

## Multi-Modal Support

The modality framework enables domain-specific model adaptations:

```python
# Create a model adapted for protein data
protein_model = create_model(
    config,
    modality="protein",
    rngs=rngs
)
```

Available modalities: image, text, audio, protein, tabular, timeseries, multi_modal

See the [modalities README](modalities/README.md) for more details.

## Device Management

The library includes device management for GPU/CPU optimization:

```python
from artifex.generative_models.core.device_manager import get_device_manager

manager = get_device_manager()
print(f"GPU Available: {manager.has_gpu}")
print(f"Device Count: {manager.device_count}")
```

See the [core README](core/README.md) for device management details.

## Documentation

For detailed information, see:

- [Factory System](factory/README.md) - Centralized model creation
- [Core Module](core/README.md) - Device management and core utilities
- [Extensions](extensions/README.md) - Domain-specific functionality
- [Modalities](modalities/README.md) - Multi-modal support
- [Energy Models](models/energy/README.md) - Energy-based models

For general documentation, see the [main documentation](../../../docs/).
