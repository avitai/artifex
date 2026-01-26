# Architecture Overview

Artifex follows a modular, protocol-based architecture designed for research experimentation and production deployment.

## High-Level Structure

```
artifex/
├── benchmarks/             # Evaluation framework and metrics
├── cli/                    # Command-line interface
├── configs/                # Configuration schemas and defaults
├── data/                   # Data loading and preprocessing
├── generative_models/
│   ├── core/               # Core abstractions
│   │   ├── protocols/      # Type-safe interfaces
│   │   ├── configuration/  # Frozen dataclass configs
│   │   ├── losses/         # Modular loss functions
│   │   ├── sampling/       # Sampling strategies
│   │   ├── distributions/  # Probability distributions
│   │   └── evaluation/     # Metrics and benchmarks
│   ├── models/             # Model implementations
│   │   ├── vae/            # VAE variants
│   │   ├── gan/            # GAN variants
│   │   ├── diffusion/      # Diffusion models
│   │   ├── flow/           # Normalizing flows
│   │   ├── energy/         # Energy-based models
│   │   ├── autoregressive/ # Autoregressive models
│   │   ├── geometric/      # Geometric models
│   │   ├── audio/          # Audio models (WaveNet)
│   │   └── backbones/      # Shared architectures
│   ├── modalities/         # Multi-modal support
│   ├── training/           # Training infrastructure
│   ├── inference/          # Generation and serving
│   ├── factory/            # Model creation
│   ├── extensions/         # Domain-specific extensions
│   ├── fine_tuning/        # LoRA, adapters, RL
│   ├── scaling/            # Distributed training
│   └── zoo/                # Pre-configured models
├── utils/                  # Shared utilities
└── visualization/          # Plotting and visualization
```

## Core Design Patterns

### Protocol-Based Interfaces

All major components implement Python Protocols for type-safe interfaces:

```python
from typing import Protocol

class GenerativeModel(Protocol):
    """Protocol for all generative models."""

    def generate(self, n_samples: int, *, rngs: nnx.Rngs) -> jax.Array:
        """Generate samples from the model."""
        ...

    def loss(self, batch: jax.Array, *, rngs: nnx.Rngs) -> LossOutput:
        """Compute loss for a batch."""
        ...
```

**Benefits:**

- Clear contracts between components
- Type checking at development time
- Easy to swap implementations
- Facilitates testing with mocks

### Frozen Dataclass Configuration

All models use immutable configuration objects:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class VAEConfig:
    name: str
    encoder: EncoderConfig
    decoder: DecoderConfig
    kl_weight: float = 1.0
```

**Benefits:**

- Immutable during training
- Automatic validation
- Serializable for reproducibility
- IDE support with autocomplete

### Factory Pattern

Models are created through factories for consistent initialization:

```python
from artifex.generative_models.factory import create_model

model = create_model(config, rngs=rngs)
```

The factory:

1. Validates configuration
2. Selects appropriate model class
3. Initializes with proper RNG management
4. Returns fully configured model

## Component Details

### Core (`generative_models/core/`)

- **protocols/**: Interface definitions for models, trainers, data loaders
- **configuration/**: Frozen dataclass configs for all model types
- **losses/**: Composable loss functions (MSE, adversarial, perceptual)
- **sampling/**: MCMC, ancestral, ODE/SDE samplers
- **distributions/**: Probability distributions for generative modeling
- **evaluation/**: Metrics and benchmarks

### Models (`generative_models/models/`)

Each model type has its own subdirectory:

- **vae/**: VAE, Beta-VAE, VQ-VAE, Conditional VAE
- **gan/**: DCGAN, WGAN, StyleGAN, PatchGAN
- **diffusion/**: DDPM, DDIM, DiT, Score-based
- **flow/**: RealNVP, Glow, MAF, IAF, NSF
- **energy/**: Energy-based models with MCMC
- **autoregressive/**: PixelCNN, WaveNet, Transformers
- **geometric/**: Point clouds, meshes, voxels
- **audio/**: WaveNet and audio generation
- **backbones/**: UNet, ResNet, Transformer blocks

### Modalities (`generative_models/modalities/`)

Each modality provides:

- **datasets.py**: Data loading and preprocessing
- **evaluation.py**: Modality-specific metrics
- **representations.py**: Feature representations

Supported modalities:

- Image, Text, Audio
- Protein, Tabular, Timeseries
- Geometric (point clouds, meshes)
- Multimodal

### Training (`generative_models/training/`)

- Training loops and optimization
- Logging and checkpointing
- Learning rate scheduling
- Multi-GPU coordination

### Factory (`generative_models/factory/`)

- Model creation from configurations
- Automatic class selection
- Validation and error handling

## Data Flow

### Training Flow

```
Configuration → Factory → Model
                           ↓
Data Loader → Batch → Model.loss() → Optimizer → Updated Parameters
                           ↓
                      Metrics Logger
```

### Generation Flow

```
Configuration → Factory → Model
                           ↓
Latent Space → Model.generate() → Samples → Post-processing
```

## Extension Points

### Adding New Models

1. Define configuration in `core/configuration/`
2. Implement model in appropriate `models/` subdirectory
3. Register in factory
4. Add tests mirroring source structure

### Adding New Modalities

1. Create directory in `modalities/`
2. Implement datasets, evaluation, representations
3. Register in modality factory
4. Add comprehensive tests

### Custom Losses

```python
from artifex.generative_models.core.losses import CompositeLoss, WeightedLoss

custom_loss = CompositeLoss([
    WeightedLoss(mse_loss, weight=1.0, name="reconstruction"),
    WeightedLoss(custom_fn, weight=0.1, name="custom"),
])
```

## Hardware Management

### Device Manager

```python
from artifex.generative_models.core import DeviceManager

device_manager = DeviceManager()
device = device_manager.get_device()  # Auto-selects GPU/CPU
```

Handles:

- GPU/CPU detection and fallback
- Memory management
- Multi-device coordination

## See Also

- [Design Philosophy](../development/philosophy.md) - Guiding principles
- [Core Concepts](../getting-started/core-concepts.md) - Getting started guide
- [Testing Guide](../development/testing.md) - Testing practices
