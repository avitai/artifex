# Generative Models

The main module containing all generative model implementations, core infrastructure, modalities, extensions, and training systems in Artifex.

## Overview

<div class="grid cards" markdown>

- :material-cube-outline:{ .lg .middle } **Model Architectures**

    ---

    VAE, GAN, Diffusion, Flow, EBM, and Autoregressive models

- :material-cogs:{ .lg .middle } **Core Infrastructure**

    ---

    Configuration, losses, distributions, sampling, and metrics

- :material-view-grid:{ .lg .middle } **Modalities**

    ---

    Image, text, audio, protein, and multimodal support

- :material-puzzle:{ .lg .middle } **Extensions**

    ---

    Domain-specific extensions for proteins, geometric data, and more

</div>

## Quick Start

### Creating a Model

```python
from artifex.generative_models.factories import create_model
from artifex.generative_models.core.configuration import VAEConfig
from flax import nnx

# Create VAE model
rngs = nnx.Rngs(0)
config = VAEConfig(
    name="my_vae",
    latent_dim=64,
    encoder_hidden_dims=[256, 128],
    decoder_hidden_dims=[128, 256],
)

model = create_model("vae", config=config, rngs=rngs)
```

### Training a Model

```python
from artifex.generative_models.training import VAETrainer

trainer = VAETrainer(
    model=model,
    config=training_config,
    train_dataset=train_data,
)

trainer.train()
```

### Generating Samples

```python
from artifex.inference import InferencePipeline

pipeline = InferencePipeline(model)
samples = pipeline.generate(num_samples=16)
```

## Module Structure

The `generative_models` package is organized into the following submodules:

### Models

Implementation of all generative model architectures.

| Model Type | Description |
|------------|-------------|
| [VAE](../models/beta_vae.md) | Variational Autoencoders |
| [GAN](../models/dcgan.md) | Generative Adversarial Networks |
| [Diffusion](../models/diffusion.md) | Denoising Diffusion Models |
| [Flow](../models/glow.md) | Normalizing Flow Models |
| [EBM](../models/ebm.md) | Energy-Based Models |
| [Autoregressive](../models/pixel_cnn.md) | Autoregressive Models |

[:octicons-arrow-right-24: Models Reference](../models/index.md)

### Core

Foundational abstractions and utilities.

| Component | Description |
|-----------|-------------|
| [Configuration](../core/unified.md) | Unified configuration system |
| [Losses](../core/reconstruction.md) | Loss functions |
| [Distributions](../core/continuous.md) | Probability distributions |
| [Sampling](../core/ancestral.md) | Sampling methods |
| [Metrics](../core/fid.md) | Evaluation metrics |
| [Layers](../core/flash_attention.md) | Neural network layers |

[:octicons-arrow-right-24: Core Reference](../core/index.md)

### Modalities

Data modality implementations.

| Modality | Description |
|----------|-------------|
| [Base](../modalities/base.md) | Base modality classes |
| [Adapters](../modalities/adapters.md) | Model adapters for modalities |
| [Evaluation](../modalities/evaluation.md) | Modality evaluation metrics |
| [Datasets](../modalities/datasets.md) | Dataset utilities |
| [Representations](../modalities/representations.md) | Feature representations |

[:octicons-arrow-right-24: Modalities Reference](../modalities/index.md)

### Training

Training infrastructure and utilities.

| Component | Description |
|-----------|-------------|
| [VAE Trainer](../training/vae_trainer.md) | VAE model trainer |
| [GAN Trainer](../training/gan_trainer.md) | GAN model trainer |
| [Diffusion Trainer](../training/diffusion_trainer.md) | Diffusion model trainer |
| [Checkpoint](../training/checkpoint.md) | Training checkpointing |
| [Data Parallel](../training/data_parallel.md) | Multi-device training |
| [AdamW](../training/adamw.md) | Optimization algorithms |
| [Scheduler](../training/scheduler.md) | Learning rate schedules |

[:octicons-arrow-right-24: Training Reference](../training/index.md)

### Extensions

Domain-specific extensions.

| Extension | Description |
|-----------|-------------|
| [Extensions](../extensions/extensions.md) | Extension system overview |
| [Backbone](../extensions/backbone.md) | Model backbones |
| [Embeddings](../extensions/embeddings.md) | Embedding layers |
| [Features](../extensions/features.md) | Feature processing |
| [Registry](../extensions/registry.md) | Extension registry |

[:octicons-arrow-right-24: Extensions Reference](../extensions/index.md)

### Factories

Model creation and registration.

```python
from artifex.generative_models.factories import (
    create_model,
    register_model,
    list_models,
)

# List available models
available = list_models()
# ['vae', 'beta_vae', 'vq_vae', 'gan', 'wgan', ...]

# Create model by name
model = create_model("vae", config=config, rngs=rngs)

# Register custom model
register_model("my_model", MyModelClass)
```

[:octicons-arrow-right-24: Factory Reference](../factory/index.md)

See also: [VAE Factory](../factory/vae.md) | [GAN Factory](../factory/gan.md) | [Diffusion Factory](../factory/diffusion.md) | [Flow Factory](../factory/flow.md)

## Architecture

```
generative_models/
├── core/                 # Core infrastructure
│   ├── configuration/    # Configuration system
│   ├── losses/           # Loss functions
│   ├── distributions/    # Probability distributions
│   ├── sampling/         # Sampling methods
│   ├── metrics/          # Evaluation metrics
│   └── layers/           # Neural network layers
├── models/               # Model implementations
│   ├── vae/              # VAE variants
│   ├── gan/              # GAN variants
│   ├── diffusion/        # Diffusion models
│   ├── flow/             # Flow models
│   ├── ebm/              # Energy-based models
│   └── autoregressive/   # Autoregressive models
├── modalities/           # Data modality support
│   ├── image/            # Image modality
│   ├── text/             # Text modality
│   ├── audio/            # Audio modality
│   ├── protein/          # Protein modality
│   └── multi_modal/      # Multimodal support
├── training/             # Training infrastructure
│   ├── trainers/         # Model trainers
│   ├── callbacks/        # Training callbacks
│   ├── distributed/      # Distributed training
│   └── optimizers/       # Optimizers and schedulers
├── extensions/           # Domain extensions
│   ├── protein/          # Protein modeling
│   └── geometric/        # Geometric deep learning
└── factories/            # Model creation
    ├── model_factory.py  # Model factory
    └── registry.py       # Model registry
```

## Design Principles

### 1. Protocol-Based Interfaces

All components use Python Protocols for type-safe interfaces:

```python
from artifex.generative_models.core.protocols import GenerativeModel

class MyModel(GenerativeModel):
    def generate(self, num_samples: int, **kwargs) -> jax.Array:
        ...

    def loss_fn(self, batch: jax.Array, **kwargs) -> jax.Array:
        ...
```

### 2. Unified Configuration

All models use the unified configuration system:

```python
from artifex.generative_models.core.configuration import VAEConfig

config = VAEConfig(
    name="my_vae",
    latent_dim=64,
    # Type-safe, validated configuration
)
```

### 3. Modality-Agnostic Design

Models work with any data modality through adapters:

```python
from artifex.generative_models.modalities import get_modality

# Get modality handler
image_modality = get_modality("image", rngs=rngs)

# Adapt model for modality
adapted_model = image_modality.get_adapter("vae").adapt(model)
```

### 4. Hardware-Aware

All components are hardware-aware with automatic device management:

```python
from artifex.generative_models.core import DeviceManager

device_manager = DeviceManager()
device = device_manager.get_device()  # Auto-selects GPU/CPU
```

## Related Documentation

- [Models Reference](../models/index.md) - Model implementations
- [Core Reference](../core/index.md) - Core infrastructure
- [Modalities Reference](../modalities/index.md) - Data modalities
- [Training Reference](../training/index.md) - Training systems
- [Factory Reference](../factory/index.md) - Model creation factories
