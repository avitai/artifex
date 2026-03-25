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
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import DecoderConfig, EncoderConfig, VAEConfig
from flax import nnx

# Create VAE model
encoder = EncoderConfig(
    name="vae_encoder",
    input_shape=(28, 28, 1),
    latent_dim=64,
    hidden_dims=(256, 128),
)
decoder = DecoderConfig(
    name="vae_decoder",
    output_shape=(28, 28, 1),
    latent_dim=64,
    hidden_dims=(128, 256),
)
config = VAEConfig(
    name="my_vae",
    encoder=encoder,
    decoder=decoder,
)

rngs = nnx.Rngs(0)
model = create_model(config, rngs=rngs)
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

There is no top-level `artifex.inference` namespace or one shared
inference pipeline in the current runtime. Generation remains family-owned.

```python
# `model` above is a VAE built from `VAEConfig`.
samples = model.sample(num_samples=16)
```

See [Inference Overview](../user-guide/inference/overview.md) for the retained
loading and generation workflow, and [Inference Reference](../inference/index.md)
for the one shared production-optimization surface.

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

Registry-backed modalities plus family-scoped owner pages.

Use [Modalities Overview](../modalities/index.md) for the retained registry-backed surface and the owner pages below only for family-scoped helper details.

| Reference | Description |
|-----------|-------------|
| [Registry Owner](../modalities/registry.md) | Shared registry-backed surface for `image`, `molecular`, and `protein` |
| [Timeseries Base](../modalities/base.md) | Timeseries helper owner page |
| [Timeseries Datasets](../modalities/datasets.md) | Synthetic timeseries data factories |
| [Protein Modality](../modalities/modality.md) | Protein-specific adapter and extension lookup |
| [Protein Losses](../modalities/losses.md) | Protein structure loss builders |

[:octicons-arrow-right-24: Modalities Overview](../modalities/index.md)

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

Domain-specific extensions with one curated overview plus live owner pages.

Use [Extensions Overview](../extensions/index.md) for the curated scope and the owner pages below for live module details.

| Reference | Description |
|-----------|-------------|
| [Base Extensions](../extensions/extensions.md) | Shared extension hierarchy and base contracts |
| [Registry Owner](../extensions/registry.md) | Registry enum, discovery helpers, and factory surface |
| [Protein Constraints](../extensions/constraints.md) | Protein constraint owners and measurement helpers |
| [NLP Embeddings](../extensions/embeddings.md) | RoPE, sinusoidal, and text embedding owners |
| [Audio Analysis](../extensions/temporal.md) | Temporal audio-analysis owner page |

[:octicons-arrow-right-24: Extensions Overview](../extensions/index.md)

### Factory

Model creation and registration.

```python
from artifex.generative_models.factory import create_model, create_model_with_extensions

# `config` is a family-specific typed config such as VAEConfig, DDPMConfig,
# WGANConfig, or PointCloudConfig.

# Create a model from a dataclass config
model = create_model(config, rngs=rngs)

# Create a model with extensions
model, extensions = create_model_with_extensions(
    config,
    extensions_config=extension_configs,
    rngs=rngs,
)
```

[:octicons-arrow-right-24: Factory Reference](../factory/index.md)

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
├── factory/              # Model creation
│   ├── core.py           # ModelFactory implementation
│   ├── registry.py       # Builder registry
│   └── builders/         # Model-family builders
└── zoo/                  # Retired preset compatibility boundary
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
