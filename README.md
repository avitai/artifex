# Artifex

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-green)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-NNX-orange)](https://github.com/google/flax)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## A research-focused modular generative modeling library built on JAX/Flax NNX

*From Latin "artifex" — craftsman, artist, maker*

[Documentation](https://docs.avitai.bio/artifex) • [Getting Started](docs/getting-started/installation.md) • [Examples](docs/examples) • [Contributing](CONTRIBUTING.md)

</div>

---

> **⚠️ Major Refactoring in Progress**
>
> Artifex is currently undergoing a significant architectural refactoring. Please be aware of the following implications:
>
> | Area | Status | Impact |
> |------|--------|--------|
> | **API** | 🔄 Unstable | Breaking changes are expected. Public interfaces may change without deprecation warnings. Pin to specific commits if stability is required. |
> | **Tests** | 🔄 In Flux | Test suite is being restructured. Some tests may fail or be skipped. Coverage metrics are temporarily unreliable. |
> | **Documentation** | 🔄 Outdated | Docs may not reflect current implementation. Code examples might not work. Refer to source code and tests for accurate usage. |
>
> We recommend waiting for a stable release before using Artifex in production. For research and experimentation, proceed with the understanding that APIs will evolve.

---

## Overview

Artifex is a modular library for generative modeling research, providing implementations of various state-of-the-art generative models with a focus on modularity, type safety, and scientific reproducibility. Built on JAX and Flax NNX, it emphasizes clean abstractions and extensible design for research experimentation.

### Why Artifex?

- **Research First**: Designed for experimentation with clean, modular architecture
- **Modern Stack**: Built on JAX/Flax NNX with full JIT compilation and automatic differentiation
- **Type Safe**: Protocol-based design with comprehensive type annotations
- **Multi-Modal**: Unified interface across images, text, audio, proteins, and more
- **Extensible**: Easy to add new models, losses, and domain-specific constraints
- **Well Tested**: Extensive test suite covering core functionality

## Design Philosophy

### Research-Focused

Artifex prioritizes:

- **Modularity**: Easy to swap components and experiment
- **Clarity**: Clean, readable implementations over clever optimizations
- **Extensibility**: Simple to add new models and functionality
- **Reproducibility**: Deterministic with clear configuration management

### Technical Principles

- **Type Safety**: Protocol-based design with full type annotations
- **JAX Native**: Leverages JAX's functional programming paradigm
- **Flax NNX**: Modern object-oriented API for neural networks
- **Configuration Management**: Frozen dataclass configs with validation
- **Testing**: Test-driven development with comprehensive coverage

See [Design Philosophy](docs/development/philosophy.md) for detailed discussion.

## Features

### Generative Models

- **VAE Family**: VAE, β-VAE, VQ-VAE, Conditional VAE
- **GAN Family**: DCGAN, WGAN, StyleGAN, CycleGAN, PatchGAN
- **Diffusion Models**: DDPM, DDIM, Score-based models, DiT, Latent Diffusion
- **Normalizing Flows**: RealNVP, Glow, MAF, IAF, Neural Spline Flows
- **Energy-Based Models**: Langevin dynamics, MCMC sampling with BlackJAX
- **Autoregressive Models**: PixelCNN, WaveNet, Transformer-based
- **Geometric Models**: Point clouds, meshes, protein structures, SE(3) molecular flows

### Modality Support

- **Image**: Multi-scale architectures, various loss functions, quality metrics
- **Text**: Tokenization, language modeling, text generation
- **Audio**: Spectral processing, waveform generation, WaveNet
- **Protein**: Structure generation with physical constraints
- **Tabular**: Mixed data types, privacy-preserving generation
- **Timeseries**: Sequential patterns, temporal dynamics
- **Multi-Modal**: Cross-modal generation and alignment

### Core Components

- **Unified Configuration**: Frozen dataclass configs with nested validation
- **Protocol-Based Design**: Clear interfaces for models, trainers, and data
- **Modular Losses**: Composable loss functions (reconstruction, adversarial, perceptual)
- **Flexible Sampling**: Multiple sampling strategies (ancestral, MCMC, ODE/SDE)
- **Extension System**: Domain-specific constraints and functionality
- **Evaluation Framework**: Standardized metrics and benchmarks

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/avitai/artifex.git
cd artifex

# Run setup script (creates venv, installs dependencies, detects GPU)
./setup.sh

# Activate the environment (must use 'source')
source ./activate.sh
```

The setup script automatically:

- Detects GPU/CUDA availability
- Creates a virtual environment with uv
- Installs appropriate dependencies (CPU or GPU)
- Configures environment variables
- Creates the `activate.sh` script for future use

For detailed setup options and GPU configuration, see the [Installation Guide](docs/getting-started/installation.md).

### Your First Model

```python
import jax
from flax import nnx
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
)

# Create nested configuration using frozen dataclasses
encoder_config = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=64,
    hidden_dims=(256, 128),
    activation="relu",
)

decoder_config = DecoderConfig(
    name="decoder",
    output_shape=(28, 28, 1),
    latent_dim=64,
    hidden_dims=(128, 256),
    activation="relu",
)

config = VAEConfig(
    name="mnist_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    encoder_type="dense",  # Options: dense, cnn, resnet
    kl_weight=1.0,
)

# Initialize model directly
rngs = nnx.Rngs(0)
model = VAE(config, rngs=rngs)

# Forward pass
batch = jax.random.normal(jax.random.key(0), (16, 28, 28, 1))
outputs = model(batch)

# Generate samples
samples = model.sample(n_samples=16)
```

See the [Quickstart Guide](docs/getting-started/quickstart.md) for more examples.

## Documentation

### Getting Started

- [Installation Guide](docs/getting-started/installation.md) - Setup instructions and requirements
- [Quickstart Guide](docs/getting-started/quickstart.md) - Train your first model
- [Core Concepts](docs/getting-started/core-concepts.md) - Architecture and design principles

### User Guides

- [Models Guide](docs/user-guide/models) - Detailed guides for each model type
- [Data Pipeline](docs/user-guide/data) - Loading and preprocessing data
- [Training Guide](docs/user-guide/training) - Training workflows and best practices
- [Evaluation Guide](docs/benchmarks) - Metrics and benchmarking

### API Reference

- [Core API](docs/api/core) - Base classes and protocols
- [Models API](docs/api/models) - Model implementations
- [Training API](docs/api/training) - Trainers and optimization
- [Data API](docs/api/data) - Datasets and loaders

### Advanced Topics

- [Scaling & Distribution](docs/scaling) - Multi-GPU/TPU training strategies
- [Extensions System](docs/extensions) - Domain-specific functionality
- [Fine-Tuning](docs/fine_tuning) - LoRA, prefix tuning, and adaptation

## Architecture

Artifex follows a modular, protocol-based architecture:

```
artifex/
├── benchmarks/             # Evaluation framework and metrics
├── cli/                    # Command-line interface
├── configs/                # Configuration schemas and defaults
├── data/                   # Data loading and preprocessing
│   ├── augmentation/       # Data augmentation utilities
│   ├── datasets/           # Dataset implementations (image, text, audio, video)
│   ├── loaders/            # Data loading utilities
│   ├── preprocessing/      # Preprocessing pipelines
│   ├── protein/            # Protein-specific data handling
│   └── tokenizers/         # Text tokenization
├── generative_models/
│   ├── core/               # Core abstractions
│   │   ├── configuration/  # Frozen dataclass configs
│   │   ├── distributions/  # Probability distributions
│   │   ├── evaluation/     # Metrics and benchmarks
│   │   ├── layers/         # Custom neural network layers
│   │   ├── losses/         # Modular loss functions
│   │   ├── protocols/      # Type-safe interfaces
│   │   └── sampling/       # Sampling strategies
│   ├── models/             # Model implementations
│   │   ├── audio/          # Audio models (WaveNet)
│   │   ├── autoregressive/ # Autoregressive models (PixelCNN, Transformer)
│   │   ├── backbones/      # Shared architectures (UNet, DiT)
│   │   ├── common/         # Shared model components
│   │   ├── diffusion/      # Diffusion models (DDPM, DDIM, DiT)
│   │   ├── energy/         # Energy-based models
│   │   ├── flow/           # Normalizing flows (RealNVP, Glow, MAF)
│   │   ├── gan/            # GAN variants (DCGAN, WGAN, StyleGAN)
│   │   ├── geometric/      # Geometric models (point clouds, meshes)
│   │   └── vae/            # VAE variants (VAE, β-VAE, VQ-VAE)
│   ├── modalities/         # Multi-modal support
│   │   ├── audio/          # Audio modality
│   │   ├── image/          # Image modality
│   │   ├── molecular/      # Molecular modality
│   │   ├── multi_modal/    # Cross-modal generation
│   │   ├── protein/        # Protein structure modality
│   │   ├── tabular/        # Tabular data modality
│   │   ├── text/           # Text modality
│   │   └── timeseries/     # Time series modality
│   ├── training/           # Training infrastructure
│   │   ├── callbacks/      # Training callbacks
│   │   ├── distributed/    # Distributed training utilities
│   │   ├── optimizers/     # Custom optimizers
│   │   ├── rl/             # Reinforcement learning trainers
│   │   ├── schedulers/     # Learning rate schedulers
│   │   └── trainers/       # Model-specific trainers
│   ├── inference/          # Generation and serving
│   ├── factory/            # Model creation utilities
│   ├── extensions/         # Domain-specific extensions
│   │   ├── audio_processing/
│   │   ├── chemical/
│   │   ├── nlp/
│   │   ├── protein/
│   │   └── vision/
│   ├── fine_tuning/        # LoRA, adapters, RL
│   ├── scaling/            # Distributed training
│   ├── utils/              # Generative model utilities
│   └── zoo/                # Pre-configured models
├── utils/                  # Shared utilities
└── visualization/          # Plotting and visualization
```

See [Architecture Overview](docs/core/architecture.md) for detailed information.

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suite
uv run pytest tests/artifex/generative_models/models/vae/ -v

# Run with coverage
uv run pytest --cov=src/artifex --cov-report=html

# GPU-aware testing
./scripts/smart_test_runner.sh tests/ -v
```

### Code Quality

```bash
# Run all quality checks
uv run pre-commit run --all-files

# Individual checks
uv run ruff check src/          # Linting
uv run ruff format src/         # Formatting
uv run pyright src/             # Type checking
```

See [Testing Guide](docs/development/testing.md) and [Contributing Guide](CONTRIBUTING.md) for more information.

## Current Status

Artifex is undergoing a major refactoring effort. The library architecture is being restructured for better modularity and maintainability:

- 🔄 Core model implementations (VAE, GAN, Diffusion, Flow, EBM, Autoregressive) - refactoring
- 🔄 Multi-modal data pipeline - refactoring
- 🔄 Unified configuration system - refactoring
- 🔄 Test suite - being restructured
- 🔄 Protocol-based architecture - refactoring
- 🚧 Advanced scaling features (experimental)
- 🚧 Benchmark suite (in progress)
- 🚧 Documentation (outdated, being updated)
- 📋 API stabilization (planned)
- 📋 Performance optimizations (planned)

See [Roadmap](docs/development/roadmap.md) for planned features and improvements.

## Contributing

We welcome contributions! Artifex is an open-source research project that benefits from community involvement.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [Contributing Guide](CONTRIBUTING.md) for detailed guidelines.

## Citation

If you use Artifex in your research, please cite:

```bibtex
@software{artifex_2025,
  title = {Artifex: A Modular Generative Modeling Library},
  author = {Shafiei, Mahdi},
  year = {2025},
  url = {https://github.com/avitai/artifex},
  version = {0.1.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Artifex is built on excellent open-source projects:

- [JAX](https://github.com/google/jax) - High-performance numerical computing
- [Flax](https://github.com/google/flax) - Neural network library
- [Optax](https://github.com/deepmind/optax) - Optimization library
- [Orbax](https://github.com/google/orbax) - Checkpointing
- [BlackJAX](https://github.com/blackjax-devs/blackjax) - MCMC sampling

---

<div align="center">

**[Documentation](https://docs.avitai.bio/artifex)** • **[GitHub](https://github.com/avitai/artifex)** • **[Issues](https://github.com/avitai/artifex/issues)**

</div>
