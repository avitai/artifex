# Artifex

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-green)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-NNX-orange)](https://github.com/google/flax)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## A research-focused modular generative modeling library built on JAX/Flax NNX

From Latin "artifex": craftsman, artist, maker.

[Documentation](https://docs.avitai.bio/artifex) • [Getting Started](docs/getting-started/installation.md) • [Examples](docs/examples/index.md) • [Contributing](CONTRIBUTING.md)

</div>

---

> **⚠️ Major Refactoring in Progress**
>
> Artifex is still in a heavy rebuild cycle. Stability is not guaranteed, and breaking changes are expected between commits.
>
> | Area | Status | Current expectation |
> |------|--------|---------------------|
> | **API surface** | Unstable | Public interfaces can change without deprecation while runtime boundaries are still being simplified. |
> | **Performance** | In progress | Optimization work is ongoing across training, inference, and benchmark paths. Do not assume current throughput or memory behavior is final. |
> | **Feature breadth** | Expanding | Core model families ship today, but additional capabilities, deeper integrations, and broader examples are still being added. |
> | **Docs and workflows** | Maintained but evolving | Checked-in installation, quickstart, examples, and contributor workflows are kept aligned with the live runtime, while broader documentation continues to be revised. |
>
> Artifex is suitable for active research and repository development. It is not in a state where long-term API stability or production guarantees should be assumed.

---

## Overview

Artifex is a modular library for generative modeling research, providing implementations of various state-of-the-art generative models with a focus on modularity, type safety, and scientific reproducibility. Built on JAX and Flax NNX, it emphasizes clean abstractions and extensible design for research experimentation.

### Why Artifex?

- **Research First**: Designed for experimentation with clean, modular architecture
- **Modern Stack**: Built on JAX/Flax NNX with full JIT compilation and automatic differentiation
- **Typed Surfaces**: Protocol-based design with Pyright-checked source interfaces
- **Multi-Modal**: Unified interface across images, text, audio, proteins, and more
- **Extensible**: Easy to add new models, losses, and domain-specific constraints
- **Actively Verified**: Blocking CI enforces repository contracts, packaging checks, and focused test suites

## Design Philosophy

### Research-Focused

Artifex prioritizes:

- **Modularity**: Easy to swap components and experiment
- **Clarity**: Clean, readable implementations over clever optimizations
- **Extensibility**: Simple to add new models and functionality
- **Reproducibility**: Deterministic with clear configuration management

### Technical Principles

- **Type Checking**: Pyright basic-mode reports track the supported source surface while repo-wide blocking enforcement is still being rebuilt
- **JAX Native**: Leverages JAX's functional programming paradigm
- **Flax NNX**: Modern object-oriented API for neural networks
- **Configuration Management**: Frozen dataclass configs with validation
- **Testing**: Blocking CI enforces repository contracts and a 70% repo-wide coverage floor while new changes target 80% coverage

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
- **Evaluation Framework**: Standardized metrics and benchmarks with CalibraX-aligned composition

## Quick Start

### Installation

```bash
# Package users
pip install artifex

# Optional Linux NVIDIA GPU support
pip install "artifex[cuda12]"
```

If you are contributing from a source checkout instead:

```bash
git clone https://github.com/avitai/artifex.git
cd artifex

# Run setup script (creates .venv, syncs extras, chooses a backend policy)
./setup.sh

# Activate the environment (must use 'source')
source ./activate.sh
```

The setup script automatically:

- Detects an appropriate backend policy
- Creates a virtual environment with uv
- Syncs the right extras for CPU, CUDA 12, or Metal development
- Writes a generated `.artifex.env` file and leaves `.env` for user-owned overrides
- Re-sourcing `activate.sh` refreshes the managed backend state before applying user overrides

For an explicit choice, use `./setup.sh --backend cpu`, `./setup.sh --backend cuda12`, or `./setup.sh --backend metal`.

If you need to rebuild from scratch, use `./setup.sh --recreate`. If you also want to clear repo-local test and coverage artifacts without touching user-owned `.env` files, use `./setup.sh --force-clean`.

For detailed package-user and source-checkout options, see the [Installation Guide](docs/getting-started/installation.md).

### Start with the checked-in VAE quickstart

The primary onboarding path is the live VAE quickstart under `docs/getting-started/quickstart.py` and `docs/getting-started/quickstart.ipynb`. It trains a VAE on MNIST with `TFDSEagerSource`, `VAETrainer`, and `train_epoch_staged`.

```python
from datarax.sources import TFDSEagerSource
from datarax.sources.tfds_source import TFDSEagerConfig
from artifex.generative_models.core.configuration import DecoderConfig, EncoderConfig, VAEConfig
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.training import train_epoch_staged
from artifex.generative_models.training.trainers import VAETrainer, VAETrainingConfig
```

From a source checkout, run the maintained quickstart pair directly:

```bash
uv run python docs/getting-started/quickstart.py
uv run jupyter lab docs/getting-started/quickstart.ipynb
```

For the full walkthrough, see the [Quickstart Guide](docs/getting-started/quickstart.md).

## Documentation

### Start Here

- [Installation Guide](docs/getting-started/installation.md) - Environment setup, backend policy, and package installation
- [Quickstart Guide](docs/getting-started/quickstart.md) - VAE-first onboarding on MNIST
- [Core Concepts](docs/getting-started/core-concepts.md) - Architecture, configuration, and runtime model

### User and API Guides

- [Examples Catalog](docs/examples/index.md) - Executable and documented example inventory
- [Benchmarks](docs/benchmarks/index.md) - Evaluation suites and benchmark guidance
- [Model Guides](docs/user-guide/models/vae-guide.md) - User-facing guides across model families
- [Core API](docs/api/core/base.md) - Core runtime and protocol surfaces
- [Models API](docs/api/models/vae.md) - Model-family API reference
- [Training API](docs/api/training/trainer.md) - Training and optimization surfaces

### Contributor References

- [Contributing Guide](CONTRIBUTING.md) - Setup, workflow, and contribution expectations
- [Testing Guide](TESTING.md) - Supported pytest workflow and backend guidance
- [Example Documentation Design](docs/development/example-documentation-design.md) - Reader-facing example standards
- [Planned Modules](docs/roadmap/planned-modules.md) - Areas that remain intentionally unshipped or planned

## Architecture

Artifex keeps the public package surface relatively small at the top level and concentrates most runtime code under `artifex.generative_models`.

```text
artifex/
├── src/artifex/
│   ├── benchmarks/         # Benchmark foundations, adapters, datasets, and suites
│   ├── cli/                # Supported `artifex` command-line entrypoint
│   ├── configs/            # Checked-in config defaults and loader utilities
│   ├── data/               # Shared data helpers and retained dataset surfaces
│   ├── generative_models/
│   │   ├── core/           # Configuration, protocols, losses, layers, sampling, evaluation
│   │   ├── extensions/     # Audio, chemical, NLP, protein, and vision extensions
│   │   ├── factory/        # Canonical model creation surface
│   │   ├── inference/      # Inference and optimization helpers
│   │   ├── modalities/     # Image, text, audio, protein, tabular, timeseries, multimodal
│   │   ├── models/         # VAE, GAN, diffusion, flow, energy, autoregressive, geometric
│   │   ├── scaling/        # Distributed and scaling helpers
│   │   ├── training/       # Loops, callbacks, optimizers, schedulers, RL, trainers
│   │   ├── utils/          # Logging, JAX helpers, visualization, analysis utilities
│   │   └── zoo/            # Checked-in model zoo configs
│   ├── utils/              # Shared package utilities
│   └── visualization/      # Public visualization helpers
├── docs/                   # User, API, and contributor documentation
├── examples/               # Executable scripts and notebook pairs
└── tests/                  # Package, integration, unit, and repo-contract coverage
```

See [Architecture Overview](docs/core/architecture.md) for more detail.

## Development

### Verification workflow

```bash
# Standard test suite
uv run pytest

# Focused contract checks
uv run pytest tests/artifex/repo_contracts -q

# Docs validation
uv run python scripts/validate_docs.py --check-only --config-path mkdocs.yml --docs-path docs --src-path src
```

### Code quality

```bash
# Run the repository hooks
uv run pre-commit run --all-files

# Targeted quality tools
uv run ruff check src tests
uv run ruff format src tests
uv run pyright
```

See [Testing Guide](TESTING.md) and [Contributing Guide](CONTRIBUTING.md) for the maintained contributor workflow.

## Project Status

Artifex is in active alpha development.

- Checked-in installation, onboarding, example, and contributor guides are maintained against the live runtime.
- Blocking CI enforces repository contracts and build verification.
- Quality and security workflows remain reviewed but informational while broader release hardening continues.
- Package surfaces can still evolve between commits when a simpler or more truthful runtime design requires it.

Use the [Installation Guide](docs/getting-started/installation.md), [Quickstart Guide](docs/getting-started/quickstart.md), [Testing Guide](TESTING.md), and [Planned Modules](docs/roadmap/planned-modules.md) as the current source of truth for supported workflows.

## Contributing

Artifex accepts contributions through the standard repository workflow.

1. Clone the repository and run `./setup.sh`.
2. Activate the environment with `source ./activate.sh`.
3. Create a feature branch for the change.
4. Add or update tests and documentation with the code change.
5. Run `uv run pytest` and `uv run pre-commit run --all-files`.
6. Open a Pull Request.

See the [Contributing Guide](CONTRIBUTING.md) for the full contributor checklist and coding expectations.

## Citation

If you use Artifex in research, please cite:

```bibtex
@software{artifex_2025,
  title = {Artifex: Generative Modeling Research Library},
  author = {Shafiei, Mahdi and contributors},
  year = {2025},
  url = {https://github.com/avitai/artifex},
  version = {0.1.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Artifex builds on several strong open-source projects:

- [JAX](https://github.com/google/jax) - Numerical computing and transformations
- [Flax](https://github.com/google/flax) - Neural network modules with NNX support
- [Optax](https://github.com/deepmind/optax) - Optimization utilities
- [Orbax](https://github.com/google/orbax) - Checkpointing
- [BlackJAX](https://github.com/blackjax-devs/blackjax) - MCMC and energy-based sampling
- [CalibraX](https://github.com/avitai/calibrax) - Evaluation and benchmark composition
- [DataRax](https://github.com/avitai/datarax) - Dataset and source adapters used in onboarding workflows

---

<div align="center">

**[Documentation](https://docs.avitai.bio/artifex)** • **[GitHub](https://github.com/avitai/artifex)** • **[Issues](https://github.com/avitai/artifex/issues)**

</div>
