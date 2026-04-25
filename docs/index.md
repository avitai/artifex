# Artifex: Generative Modeling Research Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-orange.svg)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Flax-NNX-blue.svg)](https://flax.readthedocs.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A research-focused modular generative modeling library built on JAX/Flax NNX, providing implementations of state-of-the-art generative models with multi-modal support and scientific computing focus.

## Why Artifex?

<div class="grid cards" markdown>

- :material-flask-outline:{ .lg .middle } __State-of-the-Art Models__

    ---

    VAEs, GANs, Diffusion, Flows, EBMs, Autoregressive, and Geometric models with 2025 research compliance

    [:octicons-arrow-right-24: Model Gallery](models/index.md)

- :material-rocket-launch:{ .lg .middle } __Research-Focused__

    ---

    Hardware-aware optimization, distributed training, mixed precision validated through 2150+ tests

    [:octicons-arrow-right-24: Getting Started](getting-started/installation.md)

- :material-view-grid:{ .lg .middle } __Multi-Modal Support__

    ---

    Native support for images, text, audio, proteins, and multi-modal data

    [:octicons-arrow-right-24: Modalities Guide](user-guide/data/overview.md)

- :material-scale-balance:{ .lg .middle } __Scalable Architecture__

    ---

    From single GPU to multi-node distributed training with FSDP and tensor parallelism

    [:octicons-arrow-right-24: Scaling Guide](user-guide/advanced/distributed.md)

</div>

## Quick Start

### Installation

```bash
# CPU-only version
pip install avitai-artifex

# With GPU support (CUDA 12.0+)
pip install "avitai-artifex[cuda12]"
```

See the [Installation Guide](getting-started/installation.md) for detailed setup instructions including Docker and source installation.

### Start with the VAE Quickstart

The primary onboarding path is the checked-in VAE quickstart. It uses
`TFDSEagerSource`, `VAEConfig`, `VAETrainer`, and `train_epoch_staged` to train on
MNIST while keeping the hot path in JAX.

```python
from datarax.sources import TFDSEagerSource
from datarax.sources.tfds_source import TFDSEagerConfig
from artifex.generative_models.core.configuration import DecoderConfig, EncoderConfig, VAEConfig
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.training import train_epoch_staged
from artifex.generative_models.training.trainers import VAETrainer, VAETrainingConfig
```

See the [Quickstart Guide](getting-started/quickstart.md) for the full walkthrough and
refer to the checked-in executable pair under `docs/getting-started/quickstart.py` and
`docs/getting-started/quickstart.ipynb` in the repository.

### Next Steps

<div class="grid cards" markdown>

- :material-school:{ .lg .middle } __Core Concepts__

    ---

    Understand generative modeling concepts and Artifex architecture

    [:octicons-arrow-right-24: Learn](getting-started/core-concepts.md)

- :material-rocket-launch:{ .lg .middle } __Quickstart Guide__

    ---

    Train your first generative model with Artifex

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

- :material-bookshelf:{ .lg .middle } __Model Guides__

    ---

    Deep dive into VAEs, GANs, Diffusion, Flows, and more

    [:octicons-arrow-right-24: Guides](user-guide/models/vae-guide.md)

- :material-chart-line:{ .lg .middle } __Examples__

    ---

    Working examples for all model types and use cases

    [:octicons-arrow-right-24: Examples](examples/index.md)

</div>

## Model Types

| Model Type         | Best For                                          | Guide                                                         |
| ------------------ | ------------------------------------------------- | ------------------------------------------------------------- |
| __VAE__            | Latent space exploration, representation learning | [VAE Guide](user-guide/concepts/vae-explained.md)             |
| __GAN__            | High-quality image synthesis, style transfer      | [GAN Guide](user-guide/concepts/gan-explained.md)             |
| __Diffusion__      | State-of-the-art generation, inpainting           | [Diffusion Guide](user-guide/concepts/diffusion-explained.md) |
| __Flow__           | Exact likelihood, density estimation              | [Flow Guide](user-guide/concepts/flow-explained.md)           |
| __EBM__            | Compositional generation, constraints             | [EBM Guide](user-guide/concepts/ebm-explained.md)             |
| __Autoregressive__ | Text, sequential data                             | [AR Guide](user-guide/concepts/autoregressive-explained.md)   |
| __Geometric__      | Proteins, molecules, 3D structures                | [Examples](examples/geometric/geometric-models-demo.md)       |

## Architecture

See [Architecture Overview](core/architecture.md) for detailed system design, component structure, and extension points.

## Citation

```bibtex
@software{artifex_2025,
  title = {Artifex: Generative Modeling Research Library},
  author = {Shafiei, Mahdi and contributors},
  year = {2025},
  url = {https://github.com/avitai/artifex},
  version = {0.1.0}
}
```

## Contributing

We welcome contributions! See the [Contributing Guide](community/contributing.md) for guidelines.

---

<p align="center">
  <strong><a href="https://github.com/avitai/artifex">GitHub</a></strong> |
  <strong><a href="https://github.com/avitai/artifex/issues">Issues</a></strong> |
  <strong><a href="https://github.com/avitai/artifex/discussions">Discussions</a></strong>
</p>
