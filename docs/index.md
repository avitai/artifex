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
pip install artifex

# With GPU support (CUDA 12.0+)
pip install artifex[cuda]
```

See the [Installation Guide](getting-started/installation.md) for detailed setup instructions including Docker and source installation.

### Train a Diffusion Model on Fashion-MNIST

```python
import jax
import jax.numpy as jnp
import optax
from datarax import from_source
from datarax.core.config import ElementOperatorConfig
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator
from datarax.sources import TfdsDataSourceConfig, TFDSSource
from flax import nnx

from artifex.generative_models.core.configuration import (
    DDPMConfig, NoiseScheduleConfig, UNetBackboneConfig,
)
from artifex.generative_models.models.diffusion import DDPMModel
from artifex.generative_models.training.trainers import (
    DiffusionTrainer, DiffusionTrainingConfig,
)

# 1. Load Fashion-MNIST with datarax
def normalize(element, _key):
    image = element.data["image"].astype(jnp.float32) / 127.5 - 1.0
    return element.replace(data={**element.data, "image": image})

source = TFDSSource(
    TfdsDataSourceConfig(name="fashion_mnist", split="train", shuffle=True),
    rngs=nnx.Rngs(0),
)
normalize_op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(1))
pipeline = from_source(source, batch_size=64) >> OperatorNode(normalize_op)

# 2. Create DDPM model
config = DDPMConfig(
    name="fashion_ddpm",
    input_shape=(28, 28, 1),
    backbone=UNetBackboneConfig(
        name="unet", in_channels=1, out_channels=1,
        hidden_dims=(32, 64, 128), channel_mult=(1, 2, 4), activation="silu",
    ),
    noise_schedule=NoiseScheduleConfig(
        name="cosine", schedule_type="cosine", num_timesteps=1000,
    ),
)
model = DDPMModel(config, rngs=nnx.Rngs(42))
optimizer = nnx.Optimizer(model, optax.adamw(1e-4), wrt=nnx.Param)

# 3. Configure trainer with SOTA techniques
trainer = DiffusionTrainer(
    noise_schedule=model.noise_schedule,
    config=DiffusionTrainingConfig(
        loss_weighting="min_snr", snr_gamma=5.0, ema_decay=0.9999,
    ),
)
jit_train_step = nnx.jit(trainer.train_step)

# 4. Training loop
rng = jax.random.PRNGKey(0)
for batch in pipeline:
    rng, step_rng = jax.random.split(rng)
    _, metrics = jit_train_step(model, optimizer, {"image": batch["image"]}, step_rng)
    trainer.update_ema(model)

# 5. Generate samples
samples = model.sample(n_samples_or_shape=16, steps=100)
```

See the [Quickstart Guide](getting-started/quickstart.md) for complete examples including VAE training, visualization, and more.

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
