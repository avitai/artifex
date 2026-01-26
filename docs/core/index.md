# Core Components

The core module provides foundational abstractions, protocols, and utilities for building generative models in Artifex.

## Overview

<div class="grid cards" markdown>

- :material-cube-outline:{ .lg .middle } **Configuration**

    ---

    Unified dataclass-based configuration system for all models

- :material-function:{ .lg .middle } **Loss Functions**

    ---

    Composable losses for reconstruction, adversarial, divergence, and more

- :material-chart-scatter-plot:{ .lg .middle } **Distributions**

    ---

    Probability distributions for latent spaces and sampling

- :material-dice-multiple:{ .lg .middle } **Sampling**

    ---

    MCMC, ODE/SDE solvers, ancestral sampling, and BlackJAX integration

</div>

## Configuration System

Type-safe, validated configuration for all model components.

```python
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
)

encoder = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(256, 128),
)

config = VAEConfig(
    name="my_vae",
    encoder=encoder,
    decoder=decoder,
)
```

[:octicons-arrow-right-24: Configuration Reference](unified.md)

### Configuration Modules

| Module | Description |
|--------|-------------|
| [unified](unified.md) | Unified configuration classes |
| [environment](environment.md) | Environment configuration |
| [gan](gan.md) | GAN-specific configs |
| [validation](validation.md) | Config validation |
| [migrate_configs](migrate_configs.md) | Config migration tools |

## Loss Functions

Composable loss functions for training generative models.

### Reconstruction Losses

```python
from artifex.generative_models.core.losses import (
    mse_loss,
    binary_cross_entropy,
    gaussian_nll,
)

# Mean squared error
loss = mse_loss(prediction, target)

# Binary cross entropy for images
loss = binary_cross_entropy(prediction, target)
```

[:octicons-arrow-right-24: Reconstruction Losses](reconstruction.md)

### Adversarial Losses

```python
from artifex.generative_models.core.losses import (
    discriminator_loss,
    generator_loss,
    wasserstein_loss,
    gradient_penalty,
)

# Standard GAN loss
d_loss = discriminator_loss(real_scores, fake_scores)
g_loss = generator_loss(fake_scores)

# WGAN with gradient penalty
gp = gradient_penalty(discriminator, real, fake)
```

[:octicons-arrow-right-24: Adversarial Losses](adversarial.md)

### Divergence Losses

```python
from artifex.generative_models.core.losses import (
    kl_divergence,
    kl_divergence_gaussian,
    mmd_loss,
)

# KL divergence for VAE
kl_loss = kl_divergence_gaussian(mean, log_var)

# Maximum Mean Discrepancy
mmd = mmd_loss(samples1, samples2)
```

[:octicons-arrow-right-24: Divergence Losses](divergence.md)

### Perceptual Losses

```python
from artifex.generative_models.core.losses import (
    perceptual_loss,
    lpips_loss,
)

# VGG-based perceptual loss
loss = perceptual_loss(generated, target, feature_extractor)
```

[:octicons-arrow-right-24: Perceptual Losses](perceptual.md)

### Composable Losses

```python
from artifex.generative_models.core.losses import CompositeLoss

loss_fn = CompositeLoss(
    losses=[
        ("reconstruction", mse_loss, 1.0),
        ("kl", kl_divergence_gaussian, 0.1),
        ("perceptual", perceptual_loss, 0.01),
    ]
)

total_loss, loss_dict = loss_fn(model_outputs, targets)
```

[:octicons-arrow-right-24: Composable Losses](composable.md)

## Distributions

Probability distributions for generative modeling.

```python
from artifex.generative_models.core.distributions import (
    Normal,
    Categorical,
    MixtureDiagonal,
)

# Gaussian distribution
dist = Normal(mean, std)
samples = dist.sample(key, shape=(100,))
log_prob = dist.log_prob(samples)

# Mixture of Gaussians
mixture = MixtureDiagonal(means, stds, weights)
```

### Distribution Modules

| Module | Description |
|--------|-------------|
| [base](base.md) | Base distribution class |
| [continuous](continuous.md) | Normal, Laplace, etc. |
| [discrete](discrete.md) | Categorical, Bernoulli |
| [mixture](mixture.md) | Mixture distributions |
| [transformations](transformations.md) | Distribution transforms |

## Sampling Methods

Advanced sampling algorithms for generative models.

### Ancestral Sampling

```python
from artifex.generative_models.core.sampling import ancestral_sample

samples = ancestral_sample(
    model=diffusion_model,
    shape=(batch_size, *image_shape),
    num_steps=1000,
    key=prng_key,
)
```

[:octicons-arrow-right-24: Ancestral Sampling](ancestral.md)

### MCMC Sampling

```python
from artifex.generative_models.core.sampling import (
    langevin_dynamics,
    hmc_sample,
)

# Langevin dynamics for EBM
samples = langevin_dynamics(
    energy_fn=model.energy,
    initial=noise,
    step_size=0.01,
    num_steps=100,
)
```

[:octicons-arrow-right-24: MCMC Sampling](mcmc.md)

### BlackJAX Integration

```python
from artifex.generative_models.core.sampling import BlackJAXSampler

sampler = BlackJAXSampler(
    algorithm="nuts",
    step_size=0.1,
    num_warmup=500,
)

samples = sampler.sample(
    log_prob_fn=model.log_prob,
    initial_state=initial,
    num_samples=1000,
)
```

[:octicons-arrow-right-24: BlackJAX Samplers](blackjax_samplers.md)

### ODE/SDE Solvers

```python
from artifex.generative_models.core.sampling import (
    ode_solver,
    sde_solver,
)

# Probability flow ODE
samples = ode_solver(
    drift_fn=model.drift,
    initial=noise,
    t_span=(1.0, 0.0),
)

# Reverse-time SDE
samples = sde_solver(
    drift_fn=model.drift,
    diffusion_fn=model.diffusion,
    initial=noise,
)
```

[:octicons-arrow-right-24: ODE Solvers](ode.md) | [:octicons-arrow-right-24: SDE Solvers](sde.md)

## Metrics

Evaluation metrics for generative models.

### Image Metrics

```python
from artifex.generative_models.core.metrics import (
    compute_fid,
    compute_inception_score,
)

fid = compute_fid(real_images, generated_images)
is_score = compute_inception_score(generated_images)
```

[:octicons-arrow-right-24: FID](fid.md) | [:octicons-arrow-right-24: Inception Score](inception_score.md)

### Statistical Metrics

```python
from artifex.generative_models.core.metrics import (
    precision_recall,
    coverage,
    density,
)

pr = precision_recall(real_features, generated_features)
```

[:octicons-arrow-right-24: Precision/Recall](precision_recall.md)

## Layers

Reusable neural network layers.

### Attention

```python
from artifex.generative_models.core.layers import (
    MultiHeadAttention,
    FlashAttention,
    CrossAttention,
)

attention = MultiHeadAttention(
    num_heads=8,
    head_dim=64,
    rngs=rngs,
)
```

[:octicons-arrow-right-24: Flash Attention](flash_attention.md)

### Positional Encoding

```python
from artifex.generative_models.core.layers import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
)
```

[:octicons-arrow-right-24: Positional Encoding](positional.md)

### Transformer Blocks

```python
from artifex.generative_models.core.layers import (
    TransformerBlock,
    TransformerEncoder,
    TransformerDecoder,
)
```

[:octicons-arrow-right-24: Transformers](transformers.md)

### ResNet Blocks

```python
from artifex.generative_models.core.layers import (
    ResNetBlock,
    ResNetBlockV2,
)
```

[:octicons-arrow-right-24: ResNet](resnet.md)

## Device Management

Hardware-aware device management.

```python
from artifex.generative_models.core import DeviceManager

device_manager = DeviceManager()
device = device_manager.get_device()

# Check GPU availability
if device_manager.has_gpu():
    print(f"Using GPU: {device_manager.gpu_info()}")
```

[:octicons-arrow-right-24: Device Manager](device_manager.md)

## Protocols

Protocol interfaces for type-safe implementations.

```python
from artifex.generative_models.core.protocols import (
    GenerativeModel,
    Encoder,
    Decoder,
    LossFunction,
)
```

### Protocol Modules

| Module | Description |
|--------|-------------|
| [benchmarks](benchmarks.md) | Benchmark protocols |
| [configuration](configuration.md) | Config protocols |
| [evaluation](evaluation.md) | Evaluation protocols |
| [metrics](metrics.md) | Metric protocols |

## Module Reference

| Category | Modules |
|----------|---------|
| **Configuration** | [unified](unified.md), [environment](environment.md), [gan](gan.md), [validation](validation.md), [migrate_configs](migrate_configs.md) |
| **Losses** | [adversarial](adversarial.md), [base](base.md), [composable](composable.md), [divergence](divergence.md), [geometric](geometric.md), [perceptual](perceptual.md), [reconstruction](reconstruction.md), [regularization](regularization.md) |
| **Distributions** | [base](base.md), [continuous](continuous.md), [discrete](discrete.md), [mixture](mixture.md), [transformations](transformations.md) |
| **Sampling** | [ancestral](ancestral.md), [base](base.md), [blackjax_samplers](blackjax_samplers.md), [diffusion](diffusion.md), [mcmc](mcmc.md), [ode](ode.md), [sde](sde.md) |
| **Metrics** | [base](base.md), [distance](distance.md), [fid](fid.md), [inception_score](inception_score.md), [information](information.md), [precision_recall](precision_recall.md), [quality](quality.md), [statistical](statistical.md) |
| **Layers** | [causal](causal.md), [flash_attention](flash_attention.md), [positional](positional.md), [residual](residual.md), [resnet](resnet.md), [transformers](transformers.md) |
| **Core** | [adapters](adapters.md), [base](base.md), [checkpointing](checkpointing.md), [device_manager](device_manager.md), [interfaces](interfaces.md), [logging](logging.md), [parallelism](parallelism.md), [performance](performance.md), [types](types.md) |

## Related Documentation

- [API Reference](../api/core/base.md) - Complete API documentation
- [Configuration Guide](../user-guide/training/configuration.md) - Configuration system guide
- [Loss Functions Guide](../api/core/losses.md) - Loss function details
