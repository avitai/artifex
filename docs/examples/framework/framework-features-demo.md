# Artifex Framework Features Demonstration

**Level:** Intermediate | **Runtime:** ~1-2 minutes (CPU) | **Format:** Python + Jupyter

**Prerequisites:** Basic understanding of generative models and JAX | **Target Audience:** Users learning the framework's architecture

## Overview

This example provides a comprehensive tour of the Artifex framework's core features and design patterns. Learn how to leverage the unified configuration system, factory pattern, composable losses, sampling methods, and modality adapters for building production-ready generative models.

## What You'll Learn

<div class="grid cards" markdown>

- :material-cog-outline: **Unified Configuration**

    ---

    Type-safe model, training, and data configurations with Pydantic validation

- :material-factory: **Factory Pattern**

    ---

    Consistent model creation interface across all model types (VAE, GAN, diffusion)

- :material-function-variant: **Composable Losses**

    ---

    Flexible loss composition with weighted components and tracking

- :material-dice-multiple: **Sampling Methods**

    ---

    MCMC and SDE sampling for energy-based and diffusion models

- :material-view-module-outline: **Modality System**

    ---

    Domain-specific adapters for images, text, audio, proteins, and 3D data

</div>

## Files

This example is available in two formats:

- **Python Script**: [`framework_features_demo.py`](https://github.com/avitai/artifex/blob/main/examples/generative_models/framework_features_demo.py)
- **Jupyter Notebook**: [`framework_features_demo.ipynb`](https://github.com/avitai/artifex/blob/main/examples/generative_models/framework_features_demo.ipynb)

## Quick Start

### Run the Python Script

```bash
# Activate environment
source activate.sh

# Run the example
python examples/generative_models/framework_features_demo.py
```

### Run the Jupyter Notebook

```bash
# Activate environment
source activate.sh

# Launch Jupyter
jupyter lab examples/generative_models/framework_features_demo.ipynb
```

## Key Concepts

### 1. Unified Configuration System

Artifex uses frozen dataclass configuration classes for type-safe, validated model definitions:

```python
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
)

# Create encoder config (nested configuration)
encoder_config = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(256, 128),  # Tuple for frozen dataclass
    activation="relu",
)

# Create decoder config
decoder_config = DecoderConfig(
    name="decoder",
    output_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(128, 256),  # Tuple for frozen dataclass
    activation="relu",
)

# Create a model configuration with nested configs
config = VAEConfig(
    name="my_vae",
    encoder=encoder_config,
    decoder=decoder_config,
    kl_weight=1.0,
)
```

**Benefits:**

- Type-safe with automatic validation
- Immutable (frozen) for consistency
- Serialization to JSON/YAML for reproducibility
- Nested composition for complex models
- Easy parameter sweeps for hyperparameter tuning

### 2. Factory Pattern

The factory pattern provides unified model creation:

```python
from artifex.generative_models.factory import create_model
from flax import nnx

# Setup RNGs
rngs = nnx.Rngs(params=42, dropout=42)

# Create any model from configuration
model = create_model(config, rngs=rngs)

# Test forward pass
outputs = model(test_data, rngs=rngs)
```

**Why Use Factories?**

- Consistency across all model types
- Validation before instantiation
- Easy to swap models for experimentation
- Proper RNG management

### 3. Composable Loss System

Combine multiple loss functions with different weights:

$$L_{\text{total}} = \sum_{i=1}^{n} w_i \cdot L_i(\text{pred}, \text{target})$$

```python
from artifex.generative_models.core.losses import (
    CompositeLoss,
    WeightedLoss,
    mse_loss,
    mae_loss,
)

# Create composite loss
composite = CompositeLoss([
    WeightedLoss(mse_loss, weight=1.0, name="reconstruction"),
    WeightedLoss(mae_loss, weight=0.5, name="l1_penalty"),
], return_components=True)

# Compute loss with component tracking
total_loss, components = composite(predictions, targets)
# components = {"reconstruction": 0.15, "l1_penalty": 0.08}
```

### 4. Sampling Methods

#### MCMC Sampling (Energy-Based Models)

```python
from artifex.generative_models.core.sampling import mcmc_sampling

def log_prob_fn(x):
    return -0.5 * jnp.sum(x**2)  # Log probability

samples = mcmc_sampling(
    log_prob_fn=log_prob_fn,
    init_state=jnp.zeros(10),
    key=jax.random.key(42),
    n_samples=1000,
    n_burnin=200,
    step_size=0.1,
)
```

#### SDE Sampling (Diffusion Models)

```python
from artifex.generative_models.core.sampling import sde_sampling

def drift_fn(x, t):
    return -x  # Drift function

def diffusion_fn(x, t):
    return jnp.ones_like(x) * 0.1  # Diffusion coefficient

sample = sde_sampling(
    drift_fn=drift_fn,
    diffusion_fn=diffusion_fn,
    init_state=x0,
    t_span=(0.0, 1.0),
    key=key,
    n_steps=100,
)
```

### 5. Modality System

Domain-specific features for different data types:

```python
from artifex.generative_models.modalities import get_modality

# Get image modality
image_modality = get_modality('image', rngs=rngs)

# Create dataset
dataset = image_modality.create_dataset(data_config)

# Compute metrics
metrics = image_modality.evaluate(model, test_data)
# metrics = {"fid": 12.5, "is_score": 8.3, ...}

# Get modality adapter
adapter = image_modality.get_adapter('vae')
adapted_model = adapter.adapt(base_model, config)
```

**Available Modalities:**

- `image`: Convolutional layers, FID, IS metrics
- `text`: Tokenization, perplexity, BLEU
- `audio`: Spectrograms, MFCCs
- `protein`: Structure prediction, sequence modeling
- `geometric`: Point clouds, mesh processing

## Code Structure

The example demonstrates framework features in five sections:

1. **Configuration System** - Create type-safe configs for models, training, data
2. **Factory Pattern** - Instantiate models from configurations
3. **Composable Losses** - Combine weighted loss functions
4. **Sampling Methods** - MCMC and SDE sampling for generation
5. **Modality System** - Domain-specific adapters and evaluation

Each section is self-contained and can be run independently.

## Features Demonstrated

- ✅ Type-safe configuration with automatic validation
- ✅ Unified model creation across all types (VAE, GAN, diffusion, flow, EBM)
- ✅ Flexible loss composition with component tracking
- ✅ MCMC sampling for energy-based models
- ✅ SDE sampling for diffusion models
- ✅ Modality-specific dataset loading and evaluation
- ✅ Proper RNG management with `nnx.Rngs`
- ✅ JIT compilation for performance

## Experiments to Try

1. **Create Different Model Types**

   ```python
   from artifex.generative_models.core.configuration import (
       GANConfig,
       GeneratorConfig,
       DiscriminatorConfig,
   )

   # Create GAN config with nested generator/discriminator
   gen_config = GeneratorConfig(
       name="generator",
       latent_dim=100,
       output_shape=(28, 28, 1),
       hidden_dims=(256, 512),
   )
   disc_config = DiscriminatorConfig(
       name="discriminator",
       input_shape=(28, 28, 1),
       hidden_dims=(512, 256),
   )
   gan_config = GANConfig(
       name="my_gan",
       generator=gen_config,
       discriminator=disc_config,
   )
   gan = create_model(gan_config, rngs=rngs)
   ```

2. **Custom Loss Combinations**

   ```python
   # Add perceptual loss to composite
   from artifex.generative_models.core.losses import PerceptualLoss

   composite = CompositeLoss([
       WeightedLoss(mse_loss, weight=1.0, name="recon"),
       WeightedLoss(PerceptualLoss(), weight=0.1, name="perceptual"),
   ])
   ```

3. **Adjust Sampling Parameters**

   ```python
   # Try different MCMC settings
   samples = mcmc_sampling(
       log_prob_fn=log_prob_fn,
       init_state=x0,
       key=key,
       n_samples=5000,  # More samples
       step_size=0.01,  # Smaller steps
   )
   ```

4. **Experiment with Modalities**

   ```python
   # Try different modalities
   audio_modality = get_modality('audio', rngs=rngs)
   audio_dataset = audio_modality.create_dataset(audio_config)
   ```

## Next Steps

<div class="grid cards" markdown>

- :material-arrow-right: **VAE Examples**

    ---

    Learn VAE implementation patterns

    [:octicons-arrow-right-24: Basic VAE Tutorial](../basic/vae-mnist.md)

- :material-arrow-right: **GAN Examples**

    ---

    Explore GAN training

    [:octicons-arrow-right-24: Basic GAN Tutorial](../basic/simple-gan.md)

- :material-arrow-right: **Diffusion Examples**

    ---

    Understand diffusion models

    [:octicons-arrow-right-24: Simple Diffusion](../diffusion/simple-diffusion.md)

- :material-arrow-right: **Loss Examples**

    ---

    Deep dive into loss functions

    [:octicons-arrow-right-24: Loss Function Guide](../losses/loss-examples.md)

</div>

## Troubleshooting

### Missing Configuration Fields

**Symptom:** `ValidationError` or `TypeError` when creating configuration

**Solution:** Check required fields in the specific config class. Each model type has its own config class with nested configs.

```python
# View VAE config structure
from artifex.generative_models.core.configuration import VAEConfig, EncoderConfig
print(VAEConfig.__dataclass_fields__)  # See required fields
print(EncoderConfig.__dataclass_fields__)  # See nested config fields
```

### Factory Creation Fails

**Symptom:** `TypeError` or `AttributeError` during model creation

**Solution:** Use the specific config class for your model type with properly nested configs

```python
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
)

# Create with proper nested config structure
encoder = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(256, 128),
)
decoder = DecoderConfig(
    name="decoder",
    output_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(128, 256),
)
config = VAEConfig(
    name="my_vae",
    encoder=encoder,
    decoder=decoder,
    kl_weight=1.0,
)
```

### RNG Key Errors

**Symptom:** `KeyError` for missing RNG streams

**Solution:** Initialize all required RNG streams

```python
# VAE needs params, dropout, and sample streams
rngs = nnx.Rngs(
    params=42,
    dropout=43,
    sample=44,
)
```

## Additional Resources

### Documentation

- [Configuration System Guide](../../user-guide/training/configuration.md) - Deep dive into configurations
- [Factory Pattern Guide](../../factory/index.md) - Advanced factory usage
- [Loss Functions API](../../api/core/losses.md) - Complete loss function reference
- [Sampling Methods](../../api/sampling.md) - Sampling algorithm details

### Related Examples

- [Loss Examples](../losses/loss-examples.md) - Complete loss function showcase
- [VAE MNIST Tutorial](../basic/vae-mnist.md) - Step-by-step VAE implementation
- [Simple Diffusion](../diffusion/simple-diffusion.md) - Diffusion model basics

### Papers

- **Pydantic**: [Pydantic Documentation](https://docs.pydantic.dev/) - Configuration validation
- **JAX**: [JAX Documentation](https://jax.readthedocs.io/) - Array programming and JIT compilation
- **Flax NNX**: [Flax NNX Guide](https://flax.readthedocs.io/en/latest/nnx/index.html) - Neural network library
