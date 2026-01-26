# Centralized Factory System

The Artifex repository uses a centralized factory system for creating all generative models. This provides a single, consistent API for model creation with full type safety and modality support.

## Migration Notice

⚠️ **IMPORTANT**: The old factory modules have been deprecated and removed:

- `artifex.generative_models.factories.*` → Use `artifex.generative_models.factory`
- `artifex.generative_models.models.*/factory.py` → Use `artifex.generative_models.factory`
- Model-specific factory functions → Use `create_model()` or `zoo.create_model()`

## Architecture

```
factory/
├── __init__.py         # Public API (create_model function)
├── core.py            # ModelFactory implementation
├── registry.py        # ModelTypeRegistry for builders
└── builders/          # Model-specific builders
    ├── vae.py         # VAE models (VAE, BetaVAE, CVAE, VQVAE)
    ├── gan.py         # GAN models (DCGAN, WGAN, CycleGAN, etc.)
    ├── diffusion.py   # Diffusion models (DDPM, DDIM, Score-based)
    ├── flow.py        # Flow models (RealNVP, Glow, MAF)
    ├── ebm.py         # Energy-based models
    ├── autoregressive.py  # Autoregressive models (Transformer, PixelCNN)
    └── geometric.py   # Geometric models (PointCloud, Mesh, Graph)
```

## Basic Usage

### Creating Models with Configuration

```python
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import ModelConfiguration
from flax import nnx

# Create configuration
config = ModelConfiguration(
    name="my_vae",
    model_class="artifex.generative_models.models.vae.VAE",
    input_dim=(28, 28, 1),
    hidden_dims=[512, 256],
    output_dim=64,  # latent_dim for VAE
    activation="relu",
    # Model-specific functional parameters go in 'parameters'
    parameters={
        "encoder_type": "mlp",
        "decoder_type": "mlp",
        "beta": 1.0,
        "kl_weight": 1.0,
    },
    # Non-functional metadata for tracking and documentation
    metadata={
        "experiment_id": "exp_001",
        "dataset_version": "v2.1",
    }
)

# Create model
rngs = nnx.Rngs(42)
model = create_model(config, rngs=rngs)
```

### ⚠️ Parameter Handling Guidelines

**IMPORTANT**: Always use the `parameters` field for model configuration, NOT `metadata`:

```python
# ✅ CORRECT - Use parameters field
config = ModelConfiguration(
    parameters={
        "beta": 1.0,
        "kl_weight": 0.5,
    }
)

# ❌ WRONG - Don't nest model params in metadata
config = ModelConfiguration(
    metadata={"vae_params": {"beta": 1.0}}  # DON'T DO THIS
)
```

### Model Types and Configurations

#### VAE Models

```python
# Standard VAE
vae_config = ModelConfiguration(
    name="vae",
    model_class="artifex.generative_models.models.vae.VAE",
    input_dim=(28, 28, 1),
    hidden_dims=[512, 256],
    output_dim=64,  # latent_dim
    parameters={
        "encoder_type": "cnn",  # or "mlp"
        "decoder_type": "cnn",  # or "mlp"
        "kl_weight": 1.0,
    }
)

# Beta-VAE
beta_vae_config = ModelConfiguration(
    name="beta_vae",
    model_class="artifex.generative_models.models.vae.BetaVAE",
    input_dim=(64, 64, 3),
    hidden_dims=[256, 128, 64],
    output_dim=32,
    parameters={
        "beta": 4.0,
        "beta_warmup_steps": 1000,
    }
)
```

#### GAN Models

```python
# DCGAN
gan_config = ModelConfiguration(
    name="dcgan",
    model_class="artifex.generative_models.models.gan.DCGAN",
    input_dim=100,  # latent_dim
    hidden_dims=[256, 128, 64],
    output_dim=(3, 32, 32),  # image shape
    parameters={
        "discriminator_features": [64, 128, 256],
        "use_batch_norm": True,
        "dropout_rate": 0.2,
    }
)

# Create GAN
gan = create_model(gan_config, rngs=rngs)
```

#### Diffusion Models

```python
# DDPM
diffusion_config = ModelConfiguration(
    name="ddpm",
    model_class="artifex.generative_models.models.diffusion.DDPM",
    input_dim=(32, 32, 3),
    hidden_dims=[128, 256, 512, 256, 128],
    output_dim=(32, 32, 3),
    activation="gelu",
    parameters={
        "noise_steps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "cosine",
    }
)

# Diffusion models need a backbone function
def backbone_fn(x, time_emb, rngs):
    # Your UNet or other backbone implementation
    return processed_x

diffusion = create_model(diffusion_config, rngs=rngs, backbone_fn=backbone_fn)
```

### Modality Adaptation

The factory supports modality-specific adaptations:

```python
# Create a VAE adapted for protein data
protein_vae = create_model(
    config,
    modality="protein",  # Apply protein-specific adaptations
    rngs=rngs
)

# Available modalities:
# - "protein": Protein structure modeling
# - "image": Image generation
# - "text": Text generation
# - "audio": Audio synthesis
# - "tabular": Tabular data
# - "multi_modal": Multi-modal learning
```

## Model Zoo

Pre-configured models are available through the model zoo:

```python
from artifex.generative_models.zoo import zoo

# List available configurations
configs = zoo.list_configs()
# ['vae_mnist', 'gan_cifar10', 'ebm_cifar', 'pixel_cnn_cifar10', ...]

# List by category
vision_configs = zoo.list_configs(category="vision")
protein_configs = zoo.list_configs(category="protein")

# Get configuration details
info = zoo.get_info("vae_mnist")

# Create model from zoo
model = zoo.create_model("vae_mnist", rngs=rngs)

# Create with overrides
model = zoo.create_model(
    "gan_cifar10",
    rngs=rngs,
    hidden_dims=[1024, 512, 256, 128],  # Override architecture
    learning_rate=0.0001,  # Override training params
)
```

### Available Zoo Configurations

#### Vision Models

- `vae_mnist`: VAE optimized for MNIST
- `gan_mnist`: DCGAN for MNIST
- `gan_cifar10`: DCGAN for CIFAR-10
- `ebm_cifar`: Energy-based model for CIFAR-10
- `pixel_cnn_cifar10`: PixelCNN for CIFAR-10
- `diffusion_celeba`: Diffusion model for CelebA

#### Protein Models

- `vae_protein`: VAE for protein sequences
- `diffusion_foldingdiff`: Diffusion model for protein structure

## Adding New Model Types

### 1. Create a New Builder

Create a new builder in `factory/builders/`:

```python
from artifex.generative_models.factory.registry import ModelBuilder
from artifex.generative_models.core.configuration import ModelConfiguration
from flax import nnx
from typing import Any

class MyModelBuilder(ModelBuilder):
    """Builder for my custom model type."""

    def build(
        self,
        config: ModelConfiguration,
        *,
        rngs: nnx.Rngs,
        **kwargs
    ) -> Any:
        """Build model from configuration."""
        # Parse model class
        module_path, class_name = config.model_class.rsplit(".", 1)

        # Import the model class
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # Prepare configuration
        modified_config = self._prepare_config(config, class_name)

        # Create and return model
        return model_class(config=modified_config, rngs=rngs, **kwargs)

    def _prepare_config(self, config, class_name):
        """Prepare configuration for specific model variant.

        Follow the clean parameter handling pattern:
        1. Extract existing parameters once
        2. Build complete parameters dict in one step
        3. Always preserve user parameters
        """
        config_dict = config.model_dump()

        # Get existing parameters or create empty dict
        existing_params = config.parameters or {}

        # Create new parameters dict with both existing and computed values
        config_dict["parameters"] = {
            **existing_params,  # Always preserve user parameters
            # Add defaults for missing parameters
            "my_param": existing_params.get("my_param", "default"),
            "another_param": existing_params.get("another_param", 100),
        }

        return ModelConfiguration(**config_dict)
```

### 2. Register the Builder

In `factory/core.py`, add to `_register_default_builders()`:

```python
from artifex.generative_models.factory.builders.my_model import MyModelBuilder

self.registry.register("my_model", MyModelBuilder())
```

### 3. Add Model Type Detection

Update `_extract_model_type()` in `factory/core.py`:

```python
patterns = [
    # ... existing patterns ...
    (r".*MyModel$", "my_model"),
]
```

## Adding Zoo Configurations

Create a YAML file in `zoo/configs/{category}/`:

```yaml
name: my_model_config
model_class: artifex.generative_models.models.my_module.MyModel
type: model
description: My custom model configuration
version: 1.0.0

# Architecture
input_dim: [28, 28, 1]
hidden_dims: [256, 128]
output_dim: 64
activation: relu

# Model parameters (functional configuration)
parameters:
  param1: value1
  param2: value2

# Metadata (non-functional information)
metadata:
  training_defaults:
    batch_size: 128
    learning_rate: 0.001
    epochs: 100
  experiment_notes: "Initial configuration for testing"

tags:
  - category
  - model_type
  - dataset
```

## Error Handling

The factory provides clear error messages:

```python
# Invalid model type
config = ModelConfiguration(
    model_class="artifex.generative_models.models.unknown.Unknown",
    ...
)
create_model(config, rngs=rngs)
# ValueError: No builder registered for type 'unknown'. Available types: vae, gan, ...

# Missing required configuration
config = ModelConfiguration(
    model_class="",  # Empty
    ...
)
create_model(config, rngs=rngs)
# ValueError: model_class must be specified in configuration

# Invalid modality
create_model(config, modality="nonexistent", rngs=rngs)
# ValueError: Failed to apply modality 'nonexistent': ...
```

## Migration Guide

### From Old Factory Functions

```python
# Old way (DEPRECATED)
from artifex.generative_models.factories.vae import create_vae
vae = create_vae(config=config, rngs=rngs)

# New way
from artifex.generative_models.factory import create_model
vae = create_model(config, rngs=rngs)
```

### From Dataset-Specific Factories

```python
# Old way (DEPRECATED)
from artifex.generative_models.factories.vae import create_vae_for_mnist
vae = create_vae_for_mnist(rngs=rngs)

# New way
from artifex.generative_models.zoo import zoo
vae = zoo.create_model("vae_mnist", rngs=rngs)
```

### From Protein-Specific Factories

```python
# Old way (DEPRECATED)
from artifex.generative_models.factories.protein import create_protein_vae
protein_vae = create_protein_vae(config=config, rngs=rngs)

# New way
from artifex.generative_models.factory import create_model
protein_vae = create_model(config, modality="protein", rngs=rngs)
```

## Testing

Test your models with the new factory:

```python
import pytest
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import ModelConfiguration
from flax import nnx

def test_create_model():
    config = ModelConfiguration(
        name="test_model",
        model_class="artifex.generative_models.models.vae.VAE",
        input_dim=(28, 28, 1),
        hidden_dims=[128],
        output_dim=32,
    )

    rngs = nnx.Rngs(42)
    model = create_model(config, rngs=rngs)

    assert model is not None
    assert hasattr(model, "encode")
    assert hasattr(model, "decode")
    assert model.latent_dim == 32
```

## Best Practices

1. **Always use ModelConfiguration**: Never pass raw dictionaries
2. **Use `parameters` for model config**: Put functional parameters in the `parameters` field, not `metadata`
3. **Use `metadata` for documentation**: Only non-functional information goes in `metadata`
4. **Use the zoo for standard models**: Don't recreate common configurations
5. **Apply modalities when needed**: Use the `modality` parameter for domain-specific adaptations
6. **Handle errors gracefully**: The factory provides detailed error messages
7. **Preserve user parameters**: Builders should always preserve user-provided parameters with `**existing_params`

## Technical Details

### Model Type Detection

The factory automatically detects the model type from the `model_class` path:

1. First, it looks for the module name (e.g., `models.vae.VAE` → `vae`)
2. If not found, it uses pattern matching on the class name
3. Falls back to "unknown" if no pattern matches

### Builder Pattern

Each builder:

1. Parses the model class from the configuration
2. Imports the model class dynamically
3. Prepares configuration with model-specific parameters
4. Instantiates the model with proper arguments

### Configuration Preparation

Builders MUST follow the clean parameter pattern:

```python
def _prepare_config(self, config: ModelConfiguration, class_name: str) -> ModelConfiguration:
    config_dict = config.model_dump()

    # Extract existing parameters once
    existing_params = config.parameters or {}

    # Build complete parameters dict in one step
    config_dict["parameters"] = {
        **existing_params,  # Preserve user parameters
        # Add defaults
        "param1": existing_params.get("param1", default1),
        "param2": existing_params.get("param2", default2),
    }

    return ModelConfiguration(**config_dict)
```

Key principles:

- Extract parameters from `config.parameters`, NOT `config.metadata`
- Always preserve user parameters with `**existing_params`
- Build the complete parameters dict in one step
- Never use nested metadata patterns

## Support

For issues or questions:

- Check the test files in `tests/artifex/generative_models/factory/`
- Review builder implementations in `factory/builders/`
- Consult the unified configuration documentation
