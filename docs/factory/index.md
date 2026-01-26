# Model Factory

The factory module provides a centralized, type-safe interface for creating generative models in Artifex. It uses dataclass-based configurations to determine model type automatically, eliminating the need for string-based model class specifications.

## Overview

<div class="grid cards" markdown>

- :material-factory:{ .lg .middle } **Unified Interface**

    ---

    Single `create_model()` function for all model types

- :material-shield-check:{ .lg .middle } **Type-Safe**

    ---

    Dataclass configs with automatic validation

- :material-puzzle:{ .lg .middle } **Extensible**

    ---

    Register custom builders for new model types

- :material-view-module:{ .lg .middle } **Modality Support**

    ---

    Optional modality adaptation for domain-specific models

</div>

## Quick Start

### Basic Model Creation

```python
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import (
    VAEConfig,
    EncoderConfig,
    DecoderConfig,
)
from flax import nnx

# Create configuration
encoder = EncoderConfig(
    name="encoder",
    input_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(256, 128),
    activation="relu",
)
decoder = DecoderConfig(
    name="decoder",
    output_shape=(28, 28, 1),
    latent_dim=32,
    hidden_dims=(128, 256),
    activation="relu",
)
config = VAEConfig(
    name="my_vae",
    encoder=encoder,
    decoder=decoder,
    kl_weight=1.0,
)

# Create model - type is inferred from config
rngs = nnx.Rngs(params=42, dropout=43, sample=44)
model = create_model(config, rngs=rngs)
```

### Model Type Inference

The factory automatically infers model type from the configuration class:

| Config Class | Model Type | Created Model |
|--------------|------------|---------------|
| `VAEConfig`, `BetaVAEConfig`, `ConditionalVAEConfig`, `VQVAEConfig` | `vae` | VAE variants |
| `GANConfig`, `DCGANConfig`, `WGANConfig`, `LSGANConfig` | `gan` | GAN variants |
| `DiffusionConfig`, `DDPMConfig`, `ScoreDiffusionConfig` | `diffusion` | Diffusion models |
| `EBMConfig`, `DeepEBMConfig` | `ebm` | Energy-based models |
| `FlowConfig` | `flow` | Normalizing flows |
| `AutoregressiveConfig`, `TransformerConfig`, `PixelCNNConfig`, `WaveNetConfig` | `autoregressive` | Autoregressive models |
| `GeometricConfig`, `PointCloudConfig`, `MeshConfig`, `VoxelConfig`, `GraphConfig` | `geometric` | Geometric models |

## API Reference

### `create_model`

The main function for model creation.

```python
def create_model(
    config: DataclassConfig,
    *,
    modality: str | None = None,
    rngs: nnx.Rngs,
    **kwargs,
) -> Any:
    """Create a model from configuration.

    Args:
        config: Dataclass model configuration (DDPMConfig, VAEConfig, etc.)
        modality: Optional modality for adaptation ('image', 'text', 'audio', etc.)
        rngs: Random number generators
        **kwargs: Additional arguments passed to the builder

    Returns:
        Created model instance

    Raises:
        TypeError: If config is not a supported dataclass config
        ValueError: If builder not found for model type
    """
```

**Example:**

```python
from artifex.generative_models.factory import create_model
from artifex.generative_models.core.configuration import DDPMConfig, UNetBackboneConfig, NoiseScheduleConfig

# Create diffusion model config
backbone = UNetBackboneConfig(
    name="unet",
    in_channels=3,
    out_channels=3,
    base_channels=64,
    channel_mults=(1, 2, 4),
)
noise_schedule = NoiseScheduleConfig(
    name="schedule",
    schedule_type="linear",
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=2e-2,
)
config = DDPMConfig(
    name="ddpm",
    input_shape=(3, 32, 32),
    backbone=backbone,
    noise_schedule=noise_schedule,
)

# Create model
model = create_model(config, rngs=rngs)
```

### `create_model_with_extensions`

Create a model with extensions for enhanced functionality.

```python
def create_model_with_extensions(
    config: DataclassConfig,
    *,
    extensions_config: dict[str, ExtensionConfig] | None = None,
    modality: str | None = None,
    rngs: nnx.Rngs,
    **kwargs,
) -> tuple[Any, dict[str, ModelExtension]]:
    """Create a model and its extensions from configuration.

    Returns:
        Tuple of (model, extensions_dict)
    """
```

**Example:**

```python
from artifex.generative_models.factory import create_model_with_extensions

# Create model with extensions
model, extensions = create_model_with_extensions(
    config,
    extensions_config={
        "augmentation": augmentation_config,
        "regularization": reg_config,
    },
    rngs=rngs,
)
```

### `ModelFactory`

The underlying factory class for advanced usage.

```python
class ModelFactory:
    """Centralized factory for all generative models."""

    def __init__(self):
        """Initialize with default builders."""

    def create(
        self,
        config: DataclassConfig,
        *,
        modality: str | None = None,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> Any:
        """Create a model from dataclass configuration."""
```

## Builders

Each model family has a dedicated builder that handles model instantiation:

### VAE Builder

Creates VAE variants based on configuration type:

- `VAEConfig` → `VAE`
- `BetaVAEConfig` → `BetaVAE`
- `ConditionalVAEConfig` → `ConditionalVAE`
- `VQVAEConfig` → `VQVAE`

[:octicons-arrow-right-24: VAE Builder Reference](vae.md)

### GAN Builder

Creates GAN variants:

- `GANConfig` → `GAN`
- `DCGANConfig` → `DCGAN`
- `WGANConfig` → `WGAN`
- `LSGANConfig` → `LSGAN`

[:octicons-arrow-right-24: GAN Builder Reference](gan.md)

### Diffusion Builder

Creates diffusion models:

- `DDPMConfig` → `DDPMModel`
- `ScoreDiffusionConfig` → `ScoreDiffusionModel`
- `DiffusionConfig` → `DiffusionModel`

[:octicons-arrow-right-24: Diffusion Builder Reference](diffusion.md)

### Flow Builder

Creates normalizing flows:

- `FlowConfig` → `NormalizingFlow`

[:octicons-arrow-right-24: Flow Builder Reference](flow.md)

### EBM Builder

Creates energy-based models:

- `EBMConfig` → `EBM`
- `DeepEBMConfig` → `DeepEBM`

[:octicons-arrow-right-24: EBM Builder Reference](ebm.md)

### Autoregressive Builder

Creates autoregressive models:

- `TransformerConfig` → `Transformer`
- `PixelCNNConfig` → `PixelCNN`
- `WaveNetConfig` → `WaveNet`

[:octicons-arrow-right-24: Autoregressive Builder Reference](autoregressive.md)

### Geometric Builder

Creates geometric models:

- `PointCloudConfig` → `PointCloudModel`
- `MeshConfig` → `MeshModel`
- `VoxelConfig` → `VoxelModel`
- `GraphConfig` → `GraphModel`

[:octicons-arrow-right-24: Geometric Builder Reference](geometric.md)

## Registry

The model type registry manages builder registration:

```python
from artifex.generative_models.factory.registry import ModelTypeRegistry

# Create custom registry
registry = ModelTypeRegistry()

# Register custom builder
registry.register("custom_type", CustomBuilder())

# Get builder
builder = registry.get_builder("custom_type")
```

[:octicons-arrow-right-24: Registry Reference](registry.md)

## Modality Adaptation

The factory supports optional modality adaptation for domain-specific models:

```python
# Create image-adapted model
model = create_model(config, modality="image", rngs=rngs)

# Create text-adapted model
model = create_model(config, modality="text", rngs=rngs)

# Create audio-adapted model
model = create_model(config, modality="audio", rngs=rngs)
```

**Available Modalities:**

- `image`: Convolutional layers, FID/IS metrics
- `text`: Tokenization, perplexity metrics
- `audio`: Spectrograms, MFCCs
- `protein`: Structure prediction, sequence modeling
- `geometric`: Point clouds, meshes

## Custom Builders

Create custom builders for new model types:

```python
from artifex.generative_models.factory.registry import ModelBuilder
from flax import nnx

class CustomBuilder(ModelBuilder):
    """Builder for custom model type."""

    def build(self, config, *, rngs: nnx.Rngs, **kwargs):
        """Build the model from configuration."""
        return CustomModel(config, rngs=rngs, **kwargs)

# Register with factory
from artifex.generative_models.factory import ModelFactory

factory = ModelFactory()
factory.registry.register("custom", CustomBuilder())
```

## Best Practices

### DO

- ✅ Use dataclass configs instead of dictionaries
- ✅ Validate configs before passing to factory
- ✅ Use type hints for better IDE support
- ✅ Pass all required RNG streams to `nnx.Rngs`

### DON'T

- ❌ Pass dictionary configs (will raise `TypeError`)
- ❌ Use string-based model class specifications
- ❌ Forget to provide `rngs` parameter

## Error Handling

The factory provides clear error messages:

```python
# TypeError: Dictionary configs not supported
create_model({"model_class": "vae"}, rngs=rngs)
# Raises: TypeError: Expected dataclass config, got dict.

# TypeError: Unknown config type
create_model(UnknownConfig(), rngs=rngs)
# Raises: TypeError: Unknown config type: UnknownConfig

# ValueError: Builder not found
# (Only possible with custom registries)
```

## Module Reference

| Module | Description |
|--------|-------------|
| [core](core.md) | Core factory implementation and `create_model` function |
| [registry](registry.md) | Model type registry and builder base class |
| [vae](vae.md) | VAE model builder |
| [gan](gan.md) | GAN model builder |
| [diffusion](diffusion.md) | Diffusion model builder |
| [flow](flow.md) | Normalizing flow builder |
| [ebm](ebm.md) | Energy-based model builder |
| [autoregressive](autoregressive.md) | Autoregressive model builder |
| [geometric](geometric.md) | Geometric model builder |

## Related Documentation

- [Configuration System](../user-guide/training/configuration.md) - Understanding dataclass configs
- [Model Gallery](../models/index.md) - Available model implementations
- [Factory Guide](../guides/factory.md) - Detailed factory usage guide
