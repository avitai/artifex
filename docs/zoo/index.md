# Model Zoo

The Model Zoo provides pre-configured model configurations for common use cases, enabling quick experimentation with state-of-the-art generative models without writing configuration from scratch.

## Overview

<div class="grid cards" markdown>

- :material-paw:{ .lg .middle } **Pre-configured Models**

    ---

    Ready-to-use configurations for VAEs, GANs, Diffusion, and more

- :material-tune:{ .lg .middle } **Easy Customization**

    ---

    Override any configuration parameter while keeping defaults

- :material-file-document:{ .lg .middle } **YAML-based**

    ---

    Configurations stored in human-readable YAML files

- :material-tag:{ .lg .middle } **Categorized**

    ---

    Filter models by category (vision, protein, audio, etc.)

</div>

## Quick Start

### Using Pre-configured Models

```python
from artifex.generative_models.zoo import zoo
from flax import nnx

# List available configurations
available = zoo.list_configs()
print(f"Available models: {available}")

# Create a model from zoo configuration
rngs = nnx.Rngs(params=42, dropout=43, sample=44)
model = zoo.create_model("vae-mnist", rngs=rngs)

# Create with overrides
model = zoo.create_model(
    "vae-mnist",
    rngs=rngs,
    latent_dim=64,  # Override latent dimension
)
```

### Getting Configuration Info

```python
# Get detailed info about a configuration
info = zoo.get_info("vae-mnist")
print(f"Model: {info['name']}")
print(f"Description: {info['description']}")
print(f"Tags: {info['tags']}")

# Get the raw configuration
config = zoo.get_config("vae-mnist")
```

### Filtering by Category

```python
# List models in a specific category
vision_models = zoo.list_configs(category="vision")
protein_models = zoo.list_configs(category="protein")
```

## API Reference

### `ModelZoo`

The main class for managing pre-configured models.

```python
class ModelZoo:
    """Registry for pre-configured model configurations."""

    def __init__(self):
        """Initialize the model zoo."""

    def get_config(self, name: str) -> ModelConfig:
        """Get a pre-configured model configuration.

        Args:
            name: Name of the configuration

        Returns:
            Model configuration

        Raises:
            KeyError: If configuration not found
        """

    def list_configs(self, category: str | None = None) -> list[str]:
        """List all available configurations.

        Args:
            category: Optional category filter

        Returns:
            List of configuration names
        """

    def create_model(
        self,
        name: str,
        *,
        rngs: nnx.Rngs,
        modality: str | None = None,
        **overrides
    ) -> Any:
        """Create a model from zoo configuration.

        Args:
            name: Name of the configuration
            rngs: Random number generators
            modality: Optional modality adaptation
            **overrides: Configuration overrides

        Returns:
            Created model instance
        """

    def register_config(self, config: ModelConfig) -> None:
        """Register a configuration in the zoo."""

    def get_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a configuration."""
```

### Global Instance

A global zoo instance is available for convenience:

```python
from artifex.generative_models.zoo import zoo

# Use the global instance
models = zoo.list_configs()
```

## Adding Custom Configurations

### Register at Runtime

```python
from artifex.generative_models.zoo import zoo
from artifex.generative_models.core.configuration import ModelConfig

# Create a custom configuration
my_config = ModelConfig(
    name="my-custom-vae",
    description="Custom VAE for my project",
    model_class="vae",
    input_dim=784,
    hidden_dims=(512, 256, 128),
    latent_dim=32,
    metadata={"tags": ["custom", "vision"]},
)

# Register in the zoo
zoo.register_config(my_config)

# Now you can use it
model = zoo.create_model("my-custom-vae", rngs=rngs)
```

### Add YAML Configuration

Create a YAML file in `zoo/configs/<category>/<name>.yaml`:

```yaml
# zoo/configs/vision/my-vae.yaml
name: my-vae
description: Custom VAE for image generation
model_class: vae
version: "1.0"
input_dim: 784
output_dim: 784
hidden_dims: [512, 256, 128]
latent_dim: 32
metadata:
  tags:
    - vision
    - vae
  author: My Name
```

## Available Categories

| Category | Description | Example Models |
|----------|-------------|----------------|
| `vision` | Image generation models | VAE-MNIST, DCGAN-CIFAR |
| `protein` | Protein structure models | ProteinVAE, SE3Flow |
| `audio` | Audio generation models | WaveNet, AudioVAE |
| `text` | Text generation models | GPT-style, PixelCNN |
| `molecular` | Molecular generation | MolFlow, GraphVAE |

## Configuration Overrides

When creating a model, you can override any configuration parameter:

```python
# Override multiple parameters
model = zoo.create_model(
    "vae-mnist",
    rngs=rngs,
    latent_dim=128,
    hidden_dims=(1024, 512, 256),
    kl_weight=0.5,
)
```

Overrides that don't match existing fields are added to metadata:

```python
# Custom metadata
model = zoo.create_model(
    "vae-mnist",
    rngs=rngs,
    experiment_name="exp-001",  # Added to metadata
)
```

## Integration with Factory

The zoo uses the [Model Factory](../factory/index.md) internally:

```python
# These are equivalent:
model1 = zoo.create_model("vae-mnist", rngs=rngs)

# Manual approach
config = zoo.get_config("vae-mnist")
model2 = create_model(config, rngs=rngs)
```

## Best Practices

### DO

- ✅ Use the zoo for quick prototyping
- ✅ Override only the parameters you need
- ✅ Use categories to organize custom configs
- ✅ Include descriptive metadata in custom configs

### DON'T

- ❌ Modify zoo configs directly (use overrides instead)
- ❌ Use generic names that might conflict

## Related Documentation

- [Model Factory](../factory/index.md) - Low-level model creation
- [Configuration System](../user-guide/training/configuration.md) - Configuration details
- [Model Gallery](../models/index.md) - All available models
