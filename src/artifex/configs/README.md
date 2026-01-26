# Configuration System

A type-safe configuration system for generative models, providing Pydantic-based schemas for model architectures, training, and inference.

## Key Features

- **Type Safety**: Pydantic-based schema validation with type hints
- **Hierarchical Configurations**: Base configs with model-specific extensions
- **Default Values**: Sensible defaults for all configuration options
- **Schema Validation**: Automatic validation of configuration values
- **Configuration Versioning**: Track and reproduce configurations
- **Command-Line Interface**: Manage configurations via CLI

## Directory Structure

```
configs/
├── defaults/            # Default configuration values
│   ├── data/            # Dataset configurations
│   ├── distributed/     # Distributed training configs
│   ├── env/             # Environment-specific configs
│   ├── inference/       # Inference configurations
│   ├── models/          # Model architecture configs
│   └── training/        # Training configurations
├── experiments/         # Full experiment configurations
├── schema/              # Configuration schema definitions
└── utils/               # Configuration utilities
```

## Usage Examples

### Loading a Configuration

```python
from artifex.generative_models.core.configuration import ModelConfiguration
from artifex.generative_models.factory import create_model

# Create a model configuration
config = ModelConfiguration(
    name="protein_diffusion",
    model_class="artifex.generative_models.models.diffusion.DDPMModel",
    input_dim=(1024, 3),
    hidden_dims=[256, 512, 256],
    output_dim=(1024, 3),
    parameters={
        "noise_steps": 1000,
        "beta_schedule": "cosine",
    }
)

# Create model from configuration
model = create_model(config, rngs=rngs)
```

### Creating a Configuration

```python
from artifex.configs import (
    PointCloudDiffusionConfig,
    TrainingConfig,
)

# Model configuration
model_config = PointCloudDiffusionConfig(
    name="custom_diffusion",
    model_dim=256,
    num_layers=8,
    num_heads=8,
    dropout=0.1,
    timesteps=1000,
    beta_schedule="linear",
)

# Training configuration
training_config = TrainingConfig(
    name="custom_training",
    batch_size=64,
    num_epochs=100,
)
```

### Using the CLI

```bash
# Create configuration from template
python -m artifex.generative_models.configs.cli create \
    templates/protein_diffusion.yaml \
    my_experiment.yaml \
    --override "model.model_dim=256"

# Validate configuration
python -m artifex.generative_models.configs.cli validate my_experiment.yaml

# Show configuration
python -m artifex.generative_models.configs.cli show my_experiment.yaml

# Version configuration
python -m artifex.generative_models.configs.cli version my_experiment.yaml \
    --registry ./config_registry \
    --description "My experiment"
```

### Configuration Versioning

```python
from artifex.generative_models.configs import ConfigVersionRegistry

# Create registry
registry = ConfigVersionRegistry("./config_registry")

# Register configuration
config = load_yaml_config("my_experiment.yaml")
version = registry.register(config, description="My experiment")

# Get configuration by version
config = registry.get_by_version("20230215-a1b2c3d4")
```

## Adding New Configuration Types

To add a new configuration type:

1. Create a new schema class in `schema/`
2. Add default configurations in `defaults/`
3. Update the imports in `__init__.py`
4. Add model registration if needed

## Testing

```bash
# Run configuration tests
./test.py standalone

# Run with coverage
./test.py standalone -c

# Run specific test
./test.py specific -f tests/standalone/test_distributed_config.py
```

For more details, see the [main documentation](../../docs/).
