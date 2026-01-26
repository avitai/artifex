# Configuration System

Unified configuration management for all Artifex components, providing type-safe, validated configurations with schema support, templates, and utilities.

## Overview

<div class="grid cards" markdown>

- :material-cog:{ .lg .middle } **Schema-Based Configs**

    ---

    Pydantic-based configuration classes with validation and type safety

- :material-file-document:{ .lg .middle } **Templates**

    ---

    Pre-built configuration templates for common use cases

- :material-swap-horizontal:{ .lg .middle } **Conversion**

    ---

    Convert between YAML, JSON, and Python configuration formats

- :material-check-circle:{ .lg .middle } **Validation**

    ---

    Automatic validation with clear error messages

</div>

## Quick Start

### Loading Configuration

```python
from artifex.configs import load_config, ConfigLoader

# Load from YAML file
config = load_config("config.yaml")

# Load with validation
loader = ConfigLoader(schema="training")
config = loader.load("training_config.yaml")
```

### Creating Configurations

```python
from artifex.configs.schema import TrainingConfig, DataConfig

# Create training configuration
training_config = TrainingConfig(
    batch_size=128,
    num_epochs=100,
    learning_rate=1e-3,
    optimizer="adamw",
    scheduler="cosine",
)

# Create data configuration
data_config = DataConfig(
    dataset="cifar10",
    train_split="train",
    val_split="test",
    augmentation=True,
)
```

## Schema Reference

### Base Schema

Foundation configuration classes.

```python
from artifex.configs.schema import BaseConfig

class BaseConfig:
    """Base configuration with common fields."""
    name: str
    seed: int = 42
    dtype: str = "float32"
    debug: bool = False
```

[:octicons-arrow-right-24: Base Schema](base.md)

### Training Schema

Training-specific configurations.

```python
from artifex.configs.schema import TrainingConfig

config = TrainingConfig(
    batch_size=128,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=0.01,
    gradient_clip=1.0,
    mixed_precision=True,
    gradient_accumulation_steps=4,
)
```

[:octicons-arrow-right-24: Training Schema](training.md)

### Data Schema

Data loading and preprocessing configurations.

```python
from artifex.configs.schema import DataConfig

config = DataConfig(
    dataset="imagenet",
    root="/path/to/data",
    batch_size=256,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
)
```

[:octicons-arrow-right-24: Data Schema](data.md)

### Distributed Schema

Multi-device training configurations.

```python
from artifex.configs.schema import DistributedConfig

config = DistributedConfig(
    strategy="data_parallel",
    num_devices=4,
    mesh_shape=(2, 2),
    axis_names=("data", "model"),
)
```

[:octicons-arrow-right-24: Distributed Schema](distributed.md)

### Inference Schema

Inference and serving configurations.

```python
from artifex.configs.schema import InferenceConfig

config = InferenceConfig(
    batch_size=1,
    max_length=512,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
)
```

[:octicons-arrow-right-24: Inference Schema](inference.md)

### Hyperparameter Schema

Hyperparameter search configurations.

```python
from artifex.configs.schema import HyperparamConfig

config = HyperparamConfig(
    search_space={
        "learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-2},
        "batch_size": {"type": "choice", "values": [32, 64, 128, 256]},
    },
    num_trials=100,
    metric="val_loss",
    direction="minimize",
)
```

[:octicons-arrow-right-24: Hyperparameter Schema](hyperparam.md)

### Extensions Schema

Extension-specific configurations.

```python
from artifex.configs.schema import ExtensionsConfig

config = ExtensionsConfig(
    protein={
        "max_length": 256,
        "include_structure": True,
    },
    geometric={
        "equivariant": True,
        "rotation_order": 2,
    },
)
```

[:octicons-arrow-right-24: Extensions Schema](extensions.md)

## Configuration Utilities

### Config Loader

Load configurations from various sources.

```python
from artifex.configs.utils import ConfigLoader

loader = ConfigLoader()

# Load from file
config = loader.load("config.yaml")

# Load from dict
config = loader.from_dict({"batch_size": 128})

# Load with environment variable substitution
config = loader.load("config.yaml", resolve_env=True)
```

[:octicons-arrow-right-24: Config Loader](config_loader.md)

### Validation

Validate configurations against schemas.

```python
from artifex.configs.utils import validate_config

# Validate against schema
errors = validate_config(config, schema="training")

if errors:
    for error in errors:
        print(f"Validation error: {error}")
```

[:octicons-arrow-right-24: Validation](validation.md)

### Conversion

Convert between configuration formats.

```python
from artifex.configs.utils import convert_config

# YAML to JSON
convert_config("config.yaml", "config.json", format="json")

# To Python dict
config_dict = convert_config("config.yaml", format="dict")

# To Pydantic model
config_model = convert_config("config.yaml", format="pydantic")
```

[:octicons-arrow-right-24: Conversion](conversion.md)

### Merge

Merge multiple configurations.

```python
from artifex.configs.utils import merge_configs

# Merge configs with override priority
merged = merge_configs(
    base_config,
    override_config,
    strategy="deep",  # Deep merge nested dicts
)

# Merge from files
merged = merge_configs(
    "base.yaml",
    "experiment.yaml",
    "local.yaml",  # Highest priority
)
```

[:octicons-arrow-right-24: Merge](merge.md)

### Templates

Use pre-built configuration templates.

```python
from artifex.configs.utils import get_template

# Get VAE training template
template = get_template("vae_training")

# Customize template
template.model.latent_dim = 64
template.training.learning_rate = 1e-4
```

[:octicons-arrow-right-24: Templates](templates.md)

### I/O

Configuration file operations.

```python
from artifex.configs.utils import save_config, load_config

# Save configuration
save_config(config, "experiment_config.yaml")

# Load with comments preserved
config = load_config("config.yaml", preserve_comments=True)
```

[:octicons-arrow-right-24: I/O](io.md)

### Error Handling

Configuration error handling utilities.

```python
from artifex.configs.utils import ConfigError, handle_config_error

try:
    config = load_config("config.yaml")
except ConfigError as e:
    handle_config_error(e, show_suggestions=True)
```

[:octicons-arrow-right-24: Error Handling](error_handling.md)

## CLI Configuration

Command-line interface configuration.

```python
from artifex.configs import CLIConfig

cli_config = CLIConfig(
    verbose=True,
    log_level="INFO",
    output_dir="./outputs",
)
```

[:octicons-arrow-right-24: CLI Config](cli.md)

## Extension Configuration

Configuration for domain extensions.

```python
from artifex.configs import ExtensionConfig

extension_config = ExtensionConfig(
    name="protein",
    enabled=True,
    settings={
        "use_esm": True,
        "structure_prediction": True,
    },
)
```

[:octicons-arrow-right-24: Extension Config](extension_config.md)

## Module Reference

| Category | Modules |
|----------|---------|
| **Schema** | [base](base.md), [data](data.md), [distributed](distributed.md), [extensions](extensions.md), [hyperparam](hyperparam.md), [inference](inference.md), [training](training.md) |
| **Utils** | [config_loader](config_loader.md), [conversion](conversion.md), [error_handling](error_handling.md), [io](io.md), [merge](merge.md), [templates](templates.md), [validation](validation.md) |
| **Other** | [cli](cli.md), [extension_config](extension_config.md) |

## Related Documentation

- [Configuration Guide](../user-guide/training/configuration.md) - Complete configuration guide
- [Training Guide](../user-guide/training/training-guide.md) - Using configs in training
- [Core Configuration](../core/unified.md) - Core configuration system
