# Configuration System

`artifex.configs` is the public convenience surface over the typed configuration
runtime in `artifex.generative_models.core.configuration`.

It exposes frozen dataclass configs, YAML loading helpers, template generation,
environment-aware config utilities, and config versioning helpers.

## Quick Start

### Create Typed Configs

```python
from artifex.configs import OptimizerConfig, SchedulerConfig, TrainingConfig

training_config = TrainingConfig(
    name="custom_training",
    batch_size=64,
    num_epochs=100,
    optimizer=OptimizerConfig(
        name="adamw",
        optimizer_type="adamw",
        learning_rate=2e-4,
    ),
    scheduler=SchedulerConfig(
        name="cosine",
        scheduler_type="cosine",
        warmup_steps=500,
    ),
)
```

### Load YAML Assets

```python
from artifex.configs import (
    ExperimentTemplateConfig,
    get_training_config,
    load_experiment_config,
    TrainingConfig,
)

raw_training = TrainingConfig.from_yaml(
    "src/artifex/configs/defaults/training/protein_diffusion_training.yaml"
)
training_config = get_training_config("protein_diffusion_training")
experiment_template = load_experiment_config("protein_diffusion_experiment")
```

The typed helpers such as `TrainingConfig.from_yaml()` and
`get_training_config()` validate against the current dataclass schema.
`load_experiment_config()` returns an `ExperimentTemplateConfig`, not a fully
resolved `ExperimentConfig`. The retained experiment templates still reference
other config assets and use explicit placeholder roots for machine-specific
output, data, and cache paths.

### Generate Supported Templates

```python
from artifex.configs import template_manager

training_template = template_manager.generate_config(
    "simple_training",
    batch_size=64,
    learning_rate=2e-4,
    scheduler_type="linear",
    total_steps=5000,
)
```

The supported built-in templates are:

- `simple_training`
- `distributed_training`

### Track Config Versions

```python
from artifex.configs import ConfigVersionRegistry, compute_config_hash, load_experiment_config

registry = ConfigVersionRegistry("./temp/config_registry")
config = load_experiment_config("protein_diffusion_experiment")
version = registry.register(config, description="Protein diffusion baseline")
config_hash = compute_config_hash(config)
```

## Public Surface

The top-level `artifex.configs` package re-exports the supported runtime API:

- typed configs such as `BaseConfig`, `TrainingConfig`,
  `DataConfig`, `InferenceConfig`, `DistributedConfig`, `ExperimentConfig`,
  and `ExperimentTemplateConfig`
- nested config types such as `OptimizerConfig` and `SchedulerConfig`
- hyperparameter search types such as `HyperparamSearchConfig`,
  `ParameterDistribution`, `CategoricalDistribution`, `UniformDistribution`,
  and `ChoiceDistribution`
- YAML and template loading helpers such as `TrainingConfig.from_yaml()`,
  `get_training_config`, `get_data_config`, `get_inference_config`, and
  `load_experiment_config`
- versioning helpers such as `ConfigVersionRegistry` and `compute_config_hash`
- template utilities such as `template_manager`, `SIMPLE_TRAINING_TEMPLATE`,
  and `DISTRIBUTED_TEMPLATE`

Prefer imports from `artifex.configs` over deeper re-export paths in docs and
examples.

## Current Docs Map

- [Base Config](base.md)
- [Training Config](training.md)
- [Data Config](data.md)
- [Distributed Config](distributed.md)
- [Inference Config](inference.md)
- [Hyperparameter Search](hyperparam.md)
- [Extensions](extensions.md)
- [Config Loader](config_loader.md)
- [Templates](templates.md)
- [Error Handling](error_handling.md)

## Related Documentation

- [Configuration Guide](../user-guide/training/configuration.md)
- [Training Guide](../user-guide/training/training-guide.md)
- [Factory Guide](../guides/factory.md)
