# Configuration System

`artifex.configs` is the public convenience surface over the typed configuration
runtime in `artifex.generative_models.core.configuration`.

The configuration system uses frozen dataclass configs and typed helper
utilities.

## What This Package Exposes

- typed config classes such as `TrainingConfig`, `DistributedConfig`,
  `DataConfig`, and `InferenceConfig`
- config loading helpers such as `get_training_config`, `get_data_config`,
  and `load_experiment_config`
- versioning helpers such as `ConfigVersionRegistry` and
  `compute_config_hash`
- the supported built-in template surface via `template_manager`,
  `SIMPLE_TRAINING_TEMPLATE`, and `DISTRIBUTED_TEMPLATE`

## Typed Config Example

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

## Loading YAML

```python
from artifex.configs import TrainingConfig, get_training_config

training_from_file = TrainingConfig.from_yaml(
    "src/artifex/configs/defaults/training/protein_diffusion_training.yaml"
)
training_config = get_training_config("protein_diffusion_training")
```

`load_experiment_config()` returns a typed `ExperimentTemplateConfig`. The
retained shipped experiment templates still reference other config assets and
use explicit placeholder roots such as `{ARTIFEX_OUTPUT_ROOT}`,
`{ARTIFEX_DATA_ROOT}`, and `{ARTIFEX_CACHE_ROOT}` for machine-specific paths.
Those retained template documents keep their own `experiment_name` schema
instead of being forced into the named runtime `BaseConfig` contract.

## Using Templates

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

The supported built-ins are `simple_training` and `distributed_training`, and
they materialize typed frozen configs.

## Versioning

```python
from artifex.configs import (
    ConfigVersionRegistry,
    compute_config_hash,
    load_experiment_config,
)

registry = ConfigVersionRegistry("./temp/config_registry")
experiment_template = load_experiment_config("protein_diffusion_experiment")
version = registry.register(experiment_template, description="Protein diffusion baseline")
config_hash = compute_config_hash(experiment_template)
```

## Tests

```bash
uv run pytest tests/artifex/configs -q
uv run pytest tests/artifex/repo_contracts/test_config_surface_contracts.py -q
```

Package-local docs should stay aligned with the public `artifex.configs` export
surface and describe the symbols that are re-exported there as the supported
API.
