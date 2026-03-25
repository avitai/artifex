# Configuration API

The Artifex configuration backend is built on frozen dataclasses. Public code
should prefer `artifex.configs` for imports and retained config loading.

## Public Surfaces

- `artifex.configs`
  - reviewed convenience surface for config classes and retained config assets
- `artifex.generative_models.core.configuration`
  - lazy package surface for concrete config families used by runtime code

Internal serializer bases such as `ConfigDocument` stay in their concrete
modules and are not part of the normal user-facing config API.

## Runtime Config Families

Representative runtime config families include:

- `TrainingConfig`
- `OptimizerConfig`
- `SchedulerConfig`
- `DataConfig`
- `EvaluationConfig`
- `InferenceConfig`
- `DistributedConfig`
- model-family-specific configs such as `VAEConfig`, `DDPMConfig`, `EBMConfig`,
  and `PointCloudConfig`

There is no public catch-all generic model config on the supported
model-creation path. Runtime code should use the concrete family config that
matches the model being materialized.

All runtime configs share the same construction and validation pattern:

```python
from artifex.configs import TrainingConfig

training_config = TrainingConfig.from_yaml(
    "src/artifex/configs/defaults/training/protein_diffusion_training.yaml"
)
payload = training_config.to_dict()
```

## Retained Template Documents

Some retained config assets are reference templates rather than runtime config
objects. Those load as typed template documents such as
`ExperimentTemplateConfig`.

```python
from artifex.configs import ExperimentTemplateConfig, load_experiment_config

template = load_experiment_config("protein_diffusion_experiment")
assert isinstance(template, ExperimentTemplateConfig)
```

## Design Contract

- configs are immutable, slotted, keyword-only dataclasses
- `from_dict()` and `from_yaml()` are the canonical materialization paths
- validation lives in `__post_init__(self) -> None`
- nested config sections stay typed all the way through template generation and
  YAML loading
