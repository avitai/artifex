# Base Configs

The base configuration layer is exposed through `artifex.configs`.

## Core Types

- `BaseConfig`: shared frozen-dataclass foundation for named runtime configs
- `ExperimentConfig`: composite config that combines model, training, data, and optional evaluation configs
- `ExperimentTemplateConfig`: typed reference template for retained experiment YAML assets

## Shared Responsibilities

`BaseConfig` provides the shared runtime metadata used by named runtime configs:

- `name`
- `description`
- `tags`
- `metadata`

All typed configs inherit the same dataclass serialization helpers:

- `from_dict()`
- `to_dict()`
- `from_yaml()`
- `to_yaml()`

## Public Import

```python
from artifex.configs import (
    BaseConfig,
    ExperimentConfig,
    ExperimentTemplateConfig,
)
```

`ExperimentConfig` is a typed runtime object built from already-instantiated
component configs. `ExperimentTemplateConfig` is a typed retained template
document: it uses the same frozen-dataclass serialization helpers, but it is
not a `BaseConfig` and does not carry runtime `name` metadata. Those retained
experiment templates use explicit placeholder roots for output, data, and cache
paths.
