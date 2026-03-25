# Config Loader

`artifex.configs` exposes a small set of typed loader helpers for the retained
runtime config assets under `src/artifex/configs/`.

## Public Imports

```python
from artifex.configs import (
    get_config_path,
    get_data_config,
    get_inference_config,
    get_protein_extensions_config,
    get_training_config,
    load_experiment_config,
)
```

## Current Contract

- `get_config_path(name, config_type=...)` resolves the retained config asset path
- `get_data_config(name)` returns a typed `DataConfig`
- `get_training_config(name)` returns a typed `TrainingConfig`
- `get_inference_config(name)` returns the narrowest typed inference config that matches the YAML shape
- `get_protein_extensions_config(name="protein")` returns a typed `ProteinExtensionsConfig`
- `load_experiment_config(name)` returns a typed `ExperimentTemplateConfig`
- concrete frozen dataclass configs can also load themselves directly with `TrainingConfig.from_yaml(...)` and the corresponding `from_yaml()` method on other config classes

## Example

```python
from artifex.configs import (
    ExperimentTemplateConfig,
    TrainingConfig,
    get_inference_config,
    get_protein_extensions_config,
    get_training_config,
    load_experiment_config,
)

training_config = get_training_config("protein_diffusion_training")
training_config_from_yaml = TrainingConfig.from_yaml(
    "src/artifex/configs/defaults/training/protein_diffusion_training.yaml"
)
inference_config = get_inference_config("protein_diffusion_inference")
protein_extensions = get_protein_extensions_config("protein")
experiment_template = load_experiment_config("protein_diffusion_experiment")
```

The loader rejects empty YAML files and non-mapping top-level YAML payloads.
Retained shipped experiment templates resolve to `ExperimentTemplateConfig`
objects and use explicit placeholder roots such as `{ARTIFEX_OUTPUT_ROOT}`,
`{ARTIFEX_DATA_ROOT}`, and `{ARTIFEX_CACHE_ROOT}` for machine-specific paths.
Typed loader helpers raise the repo-owned `ConfigError` hierarchy for missing
files, parse failures, and typed schema validation failures.
