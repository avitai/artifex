# Training Configuration

Artifex training is configured with typed frozen dataclasses. The supported
public import surface is `artifex.configs`.

## Core Rules

- runtime configs are `@dataclass(frozen=True, slots=True, kw_only=True)`
- validation happens in `__post_init__(self) -> None`
- YAML loading materializes the same typed dataclasses used at runtime
- training config owns nested optimizer and scheduler configs explicitly

## Main Training Types

```python
from artifex.configs import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
```

- `OptimizerConfig`: optimizer algorithm and optimizer-specific settings
- `SchedulerConfig`: optional learning-rate schedule
- `TrainingConfig`: batch size, epochs, checkpointing, logging, and the nested
  optimizer/scheduler objects

## Constructing A TrainingConfig

```python
from artifex.configs import OptimizerConfig, SchedulerConfig, TrainingConfig

optimizer = OptimizerConfig(
    name="adamw_optimizer",
    optimizer_type="adamw",
    learning_rate=3e-4,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
)

scheduler = SchedulerConfig(
    name="cosine_schedule",
    scheduler_type="cosine",
    warmup_steps=1000,
    min_lr_ratio=0.1,
)

training_config = TrainingConfig(
    name="diffusion_training",
    batch_size=64,
    num_epochs=200,
    optimizer=optimizer,
    scheduler=scheduler,
    gradient_clip_norm=1.0,
    save_frequency=5000,
    log_frequency=100,
)
```

## Loading Retained YAML Assets

The retained shipped defaults under `src/artifex/configs/defaults/` load as
typed dataclasses:

```python
from artifex.configs import TrainingConfig, get_training_config

typed_default = get_training_config("protein_diffusion_training")
same_config = TrainingConfig.from_yaml(
    "src/artifex/configs/defaults/training/protein_diffusion_training.yaml"
)
```

## Experiment Templates

Experiment YAML assets under `src/artifex/configs/experiments/` are retained
reference templates, not direct runtime `TrainingConfig` or `ExperimentConfig`
objects.

```python
from artifex.configs import ExperimentTemplateConfig, load_experiment_config

template = load_experiment_config("protein_diffusion_experiment")
assert isinstance(template, ExperimentTemplateConfig)
```

These template documents reference runtime config files plus optional override
sections. They stay separate from named runtime configs on purpose.

## Template Generation

Use the template manager when you want to materialize a typed config from one
of the shipped template families:

```python
from artifex.configs import template_manager

training_config = template_manager.generate_config(
    "simple_training",
    batch_size=64,
    learning_rate=2e-4,
)
```

Template generation routes through the same typed config backend as YAML
loading, so nested optimizer and scheduler sections remain typed dataclasses.
