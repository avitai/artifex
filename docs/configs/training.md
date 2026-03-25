# Training Configs

Training uses one top-level dataclass plus nested optimizer and scheduler
configs.

## Public Imports

```python
from artifex.configs import OptimizerConfig, SchedulerConfig, TrainingConfig

training_config = TrainingConfig(
    name="baseline_training",
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

## Available Types

- `TrainingConfig`
- `OptimizerConfig`
- `SchedulerConfig`

## Key Training Fields

- `batch_size`
- `num_epochs`
- `gradient_clip_norm`
- `checkpoint_dir`
- `save_frequency`
- `max_checkpoints`
- `log_frequency`
- `use_wandb`
- `wandb_project`

The nested optimizer and scheduler configs hold the optimizer-specific and
scheduler-specific parameters.
