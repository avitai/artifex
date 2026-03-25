# Training Factories

**Status:** `Supported runtime training surface`

**Modules:** `artifex.generative_models.training.optimizers.factory`, `artifex.generative_models.training.schedulers.factory`

**Sources:** `src/artifex/generative_models/training/optimizers/factory.py`, `src/artifex/generative_models/training/schedulers/factory.py`

Artifex owns one shared optimizer factory and one shared scheduler factory. The public contract is typed: pass `OptimizerConfig` and `SchedulerConfig`, then feed the resulting Optax objects into `Trainer` or a family-specific loop.

## Optimizer Factory

```python
from artifex.generative_models.core.configuration import OptimizerConfig
from artifex.generative_models.training import create_optimizer

optimizer = create_optimizer(
    OptimizerConfig(
        name="adamw",
        optimizer_type="adamw",
        learning_rate=1e-3,
        weight_decay=0.01,
    )
)
```

Supported optimizer types today: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `lamb`, `radam`, and `nadam`.

## Scheduler Factory

```python
from artifex.generative_models.core.configuration import SchedulerConfig
from artifex.generative_models.training import create_scheduler

scheduler = create_scheduler(
    SchedulerConfig(
        name="cosine",
        scheduler_type="cosine",
        warmup_steps=1_000,
        cycle_length=100_000,
        min_lr_ratio=0.1,
    ),
    base_lr=1e-3,
)
```

Supported scheduler types today: `constant`, `linear`, `cosine`, `exponential`, `polynomial`, `step`, `multistep`, `cyclic`, `one_cycle`, and `none`.

## Shared Trainer Integration

```python
from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from artifex.generative_models.training import Trainer, create_optimizer, create_scheduler

optimizer_config = OptimizerConfig(
    name="adamw",
    optimizer_type="adamw",
    learning_rate=1e-3,
    weight_decay=0.01,
)
scheduler_config = SchedulerConfig(
    name="cosine",
    scheduler_type="cosine",
    warmup_steps=1_000,
    cycle_length=100_000,
    min_lr_ratio=0.1,
)
training_config = TrainingConfig(
    name="factory-demo",
    optimizer=optimizer_config,
    scheduler=scheduler_config,
    batch_size=32,
    num_epochs=10,
)

scheduler = create_scheduler(
    SchedulerConfig(
        name="cosine",
        scheduler_type="cosine",
        warmup_steps=1_000,
        cycle_length=100_000,
        min_lr_ratio=0.1,
    ),
    base_lr=optimizer_config.learning_rate,
)
optimizer = create_optimizer(
    OptimizerConfig(
        name="adamw",
        optimizer_type="adamw",
        learning_rate=1e-3,
        weight_decay=0.01,
    ),
    schedule=scheduler,
)

trainer = Trainer(
    model=model,
    training_config=training_config,
    optimizer=optimizer,
    loss_fn=loss_fn,
)
```

Standalone pages like `adamw.md`, `cosine.md`, and `scheduler.md` stay in the coming-soon bucket because Artifex does not ship those as standalone runtime training modules today.
