# Training Systems

**Status:** `Supported runtime training reference`

`artifex.generative_models.training` keeps the shared owner set narrow: the shared package owns `Trainer`, typed optimizer and scheduler factories, callback modules, gradient accumulation helpers, distributed utilities, staged and streaming loop helpers, and typed RL trainer contracts. Family-specific trainer implementations live under `artifex.generative_models.training.trainers`.

## Shared Trainer

Use `Trainer` when you want an explicit objective boundary and callback-aware training loop:

```python
from artifex.generative_models.core.configuration import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from artifex.generative_models.training import Trainer, create_optimizer, create_scheduler
from artifex.generative_models.training.callbacks import (
    CallbackList,
    ProgressBarCallback,
    ProgressBarConfig,
)

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
    name="baseline-training",
    optimizer=optimizer_config,
    scheduler=scheduler_config,
    batch_size=64,
    num_epochs=20,
)

schedule = create_scheduler(
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
    schedule=schedule,
)

trainer = Trainer(
    model=model,
    training_config=training_config,
    optimizer=optimizer,
    loss_fn=loss_fn,
    callbacks=CallbackList([
        ProgressBarCallback(ProgressBarConfig(show_metrics=True)),
    ]),
)
```

## Family Trainers

The shared package does not hide model-specific objectives behind one universal trainer class. Use the trainer family that matches the model runtime you are actually training:

- [VAE Trainer](vae_trainer.md)
- [GAN Trainer](gan_trainer.md)
- [Diffusion Trainer](diffusion_trainer.md)
- [Flow Trainer](flow_trainer.md) using `FlowTrainingConfig(time_sampling="logit_normal")` when you want the retained shared flow-matching configuration surface
- [Energy Trainer](energy_trainer.md)
- [Autoregressive Trainer](autoregressive_trainer.md)
- [REINFORCE Trainer](reinforce.md)
- [PPO Trainer](ppo.md)
- [GRPO Trainer](grpo.md)
- [DPO Trainer](dpo.md)

## Distributed Utilities

Artifex ships distributed helpers as utilities, not as trainer subclasses. The retained owners are:

- `DeviceMeshManager` in [mesh.md](mesh.md)
- `DataParallel` in [data_parallel.md](data_parallel.md)
- `DevicePlacement` in [device_placement.md](device_placement.md)
- `DistributedMetrics` in [distributed_metrics.md](distributed_metrics.md)

## Advanced Shared Utilities

- `GradientAccumulator` and `DynamicLossScaler` live in [gradient_accumulation.md](gradient_accumulation.md)
- shared helper functions such as `sample_logit_normal` live in [utils.md](utils.md)
- callback surfaces live in [base.md](base.md), [checkpoint.md](checkpoint.md), [early_stopping.md](early_stopping.md), [logging.md](logging.md), and [profiling.md](profiling.md)

## Current Training Pages

- Callbacks: [base](base.md), [checkpoint](checkpoint.md), [early_stopping](early_stopping.md), [logging](logging.md), [profiling](profiling.md)
- Factories and helpers: [factory](factory.md), [gradient_accumulation](gradient_accumulation.md), [utils](utils.md)
- Distributed utilities: [data_parallel](data_parallel.md), [device_placement](device_placement.md), [distributed_metrics](distributed_metrics.md), [mesh](mesh.md)
- Family trainers: [vae_trainer](vae_trainer.md), [gan_trainer](gan_trainer.md), [diffusion_trainer](diffusion_trainer.md), [flow_trainer](flow_trainer.md), [energy_trainer](energy_trainer.md), [autoregressive_trainer](autoregressive_trainer.md)
- RL trainers: [reinforce](reinforce.md), [ppo](ppo.md), [grpo](grpo.md), [dpo](dpo.md)

## Coming Soon

Standalone optimizer and scheduler module pages remain roadmap-only until real modules exist. Use the current factory owners instead.

- Planned-only or future pages: [adamw](adamw.md), [adafactor](adafactor.md), [lion](lion.md), [scheduler](scheduler.md), [optax_wrappers](optax_wrappers.md), [exponential](exponential.md), [linear](linear.md), [cosine](cosine.md), [mixed_precision](mixed_precision.md), [tracking](tracking.md), [visualization](visualization.md), [model_parallel](model_parallel.md)
