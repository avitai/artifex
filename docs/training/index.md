# Training Systems

Comprehensive training infrastructure for generative models, including model-specific trainers, distributed training, callbacks, and optimization utilities.

## Overview

<div class="grid cards" markdown>

- :material-school:{ .lg .middle } **Model Trainers**

    ---

    Specialized trainers for VAE, GAN, Diffusion, Flow, EBM, and Autoregressive models

- :material-server-network:{ .lg .middle } **Distributed Training**

    ---

    Data parallel and model parallel training across multiple GPUs/TPUs

- :material-bell:{ .lg .middle } **Callbacks**

    ---

    Checkpointing, early stopping, logging, and custom callbacks

- :material-speedometer:{ .lg .middle } **Optimizers**

    ---

    AdamW, Lion, Adafactor with learning rate schedulers

</div>

## Quick Start

### Basic Training

```python
from artifex.generative_models.training import VAETrainer
from artifex.generative_models.core.configuration import TrainingConfig

# Create training configuration
training_config = TrainingConfig(
    batch_size=128,
    num_epochs=100,
    optimizer={"type": "adam", "learning_rate": 1e-3},
    scheduler={"type": "cosine", "warmup_steps": 1000},
)

# Create trainer
trainer = VAETrainer(
    model=model,
    config=training_config,
    train_dataset=train_data,
    val_dataset=val_data,
)

# Train
trainer.train()
```

## Model-Specific Trainers

Each model family has a specialized trainer that handles its unique training requirements.

### VAE Trainer

Handles ELBO loss, KL annealing, and reconstruction metrics.

```python
from artifex.generative_models.training import VAETrainer

trainer = VAETrainer(
    model=vae_model,
    config=training_config,
    train_dataset=train_data,
    kl_annealing=True,
    kl_warmup_epochs=10,
)
```

[:octicons-arrow-right-24: VAE Trainer Reference](vae_trainer.md)

### GAN Trainer

Manages generator/discriminator alternating updates.

```python
from artifex.generative_models.training import GANTrainer

trainer = GANTrainer(
    model=gan_model,
    config=training_config,
    train_dataset=train_data,
    d_steps=5,  # Discriminator steps per generator step
    gp_weight=10.0,  # Gradient penalty weight
)
```

[:octicons-arrow-right-24: GAN Trainer Reference](gan_trainer.md)

### Diffusion Trainer

Handles noise scheduling and denoising score matching.

```python
from artifex.generative_models.training import DiffusionTrainer

trainer = DiffusionTrainer(
    model=diffusion_model,
    config=training_config,
    train_dataset=train_data,
    ema_decay=0.9999,  # Exponential moving average
)
```

[:octicons-arrow-right-24: Diffusion Trainer Reference](diffusion_trainer.md)

### Flow Trainer

Manages exact likelihood training for normalizing flows.

```python
from artifex.generative_models.training import FlowTrainer

trainer = FlowTrainer(
    model=flow_model,
    config=training_config,
    train_dataset=train_data,
)
```

[:octicons-arrow-right-24: Flow Trainer Reference](flow_trainer.md)

### Energy Trainer

Handles contrastive divergence and MCMC sampling.

```python
from artifex.generative_models.training import EnergyTrainer

trainer = EnergyTrainer(
    model=ebm_model,
    config=training_config,
    train_dataset=train_data,
    mcmc_steps=10,
)
```

[:octicons-arrow-right-24: Energy Trainer Reference](energy_trainer.md)

### Autoregressive Trainer

Manages sequential likelihood training.

```python
from artifex.generative_models.training import AutoregressiveTrainer

trainer = AutoregressiveTrainer(
    model=ar_model,
    config=training_config,
    train_dataset=train_data,
)
```

[:octicons-arrow-right-24: Autoregressive Trainer Reference](autoregressive_trainer.md)

## Callbacks

Callbacks allow customization of the training loop.

### Built-in Callbacks

| Callback | Description |
|----------|-------------|
| [CheckpointCallback](checkpoint.md) | Save model checkpoints |
| [EarlyStoppingCallback](early_stopping.md) | Stop training when validation plateaus |
| [LoggingCallback](logging.md) | Log metrics to console/file |
| [ProfilingCallback](profiling.md) | Profile training performance |
| [VisualizationCallback](visualization.md) | Generate sample visualizations |

### Using Callbacks

```python
from artifex.generative_models.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
)

callbacks = [
    CheckpointCallback(
        save_dir="checkpoints/",
        save_every_n_epochs=10,
        save_best=True,
        metric="val_loss",
    ),
    EarlyStoppingCallback(
        patience=20,
        metric="val_loss",
        mode="min",
    ),
    LoggingCallback(
        log_every_n_steps=100,
        use_wandb=True,
    ),
]

trainer = VAETrainer(
    model=model,
    config=config,
    train_dataset=train_data,
    callbacks=callbacks,
)
```

### Custom Callbacks

```python
from artifex.generative_models.training.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def on_epoch_start(self, trainer, epoch):
        print(f"Starting epoch {epoch}")

    def on_epoch_end(self, trainer, epoch, metrics):
        print(f"Epoch {epoch} completed: {metrics}")

    def on_train_batch_end(self, trainer, batch_idx, loss):
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: loss={loss:.4f}")
```

## Distributed Training

### Data Parallel

```python
from artifex.generative_models.training.distributed import DataParallelTrainer

trainer = DataParallelTrainer(
    model=model,
    config=config,
    train_dataset=train_data,
    num_devices=4,  # Use 4 GPUs
)
```

[:octicons-arrow-right-24: Data Parallel Reference](data_parallel.md)

### Model Parallel

```python
from artifex.generative_models.training.distributed import ModelParallelTrainer

trainer = ModelParallelTrainer(
    model=large_model,
    config=config,
    train_dataset=train_data,
    mesh_shape=(2, 4),  # 2x4 device mesh
)
```

[:octicons-arrow-right-24: Model Parallel Reference](model_parallel.md)

### Device Mesh

```python
from artifex.generative_models.training.distributed import DeviceMesh

mesh = DeviceMesh(
    shape=(2, 2),  # 2x2 mesh
    axis_names=("data", "model"),
)
```

[:octicons-arrow-right-24: Device Mesh Reference](mesh.md)

## Optimizers

### Available Optimizers

| Optimizer | Description | Best For |
|-----------|-------------|----------|
| [AdamW](adamw.md) | Adam with weight decay | General use |
| [Lion](lion.md) | Memory-efficient optimizer | Large models |
| [Adafactor](adafactor.md) | Low memory optimizer | Very large models |

### Using Optimizers

```python
from artifex.generative_models.training.optimizers import create_optimizer

optimizer = create_optimizer(
    optimizer_type="adamw",
    learning_rate=1e-3,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999,
)
```

## Learning Rate Schedulers

### Available Schedulers

| Scheduler | Description |
|-----------|-------------|
| [Cosine](cosine.md) | Cosine annealing with warmup |
| [Linear](linear.md) | Linear warmup and decay |
| [Exponential](exponential.md) | Exponential decay |

### Using Schedulers

```python
from artifex.generative_models.training.schedulers import create_scheduler

scheduler = create_scheduler(
    scheduler_type="cosine",
    warmup_steps=1000,
    total_steps=100000,
    min_lr=1e-6,
)
```

[:octicons-arrow-right-24: Scheduler Reference](scheduler.md)

## RL Training

Reinforcement learning trainers for fine-tuning generative models.

| Trainer | Description |
|---------|-------------|
| [REINFORCE](reinforce.md) | Policy gradient training |
| [PPO](ppo.md) | Proximal Policy Optimization |
| [DPO](dpo.md) | Direct Preference Optimization |
| [GRPO](grpo.md) | Group Relative Policy Optimization |

[:octicons-arrow-right-24: RL Training Guide](../user-guide/training/rl-training.md)

## Advanced Features

### Gradient Accumulation

```python
trainer = VAETrainer(
    model=model,
    config=config,
    train_dataset=train_data,
    gradient_accumulation_steps=4,
)
```

[:octicons-arrow-right-24: Gradient Accumulation](gradient_accumulation.md)

### Mixed Precision Training

```python
trainer = VAETrainer(
    model=model,
    config=config,
    train_dataset=train_data,
    mixed_precision=True,  # Use bfloat16
)
```

[:octicons-arrow-right-24: Mixed Precision](mixed_precision.md)

### Experiment Tracking

```python
trainer = VAETrainer(
    model=model,
    config=config,
    train_dataset=train_data,
    tracking={
        "wandb": {"project": "my-project"},
        "mlflow": {"experiment": "vae-experiments"},
    },
)
```

[:octicons-arrow-right-24: Experiment Tracking](tracking.md)

## Module Reference

| Category | Modules |
|----------|---------|
| **Trainers** | [vae_trainer](vae_trainer.md), [gan_trainer](gan_trainer.md), [diffusion_trainer](diffusion_trainer.md), [flow_trainer](flow_trainer.md), [energy_trainer](energy_trainer.md), [autoregressive_trainer](autoregressive_trainer.md) |
| **Callbacks** | [base](base.md), [checkpoint](checkpoint.md), [early_stopping](early_stopping.md), [logging](logging.md), [profiling](profiling.md), [visualization](visualization.md) |
| **Distributed** | [data_parallel](data_parallel.md), [model_parallel](model_parallel.md), [mesh](mesh.md), [device_placement](device_placement.md), [distributed_metrics](distributed_metrics.md) |
| **Optimizers** | [adamw](adamw.md), [lion](lion.md), [adafactor](adafactor.md), [optax_wrappers](optax_wrappers.md) |
| **Schedulers** | [cosine](cosine.md), [linear](linear.md), [exponential](exponential.md), [factory](factory.md), [scheduler](scheduler.md) |
| **RL** | [reinforce](reinforce.md), [ppo](ppo.md), [dpo](dpo.md), [grpo](grpo.md) |
| **Utilities** | [base](base.md), [gradient_accumulation](gradient_accumulation.md), [mixed_precision](mixed_precision.md), [tracking](tracking.md), [trainer](trainer.md), [utils](utils.md) |

## Related Documentation

- [Training Guide](../user-guide/training/training-guide.md) - Complete training guide
- [Configuration System](../user-guide/training/configuration.md) - Training configuration
- [Distributed Training](../user-guide/advanced/distributed.md) - Multi-GPU/TPU guide
