# Logging and Experiment Tracking

This guide covers integrating logging backends with Artifex training loops for experiment tracking, metric visualization, and debugging.

## Overview

Artifex provides logging callbacks that seamlessly integrate with popular experiment tracking platforms:

- **Weights & Biases (W&B)**: Full-featured experiment tracking with rich visualizations
- **TensorBoard**: Google's visualization toolkit for machine learning
- **Console/Progress Bar**: Real-time training feedback with Rich progress bars

All callbacks follow the same interface and can be combined in a single training run.

## Quick Start

```python
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
    ProgressBarCallback,
    ProgressBarConfig,
)
from artifex.generative_models.training import Trainer

# Create callbacks
wandb_callback = WandbLoggerCallback(WandbLoggerConfig(
    project="my-project",
    name="experiment-1",
))

tensorboard_callback = TensorBoardLoggerCallback(TensorBoardLoggerConfig(
    log_dir="logs/tensorboard",
))

progress_callback = ProgressBarCallback(ProgressBarConfig(
    show_metrics=True,
))

# Use with trainer
trainer = Trainer(
    model=model,
    training_config=training_config,
    callbacks=[wandb_callback, tensorboard_callback, progress_callback],
)
```

## Weights & Biases Integration

### Configuration

```python
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
)

config = WandbLoggerConfig(
    project="vae-experiments",           # W&B project name (required)
    entity="my-team",                     # Team or username (optional)
    name="vae-cifar10-beta4",            # Run name (auto-generated if None)
    tags=["vae", "cifar10", "baseline"],  # Tags for filtering runs
    notes="Testing beta=4.0 with cyclical annealing",  # Run description
    config={                              # Hyperparameters to log
        "learning_rate": 1e-4,
        "beta": 4.0,
        "kl_annealing": "cyclical",
    },
    mode="online",                        # "online", "offline", or "disabled"
    resume=None,                          # Resume previous run
    log_every_n_steps=10,                # Log frequency
    log_on_epoch_end=True,               # Log epoch summaries
    log_dir="./wandb",                   # Local directory for artifacts
)

callback = WandbLoggerCallback(config)
```

### WandbLoggerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | `str` | Required | W&B project name |
| `entity` | `str` | `None` | W&B team/username |
| `name` | `str` | `None` | Run name (auto-generated if None) |
| `tags` | `list[str]` | `[]` | Tags for filtering |
| `notes` | `str` | `None` | Run description |
| `config` | `dict` | `{}` | Hyperparameters to log |
| `mode` | `str` | `"online"` | `"online"`, `"offline"`, or `"disabled"` |
| `resume` | `str\|bool` | `None` | Resume options: `"allow"`, `"never"`, `"must"`, `"auto"` |
| `log_every_n_steps` | `int` | `1` | Logging frequency |
| `log_on_epoch_end` | `bool` | `True` | Log epoch summaries |
| `log_dir` | `str` | `None` | Local directory for W&B files |

### W&B Features

```python
# Automatic metric logging
# All metrics returned by the trainer are logged automatically:
# - loss, recon_loss, kl_loss (for VAE)
# - d_loss, g_loss (for GAN)
# - perplexity, accuracy (for autoregressive models)

# Hyperparameter tracking
config = WandbLoggerConfig(
    project="my-project",
    config={
        "model_type": "VAE",
        "latent_dim": 128,
        "hidden_dims": [64, 128, 256],
        "learning_rate": 1e-4,
        "batch_size": 64,
        "optimizer": "adam",
    },
)

# Run comparison
# W&B automatically enables comparing runs via:
# - Parallel coordinates plots
# - Hyperparameter importance analysis
# - Custom visualizations
```

### W&B Installation

```bash
pip install wandb

# First-time setup
wandb login
```

## TensorBoard Integration

### Configuration

```python
from artifex.generative_models.training.callbacks import (
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
)

config = TensorBoardLoggerConfig(
    log_dir="logs/tensorboard/experiment-1",  # Directory for logs
    flush_secs=120,                            # Flush interval (seconds)
    max_queue=10,                              # Max queued events
    log_every_n_steps=1,                       # Logging frequency
    log_on_epoch_end=True,                     # Log epoch summaries
    log_graph=False,                           # Log model graph (experimental)
)

callback = TensorBoardLoggerCallback(config)
```

### TensorBoardLoggerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | `str` | `"logs/tensorboard"` | Directory for TensorBoard logs |
| `flush_secs` | `int` | `120` | Flush to disk interval (seconds) |
| `max_queue` | `int` | `10` | Maximum queued events |
| `log_every_n_steps` | `int` | `1` | Logging frequency |
| `log_on_epoch_end` | `bool` | `True` | Log epoch summaries |
| `log_graph` | `bool` | `False` | Log model graph (experimental) |

### Viewing TensorBoard Logs

```bash
# Start TensorBoard server
tensorboard --logdir logs/tensorboard

# View in browser at http://localhost:6006
```

### TensorBoard Installation

```bash
pip install tensorboard
```

## Progress Bar Callback

### Configuration

```python
from artifex.generative_models.training.callbacks import (
    ProgressBarCallback,
    ProgressBarConfig,
)

config = ProgressBarConfig(
    refresh_rate=10,      # Refresh every N steps
    show_eta=True,        # Show estimated time remaining
    show_metrics=True,    # Display metrics in progress bar
    leave=True,           # Keep progress bar after completion
    disable=False,        # Disable progress bar entirely
)

callback = ProgressBarCallback(config)
```

### ProgressBarConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `refresh_rate` | `int` | `10` | Refresh frequency (steps) |
| `show_eta` | `bool` | `True` | Show estimated time remaining |
| `show_metrics` | `bool` | `True` | Display metrics inline |
| `leave` | `bool` | `True` | Keep bar after completion |
| `disable` | `bool` | `False` | Disable progress bar |

### Progress Bar Installation

```bash
pip install rich
```

## Generic Logger Callback

For custom logging backends, use the base `LoggerCallback`:

```python
from artifex.generative_models.training.callbacks import (
    LoggerCallback,
    LoggerCallbackConfig,
)
from artifex.generative_models.utils.logging import ConsoleLogger

# Create a custom logger
logger = ConsoleLogger(name="training")

# Wrap in callback
config = LoggerCallbackConfig(
    log_every_n_steps=10,
    log_on_epoch_end=True,
    prefix="train/",  # Prefix for metric names
)
callback = LoggerCallback(logger=logger, config=config)
```

## Complete Training Example

```python
import jax
import optax
from flax import nnx

from artifex.generative_models.training import Trainer, TrainingConfig
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
    ProgressBarCallback,
    ProgressBarConfig,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
)
from artifex.generative_models.training.trainers import VAETrainer, VAETrainingConfig


def train_vae_with_logging(
    model: nnx.Module,
    train_data,
    val_data,
    num_epochs: int = 100,
):
    """Train VAE with comprehensive logging."""

    # Setup optimizer
    optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

    # VAE-specific trainer
    vae_config = VAETrainingConfig(
        kl_annealing="cyclical",
        beta=4.0,
        free_bits=0.25,
    )
    vae_trainer = VAETrainer(model, optimizer, vae_config)

    # Logging callbacks
    callbacks = [
        # W&B for experiment tracking
        WandbLoggerCallback(WandbLoggerConfig(
            project="vae-training",
            name="vae-experiment",
            config={
                "kl_annealing": "cyclical",
                "beta": 4.0,
                "free_bits": 0.25,
                "learning_rate": 1e-4,
            },
            log_every_n_steps=50,
        )),

        # TensorBoard for local visualization
        TensorBoardLoggerCallback(TensorBoardLoggerConfig(
            log_dir="logs/vae-experiment",
            log_every_n_steps=50,
        )),

        # Progress bar for real-time feedback
        ProgressBarCallback(ProgressBarConfig(
            show_metrics=True,
            refresh_rate=10,
        )),

        # Early stopping
        EarlyStopping(EarlyStoppingConfig(
            monitor="val_loss",
            patience=10,
            min_delta=1e-4,
        )),

        # Checkpointing
        ModelCheckpoint(CheckpointConfig(
            dirpath="checkpoints/vae",
            monitor="val_loss",
            save_top_k=3,
        )),
    ]

    # Training configuration
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=64,
    )

    # Create base trainer with VAE loss function
    trainer = Trainer(
        model=model,
        training_config=training_config,
        loss_fn=vae_trainer.create_loss_fn(step=0),
        callbacks=callbacks,
    )

    # Train
    trainer.train(train_data, val_data)

    return model
```

## Metric Types

### Automatically Logged Metrics

Different trainers log different metrics:

**VAE Trainer**:

- `loss`: Total ELBO loss
- `recon_loss`: Reconstruction loss
- `kl_loss`: KL divergence
- `kl_weight`: Current KL annealing weight

**GAN Trainer**:

- `d_loss`: Discriminator loss
- `g_loss`: Generator loss
- `d_real`: Mean discriminator output on real data
- `d_fake`: Mean discriminator output on fake data

**Diffusion Trainer**:

- `loss`: Weighted diffusion loss
- `loss_unweighted`: Unweighted MSE loss

**Flow Trainer**:

- `loss`: Flow matching loss

**Energy Trainer**:

- `loss`: Contrastive divergence loss
- `energy_data`: Mean energy on data
- `energy_neg`: Mean energy on negatives
- `energy_gap`: Energy gap (neg - data)

**Autoregressive Trainer**:

- `loss`: Cross-entropy loss
- `perplexity`: exp(loss)
- `accuracy`: Token prediction accuracy
- `teacher_forcing_prob`: Current teacher forcing probability

### Custom Metrics

Add custom metrics by returning them from your loss function:

```python
def custom_loss_fn(model, batch, rng):
    loss = compute_loss(model, batch)

    # Return additional metrics
    metrics = {
        "loss": loss,
        "custom_metric_1": value1,
        "custom_metric_2": value2,
    }
    return loss, metrics
```

## Best Practices

### 1. Logging Frequency

```python
# For fast training loops (>100 steps/sec)
log_every_n_steps=100

# For slow training loops (<10 steps/sec)
log_every_n_steps=1

# For validation metrics
log_on_epoch_end=True
```

### 2. Organizing Runs

```python
# Use meaningful tags
config = WandbLoggerConfig(
    project="my-project",
    tags=[
        "model:vae",
        "dataset:cifar10",
        "experiment:ablation",
    ],
)

# Use descriptive names
config = WandbLoggerConfig(
    name=f"vae-beta{beta}-lr{lr}-{timestamp}",
)
```

### 3. Hyperparameter Tracking

```python
# Log all relevant hyperparameters
config = WandbLoggerConfig(
    config={
        # Model architecture
        "latent_dim": 128,
        "hidden_dims": [64, 128, 256],

        # Training
        "learning_rate": 1e-4,
        "batch_size": 64,
        "optimizer": "adam",

        # VAE-specific
        "beta": 4.0,
        "kl_annealing": "cyclical",
        "free_bits": 0.25,
    },
)
```

### 4. Multiple Loggers

```python
# Use multiple loggers for different purposes
callbacks = [
    # W&B for long-term tracking and comparison
    WandbLoggerCallback(WandbLoggerConfig(
        project="my-project",
        log_every_n_steps=100,
    )),

    # TensorBoard for quick local visualization
    TensorBoardLoggerCallback(TensorBoardLoggerConfig(
        log_every_n_steps=10,
    )),

    # Progress bar for real-time feedback
    ProgressBarCallback(),
]
```

## Related Documentation

- [Training Guide](training-guide.md) - Core training patterns
- [Advanced Features](advanced-features.md) - Gradient accumulation, mixed precision
- [Profiling](profiling.md) - Performance analysis
