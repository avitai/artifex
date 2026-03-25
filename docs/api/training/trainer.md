# Trainer API Reference

API reference for the low-level generic `Trainer`.

## Overview

`Trainer` is the shared low-level executor for NNX models when you already have
an explicit task or family objective.

It does not invent an objective from the model. Callers must provide a
step-aware objective with the signature:

```python
loss_fn(model, batch, rng, step) -> (loss, metrics)
```

That keeps the train-step boundary explicit and compatible with family-specific
trainer closures, callbacks, logging, and checkpointing.

## Constructor

```python
Trainer(
    model: nnx.Module,
    training_config: TrainingConfig,
    optimizer: optax.GradientTransformation | None = None,
    train_data_loader: Callable | None = None,
    val_data_loader: Callable | None = None,
    workdir: str | None = None,
    rng: jax.Array | None = None,
    *,
    loss_fn: Callable,
    metrics_logger: MetricsLogger | None = None,
    logger: Logger | None = None,
    checkpoint_dir: str | None = None,
    save_interval: int = 1000,
    log_callback: Callable | None = None,
    callbacks: CallbackList | None = None,
    extensions: dict[str, Extension] | None = None,
)
```

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `nnx.Module` | NNX model to train. |
| `training_config` | `TrainingConfig` | Typed training configuration. |
| `optimizer` | `optax.GradientTransformation \| None` | Optional explicit optimizer. If omitted, one is built from `training_config`. |
| `train_data_loader` | `Callable \| None` | Optional batch loader for `train_epoch()`. |
| `val_data_loader` | `Callable \| None` | Optional validation loader for `train_epoch()`. |
| `workdir` | `str \| None` | Output/work directory. |
| `rng` | `jax.Array \| None` | Initial RNG key. Defaults to `jax.random.PRNGKey(0)`. |
| `loss_fn` | `Callable` | Required explicit objective: `loss_fn(model, batch, rng, step) -> (loss, metrics)`. |
| `metrics_logger` | `MetricsLogger \| None` | Optional structured metrics logger. |
| `logger` | `Logger \| None` | Optional general logger. |
| `checkpoint_dir` | `str \| None` | Directory for the trainer’s built-in pickle checkpoints. |
| `save_interval` | `int` | Save frequency for `train()`. |
| `log_callback` | `Callable \| None` | Optional callback for train/validation metric emission. |
| `callbacks` | `CallbackList \| None` | Optional training lifecycle callbacks. |
| `extensions` | `dict[str, Extension] \| None` | Optional extensions that contribute losses or hooks. |

### Example

```python
from artifex.generative_models.training import Trainer

trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=task_loss_fn,
    train_data_loader=train_loader,
    val_data_loader=val_loader,
    workdir="./experiments/run-01",
)
```

## Methods

### train_step

Run one gradient-update step.

```python
def train_step(batch: dict[str, Any]) -> dict[str, Any]
```

Returns a metrics dictionary containing at least:

- `loss`
- `step`

The method:

- splits RNG state
- evaluates the explicit objective
- adds enabled extension losses
- computes gradients with `nnx.value_and_grad`
- updates parameters through the configured optimizer
- emits callback and logging hooks

### validate_step

Run one validation step without optimizer updates.

```python
def validate_step(batch: dict[str, Any]) -> dict[str, Any]
```

Returns validation metrics including `loss` and `step`.

### train_epoch

Train for one epoch through `train_data_loader`.

```python
def train_epoch() -> dict[str, Any]
```

Returns epoch-averaged metrics.

Notes:

- requires `train_data_loader`
- uses `training_config.save_frequency` for checkpoint cadence during this path

### train

Run the full in-memory training loop backed by a `datarax.MemorySource`.

```python
def train(
    train_data: dict[str, Any],
    num_epochs: int,
    batch_size: int,
    val_data: dict[str, Any] | None = None,
    val_interval: int = 100,
) -> dict[str, Any]
```

This path:

- shuffles training data each epoch
- logs via `metrics_logger` and `logger` when configured
- validates every `val_interval` steps when validation data is provided
- saves built-in trainer checkpoints every `save_interval` steps

### evaluate

Evaluate on in-memory data using the current explicit objective.

```python
def evaluate(data: dict[str, Any], batch_size: int) -> dict[str, Any]
```

Returns averaged evaluation metrics.

### save_checkpoint

Save the trainer’s local checkpoint bundle.

```python
def save_checkpoint(path: str | None = None) -> None
```

This is the generic trainer’s local pickle-based checkpoint path. It stores:

- `step`
- optimizer state
- model state
- RNG state
- extension state

If you want Orbax-managed best-model checkpointing during training, use
[`ModelCheckpoint`](../../training/checkpoint.md) via callbacks.

### load_checkpoint

Load a checkpoint previously written by `save_checkpoint`.

```python
def load_checkpoint(path: str) -> None
```

Restores:

- trainer step
- optimizer state
- RNG state
- model state
- extension state when present

## Design Notes

- `Trainer` is intentionally low-level.
- It is appropriate when you already own the objective boundary.
- Family-specific trainers such as VAE, diffusion, flow, energy, or
  autoregressive trainers should build explicit objective closures and hand
  those to the shared training infrastructure when that boundary is useful.

For higher-level guidance, see:

- [Training Guide](../../user-guide/training/training-guide.md)
- [Training Overview](../../user-guide/training/overview.md)
- [Checkpointing Callbacks](../../training/checkpoint.md)
