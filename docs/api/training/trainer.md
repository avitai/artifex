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
    optimizer: optax.GradientTransformation | nnx.Optimizer | None = None,
    train_data_loader: Callable | None = None,
    val_data_loader: Callable | None = None,
    workdir: str | None = None,
    rng: jax.Array | None = None,
    *,
    loss_fn: Callable,
    metrics_logger: MetricsLogger | None = None,
    logger: Logger | None = None,
    log_callback: Callable | None = None,
    callbacks: CallbackList | None = None,
    extensions: dict[str, Extension] | None = None,
    check_finite_every: int | None = 1,
)
```

### Parameters

| Parameter | Type | Description |
| --- | --- | --- |
| `model` | `nnx.Module` | NNX model to train. |
| `training_config` | `TrainingConfig` | Typed training configuration. |
| `optimizer` | `optax.GradientTransformation \| nnx.Optimizer \| None` | A native optimizer retains its filter and transformation. Raw transformations receive native finite-gradient skipping; configuration uses Substrax’s guarded builder. |
| `train_data_loader` | `Callable \| None` | Optional batch loader for `train_epoch()`. |
| `val_data_loader` | `Callable \| None` | Optional validation loader for `train_epoch()`. |
| `workdir` | `str \| None` | Output/work directory; when `training_config.checkpoint_dir` is unset, checkpoints go to its `checkpoints` subdirectory. |
| `rng` | `jax.Array \| None` | Initial RNG key. Defaults to `jax.random.PRNGKey(0)`. |
| `loss_fn` | `Callable` | Required explicit objective: `loss_fn(model, batch, rng, step) -> (loss, metrics)`. |
| `metrics_logger` | `MetricsLogger \| None` | Optional structured metrics logger. |
| `logger` | `Logger \| None` | Optional general logger. |
| `log_callback` | `Callable \| None` | Optional callback for train/validation metric emission. |
| `callbacks` | `CallbackList \| None` | Optional training lifecycle callbacks. |
| `extensions` | `dict[str, Extension] \| None` | Optional extensions that contribute losses or hooks. |
| `check_finite_every` | `int \| None` | Observe accumulated failed losses or skipped updates before this many further attempts. Default `1`; callbacks, driver completion, saves and close always observe failure. |

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

- derives the key by folding the attempted-update count into the run’s base key
- evaluates the explicit objective
- adds enabled extension losses
- delegates differentiation and updates to the step-owned `GradientStep`
- donates model and step buffers through native `nnx.jit_partial`

`train_step` emits device metrics without driver callbacks, logging, history or
saves. `train` and `train_epoch` own those host boundaries. Both batch hooks
receive the attempted-update index before the update.

### validate_step

Run one validation step without optimizer updates.

```python
def validate_step(batch: dict[str, Any]) -> dict[str, Any]
```

Returns validation metrics including `loss` and `step`.

### train_epoch

Train for one epoch through `train_data_loader`.

```python
def train_epoch(steps: int | None = None) -> dict[str, Any]
```

Returns epoch-averaged metrics. The epoch runs until the loader's iterator is exhausted, or
for `steps` batches when given.

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
- saves built-in trainer checkpoints every `training_config.save_frequency` steps

### evaluate

Evaluate on in-memory data using the current explicit objective.

```python
def evaluate(data: dict[str, Any], batch_size: int) -> dict[str, Any]
```

Returns Calibrax record-weighted scalar means with float32 totals and int32
counts, including a short final batch. A nonfinite reduced loss is refused before
selection callbacks. The evaluation stream is separate from training and is
persisted with the native run.

### save_checkpoint

Save the trainer state under a step in `checkpoint_dir`.

```python
def save_checkpoint(step: int | None = None) -> Path
```

The address must equal the trainer's current attempted-update count. One persistent
`Checkpointer` admits numerical state and writes native Orbax items beside an
Artifex producer record:

- `model`: the model state
- `rng`: the run's base key
- `extensions.step`: the complete native step, including optimizer, objective and
  auxiliary-loss state
- `extensions.extra_state`: callback and other registered host state

The separate raw optimizer item is removed. Construct fresh checkpoints for this
layout. `save_checkpoint` waits for completed persistence. `save_async()` performs
native capture before returning; its path is not completion evidence. Call
`wait_until_finished()` before publication and `close()` at shutdown, or use
`with Trainer(...) as trainer:`. Close propagates numerical and storage failures.
External references to model or step buffers are valid only until the next donated
update; retain a detached copy when needed.

Returns the directory of the saved checkpoint. If you want best-model
checkpointing during training, use
[`ModelCheckpoint`](../../training/checkpoint.md) via callbacks.

Where and how often the trainer checkpoints is the configuration's:
`training_config.checkpoint_dir` is resolved by substrax's `resolve_checkpoint_dir`
(the configured directory, else `workdir/checkpoints`; with neither the trainer
checkpoints nowhere, and the store creates the directory on the first save), `save_frequency` is
the cadence of both `train()` and `train_epoch()`, and `max_checkpoints` is how
many steps the store keeps.

### load_checkpoint

Restore a checkpoint previously written by `save_checkpoint`.

```python
def load_checkpoint(step: int | None = None) -> None
```

The step defaults to the latest one in `checkpoint_dir`. Restores the model,
optimizer, RNG and extension state, and sets the trainer step. A step the
directory holds no checkpoint at raises the store's `CheckpointNotFoundError`,
and an empty directory a `FileNotFoundError`; the former is a `FileNotFoundError`
too.

### checkpoint_state

Return the checkpoint items, for storing trainer state beside application state.

```python
def checkpoint_state() -> dict[str, Any]
```

The items are `model`, `rng` and `extensions`, as `save_checkpoint`
writes them. The outer dictionary is new on every call, but the `model` and
`extensions` entries are live `nnx.State` views whose Variables belong to the
trainer: assigning to a leaf changes the trainer. Copy the leaves first when you
need a detached snapshot:

```python
snapshot = jax.tree.map(jnp.array, trainer.checkpoint_state())
```

### apply_checkpoint_state

Apply restored checkpoint items to the live trainer.

```python
def apply_checkpoint_state(payload: dict[str, Any], *, step: int) -> None
```

This is the state-application half of `load_checkpoint`. Use it to restore
trainer state saved beside application state, such as a data iterator's state
under the `data_iterator` item, and to check the checkpoint's record before any
live state changes:

```python
with OrbaxCheckpointStore(directory) as store:
    store.save(step, {**trainer.checkpoint_state(), "data_iterator": cursor})
    ...
    checkpoint = store.restore(
        step, templates={**trainer.checkpoint_state(), "data_iterator": cursor}
    )
    validate(checkpoint.metadata)  # application-owned; raise before mutating anything
    trainer.apply_checkpoint_state(checkpoint.items, step=checkpoint.step)
```

The payload must come from a restore against `checkpoint_state()` as the
templates. Application-owned items remain the caller's responsibility. Numerical
admission precedes live changes; arbitrary host extension application remains
sequential and is not transactional. Reconstruct after a failing extension restore.
A successful restore clears the failed observation window.

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

### Numerical admission

Loss failures and native optimizer skips accumulate across the observation window;
a later success cannot erase an earlier failure. A rejected gradient holds the inner
optimizer and schedule clock while the attempted-update and RNG index advance.
A nonfinite diagnostic gradient norm alone is not the native finite-gradient
predicate. This is skip-and-report, not transactional rollback: finite gradients can
still produce nonfinite state. Checkpoint admission rejects that state before writes.

No clipping is selected implicitly. Configure clipping in `OptimizerConfig` before
constructing a guarded optimizer; a caller's `nnx.Optimizer` is not rewritten.

### Clipping configuration

`TrainingConfig.gradient_clip_norm` is removed. It was ignored by the optimizer
builder. Set `OptimizerConfig.gradient_clip_norm` explicitly to request clipping;
its default remains `None`. Move the field under `optimizer` in serialized
training documents before loading them. No old-field alias or artifact migration
is applied. The shipped protein recipe and training template explicitly select
`optimizer.gradient_clip_norm: 1.0`; that setting now reaches their optimizer.
