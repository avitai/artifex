# Trainer API Reference

Complete API reference for Artifex's training system.

## Overview

The training system provides a robust infrastructure for training generative models, with ongoing work towards production-ready status. The main components are:

- **`Trainer`**: Main training class handling the complete training loop
- **`TrainingState`**: Immutable state container for training
- **Configuration Classes**: Type-safe configuration with Pydantic

## Trainer

Main class for training generative models.

```python
from artifex.generative_models.training import Trainer
```

### Constructor

```python
Trainer(
    model: Any,
    training_config: TrainingConfig,
    optimizer: optax.GradientTransformation | None = None,
    train_data_loader: Callable | None = None,
    val_data_loader: Callable | None = None,
    workdir: str | None = None,
    rng: jax.Array | None = None,
    loss_fn: Callable | None = None,
    metrics_logger: MetricsLogger | None = None,
    logger: Logger | None = None,
    checkpoint_dir: str | None = None,
    save_interval: int = 1000,
    log_callback: Callable | None = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Any` | required | Model to train (must have parameters) |
| `training_config` | `TrainingConfig` | required | Type-safe training configuration |
| `optimizer` | `optax.GradientTransformation \| None` | `None` | Custom optimizer (auto-created if None) |
| `train_data_loader` | `Callable \| None` | `None` | Function returning training data iterator |
| `val_data_loader` | `Callable \| None` | `None` | Function returning validation data iterator |
| `workdir` | `str \| None` | `None` | Working directory for outputs |
| `rng` | `jax.Array \| None` | `None` | JAX random key (default: PRNGKey(0)) |
| `loss_fn` | `Callable \| None` | `None` | Custom loss function |
| `metrics_logger` | `MetricsLogger \| None` | `None` | Metrics logger instance |
| `logger` | `Logger \| None` | `None` | General logger instance |
| `checkpoint_dir` | `str \| None` | `None` | Checkpoint directory (default: workdir) |
| `save_interval` | `int` | `1000` | Save checkpoint every N steps |
| `log_callback` | `Callable \| None` | `None` | Custom logging callback |

**Example:**

```python
from artifex.generative_models.training import Trainer
from artifex.generative_models.core.configuration import (
    TrainingConfig,
    OptimizerConfig,
)

# Create training configuration
optimizer_config = OptimizerConfig(
    name="adam",
    optimizer_type="adam",
    learning_rate=1e-3,
)

training_config = TrainingConfig(
    name="vae_training",
    batch_size=128,
    num_epochs=100,
    optimizer=optimizer_config,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    training_config=training_config,
    train_data_loader=train_loader,
    val_data_loader=val_loader,
    workdir="./experiments/vae",
)
```

### Methods

#### train_step

Execute a single training step.

```python
def train_step(
    batch: dict[str, Any]
) -> dict[str, Any]
```

**Parameters:**

- `batch` (`dict[str, Any]`): Batch of training data

**Returns:**

- `dict[str, Any]`: Training metrics including loss

**Example:**

```python
batch = {"images": images, "labels": labels}
metrics = trainer.train_step(batch)
print(f"Loss: {metrics['loss']:.4f}")
```

#### validate_step

Execute a single validation step.

```python
def validate_step(
    batch: dict[str, Any]
) -> dict[str, Any]
```

**Parameters:**

- `batch` (`dict[str, Any]`): Batch of validation data

**Returns:**

- `dict[str, Any]`: Validation metrics including loss

**Example:**

```python
val_batch = {"images": val_images, "labels": val_labels}
val_metrics = trainer.validate_step(val_batch)
print(f"Val Loss: {val_metrics['loss']:.4f}")
```

#### train_epoch

Train for one complete epoch.

```python
def train_epoch() -> dict[str, Any]
```

**Returns:**

- `dict[str, Any]`: Average metrics for the epoch

**Example:**

```python
for epoch in range(num_epochs):
    metrics = trainer.train_epoch()
    print(f"Epoch {epoch + 1}: Loss = {metrics['loss']:.4f}")
```

**Notes:**

- Automatically saves checkpoints based on `save_frequency`
- Calls training step for each batch in the data loader
- Returns averaged metrics over all batches

#### train

Complete training loop with validation.

```python
def train(
    train_data: Any,
    num_epochs: int,
    batch_size: int,
    val_data: Any | None = None,
    val_interval: int = 100,
) -> dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data` | `Any` | required | Training data |
| `num_epochs` | `int` | required | Number of training epochs |
| `batch_size` | `int` | required | Batch size |
| `val_data` | `Any \| None` | `None` | Validation data |
| `val_interval` | `int` | `100` | Validate every N steps |

**Returns:**

- `dict[str, Any]`: Final metrics after training

**Example:**

```python
final_metrics = trainer.train(
    train_data=train_data,
    num_epochs=100,
    batch_size=128,
    val_data=val_data,
    val_interval=100,
)
```

#### evaluate

Evaluate model on a dataset.

```python
def evaluate(
    data: Any,
    batch_size: int
) -> dict[str, Any]
```

**Parameters:**

- `data` (`Any`): Evaluation data
- `batch_size` (`int`): Batch size for evaluation

**Returns:**

- `dict[str, Any]`: Average evaluation metrics

**Example:**

```python
test_metrics = trainer.evaluate(test_data, batch_size=128)
print(f"Test Loss: {test_metrics['loss']:.4f}")
```

#### generate_samples

Generate samples from the trained model.

```python
def generate_samples(
    num_samples: int,
    seed: int | None = None,
    **kwargs
) -> Any
```

**Parameters:**

- `num_samples` (`int`): Number of samples to generate
- `seed` (`int | None`): Random seed for reproducibility
- `**kwargs`: Additional arguments for model's generate method

**Returns:**

- `Any`: Generated samples

**Example:**

```python
# Generate 16 samples
samples = trainer.generate_samples(num_samples=16, seed=42)

# With temperature sampling
samples = trainer.generate_samples(
    num_samples=16,
    temperature=0.8,
)
```

**Raises:**

- `NotImplementedError`: If model doesn't have a `generate` method

#### save_checkpoint

Save training checkpoint.

```python
def save_checkpoint(
    path: str | None = None
) -> None
```

**Parameters:**

- `path` (`str | None`): Path to save checkpoint (default: auto-generated)

**Example:**

```python
# Auto-generated path
trainer.save_checkpoint()

# Custom path
trainer.save_checkpoint("./checkpoints/best_model.pkl")
```

**Notes:**

- Saves complete training state (params, opt_state, step, rng)
- Creates checkpoint directory if it doesn't exist
- Uses pickle serialization

#### load_checkpoint

Load training checkpoint.

```python
def load_checkpoint(
    path: str
) -> None
```

**Parameters:**

- `path` (`str`): Path to checkpoint file

**Example:**

```python
# Load checkpoint
trainer.load_checkpoint("./checkpoints/checkpoint_5000.pkl")

# Resume training
trainer.train_epoch()
```

**Notes:**

- Restores complete training state
- Updates internal state in-place

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `Any` | Model being trained |
| `training_config` | `TrainingConfig` | Training configuration |
| `optimizer` | `optax.GradientTransformation` | Optimizer instance |
| `state` | `dict[str, Any]` | Current training state |
| `train_metrics` | `list[dict]` | Training metrics history |
| `val_metrics` | `list[dict]` | Validation metrics history |
| `steps_per_epoch` | `int` | Number of steps per epoch |

## TrainingState

Immutable container for training state.

```python
from artifex.generative_models.training.trainer import TrainingState
```

### Constructor

```python
TrainingState(
    step: int,
    params: Any,
    opt_state: optax.OptState,
    rng: jax.Array,
    best_loss: float = float("inf"),
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step` | `int` | required | Current training step |
| `params` | `Any` | required | Model parameters |
| `opt_state` | `optax.OptState` | required | Optimizer state |
| `rng` | `jax.Array` | required | JAX random key |
| `best_loss` | `float` | `float("inf")` | Best validation loss |

### Class Methods

#### create

Create a new training state.

```python
@classmethod
def create(
    cls,
    params: Any,
    opt_state: optax.OptState,
    rng: jax.Array,
    step: int = 0,
    best_loss: float = float("inf"),
) -> "TrainingState"
```

**Example:**

```python
import jax
import optax

# Create optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(model_params)

# Create training state
state = TrainingState.create(
    params=model_params,
    opt_state=opt_state,
    rng=jax.random.PRNGKey(42),
    step=0,
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `step` | `int` | Current training step |
| `params` | `Any` | Model parameters (PyTree) |
| `opt_state` | `optax.OptState` | Optimizer state |
| `rng` | `jax.Array` | JAX random key |
| `best_loss` | `float` | Best validation loss seen |

## Configuration Classes

### TrainingConfig

Type-safe training configuration.

```python
from artifex.generative_models.core.configuration import TrainingConfig
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | ✅ | Configuration name |
| `batch_size` | `int` | ✅ | Batch size (≥ 1) |
| `num_epochs` | `int` | ✅ | Number of epochs (≥ 1) |
| `optimizer` | `OptimizerConfig` | ✅ | Optimizer config |
| `scheduler` | `SchedulerConfig \| None` | ❌ | LR scheduler config |
| `gradient_clip_norm` | `float \| None` | ❌ | Gradient clipping norm |
| `checkpoint_dir` | `Path` | ❌ | Checkpoint directory |
| `save_frequency` | `int` | ❌ | Save every N steps |
| `max_checkpoints` | `int` | ❌ | Max checkpoints to keep |
| `log_frequency` | `int` | ❌ | Log every N steps |
| `use_wandb` | `bool` | ❌ | Use W&B logging |
| `wandb_project` | `str \| None` | ❌ | W&B project name |

**Example:**

```python
training_config = TrainingConfig(
    name="vae_training",
    batch_size=128,
    num_epochs=100,
    optimizer=optimizer_config,
    scheduler=scheduler_config,
    save_frequency=5000,
    log_frequency=100,
)
```

### OptimizerConfig

Configure optimizers.

```python
from artifex.generative_models.core.configuration import OptimizerConfig
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | ✅ | Configuration name |
| `optimizer_type` | `str` | ✅ | Optimizer type |
| `learning_rate` | `float` | ✅ | Learning rate (> 0) |
| `weight_decay` | `float` | ❌ | Weight decay (≥ 0) |
| `beta1` | `float` | ❌ | Beta1 for Adam |
| `beta2` | `float` | ❌ | Beta2 for Adam |
| `eps` | `float` | ❌ | Epsilon for stability |
| `momentum` | `float` | ❌ | Momentum for SGD |
| `nesterov` | `bool` | ❌ | Use Nesterov momentum |
| `gradient_clip_norm` | `float \| None` | ❌ | Gradient clip by norm |
| `gradient_clip_value` | `float \| None` | ❌ | Gradient clip by value |

**Supported Optimizer Types:**

- `"adam"`: Adam optimizer
- `"adamw"`: AdamW with weight decay
- `"sgd"`: Stochastic Gradient Descent
- `"rmsprop"`: RMSProp
- `"adagrad"`: AdaGrad

**Example:**

```python
optimizer_config = OptimizerConfig(
    name="adamw",
    optimizer_type="adamw",
    learning_rate=3e-4,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
)
```

### SchedulerConfig

Configure learning rate schedules.

```python
from artifex.generative_models.core.configuration import SchedulerConfig
```

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | ✅ | Configuration name |
| `scheduler_type` | `str` | ✅ | Scheduler type |
| `warmup_steps` | `int` | ❌ | Warmup steps |
| `min_lr_ratio` | `float` | ❌ | Min LR ratio |
| `cycle_length` | `int \| None` | ❌ | Cosine cycle length |
| `total_steps` | `int \| None` | ❌ | Total steps (linear) |
| `decay_rate` | `float` | ❌ | Exponential decay rate |
| `decay_steps` | `int` | ❌ | Exponential decay steps |
| `step_size` | `int` | ❌ | Step schedule size |
| `gamma` | `float` | ❌ | Step/multistep gamma |
| `milestones` | `list[int]` | ❌ | Multistep milestones |

**Supported Scheduler Types:**

- `"constant"`: Constant learning rate
- `"linear"`: Linear decay
- `"cosine"`: Cosine annealing
- `"exponential"`: Exponential decay
- `"step"`: Step-wise decay
- `"multistep"`: Multiple milestone decay

**Example:**

```python
scheduler_config = SchedulerConfig(
    name="cosine_warmup",
    scheduler_type="cosine",
    warmup_steps=1000,
    cycle_length=50000,
    min_lr_ratio=0.1,
)
```

## Custom Loss Functions

Implement custom loss functions for specialized training:

```python
def custom_loss_fn(params, batch, rng):
    """Custom loss function.

    Args:
        params: Model parameters
        batch: Batch of data
        rng: JAX random key

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Forward pass
    model_outputs = model.apply(params, batch, rngs=rng)

    # Compute loss
    loss = jnp.mean((model_outputs - batch["targets"]) ** 2)

    # Additional metrics
    metrics = {
        "mse": loss,
        "mae": jnp.mean(jnp.abs(model_outputs - batch["targets"])),
    }

    return loss, metrics

# Use custom loss function
trainer = Trainer(
    model=model,
    training_config=training_config,
    loss_fn=custom_loss_fn,
)
```

## Logging Callbacks

Artifex provides built-in logging callbacks for common experiment tracking tools:

```python
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
    ProgressBarCallback,
)

# W&B logging
wandb_callback = WandbLoggerCallback(WandbLoggerConfig(
    project="my-project",
    name="experiment-1",
))

# TensorBoard logging
tb_callback = TensorBoardLoggerCallback(TensorBoardLoggerConfig(
    log_dir="logs/experiment-1",
))

# Progress bar
progress_callback = ProgressBarCallback()

# Use callbacks with trainer
trainer.fit(callbacks=[wandb_callback, tb_callback, progress_callback])
```

See [Logging Callbacks](../../training/logging.md) for full documentation.

### Custom Logging Callbacks

For custom logging needs, implement a callback:

```python
from artifex.generative_models.training.callbacks import BaseCallback

class CustomLoggerCallback(BaseCallback):
    """Custom logging callback."""

    def on_batch_end(self, trainer, batch, logs=None):
        if logs and trainer.global_step % 100 == 0:
            # Your custom logging logic
            print(f"Step {trainer.global_step}: {logs}")

# Use custom callback
trainer.fit(callbacks=[CustomLoggerCallback()])
```

## Complete Training Example

Full example with all components:

```python
from artifex.generative_models.core.configuration import (
    ModelConfig,
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.training import Trainer
from flax import nnx

# 1. Model configuration
model_config = ModelConfig(
    name="vae_mnist",
    model_class="artifex.generative_models.models.vae.base.VAE",
    input_dim=(28, 28, 1),
    hidden_dims=[512, 256],
    output_dim=64,
    parameters={"beta": 1.0},
)

# 2. Create model
rngs = nnx.Rngs(42)
model = create_model(config=model_config, rngs=rngs)

# 3. Optimizer configuration
optimizer_config = OptimizerConfig(
    name="adamw",
    optimizer_type="adamw",
    learning_rate=3e-4,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
)

# 4. Scheduler configuration
scheduler_config = SchedulerConfig(
    name="cosine_warmup",
    scheduler_type="cosine",
    warmup_steps=1000,
    cycle_length=50000,
    min_lr_ratio=0.1,
)

# 5. Training configuration
training_config = TrainingConfig(
    name="vae_training",
    batch_size=128,
    num_epochs=100,
    optimizer=optimizer_config,
    scheduler=scheduler_config,
    save_frequency=5000,
    log_frequency=100,
    checkpoint_dir="./checkpoints/vae",
)

# 6. Create trainer
trainer = Trainer(
    model=model,
    training_config=training_config,
    train_data_loader=train_loader,
    val_data_loader=val_loader,
    workdir="./experiments/vae",
)

# 7. Training loop
for epoch in range(training_config.num_epochs):
    # Train epoch
    train_metrics = trainer.train_epoch()
    print(f"Epoch {epoch + 1}: Train Loss = {train_metrics['loss']:.4f}")

    # Validate
    val_metrics = trainer.evaluate(val_data, batch_size=128)
    print(f"Epoch {epoch + 1}: Val Loss = {val_metrics['loss']:.4f}")

    # Save best model
    if val_metrics['loss'] < trainer.state.get('best_loss', float('inf')):
        trainer.save_checkpoint("./checkpoints/vae/best_model.pkl")

# 8. Generate samples
samples = trainer.generate_samples(num_samples=16, seed=42)
```

## Advanced Usage

### Custom Training Step

Override the training step for specialized training:

```python
from functools import partial
import jax
import optax
from flax import nnx

class CustomTrainer(Trainer):
    """Custom trainer with specialized training step."""

    def _train_step(self, state, batch):
        """Custom training step with additional processing."""
        rng, step_rng = jax.random.split(state["rng"])

        # Custom preprocessing
        batch = self.preprocess_batch(batch, step_rng)

        # Custom loss computation
        def loss_fn(params):
            loss, metrics = self.compute_custom_loss(params, batch, step_rng)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state["params"]
        )

        # Custom gradient processing
        grads = self.process_gradients(grads)

        # Update parameters
        updates, opt_state = self.optimizer.update(
            grads, state["opt_state"], state["params"]
        )
        params = optax.apply_updates(state["params"], updates)

        new_state = {
            "step": state["step"] + 1,
            "params": params,
            "opt_state": opt_state,
            "rng": rng,
        }

        return new_state, metrics

    def preprocess_batch(self, batch, rng):
        """Custom batch preprocessing."""
        # Your preprocessing logic
        return batch

    def compute_custom_loss(self, params, batch, rng):
        """Custom loss computation."""
        # Your loss logic
        pass

    def process_gradients(self, grads):
        """Custom gradient processing."""
        # Your gradient processing logic
        return grads

# Use custom trainer
trainer = CustomTrainer(
    model=model,
    training_config=training_config,
    train_data_loader=train_loader,
)
```

### Distributed Training

Extend trainer for distributed training:

```python
import jax

class DistributedTrainer(Trainer):
    """Trainer for distributed training across devices."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get available devices
        self.devices = jax.devices()
        print(f"Training on {len(self.devices)} devices")

        # Replicate model parameters
        self.replicated_params = jax.device_put_replicated(
            self.state["params"],
            self.devices
        )

    @partial(jax.pmap, axis_name="devices")
    def distributed_train_step(self, state, batch):
        """Training step parallelized across devices."""
        def loss_fn(params):
            outputs = self.model.apply(params, batch, training=True)
            return outputs["loss"], outputs

        (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state["params"]
        )

        # Average gradients across devices
        grads = jax.lax.pmean(grads, axis_name="devices")

        # Update
        updates, opt_state = self.optimizer.update(grads, state["opt_state"])
        params = optax.apply_updates(state["params"], updates)

        new_state = {
            "step": state["step"] + 1,
            "params": params,
            "opt_state": opt_state,
            "rng": state["rng"],
        }

        return new_state, loss

# Use distributed trainer
distributed_trainer = DistributedTrainer(
    model=model,
    training_config=training_config,
    train_data_loader=train_loader,
)
```

## Type Hints

The Trainer API uses type hints for clarity:

```python
from typing import Any, Callable
import jax
import jax.numpy as jnp
import optax

# Type aliases
Batch = dict[str, jax.Array]
Metrics = dict[str, float]
LossFn = Callable[[Any, Batch, jax.Array], tuple[float, Metrics]]

# Usage in custom code
def my_loss_fn(
    params: Any,
    batch: Batch,
    rng: jax.Array
) -> tuple[float, Metrics]:
    """Type-safe loss function."""
    pass
```

## Summary

The Trainer API provides:

✅ **Simple Interface**: Easy to use for common cases
✅ **Type-Safe**: Pydantic-based configuration
✅ **Flexible**: Extensible for custom training logic
✅ **Research-Focused**: Checkpointing, logging, monitoring for experimentation
✅ **Well-Documented**: Complete API reference with examples

## See Also

- [Training Guide](../../user-guide/training/training-guide.md) - Practical training examples
- [Training Overview](../../user-guide/training/overview.md) - Architecture and concepts
- [Configuration Guide](../../user-guide/training/configuration.md) - Configuration system details
- [Core API](../core/base.md) - Core model interfaces

---

*For practical examples, see the [Training Guide](../../user-guide/training/training-guide.md).*
