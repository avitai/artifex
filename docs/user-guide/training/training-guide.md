# Training Guide

This guide provides practical examples and patterns for training generative models with Artifex. From basic training to advanced techniques, you'll learn how to effectively train models for your specific use case.

## Quick Start

The simplest way to train a model:

```python
from artifex.generative_models.core.configuration import (
    ModelConfig,
    TrainingConfig,
    OptimizerConfig,
)
from artifex.generative_models.factory import create_model
from artifex.generative_models.training import Trainer
from flax import nnx
import jax.numpy as jnp

# Create model
model_config = ModelConfig(
    name="simple_vae",
    model_class="artifex.generative_models.models.vae.base.VAE",
    input_dim=(28, 28, 1),
    hidden_dims=[256, 128],
    output_dim=32,
)

rngs = nnx.Rngs(42)
model = create_model(config=model_config, rngs=rngs)

# Configure training
optimizer_config = OptimizerConfig(
    name="adam",
    optimizer_type="adam",
    learning_rate=1e-3,
)

training_config = TrainingConfig(
    name="quick_train",
    batch_size=128,
    num_epochs=10,
    optimizer=optimizer_config,
)

# Create trainer
trainer = Trainer(
    model=model,
    training_config=training_config,
    train_data_loader=train_loader,
)

# Train
for epoch in range(training_config.num_epochs):
    metrics = trainer.train_epoch()
    print(f"Epoch {epoch + 1}: Loss = {metrics['loss']:.4f}")
```

## Setting Up Training

### Data Loading

Create efficient data loaders for your models:

```python
import numpy as np
import jax
import jax.numpy as jnp

def create_data_loader(data, batch_size, shuffle=True):
    """Create a data loader that yields batches."""
    def data_loader(batch_size):
        num_samples = len(data)
        num_batches = num_samples // batch_size

        # Shuffle if requested
        if shuffle:
            indices = np.random.permutation(num_samples)
            data_shuffled = jax.tree_map(lambda x: x[indices], data)
        else:
            data_shuffled = data

        # Yield batches
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, num_samples)

            batch = jax.tree_map(
                lambda x: x[batch_start:batch_end],
                data_shuffled
            )
            yield batch

    return data_loader

# Example usage with MNIST
from tensorflow.datasets import load

# Load MNIST
ds_train = load('mnist', split='train', as_supervised=True)
ds_val = load('mnist', split='test', as_supervised=True)

# Convert to numpy arrays
train_images = np.array([img for img, _ in ds_train])
train_labels = np.array([label for _, label in ds_train])

val_images = np.array([img for img, _ in ds_val])
val_labels = np.array([label for _, label in ds_val])

# Normalize to [0, 1]
train_images = train_images.astype(np.float32) / 255.0
val_images = val_images.astype(np.float32) / 255.0

# Create data dictionaries
train_data = {"images": train_images, "labels": train_labels}
val_data = {"images": val_images, "labels": val_labels}

# Create data loaders
train_loader = create_data_loader(train_data, batch_size=128, shuffle=True)
val_loader = create_data_loader(val_data, batch_size=128, shuffle=False)
```

### Preprocessing

Apply preprocessing to your data:

```python
def preprocess_images(images):
    """Preprocess images for training."""
    # Normalize to [-1, 1]
    images = (images - 0.5) * 2.0

    # Add channel dimension if needed
    if images.ndim == 3:
        images = images[..., None]

    return images

def dequantize(images, rng):
    """Add uniform noise to discrete images."""
    noise = jax.random.uniform(rng, images.shape, minval=0.0, maxval=1/256.0)
    return images + noise

# Apply preprocessing
train_images = preprocess_images(train_images)
val_images = preprocess_images(val_images)

# Apply dequantization during training
def train_step_with_dequantization(state, batch, rng):
    """Training step with dequantization."""
    rng, dequant_rng = jax.random.split(rng)

    # Dequantize images
    images = dequantize(batch["images"], dequant_rng)
    batch = {**batch, "images": images}

    # Regular training step
    return train_step(state, batch, rng)
```

### Model Initialization

Properly initialize your models:

```python
from flax import nnx
from artifex.generative_models.factory import create_model

def initialize_model(model_config, seed=0):
    """Initialize a model with proper RNG handling."""
    rngs = nnx.Rngs(seed)

    # Create model
    model = create_model(config=model_config, rngs=rngs)

    # Verify model is initialized
    dummy_input = jnp.ones((1, *model_config.input_dim))

    try:
        output = model(dummy_input, rngs=rngs, training=False)
        print(f"Model initialized successfully. Output shape: {output.shape}")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        raise

    return model

# Initialize model
model = initialize_model(model_config, seed=42)
```

## Custom Training Loops

### Basic Custom Loop

Create a custom training loop for full control:

```python
import jax
import jax.numpy as jnp
import optax
from flax import nnx

def custom_training_loop(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate=1e-3,
):
    """Custom training loop with full control."""
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nnx.state(model))

    # Training state
    rng = jax.random.PRNGKey(0)
    step = 0

    # Define training step
    @nnx.jit
    def train_step(model, opt_state, batch, rng):
        def loss_fn(model):
            outputs = model(batch["images"], rngs=nnx.Rngs(rng), training=True)
            return outputs["loss"], outputs

        # Compute gradients
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, outputs), grads = grad_fn(model)

        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        model = nnx.apply_updates(model, updates)

        return model, opt_state, loss, outputs

    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []

        # Train epoch
        for batch in train_loader(batch_size=128):
            rng, step_rng = jax.random.split(rng)

            model, opt_state, loss, outputs = train_step(
                model, opt_state, batch, step_rng
            )

            epoch_losses.append(float(loss))
            step += 1

            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss:.4f}")

        # Validation
        val_losses = []
        for batch in val_loader(batch_size=128):
            outputs = model(batch["images"], rngs=nnx.Rngs(rng), training=False)
            val_losses.append(float(outputs["loss"]))

        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {np.mean(epoch_losses):.4f}")
        print(f"  Val Loss: {np.mean(val_losses):.4f}")

    return model

# Train with custom loop
model = custom_training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    learning_rate=1e-3,
)
```

### Advanced Custom Loop with Metrics

Track detailed metrics during training:

```python
from collections import defaultdict

def advanced_training_loop(
    model,
    train_loader,
    val_loader,
    num_epochs,
    optimizer_config,
    scheduler_config=None,
):
    """Advanced training loop with metrics tracking."""
    # Create optimizer
    base_lr = optimizer_config.learning_rate

    if scheduler_config:
        schedule = create_schedule(scheduler_config, base_lr)
        optimizer = optax.adam(learning_rate=schedule)
    else:
        optimizer = optax.adam(learning_rate=base_lr)

    # Apply gradient clipping if configured
    if optimizer_config.gradient_clip_norm:
        optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config.gradient_clip_norm),
            optimizer,
        )

    opt_state = optimizer.init(nnx.state(model))

    # Metrics tracking
    history = defaultdict(list)
    rng = jax.random.PRNGKey(0)
    step = 0

    @nnx.jit
    def train_step(model, opt_state, batch, rng):
        def loss_fn(model):
            outputs = model(batch["images"], rngs=nnx.Rngs(rng), training=True)
            return outputs["loss"], outputs

        (loss, outputs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Compute gradient norm
        grad_norm = optax.global_norm(grads)

        # Update
        updates, opt_state = optimizer.update(grads, opt_state)
        model = nnx.apply_updates(model, updates)

        # Add gradient norm to metrics
        metrics = {**outputs, "grad_norm": grad_norm}

        return model, opt_state, loss, metrics

    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
        for batch in train_loader(batch_size=128):
            rng, step_rng = jax.random.split(rng)

            model, opt_state, loss, metrics = train_step(
                model, opt_state, batch, step_rng
            )

            # Track metrics
            for key, value in metrics.items():
                history[f"train_{key}"].append(float(value))

            step += 1

            # Periodic logging
            if step % 100 == 0:
                recent_loss = np.mean(history["train_loss"][-100:])
                recent_grad_norm = np.mean(history["train_grad_norm"][-100:])
                print(f"Step {step}:")
                print(f"  Loss: {recent_loss:.4f}")
                print(f"  Grad Norm: {recent_grad_norm:.4f}")

        # Validation
        val_metrics = defaultdict(list)
        for batch in val_loader(batch_size=128):
            outputs = model(batch["images"], rngs=nnx.Rngs(rng), training=False)
            for key, value in outputs.items():
                val_metrics[key].append(float(value))

        # Log validation metrics
        print(f"\nEpoch {epoch + 1}:")
        for key, values in val_metrics.items():
            mean_value = np.mean(values)
            history[f"val_{key}"].append(mean_value)
            print(f"  Val {key}: {mean_value:.4f}")

    return model, history

# Train with advanced loop
model, history = advanced_training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    optimizer_config=optimizer_config,
    scheduler_config=scheduler_config,
)

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train")
plt.plot(np.arange(len(history["val_loss"])) * 100, history["val_loss"], label="Val")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(history["train_grad_norm"])
plt.xlabel("Step")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm")

plt.tight_layout()
plt.show()
```

## Learning Rate Schedules

### Warmup Schedule

Gradually increase learning rate at the start:

```python
from artifex.generative_models.core.configuration import SchedulerConfig

# Cosine schedule with warmup (recommended)
warmup_cosine = SchedulerConfig(
    name="warmup_cosine",
    scheduler_type="cosine",
    warmup_steps=1000,      # 1000 steps of warmup
    cycle_length=50000,     # Cosine cycle length
    min_lr_ratio=0.1,       # End at 10% of peak LR
)

training_config = TrainingConfig(
    name="warmup_training",
    batch_size=128,
    num_epochs=100,
    optimizer=optimizer_config,
    scheduler=warmup_cosine,
)
```

### Custom Schedules

Create custom learning rate schedules:

```python
import optax

def create_custom_schedule(
    base_lr,
    warmup_steps,
    hold_steps,
    decay_steps,
    end_lr_ratio=0.1,
):
    """Create a custom learning rate schedule.

    Schedule: warmup → hold → decay
    """
    schedules = [
        # Warmup
        optax.linear_schedule(
            init_value=0.0,
            end_value=base_lr,
            transition_steps=warmup_steps,
        ),
        # Hold
        optax.constant_schedule(base_lr),
        # Decay
        optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=decay_steps,
            alpha=end_lr_ratio,
        ),
    ]

    boundaries = [warmup_steps, warmup_steps + hold_steps]

    return optax.join_schedules(schedules, boundaries)

# Use custom schedule
custom_schedule = create_custom_schedule(
    base_lr=1e-3,
    warmup_steps=1000,
    hold_steps=5000,
    decay_steps=44000,
    end_lr_ratio=0.1,
)

optimizer = optax.adam(learning_rate=custom_schedule)
```

### One-Cycle Schedule

Implement one-cycle learning rate policy:

```python
def create_one_cycle_schedule(
    max_lr,
    total_steps,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
):
    """Create a one-cycle learning rate schedule.

    Args:
        max_lr: Maximum learning rate
        total_steps: Total training steps
        pct_start: Percentage of cycle spent increasing LR
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = max_lr / final_div_factor
    """
    initial_lr = max_lr / div_factor
    final_lr = max_lr / final_div_factor
    step_up = int(total_steps * pct_start)
    step_down = total_steps - step_up

    schedules = [
        # Increase phase
        optax.linear_schedule(
            init_value=initial_lr,
            end_value=max_lr,
            transition_steps=step_up,
        ),
        # Decrease phase
        optax.cosine_decay_schedule(
            init_value=max_lr,
            decay_steps=step_down,
            alpha=final_lr / max_lr,
        ),
    ]

    return optax.join_schedules(schedules, [step_up])

# Use one-cycle schedule
one_cycle_schedule = create_one_cycle_schedule(
    max_lr=1e-3,
    total_steps=50000,
    pct_start=0.3,
)

optimizer = optax.adam(learning_rate=one_cycle_schedule)
```

## Gradient Accumulation

Accumulate gradients to simulate larger batch sizes:

```python
def training_with_gradient_accumulation(
    model,
    train_loader,
    num_epochs,
    accumulation_steps=4,
    learning_rate=1e-3,
):
    """Training with gradient accumulation."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nnx.state(model))
    rng = jax.random.PRNGKey(0)

    @nnx.jit
    def compute_gradients(model, batch, rng):
        """Compute gradients for a batch."""
        def loss_fn(model):
            outputs = model(batch["images"], rngs=nnx.Rngs(rng), training=True)
            return outputs["loss"], outputs

        (loss, outputs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        return grads, loss, outputs

    @nnx.jit
    def apply_accumulated_gradients(model, opt_state, accumulated_grads):
        """Apply accumulated gradients."""
        # Average gradients
        averaged_grads = jax.tree_map(
            lambda g: g / accumulation_steps,
            accumulated_grads
        )

        # Update model
        updates, opt_state = optimizer.update(averaged_grads, opt_state)
        model = nnx.apply_updates(model, updates)

        return model, opt_state

    # Training loop
    for epoch in range(num_epochs):
        accumulated_grads = None
        step = 0

        for batch in train_loader(batch_size=32):  # Smaller batch size
            rng, step_rng = jax.random.split(rng)

            # Compute gradients
            grads, loss, outputs = compute_gradients(model, batch, step_rng)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree_map(
                    lambda acc, g: acc + g,
                    accumulated_grads,
                    grads
                )

            step += 1

            # Apply accumulated gradients
            if step % accumulation_steps == 0:
                model, opt_state = apply_accumulated_gradients(
                    model, opt_state, accumulated_grads
                )
                accumulated_grads = None

                if step % 100 == 0:
                    print(f"Step {step // accumulation_steps}: Loss = {loss:.4f}")

    return model

# Train with gradient accumulation
model = training_with_gradient_accumulation(
    model=model,
    train_loader=train_loader,
    num_epochs=10,
    accumulation_steps=4,  # Effective batch size = 32 * 4 = 128
)
```

!!! tip "Advanced Gradient Accumulation"
    For production use, Artifex provides a `GradientAccumulator` class with configurable normalization and step tracking. See [Advanced Features](advanced-features.md#gradient-accumulation) for details.

## Early Stopping

Artifex provides a built-in `EarlyStopping` callback to prevent overfitting:

```python
from artifex.generative_models.training.callbacks import (
    EarlyStopping,
    EarlyStoppingConfig,
    CallbackList,
)

# Configure early stopping
early_stopping_config = EarlyStoppingConfig(
    monitor="val_loss",      # Metric to monitor
    min_delta=0.001,         # Minimum change to qualify as improvement
    patience=10,             # Epochs to wait before stopping
    mode="min",              # "min" for loss, "max" for accuracy
    check_finite=True,       # Stop if metric becomes NaN/Inf
    stopping_threshold=None, # Stop immediately if metric reaches this value
    divergence_threshold=10.0,  # Stop if loss exceeds this (prevents divergence)
)

# Create callback
early_stopping = EarlyStopping(early_stopping_config)

# Use in training loop
for epoch in range(max_epochs):
    # Train epoch...
    train_metrics = train_epoch(model, train_loader)

    # Validate
    val_metrics = validate(model, val_loader)

    # Check early stopping (call on_epoch_end with metrics)
    early_stopping.on_epoch_end(trainer, epoch, {"val_loss": val_metrics["loss"]})

    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch + 1}")
        break
```

### Using Multiple Callbacks

Combine callbacks with `CallbackList`:

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
)

# Create callbacks
callbacks = CallbackList([
    EarlyStopping(EarlyStoppingConfig(patience=10, monitor="val_loss")),
    ModelCheckpoint(CheckpointConfig(
        dirpath="./checkpoints",
        monitor="val_loss",
        save_top_k=3,
    )),
])

# Dispatch events to all callbacks
callbacks.on_train_begin(trainer)
for epoch in range(max_epochs):
    callbacks.on_epoch_begin(trainer, epoch)
    # ... training ...
    callbacks.on_epoch_end(trainer, epoch, metrics)
callbacks.on_train_end(trainer)
```

## Mixed Precision Training

Use mixed precision for faster training:

```python
def mixed_precision_training(model, train_loader, num_epochs):
    """Training with mixed precision (bfloat16)."""
    # Convert model to bfloat16
    def convert_to_bfloat16(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x

    model = jax.tree_map(convert_to_bfloat16, model)

    # Use mixed precision optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale(-1e-3),  # Learning rate
    )

    opt_state = optimizer.init(nnx.state(model))
    rng = jax.random.PRNGKey(0)

    @nnx.jit
    def train_step(model, opt_state, batch, rng):
        # Convert batch to bfloat16
        batch = jax.tree_map(convert_to_bfloat16, batch)

        def loss_fn(model):
            outputs = model(batch["images"], rngs=nnx.Rngs(rng), training=True)
            # Keep loss in float32 for numerical stability
            return outputs["loss"].astype(jnp.float32), outputs

        (loss, outputs), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Update (gradients automatically in bfloat16)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = nnx.apply_updates(model, updates)

        return model, opt_state, loss

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader(batch_size=128):
            rng, step_rng = jax.random.split(rng)
            model, opt_state, loss = train_step(model, opt_state, batch, step_rng)

        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    return model

# Train with mixed precision
model = mixed_precision_training(
    model=model,
    train_loader=train_loader,
    num_epochs=10,
)
```

!!! tip "Dynamic Loss Scaling"
    For robust mixed-precision training, Artifex provides a `DynamicLossScaler` class that automatically adjusts loss scaling to prevent overflow/underflow. See [Advanced Features](advanced-features.md#dynamic-loss-scaling) for details.

## Model Checkpointing

Artifex provides robust checkpointing utilities using Orbax for saving and loading model state.

### Basic Checkpointing

```python
from artifex.generative_models.core.checkpointing import (
    setup_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
)

# Setup checkpoint manager
checkpoint_manager, checkpoint_dir = setup_checkpoint_manager(
    base_dir="./checkpoints/experiment_1"
)

# Save checkpoint during training
for step in range(num_steps):
    # ... training step ...

    if (step + 1) % 1000 == 0:
        save_checkpoint(checkpoint_manager, model, step + 1)
        print(f"Saved checkpoint at step {step + 1}")

# Load checkpoint into a model template
model_template = create_model(config, rngs=nnx.Rngs(0))
restored_model, loaded_step = load_checkpoint(
    checkpoint_manager,
    target_model_template=model_template,
)
print(f"Restored from step {loaded_step}")
```

### Checkpointing with Optimizer State

Save and restore both model and optimizer state:

```python
from artifex.generative_models.core.checkpointing import (
    setup_checkpoint_manager,
    save_checkpoint_with_optimizer,
    load_checkpoint_with_optimizer,
)

# Setup
checkpoint_manager, _ = setup_checkpoint_manager("./checkpoints")
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Save both model and optimizer
save_checkpoint_with_optimizer(checkpoint_manager, model, optimizer, step=100)

# Load both model and optimizer
model_template = create_model(config, rngs=nnx.Rngs(0))
optimizer_template = nnx.Optimizer(model_template, optax.adam(1e-4), wrt=nnx.Param)

restored_model, restored_optimizer, step = load_checkpoint_with_optimizer(
    checkpoint_manager,
    model_template,
    optimizer_template,
)
```

### ModelCheckpoint Callback

Use the `ModelCheckpoint` callback for automatic best-model saving:

```python
from artifex.generative_models.training.callbacks import (
    ModelCheckpoint,
    CheckpointConfig,
)

# Configure checkpoint callback
checkpoint_config = CheckpointConfig(
    dirpath="./checkpoints",
    filename="model-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",           # Save when val_loss decreases
    save_top_k=3,         # Keep top 3 checkpoints
    save_last=True,       # Also save the last checkpoint
)

checkpoint_callback = ModelCheckpoint(checkpoint_config)

# Use in training loop
for epoch in range(num_epochs):
    # ... training ...
    checkpoint_callback.on_epoch_end(trainer, epoch, {"val_loss": val_loss})
```

### Checkpoint Validation and Recovery

Validate checkpoints and recover from corruption:

```python
from artifex.generative_models.core.checkpointing import (
    validate_checkpoint,
    recover_from_corruption,
)

# Validate a checkpoint produces consistent outputs
is_valid = validate_checkpoint(
    checkpoint_manager,
    model,
    step=100,
    validation_data=sample_batch,
    tolerance=1e-5,
)

# Recover from corrupted checkpoints (tries newest to oldest)
recovered_model, recovered_step = recover_from_corruption(
    checkpoint_dir="./checkpoints",
    model_template=model_template,
)
```

For more details, see the [Advanced Checkpointing Guide](../advanced/checkpointing.md).

## Logging and Monitoring

Artifex provides built-in logging callbacks for seamless integration with popular experiment tracking tools. These callbacks integrate with the training callback system for automatic metric logging.

### Weights & Biases Integration

Use `WandbLoggerCallback` for experiment tracking with Weights & Biases:

```python
from artifex.generative_models.training.callbacks import (
    WandbLoggerCallback,
    WandbLoggerConfig,
    ProgressBarCallback,
)

# Configure W&B logging
wandb_config = WandbLoggerConfig(
    project="vae-experiments",
    name="experiment-1",
    tags=["vae", "baseline"],
    config={
        "learning_rate": 1e-3,
        "batch_size": 32,
        "model": "VAE",
    },
    log_every_n_steps=10,
    log_on_epoch_end=True,
)

# Create callback and add to trainer
wandb_callback = WandbLoggerCallback(config=wandb_config)
progress_callback = ProgressBarCallback()

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    callbacks=[wandb_callback, progress_callback],
)
```

**Features:**

- Automatic metric logging at configurable intervals
- Hyperparameter tracking via `config` dict
- Run tagging and notes support
- Multiple modes: `"online"`, `"offline"`, or `"disabled"`
- Run resumption support

### TensorBoard Integration

Use `TensorBoardLoggerCallback` for TensorBoard logging:

```python
from artifex.generative_models.training.callbacks import (
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
)

# Configure TensorBoard logging
tb_config = TensorBoardLoggerConfig(
    log_dir="logs/tensorboard/experiment-1",
    flush_secs=60,
    log_every_n_steps=10,
    log_on_epoch_end=True,
)

# Create callback
tb_callback = TensorBoardLoggerCallback(config=tb_config)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=10,
    callbacks=[tb_callback],
)

# View with: tensorboard --logdir logs/tensorboard
```

**Requirements:** Requires `tensorboard` package (`pip install tensorboard`)

### Progress Bar Display

Use `ProgressBarCallback` for rich console progress display:

```python
from artifex.generative_models.training.callbacks import (
    ProgressBarCallback,
    ProgressBarConfig,
)

# Configure progress bar
progress_config = ProgressBarConfig(
    refresh_rate=10,      # Update every 10 steps
    show_eta=True,        # Show estimated time
    show_metrics=True,    # Display metrics in progress bar
    leave=True,           # Keep progress bar after completion
)

progress_callback = ProgressBarCallback(config=progress_config)
trainer.fit(callbacks=[progress_callback])
```

**Requirements:** Requires `rich` package (`pip install rich`)

### Combining Multiple Loggers

Logging callbacks can be combined with other callbacks:

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    EarlyStopping,
    EarlyStoppingConfig,
    ModelCheckpoint,
    CheckpointConfig,
    WandbLoggerCallback,
    WandbLoggerConfig,
    ProgressBarCallback,
)

# Configure all callbacks
callbacks = CallbackList([
    # Logging
    WandbLoggerCallback(WandbLoggerConfig(
        project="my-project",
        name="experiment-1",
    )),
    ProgressBarCallback(),

    # Training control
    EarlyStopping(EarlyStoppingConfig(
        monitor="val_loss",
        patience=10,
    )),

    # Checkpointing
    ModelCheckpoint(CheckpointConfig(
        dirpath="checkpoints",
        monitor="val_loss",
    )),
])

trainer.fit(callbacks=callbacks)
```

### Custom Logger Callback

Wrap any custom `Logger` instance using `LoggerCallback`:

```python
from artifex.generative_models.utils.logging import ConsoleLogger
from artifex.generative_models.training.callbacks import (
    LoggerCallback,
    LoggerCallbackConfig,
)

# Use existing logger infrastructure
logger = ConsoleLogger(name="training")
config = LoggerCallbackConfig(
    log_every_n_steps=50,
    log_on_epoch_end=True,
    prefix="train/",
)

callback = LoggerCallback(logger=logger, config=config)
trainer.fit(callbacks=[callback])
```

## Performance Profiling

Artifex provides profiling callbacks for performance analysis during training.

### JAX Profiler Callback

Use `JAXProfiler` to capture traces for TensorBoard or Perfetto visualization:

```python
from artifex.generative_models.training.callbacks import (
    JAXProfiler,
    ProfilingConfig,
)

# Configure profiling (skip JIT warmup, profile 10 steps)
config = ProfilingConfig(
    log_dir="logs/profiles",
    start_step=10,  # Start after JIT compilation
    end_step=20,    # Profile for 10 steps
)

profiler = JAXProfiler(config)
trainer.fit(callbacks=[profiler])

# View traces in TensorBoard:
# tensorboard --logdir logs/profiles
```

**Best Practices:**

- Set `start_step` after JIT warmup (typically 5-10 steps)
- Keep profiling window small (10-20 steps) to minimize overhead
- Traces show XLA compilation, device execution, and memory allocation

### Memory Profiling

Track GPU/TPU memory usage with `MemoryProfiler`:

```python
from artifex.generative_models.training.callbacks import (
    MemoryProfiler,
    MemoryProfileConfig,
)

config = MemoryProfileConfig(
    log_dir="logs/memory",
    profile_every_n_steps=100,  # Collect memory stats every 100 steps
    log_device_memory=True,
)

profiler = MemoryProfiler(config)
trainer.fit(callbacks=[profiler])

# Memory profile saved to logs/memory/memory_profile.json
```

The memory profile JSON contains:

```json
[
  {"step": 0, "memory": {"cuda:0": {"bytes_in_use": 1073741824, "peak_bytes_in_use": 2147483648}}},
  {"step": 100, "memory": {"cuda:0": {"bytes_in_use": 1073741824, "peak_bytes_in_use": 2147483648}}}
]
```

### Combining Profiling with Other Callbacks

```python
from artifex.generative_models.training.callbacks import (
    CallbackList,
    JAXProfiler,
    ProfilingConfig,
    MemoryProfiler,
    MemoryProfileConfig,
    ProgressBarCallback,
    EarlyStopping,
    EarlyStoppingConfig,
)

callbacks = CallbackList([
    # Profiling
    JAXProfiler(ProfilingConfig(log_dir="logs/profiles", start_step=10, end_step=20)),
    MemoryProfiler(MemoryProfileConfig(log_dir="logs/memory", profile_every_n_steps=100)),

    # Progress display
    ProgressBarCallback(),

    # Training control
    EarlyStopping(EarlyStoppingConfig(monitor="val_loss", patience=10)),
])

trainer.fit(callbacks=callbacks)
```

See [Profiling Callbacks](../../training/profiling.md) for complete documentation.

## Common Training Patterns

### Progressive Training

Train with progressively increasing complexity:

```python
def progressive_training(model, train_loader, stages):
    """Train with progressive stages.

    Args:
        model: Model to train
        train_loader: Data loader
        stages: List of (num_epochs, learning_rate, batch_size) tuples
    """
    optimizer_state = None

    for stage_idx, (num_epochs, learning_rate, batch_size) in enumerate(stages):
        print(f"\nStage {stage_idx + 1}: LR={learning_rate}, BS={batch_size}")

        # Create optimizer for this stage
        optimizer = optax.adam(learning_rate)

        # Initialize or reuse optimizer state
        if optimizer_state is None:
            optimizer_state = optimizer.init(nnx.state(model))

        # Train for this stage
        for epoch in range(num_epochs):
            for batch in train_loader(batch_size=batch_size):
                model, optimizer_state, loss = train_step(
                    model, optimizer_state, batch, rng
                )

            print(f"  Epoch {epoch + 1}: Loss = {loss:.4f}")

    return model

# Define progressive stages
stages = [
    (10, 1e-3, 32),   # Stage 1: High LR, small batch
    (20, 5e-4, 64),   # Stage 2: Medium LR, medium batch
    (30, 1e-4, 128),  # Stage 3: Low LR, large batch
]

model = progressive_training(model, train_loader, stages)
```

### Curriculum Learning

Train with increasing data difficulty:

```python
def curriculum_learning(model, data_loader_fn, difficulty_schedule):
    """Train with curriculum learning.

    Args:
        model: Model to train
        data_loader_fn: Function that returns data loader for difficulty level
        difficulty_schedule: List of (difficulty_level, num_epochs) tuples
    """
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(nnx.state(model))
    rng = jax.random.PRNGKey(0)

    for difficulty, num_epochs in difficulty_schedule:
        print(f"\nTraining on difficulty level: {difficulty}")

        # Get data loader for this difficulty
        train_loader = data_loader_fn(difficulty)

        # Train
        for epoch in range(num_epochs):
            for batch in train_loader(batch_size=128):
                rng, step_rng = jax.random.split(rng)
                model, opt_state, loss = train_step(
                    model, opt_state, batch, step_rng
                )

            print(f"  Epoch {epoch + 1}: Loss = {loss:.4f}")

    return model

# Define curriculum
difficulty_schedule = [
    ("easy", 10),      # Train on easy examples first
    ("medium", 20),    # Then medium difficulty
    ("hard", 30),      # Finally hard examples
    ("all", 40),       # Train on all data
]

model = curriculum_learning(model, data_loader_fn, difficulty_schedule)
```

### Multi-Task Training

Train on multiple tasks simultaneously:

```python
def multi_task_training(
    model,
    task_loaders,
    task_weights,
    num_epochs,
):
    """Train on multiple tasks.

    Args:
        model: Model to train
        task_loaders: Dict of task_name -> data_loader
        task_weights: Dict of task_name -> weight
        num_epochs: Number of epochs
    """
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(nnx.state(model))
    rng = jax.random.PRNGKey(0)

    @nnx.jit
    def multi_task_step(model, opt_state, batches, rng):
        """Training step with multiple tasks."""
        def loss_fn(model):
            total_loss = 0.0
            metrics = {}

            for task_name, batch in batches.items():
                # Task-specific forward pass
                outputs = model(
                    batch,
                    task=task_name,
                    rngs=nnx.Rngs(rng),
                    training=True
                )

                # Weighted loss
                task_loss = outputs["loss"] * task_weights[task_name]
                total_loss += task_loss

                # Track metrics
                metrics[f"{task_name}_loss"] = outputs["loss"]

            return total_loss, metrics

        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        updates, opt_state = optimizer.update(grads, opt_state)
        model = nnx.apply_updates(model, updates)

        return model, opt_state, loss, metrics

    # Training loop
    for epoch in range(num_epochs):
        # Get batches from all tasks
        task_iters = {
            name: loader(batch_size=32)
            for name, loader in task_loaders.items()
        }

        for step in range(1000):  # Fixed steps per epoch
            # Get batch from each task
            batches = {
                name: next(task_iter)
                for name, task_iter in task_iters.items()
            }

            rng, step_rng = jax.random.split(rng)
            model, opt_state, loss, metrics = multi_task_step(
                model, opt_state, batches, step_rng
            )

            if step % 100 == 0:
                print(f"Step {step}: Total Loss = {loss:.4f}")
                for task_name, task_loss in metrics.items():
                    print(f"  {task_name}: {task_loss:.4f}")

    return model

# Train on multiple tasks
task_loaders = {
    "reconstruction": reconstruction_loader,
    "generation": generation_loader,
    "classification": classification_loader,
}

task_weights = {
    "reconstruction": 1.0,
    "generation": 0.5,
    "classification": 0.3,
}

model = multi_task_training(
    model=model,
    task_loaders=task_loaders,
    task_weights=task_weights,
    num_epochs=50,
)
```

## Troubleshooting

### NaN Loss

If you encounter NaN loss:

```python
# 1. Add gradient clipping
optimizer_config = OptimizerConfig(
    name="clipped_adam",
    optimizer_type="adam",
    learning_rate=1e-3,
    gradient_clip_norm=1.0,  # Clip gradients
)

# 2. Lower learning rate
optimizer_config = OptimizerConfig(
    name="lower_lr",
    optimizer_type="adam",
    learning_rate=1e-4,  # Lower LR
)

# 3. Check for numerical instability
def check_for_nans(metrics, step):
    """Check for NaNs in metrics."""
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"NaN detected at step {step} in {key}")
            # Save checkpoint before crash
            save_checkpoint(model, opt_state, step, "./emergency_checkpoint.pkl")
            raise ValueError(f"NaN in {key}")

# 4. Use mixed precision with care
# Avoid bfloat16 for loss computation
loss = loss.astype(jnp.float32)  # Keep loss in float32
```

### Slow Training

If training is slow:

```python
# 1. Use JIT compilation
@nnx.jit
def train_step(model, opt_state, batch, rng):
    # Training step logic
    pass

# 2. Profile your code with JAXProfiler callback
from artifex.generative_models.training.callbacks import JAXProfiler, ProfilingConfig
profiler = JAXProfiler(ProfilingConfig(log_dir="logs/profiles", start_step=10, end_step=20))
trainer.fit(callbacks=[profiler])
# View in TensorBoard: tensorboard --logdir logs/profiles

# 3. Increase batch size (if memory allows)
training_config = TrainingConfig(
    name="large_batch",
    batch_size=256,  # Larger batch size
    num_epochs=50,   # Fewer epochs needed
    optimizer=optimizer_config,
)

# 4. Use data prefetching
from concurrent.futures import ThreadPoolExecutor

def prefetch_data_loader(data_loader, prefetch_size=2):
    """Prefetch data in background."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        iterator = iter(data_loader(batch_size=128))
        futures = [executor.submit(lambda: next(iterator))
                   for _ in range(prefetch_size)]

        while True:
            # Get next batch from future
            batch = futures.pop(0).result()
            # Submit new prefetch
            futures.append(executor.submit(lambda: next(iterator)))
            yield batch
```

### Memory Issues

If you run out of memory:

```python
# 1. Reduce batch size
training_config = TrainingConfig(
    name="small_batch",
    batch_size=32,  # Smaller batch
    num_epochs=200,  # More epochs
    optimizer=optimizer_config,
)

# 2. Use gradient accumulation
# See "Gradient Accumulation" section above

# 3. Clear cache periodically
import jax

# Clear compilation cache
jax.clear_caches()

# 4. Use checkpointing for large models
from jax.checkpoint import checkpoint

@checkpoint
def expensive_forward_pass(model, x):
    """Forward pass with gradient checkpointing."""
    return model(x)
```

## Best Practices

### DO

- ✅ Use type-safe configuration with validation
- ✅ JIT-compile training steps for performance
- ✅ Save checkpoints regularly
- ✅ Monitor training metrics (loss, gradients)
- ✅ Use gradient clipping for stability
- ✅ Start with small learning rate and increase
- ✅ Validate periodically during training
- ✅ Save best model based on validation metrics
- ✅ Use warmup for learning rate schedules
- ✅ Profile code to find bottlenecks

### DON'T

- ❌ Skip validation - always validate your model
- ❌ Use too high learning rate initially
- ❌ Forget to shuffle training data
- ❌ Ignore NaN or infinite losses
- ❌ Train without gradient clipping
- ❌ Overwrite checkpoints without backup
- ❌ Use mixed precision for all operations
- ❌ Forget to split RNG keys properly
- ❌ Mutate training state in-place
- ❌ Skip warmup for large learning rates

## Summary

This guide covered:

- **Basic Training**: Quick start and setup
- **Custom Loops**: Full control over training
- **Learning Rate Schedules**: Warmup, cosine, one-cycle
- **Advanced Techniques**: Gradient accumulation, early stopping, mixed precision
- **Checkpointing**: Save and load model state
- **Logging**: W&B, TensorBoard integration
- **Profiling**: JAXProfiler for performance traces, MemoryProfiler for memory tracking
- **Common Patterns**: Progressive training, curriculum learning, multi-task
- **Troubleshooting**: NaN loss, slow training, memory issues

For reward-based fine-tuning and alignment, see the [RL Training Guide](rl-training.md) covering REINFORCE, PPO, GRPO, and DPO trainers.

## Next Steps

<div class="grid cards" markdown>

- :material-file-document: **[Configuration Guide](configuration.md)**

    ---

    Deep dive into configuration system and best practices

- :material-rocket: **[Training Overview](overview.md)**

    ---

    Architecture and core concepts of training system

- :material-api: **[Trainer API](../../api/training/trainer.md)**

    ---

    Complete API reference for Trainer class

- :material-brain: **[RL Training](rl-training.md)**

    ---

    REINFORCE, PPO, GRPO, and DPO for reward-based fine-tuning

- :material-cog: **[Advanced Features](advanced-features.md)**

    ---

    Gradient accumulation and dynamic loss scaling

- :material-chart-line: **[Logging & Tracking](logging.md)**

    ---

    W&B, TensorBoard, and progress bar integration

- :material-timer: **[Performance Profiling](profiling.md)**

    ---

    JAX trace profiling and memory tracking

</div>

---

*See the [Configuration Guide](configuration.md) for detailed configuration options and patterns.*
