# Checkpointing

Checkpointing strategies for saving model state and reducing memory usage during training. Artifex provides both gradient checkpointing (activation recomputation) and model checkpointing (state persistence) using Orbax.

<div class="grid cards" markdown>

- :material-content-save:{ .lg .middle } **Model Checkpointing**

    ---

    Save and restore model state with Orbax

    [:octicons-arrow-right-24: Learn more](#model-checkpointing)

- :material-memory:{ .lg .middle } **Gradient Checkpointing**

    ---

    Trade computation for memory with activation recomputation

    [:octicons-arrow-right-24: Learn more](#gradient-checkpointing)

- :material-clock-fast:{ .lg .middle } **Checkpointing Strategies**

    ---

    Optimize when and how to checkpoint

    [:octicons-arrow-right-24: Learn more](#checkpointing-strategies)

- :material-backup-restore:{ .lg .middle } **Recovery**

    ---

    Recover from failures and resume training

    [:octicons-arrow-right-24: Learn more](#recovery-and-resumption)

</div>

## Overview

Two types of checkpointing in Artifex:

1. **Model Checkpointing**: Save model state to disk for:
   - Training resumption after interruption
   - Model deployment and inference
   - Experiment tracking and reproducibility

2. **Gradient Checkpointing**: Recompute activations during backward pass to:
   - Reduce memory usage (trade compute for memory)
   - Train larger models or bigger batches
   - Enable training on memory-limited hardware

## Model Checkpointing

Save and restore model state using Orbax checkpoint manager.

### Basic Model Checkpointing

```python
import orbax.checkpoint as ocp
from flax import nnx
from artifex.generative_models.core.checkpointing import (
    setup_checkpoint_manager,
    save_checkpoint,
    load_checkpoint,
)

# Create model
model = create_vae_model(config, rngs=nnx.Rngs(0))

# Setup checkpoint manager
checkpoint_manager, checkpoint_dir = setup_checkpoint_manager(
    base_dir="./checkpoints/experiment_1"
)

# Training loop
for step in range(num_steps):
    # Training step
    model_state, loss = train_step(nnx.state(model), batch)
    nnx.update(model, model_state)

    # Save checkpoint every N steps
    if (step + 1) % save_every == 0:
        checkpoint_manager = save_checkpoint(
            checkpoint_manager,
            model,
            step=step + 1
        )
        print(f"Saved checkpoint at step {step + 1}")

print(f"Training complete. Checkpoints saved to {checkpoint_dir}")
```

### Loading Checkpoints

```python
from artifex.generative_models.core.checkpointing import (
    load_checkpoint,
    setup_checkpoint_manager,
)
from flax import nnx

# Setup checkpoint manager (same directory)
checkpoint_manager, _ = setup_checkpoint_manager(
    base_dir="./checkpoints/experiment_1"
)

# Create model template (same structure as saved model)
model_template = create_vae_model(config, rngs=nnx.Rngs(0))

# Load latest checkpoint
restored_model, step = load_checkpoint(
    checkpoint_manager,
    target_model_template=model_template,
    step=None,  # None = load latest
)

if restored_model is not None:
    print(f"Restored model from step {step}")
    model = restored_model
else:
    print("No checkpoint found, starting from scratch")
    model = model_template

# Continue training from restored state
for step in range(step + 1, num_steps):
    model_state, loss = train_step(nnx.state(model), batch)
    nnx.update(model, model_state)
```

### Loading Specific Checkpoints

```python
# Load specific checkpoint by step
restored_model, step = load_checkpoint(
    checkpoint_manager,
    target_model_template=model_template,
    step=5000,  # Load checkpoint from step 5000
)

# List available checkpoints
latest_step = checkpoint_manager.latest_step()
all_steps = checkpoint_manager.all_steps()

print(f"Latest checkpoint: step {latest_step}")
print(f"Available checkpoints: {all_steps}")

# Load best checkpoint (based on external tracking)
# You would track best step separately
best_step = 7500  # From your tracking
restored_model, step = load_checkpoint(
    checkpoint_manager,
    target_model_template=model_template,
    step=best_step,
)
```

### Checkpointing with Optimizer State

Artifex provides built-in functions for saving and loading both model and optimizer state:

```python
from flax import nnx
import optax
from artifex.generative_models.core.checkpointing import (
    setup_checkpoint_manager,
    save_checkpoint_with_optimizer,
    load_checkpoint_with_optimizer,
)

# Create model and optimizer
model = create_vae_model(config, rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

# Setup checkpoint manager
checkpoint_manager, _ = setup_checkpoint_manager(
    base_dir="./checkpoints/with_optimizer"
)

# Training with optimizer checkpointing
for step in range(num_steps):
    # Training step
    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)

    # Save checkpoint with optimizer
    if (step + 1) % save_every == 0:
        save_checkpoint_with_optimizer(
            checkpoint_manager, model, optimizer, step + 1
        )

# Load checkpoint with optimizer
model_template = create_vae_model(config, rngs=nnx.Rngs(0))
optimizer_template = nnx.Optimizer(model_template, optax.adam(1e-4), wrt=nnx.Param)

model, optimizer, step = load_checkpoint_with_optimizer(
    checkpoint_manager, model_template, optimizer_template
)

if model is not None:
    print(f"Resumed from step {step}")
else:
    print("No checkpoint found, starting from scratch")
```

### Asynchronous Checkpointing

Checkpoint without blocking training:

```python
import orbax.checkpoint as ocp
from flax import nnx

# Create checkpoint manager with async options
options = ocp.CheckpointManagerOptions(
    max_to_keep=5,
    create=True,
    save_interval_steps=1,  # Allow saving every step
    # Async saving
    enable_async_checkpointing=True,
)

checkpoint_manager = ocp.CheckpointManager(
    directory="./checkpoints/async",
    options=options,
)

# Training loop with async checkpointing
for step in range(num_steps):
    # Training step
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # NNX 0.11.0+ API

    # Save checkpoint asynchronously
    if (step + 1) % save_every == 0:
        model_state = nnx.state(model)

        save_args = ocp.args.Composite(
            model=ocp.args.StandardSave(model_state)
        )

        # Non-blocking save
        checkpoint_manager.save(step + 1, args=save_args)

        # Continue training immediately
        # Checkpoint happens in background

    # Optional: Check if previous save finished
    if checkpoint_manager.check_for_errors():
        print("Checkpoint error detected!")

# Wait for final checkpoint to finish
checkpoint_manager.wait_until_finished()
print("All checkpoints saved")
```

### Checkpoint Retention Policies

Control which checkpoints to keep:

```python
import orbax.checkpoint as ocp

# Keep only last N checkpoints
options = ocp.CheckpointManagerOptions(
    max_to_keep=5,  # Keep last 5 checkpoints
    create=True,
)

# Keep all checkpoints (be careful with disk space)
options = ocp.CheckpointManagerOptions(
    max_to_keep=None,  # Keep all
    create=True,
)

# Custom retention: Keep specific checkpoints
class CustomCheckpointManager:
    """Checkpoint manager with custom retention policy."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.manager = setup_checkpoint_manager(base_dir)[0]
        self.keep_steps = set()  # Steps to always keep

    def save(self, model, step: int, keep: bool = False):
        """Save checkpoint, optionally marking it to keep."""
        save_checkpoint(self.manager, model, step)

        if keep:
            self.keep_steps.add(step)

        # Clean up old checkpoints not in keep_steps
        all_steps = self.manager.all_steps()
        if len(all_steps) > 10:  # Keep at most 10 checkpoints
            # Remove oldest checkpoints not marked to keep
            steps_to_remove = sorted(all_steps)[:-5]  # Keep 5 recent
            for s in steps_to_remove:
                if s not in self.keep_steps:
                    self.manager.delete(s)


# Usage
manager = CustomCheckpointManager("./checkpoints/custom")

for step in range(num_steps):
    # Training
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # NNX 0.11.0+ API

    # Save checkpoint
    if (step + 1) % save_every == 0:
        # Mark checkpoints with best validation loss to keep
        is_best = (val_loss < best_val_loss)
        manager.save(model, step + 1, keep=is_best)
```

## Gradient Checkpointing

Reduce memory by recomputing activations during backward pass.

### Basic Gradient Checkpointing

```python
import jax
from jax.ad_checkpoint import checkpoint as jax_checkpoint
from flax import nnx

class CheckpointedModel(nnx.Module):
    """Model with gradient checkpointing."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        # Create layers
        self.layers = [
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            for _ in range(num_layers)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with checkpointing."""
        # Checkpoint each layer
        for layer in self.layers:
            # Activations not stored in memory
            # Will be recomputed during backward pass
            x = jax_checkpoint(lambda x: nnx.relu(layer(x)))(x)

        return x


# Create model
model = CheckpointedModel(
    num_layers=100,  # Can train much deeper models
    hidden_dim=1024,
    rngs=nnx.Rngs(0),
)

# Training step (automatic recomputation)
def loss_fn(model, x):
    output = model(x)
    return jnp.mean(output ** 2)

# Compute gradients (recomputes activations as needed)
loss, grads = nnx.value_and_grad(loss_fn)(model, x)

# Memory usage: ~50% reduction
# Training time: ~30% slower (due to recomputation)
```

### Selective Checkpointing

Checkpoint only expensive operations:

```python
from jax.ad_checkpoint import checkpoint as jax_checkpoint
from flax import nnx

class SelectiveCheckpointedTransformer(nnx.Module):
    """Transformer with selective checkpointing."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        checkpoint_attention: bool = True,
        checkpoint_ffn: bool = False,
        checkpoint_every_n: int = 1,
    ):
        super().__init__()
        self.checkpoint_attention = checkpoint_attention
        self.checkpoint_ffn = checkpoint_ffn
        self.checkpoint_every_n = checkpoint_every_n

        # Create layers
        self.layers = []
        for i in range(num_layers):
            layer = TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rngs=rngs,
            )
            self.layers.append(layer)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with selective checkpointing."""
        for i, layer in enumerate(self.layers):
            # Checkpoint every N layers
            should_checkpoint = (i % self.checkpoint_every_n == 0)

            if should_checkpoint:
                # Checkpoint entire layer
                x = jax_checkpoint(layer)(x)
            else:
                # No checkpointing
                x = layer(x)

        return x


class TransformerLayer(nnx.Module):
    """Single transformer layer with fine-grained checkpointing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
        checkpoint_attention: bool = True,
        checkpoint_ffn: bool = False,
    ):
        super().__init__()
        self.checkpoint_attention = checkpoint_attention
        self.checkpoint_ffn = checkpoint_ffn

        self.attention = MultiHeadAttention(hidden_size, num_heads, rngs=rngs)
        self.ffn = FeedForward(hidden_size, 4 * hidden_size, rngs=rngs)
        self.ln1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        self.ln2 = nnx.LayerNorm(hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with component-level checkpointing."""
        # Attention block
        residual = x
        x = self.ln1(x)

        if self.checkpoint_attention:
            # Checkpoint attention (quadratic memory in seq_len)
            x = jax_checkpoint(self.attention)(x)
        else:
            x = self.attention(x)

        x = residual + x

        # FFN block
        residual = x
        x = self.ln2(x)

        if self.checkpoint_ffn:
            # Checkpoint FFN (linear memory, fast)
            x = jax_checkpoint(self.ffn)(x)
        else:
            x = self.ffn(x)

        x = residual + x

        return x


# Usage: Checkpoint attention only (biggest memory savings)
model = SelectiveCheckpointedTransformer(
    num_layers=24,
    hidden_size=1024,
    num_heads=16,
    rngs=nnx.Rngs(0),
    checkpoint_attention=True,  # Checkpoint attention
    checkpoint_ffn=False,  # Don't checkpoint FFN
    checkpoint_every_n=2,  # Checkpoint every 2nd layer
)
```

### Checkpoint Policy Functions

Custom policies for what to checkpoint:

```python
from jax.ad_checkpoint import checkpoint_policies

def custom_checkpoint_policy(
    model: nnx.Module,
    memory_budget: float = 0.5,
) -> callable:
    """Create custom checkpoint policy based on memory budget.

    Args:
        model: The model to checkpoint
        memory_budget: Fraction of memory to use (0.5 = 50%)

    Returns:
        Policy function for selective checkpointing
    """
    # Analyze model to find expensive operations
    def get_operation_cost(op_name: str) -> float:
        """Estimate memory cost of operation."""
        if "attention" in op_name:
            return 1.0  # High cost (quadratic)
        elif "ffn" in op_name or "linear" in op_name:
            return 0.3  # Medium cost
        elif "norm" in op_name:
            return 0.1  # Low cost
        else:
            return 0.2  # Default

    # Create policy
    def should_checkpoint(primitive, *args, **kwargs):
        """Decide whether to checkpoint this operation."""
        op_name = str(primitive).lower()
        cost = get_operation_cost(op_name)

        # Checkpoint if cost exceeds budget threshold
        return cost > (1.0 - memory_budget)

    return should_checkpoint


# Use custom policy
policy = custom_checkpoint_policy(model, memory_budget=0.7)

# Apply policy to model
@jax_checkpoint(policy=policy)
def forward_with_policy(model, x):
    return model(x)

output = forward_with_policy(model, x)
```

### Remat (Rematerialization)

JAX's automatic checkpointing using `jax.checkpoint` with policies:

```python
import jax
from jax.ad_checkpoint import checkpoint as jax_checkpoint
from flax import nnx

class RematModel(nnx.Module):
    """Model using JAX remat for automatic checkpointing."""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
        checkpoint_policy: str = "everything_saveable",
    ):
        super().__init__()
        self.checkpoint_policy = checkpoint_policy

        self.layers = [
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
            for _ in range(num_layers)
        ]

    def _forward_layer(self, layer, x):
        """Forward pass through single layer."""
        return nnx.relu(layer(x))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with remat policy."""
        # Choose checkpointing policy
        if self.checkpoint_policy == "everything_saveable":
            # Save everything that doesn't require recomputation
            policy = jax.checkpoint_policies.everything_saveable
        elif self.checkpoint_policy == "nothing_saveable":
            # Recompute everything (maximum memory savings)
            policy = jax.checkpoint_policies.nothing_saveable
        elif self.checkpoint_policy == "dots_with_no_batch_dims":
            # Only checkpoint matrix multiplications
            policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
        else:
            policy = None

        # Apply checkpointing with policy
        for layer in self.layers:
            if policy:
                x = jax_checkpoint(
                    lambda x: self._forward_layer(layer, x),
                    policy=policy
                )(x)
            else:
                x = self._forward_layer(layer, x)

        return x


# Compare policies
for policy in ["everything_saveable", "nothing_saveable", "dots_with_no_batch_dims"]:
    model = RematModel(
        num_layers=50,
        hidden_dim=1024,
        rngs=nnx.Rngs(0),
        checkpoint_policy=policy,
    )

    # Measure memory and time
    x = jnp.ones((32, 1024))

    def loss_fn(model, x):
        return jnp.mean(model(x) ** 2)

    # Training step
    loss, grads = nnx.value_and_grad(loss_fn)(model, x)

    print(f"Policy: {policy}")
    print(f"  Loss: {loss}")
    # Memory and time would vary by policy
```

### Memory-Time Trade-off Analysis

```python
import time
import jax
import jax.numpy as jnp
from flax import nnx

def benchmark_checkpointing(
    num_layers: int,
    hidden_dim: int,
    batch_size: int,
    checkpoint_every_n: int = 1,
) -> dict:
    """Benchmark different checkpointing strategies."""
    results = {}

    for strategy in ["none", "all", "selective"]:
        # Create model
        if strategy == "none":
            # No checkpointing
            model = create_standard_model(num_layers, hidden_dim)
        elif strategy == "all":
            # Checkpoint every layer
            model = create_checkpointed_model(
                num_layers, hidden_dim, checkpoint_every_n=1
            )
        else:
            # Selective checkpointing
            model = create_checkpointed_model(
                num_layers, hidden_dim, checkpoint_every_n=checkpoint_every_n
            )

        # Measure time and memory
        x = jnp.ones((batch_size, hidden_dim))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        # Warmup
        loss, grads = nnx.value_and_grad(loss_fn)(model, x)

        # Benchmark
        start = time.time()
        for _ in range(10):
            loss, grads = nnx.value_and_grad(loss_fn)(model, x)
        duration = (time.time() - start) / 10

        results[strategy] = {
            "time_per_step": duration,
            "loss": float(loss),
        }

    return results


# Run benchmark
results = benchmark_checkpointing(
    num_layers=50,
    hidden_dim=1024,
    batch_size=32,
    checkpoint_every_n=5,
)

for strategy, metrics in results.items():
    print(f"\n{strategy.upper()}:")
    print(f"  Time per step: {metrics['time_per_step']:.3f}s")
    print(f"  Loss: {metrics['loss']:.4f}")

# Typical results:
# NONE: Fast (1.0x), high memory (1.0x)
# ALL: Slow (1.3x), low memory (0.5x)
# SELECTIVE: Medium (1.15x), medium memory (0.7x)
```

## Checkpointing Strategies

Optimize when and how to checkpoint for best results.

### Checkpoint Frequency

```python
class AdaptiveCheckpointing:
    """Adaptive checkpoint frequency based on training dynamics."""

    def __init__(
        self,
        base_interval: int = 1000,
        min_interval: int = 500,
        max_interval: int = 5000,
    ):
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval

        self.loss_history = []
        self.current_interval = base_interval

    def should_checkpoint(self, step: int, loss: float) -> bool:
        """Decide if we should checkpoint at this step."""
        self.loss_history.append(loss)

        # Always checkpoint at base interval
        if step % self.current_interval == 0:
            return True

        # More frequent checkpointing if loss unstable
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            loss_std = jnp.std(jnp.array(recent_losses))

            if loss_std > 0.1:
                # Unstable: Checkpoint more frequently
                self.current_interval = max(
                    self.min_interval,
                    self.current_interval // 2
                )
            else:
                # Stable: Checkpoint less frequently
                self.current_interval = min(
                    self.max_interval,
                    self.current_interval * 2
                )

        return False

    def force_checkpoint(self) -> bool:
        """Force checkpoint (e.g., at end of epoch)."""
        return True


# Usage
adaptive = AdaptiveCheckpointing(base_interval=1000)

for step in range(num_steps):
    # Training step
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)  # NNX 0.11.0+ API

    # Adaptive checkpointing
    if adaptive.should_checkpoint(step, float(loss)):
        save_checkpoint(checkpoint_manager, model, step)

    # Force checkpoint at epoch end
    if (step + 1) % steps_per_epoch == 0:
        save_checkpoint(checkpoint_manager, model, step)
```

### Checkpoint Sharding

Shard large checkpoints for faster I/O:

```python
import orbax.checkpoint as ocp
from flax import nnx
import jax

def save_sharded_checkpoint(
    checkpoint_manager,
    model,
    step: int,
    num_shards: int = 4,
):
    """Save checkpoint sharded across multiple files."""
    model_state = nnx.state(model)

    # Get all devices
    devices = jax.devices()

    # Shard model state across devices
    # This enables parallel I/O
    sharded_state = jax.tree.map(
        lambda x: jax.device_put(x, devices[0]),
        model_state
    )

    # Create save args with sharding
    save_args = ocp.args.Composite(
        model=ocp.args.StandardSave(sharded_state)
    )

    # Save (Orbax automatically shards large arrays)
    checkpoint_manager.save(step, args=save_args)
    checkpoint_manager.wait_until_finished()

    return checkpoint_manager


# Load sharded checkpoint
def load_sharded_checkpoint(
    checkpoint_manager,
    model_template,
    step=None,
):
    """Load sharded checkpoint."""
    if step is None:
        step = checkpoint_manager.latest_step()

    if step is None:
        return None, None

    model_state = nnx.state(model_template)

    restore_args = ocp.args.Composite(
        model=ocp.args.StandardRestore(model_state)
    )

    # Restore (Orbax automatically handles sharded loading)
    restored_data = checkpoint_manager.restore(step, args=restore_args)

    nnx.update(model_template, restored_data["model"])

    return model_template, step
```

### Checkpoint Validation

Artifex provides a built-in function to validate that checkpoints save and load correctly:

```python
from artifex.generative_models.core.checkpointing import (
    setup_checkpoint_manager,
    save_checkpoint,
    validate_checkpoint,
)

# Setup and save checkpoint
checkpoint_manager, _ = setup_checkpoint_manager("./checkpoints")
save_checkpoint(checkpoint_manager, model, step=100)

# Validate the checkpoint loads correctly
validation_data = jnp.ones((2, 10))  # Sample input for validation
is_valid = validate_checkpoint(
    checkpoint_manager,
    model,
    step=100,
    validation_data=validation_data,
    tolerance=1e-5,  # Maximum allowed difference
)

if is_valid:
    print("Checkpoint validated successfully")
else:
    print("Checkpoint validation failed! Investigate before continuing.")

# Use in training loop
if (step + 1) % save_every == 0:
    save_checkpoint(checkpoint_manager, model, step + 1)

    validation_sample = next(val_dataloader)
    is_valid = validate_checkpoint(
        checkpoint_manager,
        model,
        step=step + 1,
        validation_data=validation_sample["data"],
    )

    if not is_valid:
        print("Warning: Checkpoint validation failed!")
```

## Recovery and Resumption

Recover from failures and resume training.

### Training Resumption

```python
from artifex.generative_models.core.checkpointing import (
    setup_checkpoint_manager,
    load_checkpoint,
    save_checkpoint,
)
from flax import nnx
import optax

def setup_training_from_checkpoint(
    checkpoint_dir: str,
    config: dict,
) -> tuple:
    """Setup training, resuming from checkpoint if available."""
    # Setup checkpoint manager
    checkpoint_manager, _ = setup_checkpoint_manager(checkpoint_dir)

    # Create model and optimizer templates
    model = create_vae_model(config, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate), wrt=nnx.Param)

    # Try to load checkpoint
    latest_step = checkpoint_manager.latest_step()

    if latest_step is not None:
        print(f"Found checkpoint at step {latest_step}, resuming...")

        # Load model and optimizer
        model_state = nnx.state(model)
        optimizer_state = nnx.state(optimizer)

        restore_args = ocp.args.Composite(
            model=ocp.args.StandardRestore(model_state),
            optimizer=ocp.args.StandardRestore(optimizer_state),
        )

        restored_data = checkpoint_manager.restore(
            latest_step,
            args=restore_args
        )

        nnx.update(model, restored_data["model"])
        nnx.update(optimizer, restored_data["optimizer"])

        start_step = latest_step + 1
        print(f"Resumed from step {latest_step}")
    else:
        print("No checkpoint found, starting from scratch")
        start_step = 0

    return model, optimizer, start_step, checkpoint_manager


# Use in training script
model, optimizer, start_step, checkpoint_manager = setup_training_from_checkpoint(
    checkpoint_dir="./checkpoints/experiment_1",
    config=config,
)

# Continue training from start_step
for step in range(start_step, num_steps):
    # Training step
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)  # NNX 0.11.0+ API

    # Save checkpoint
    if (step + 1) % save_every == 0:
        # Save both model and optimizer
        model_state = nnx.state(model)
        optimizer_state = nnx.state(optimizer)

        save_args = ocp.args.Composite(
            model=ocp.args.StandardSave(model_state),
            optimizer=ocp.args.StandardSave(optimizer_state),
        )

        checkpoint_manager.save(step + 1, args=save_args)
        checkpoint_manager.wait_until_finished()
```

### Checkpoint Corruption Recovery

Artifex provides a built-in function to recover from corrupted checkpoints. It tries loading checkpoints from newest to oldest until one succeeds:

```python
from artifex.generative_models.core.checkpointing import recover_from_corruption

# Create a model template with the same structure as the saved model
model_template = create_vae_model(config, rngs=nnx.Rngs(0))

# Attempt to recover from any available checkpoint
model, step = recover_from_corruption(
    checkpoint_dir="./checkpoints/experiment_1",
    model_template=model_template,
)

if model is not None:
    print(f"Recovered from step {step}, continuing training...")
    # Continue training from recovered state
    for current_step in range(step + 1, num_steps):
        # Training step...
        pass
else:
    print("Recovery failed, starting from scratch...")
    model = model_template
    step = 0
```

## Best Practices

### Model Checkpointing

#### DO

- ✅ **Save checkpoints regularly** - every N steps or epochs
- ✅ **Save optimizer state** - needed for proper resumption
- ✅ **Use async checkpointing** - don't block training
- ✅ **Validate checkpoints** - ensure they load correctly
- ✅ **Keep multiple checkpoints** - protect against corruption
- ✅ **Save before evaluation** - preserve best models
- ✅ **Use absolute paths** - avoid relative path issues
- ✅ **Document checkpoint structure** - for reproducibility
- ✅ **Version checkpoint format** - handle format changes
- ✅ **Monitor disk space** - checkpoints can be large

#### DON'T

- ❌ **Don't save too frequently** - I/O overhead slows training
- ❌ **Don't keep all checkpoints** - wastes disk space
- ❌ **Don't skip validation** - corrupted checkpoints fail silently
- ❌ **Don't modify checkpoint format** - breaks compatibility
- ❌ **Don't checkpoint on all ranks** - only rank 0 in distributed
- ❌ **Don't ignore save errors** - check for failures
- ❌ **Don't use checkpoint path in model** - keep them separate
- ❌ **Don't hardcode checkpoint paths** - use configuration
- ❌ **Don't forget to wait_until_finished** - async saves need this
- ❌ **Don't checkpoint during validation** - separate concerns

### Gradient Checkpointing

#### DO

- ✅ **Profile before checkpointing** - measure actual memory usage
- ✅ **Checkpoint expensive operations** - attention, large matmuls
- ✅ **Use selective checkpointing** - balance memory vs. compute
- ✅ **Checkpoint every N layers** - for very deep models
- ✅ **Test memory savings** - verify reduction
- ✅ **Monitor training speed** - checkpointing adds overhead
- ✅ **Use with large batches** - maximize throughput
- ✅ **Combine with model parallelism** - for extreme scale
- ✅ **Document checkpoint strategy** - for reproducibility
- ✅ **Benchmark different policies** - find optimal trade-off

#### DON'T

- ❌ **Don't checkpoint everything** - excessive recomputation
- ❌ **Don't checkpoint cheap operations** - not worth overhead
- ❌ **Don't assume memory savings** - measure actual usage
- ❌ **Don't ignore speed penalty** - can be 30%+ slower
- ❌ **Don't checkpoint randomly** - use principled strategies
- ❌ **Don't checkpoint I/O operations** - data loading, logging
- ❌ **Don't over-engineer policies** - start simple
- ❌ **Don't forget to profile** - optimization without data is guessing
- ❌ **Don't checkpoint non-deterministic ops** - causes issues
- ❌ **Don't mix checkpointing styles** - keep consistent

## Summary

Checkpointing in Artifex provides:

1. **Model Checkpointing**: Save/restore model state with Orbax
   - Automatic state management
   - Async saves for efficiency
   - Validation and recovery
   - Flexible retention policies

2. **Gradient Checkpointing**: Trade compute for memory
   - Recompute activations in backward pass
   - Selective checkpointing strategies
   - Policy-based automation
   - 30-50% memory reduction

3. **Best Practices**:
   - Regular model checkpoints (every N steps)
   - Selective gradient checkpoints (expensive ops)
   - Validation and recovery procedures
   - Balance memory, speed, and reliability

## Next Steps

<div class="grid cards" markdown>

- :material-cube-outline:{ .lg .middle } **Custom Architectures**

    ---

    Build custom model architectures with checkpointing

    [:octicons-arrow-right-24: Architecture guide](architectures.md)

- :material-chart-line:{ .lg .middle } **Distributed Training**

    ---

    Combine checkpointing with distributed training

    [:octicons-arrow-right-24: Distributed guide](distributed.md)

- :material-ab-testing:{ .lg .middle } **Model Parallelism**

    ---

    Use checkpointing with model parallelism

    [:octicons-arrow-right-24: Parallelism guide](parallelism.md)

- :material-speedometer:{ .lg .middle } **Training Guide**

    ---

    Return to the comprehensive training documentation

    [:octicons-arrow-right-24: Training guide](../training/training-guide.md)

</div>
