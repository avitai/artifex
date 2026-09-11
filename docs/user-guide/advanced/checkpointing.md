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

Model state is persisted through substrax's `OrbaxCheckpointStore`, the
Orbax-backed, step-addressed store every Avitai library shares. A checkpoint is
a `PyTreeSave` payload plus a JSON metadata sidecar (step, timestamp, loss when
given, and anything passed as `additional_metadata`); restoring never executes
code.

### Basic Checkpointing

```python
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore

model = create_model(config, rngs=nnx.Rngs(0))

with OrbaxCheckpointStore("./checkpoints/experiment_1", max_to_keep=5) as store:
    for step in range(num_steps):
        # ... training step ...
        if (step + 1) % 1000 == 0:
            store.save(model, step + 1, loss=float(loss))
            print(f"Saved checkpoint at step {step + 1}")
```

`max_to_keep` is Orbax's retention: the newest checkpoints are kept, `None`
keeps them all.

### Loading Checkpoints

Build the same model template you trained and restore into it:

```python
model_template = create_model(config, rngs=nnx.Rngs(0))

with OrbaxCheckpointStore("./checkpoints/experiment_1") as store:
    step = store.latest_step()                       # or a specific step
    restored_model, metadata = store.restore(model_template, step)

print(f"Restored from step {metadata['step']}")
```

Without a target, `store.restore(step=step)` returns the payload as it was
stored; `store.list_steps()` and `store.best_step("loss")` pick a checkpoint by
step or by a metadata metric.

### Trainer Checkpoints

`Trainer.save_checkpoint()` writes the model state, the optimizer state, the RNG
key and every extension's state as one payload under the current step, and
`Trainer.load_checkpoint(step=None)` restores the latest (or a given) step into
the live trainer. Both go through the same store under `checkpoint_dir`.

To checkpoint trainer state together with application state, such as a data
iterator cursor, nest `Trainer.checkpoint_state()` inside your own payload and
store it with `OrbaxCheckpointStore`. Restore with that same nested tree as the
template, check the metadata, then call
`Trainer.apply_checkpoint_state(restored["trainer"], step=step)`. The
[Trainer API reference](../../api/training/trainer.md#apply_checkpoint_state)
shows the full sequence.

### Asynchronous Checkpointing

`store.save` returns after Orbax has finished writing, so a training loop can
mutate the model immediately afterwards. Orbax's own asynchronous save is not
exposed: an in-place `nnx.update` racing a background write would corrupt the
checkpoint silently.

### Checkpoint Retention Policies

Retention is `max_to_keep`: the store keeps the newest checkpoints and deletes
the rest. Keep a checkpoint outside that window by copying its step directory,
or run two stores over two directories (frequent, short-lived checkpoints in
one; milestones in another).

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

A checkpoint's payload describes its own tree, and a tree that does not match
the target raises `ValueError` from `store.restore`. To check that a checkpoint
reproduces the model's outputs, restore it into a fresh template and compare:

```python
import jax.numpy as jnp
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore

sample = jnp.ones((2, 10))
expected = model(sample)

with OrbaxCheckpointStore("./checkpoints") as store:
    store.save(model, step=100)
    restored, _ = store.restore(create_model(config, rngs=nnx.Rngs(0)), step=100)

assert jnp.allclose(restored(sample), expected, atol=1e-6)
```

## Recovery and Resumption

### Training Resumption

`Trainer.load_checkpoint()` restores the model, optimizer, RNG and extension
state from the latest step in `checkpoint_dir` and sets `trainer.step`, so a
training script resumes with one call:

```python
from artifex.generative_models.training import Trainer

trainer = Trainer(model=model, training_config=config, loss_fn=loss_fn, checkpoint_dir="./checkpoints")
try:
    trainer.load_checkpoint()
    print(f"Resumed from step {trainer.step}")
except FileNotFoundError:
    print("No checkpoint found, starting from scratch")

trainer.train(train_data, num_epochs=num_epochs)
```

Outside the trainer, save a dict payload and restore it into a template of the
same structure:

```python
payload = {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}
with OrbaxCheckpointStore("./checkpoints/experiment_1") as store:
    store.save(payload, step=step + 1, loss=float(loss))

with OrbaxCheckpointStore("./checkpoints/experiment_1") as store:
    template = {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}
    restored, metadata = store.restore(template, store.latest_step())
nnx.update(model, restored["model"])
nnx.update(optimizer, restored["optimizer"])
```

### Checkpoint Corruption Recovery

A checkpoint that cannot be read raises from `store.restore` rather than being
reported as missing, so recovery is a loop from the newest step to the oldest:

```python
def restore_newest_readable(store, template):
    for step in sorted(store.list_steps(), reverse=True):
        try:
            return store.restore(template, step)
        except (ValueError, OSError) as error:
            print(f"Checkpoint {step} unreadable: {error}")
    return None, {}
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

    Return to the complete training documentation

    [:octicons-arrow-right-24: Training guide](../training/training-guide.md)

</div>
