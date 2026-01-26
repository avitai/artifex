"""Streaming training loop with prefetch for large datasets.

Works with ANY trainer via the create_loss_fn() pattern.
This provides 5-20x speedup by overlapping data transfer with computation.

For datasets too large to fit in GPU memory, this loop provides:
- JIT-compiled train steps (fast computation)
- Two-stage prefetch to hide data loading latency
- Batch-level iteration from disk/CPU

NOTE: Unlike train_epoch_staged which uses nnx.fori_loop to JIT the entire
epoch, streaming mode CANNOT use fori_loop because:
1. Data comes from a Python iterator (not JAX-traceable)
2. Number of batches is unknown at compile time
3. Python's iterator protocol requires a Python loop

For maximum performance with datasets that fit in GPU memory, use
train_epoch_staged instead.

Requirements:
    - Loss function signature: (model, batch, rng, step) -> (loss, metrics)
    - All existing trainers implement create_loss_fn() with this signature

Example:
    ```python
    from artifex.generative_models.training.loops import train_epoch_streaming
    from artifex.generative_models.training.trainers import VAETrainer
    from datarax import prefetch_to_device

    # Works with ANY existing trainer
    vae_trainer = VAETrainer(config)
    loss_fn = vae_trainer.create_loss_fn(loss_type="bce")

    # Create iterator with prefetch
    data_iter = create_batch_iterator(source, batch_size=128)
    prefetched = prefetch_to_device(data_iter, size=2)

    # Train with streaming data
    step, metrics = train_epoch_streaming(model, opt, prefetched, rng, loss_fn)
    ```
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


# Type alias for loss function signature
# Note: step is passed dynamically to support KL annealing and other schedules
LossFn = Callable[
    [nnx.Module, dict[str, Any], jax.Array, jax.Array],  # (model, batch, rng, step)
    tuple[jax.Array, dict[str, jax.Array]],
]


def _create_train_step(
    loss_fn: LossFn,
) -> Callable[[nnx.Module, nnx.Optimizer, dict[str, Any], jax.Array, jax.Array], jax.Array]:
    """Create a JIT-compiled train step function.

    Defined at module level to ensure consistent JIT caching.
    """

    @nnx.jit
    def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: dict[str, Any],
        step_rng: jax.Array,
        step: jax.Array,
    ) -> jax.Array:
        def compute_loss(m: nnx.Module) -> tuple[jax.Array, dict[str, jax.Array]]:
            return loss_fn(m, batch, step_rng, step)

        (loss, _metrics), grads = nnx.value_and_grad(compute_loss, has_aux=True)(model)
        optimizer.update(model, grads)

        return loss

    return train_step


def train_epoch_streaming(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data_iterator: Iterator[dict[str, Any]],
    rng: jax.Array,
    loss_fn: LossFn,
    base_step: int = 0,
) -> tuple[int, dict[str, float]]:
    """Train one epoch with streaming data and JIT-compiled steps.

    Compatible with all existing trainers via create_loss_fn().

    This function provides optimized training for large datasets that cannot
    fit in GPU memory. It uses JIT-compiled train steps. For best performance,
    wrap your iterator with datarax.prefetch_to_device() before passing it here.

    NOTE: For datasets that fit in GPU memory, use train_epoch_staged instead -
    it uses nnx.fori_loop to JIT the entire epoch for 10-50x better performance.

    Args:
        model: NNX model to train
        optimizer: NNX optimizer
        data_iterator: Iterator yielding batch dicts (use prefetch_to_device for best perf)
        rng: Epoch RNG key
        loss_fn: From trainer.create_loss_fn() - signature: (model, batch, rng, step)
        base_step: Starting step number

    Returns:
        (final_step, averaged_metrics)

    Example:
        # Works with ANY existing trainer
        vae_trainer = VAETrainer(config)
        loss_fn = vae_trainer.create_loss_fn(loss_type="bce")

        # For best performance, prefetch before passing to this function
        from datarax import prefetch_to_device
        prefetched = prefetch_to_device(data_iter, size=2)

        step, metrics = train_epoch_streaming(model, opt, prefetched, rng, loss_fn)
    """
    # Create JIT-compiled train step
    train_step = _create_train_step(loss_fn)

    total_loss = jnp.array(0.0)
    step = base_step
    num_batches = 0

    # Python loop is unavoidable for streaming data
    # (iterator protocol is not JAX-traceable)
    for batch in data_iterator:
        rng, step_rng = jax.random.split(rng)
        loss = train_step(model, optimizer, batch, step_rng, jnp.array(step))

        total_loss = total_loss + loss
        step += 1
        num_batches += 1

    # Compute average loss
    if num_batches > 0:
        avg_loss = float(total_loss) / num_batches
        avg_metrics = {"loss": avg_loss}
    else:
        avg_metrics = {}

    return step, avg_metrics
