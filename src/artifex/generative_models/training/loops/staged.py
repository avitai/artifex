"""Staged training loop using nnx.fori_loop for maximum performance.

Works with ANY trainer via the create_loss_fn() pattern.
This provides significant speedup by JIT-compiling the ENTIRE epoch
using nnx.fori_loop, eliminating Python loop overhead.

Key optimizations (matching pure JAX performance):
1. Entire epoch wrapped in @nnx.jit
2. nnx.fori_loop replaces Python loops inside JIT
3. jax.device_put pre-stages data on GPU (done by caller)
4. lax.dynamic_slice_in_dim for batching inside JIT
5. JIT function defined at module level to enable proper caching
6. Step passed dynamically inside fori_loop for proper scheduling (KL annealing, etc.)

Requirements:
    - Data must fit in GPU memory (~10% of VRAM recommended)
    - Loss function signature: (model, batch, rng, step) -> (loss, metrics)
    - All existing trainers implement create_loss_fn() with this signature

Example:
    ```python
    from artifex.generative_models.training.loops import train_epoch_staged
    from artifex.generative_models.training.trainers import VAETrainer

    # Works with ANY existing trainer
    vae_trainer = VAETrainer(config)
    loss_fn = vae_trainer.create_loss_fn(loss_type="bce")

    # Stage data on GPU
    data = jax.device_put(train_images)

    # Train epoch - entire epoch JIT-compiled with nnx.fori_loop!
    # Step increments automatically inside the loop for proper KL annealing
    step, metrics = train_epoch_staged(model, opt, data, 128, rng, loss_fn)
    ```
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx


# Type alias for loss function signature
# Note: step is passed dynamically to support KL annealing and other schedules
LossFn = Callable[
    [nnx.Module, dict[str, Any], jax.Array, jax.Array],  # (model, batch, rng, step)
    tuple[jax.Array, dict[str, jax.Array]],
]


class _CachedEpochRunner(NamedTuple):
    """Cached epoch runner with its associated metrics structure."""

    run_epoch: Callable
    metric_keys: tuple[str, ...]


def _create_epoch_runner(
    loss_fn: LossFn,
    batch_size: int,
    num_batches: int,
    data_key: str,
) -> Callable:
    """Create a JIT-compiled epoch runner for the given configuration.

    This factory function is cached based on loss_fn identity to avoid
    recompilation when the same loss_fn is used across epochs.
    """

    @nnx.jit
    def run_epoch(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        staged_data: jax.Array,
        epoch_rng: jax.Array,
        metric_totals: dict[str, jax.Array],
        base_step: jax.Array,
    ) -> dict[str, jax.Array]:
        """Run entire epoch - JIT-compiled with fori_loop inside."""

        def body_fn(
            i: jax.Array,
            carry: tuple[nnx.Module, nnx.Optimizer, dict[str, jax.Array]],
        ) -> tuple[nnx.Module, nnx.Optimizer, dict[str, jax.Array]]:
            model, optimizer, metric_totals = carry

            # Compute actual step for this batch (for KL annealing, etc.)
            step = base_step + i

            # RNG for this step
            step_rng = jax.random.fold_in(epoch_rng, i)

            # Dynamic batch slicing inside JIT (no Python loop!)
            batch_data = jax.lax.dynamic_slice_in_dim(staged_data, i * batch_size, batch_size)
            batch = {data_key: batch_data}

            # Compute loss and gradients - step is passed dynamically!
            def compute_loss(m: nnx.Module) -> tuple[jax.Array, dict[str, jax.Array]]:
                return loss_fn(m, batch, step_rng, step)

            (_, metrics), grads = nnx.value_and_grad(compute_loss, has_aux=True)(model)

            # Update model with gradients
            optimizer.update(model, grads)

            # Accumulate all metrics using tree_map (pytree-compatible)
            new_totals = jax.tree.map(lambda a, b: a + b, metric_totals, metrics)

            return model, optimizer, new_totals

        # Initialize carry state
        init_carry = (model, optimizer, metric_totals)

        # Run fori_loop - entire epoch compiled as single XLA computation!
        _model, _optimizer, final_metrics = nnx.fori_loop(0, num_batches, body_fn, init_carry)

        return final_metrics

    return run_epoch


# Cache for epoch runners keyed by (loss_fn_id, batch_size, num_batches, data_key)
_epoch_runner_cache: dict[tuple, _CachedEpochRunner] = {}


def _get_or_create_epoch_runner(
    loss_fn: LossFn,
    batch_size: int,
    num_batches: int,
    data_key: str,
    model: nnx.Module,
    data: jax.Array,
    rng: jax.Array,
) -> _CachedEpochRunner:
    """Get or create a cached epoch runner for the given configuration.

    Only runs the probe forward pass when creating a new cache entry.
    """
    cache_key = (id(loss_fn), batch_size, num_batches, data_key)

    if cache_key not in _epoch_runner_cache:
        # Probe to discover metrics structure (only on cache miss)
        probe_batch = {data_key: data[:batch_size]}
        probe_rng = jax.random.fold_in(rng, num_batches)
        # Pass step=0 for probing (just need metrics structure)
        _, sample_metrics = loss_fn(model, probe_batch, probe_rng, jnp.array(0))
        metric_keys = tuple(sample_metrics.keys())

        # Create and cache the epoch runner
        run_epoch = _create_epoch_runner(loss_fn, batch_size, num_batches, data_key)
        _epoch_runner_cache[cache_key] = _CachedEpochRunner(run_epoch, metric_keys)

    return _epoch_runner_cache[cache_key]


def train_epoch_staged(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data: jax.Array,
    batch_size: int,
    rng: jax.Array,
    loss_fn: LossFn,
    base_step: int = 0,
    data_key: str = "image",
) -> tuple[int, dict[str, float]]:
    """Train one epoch with nnx.fori_loop - entire epoch JIT-compiled.

    Compatible with all existing trainers via create_loss_fn().

    This function uses nnx.jit + nnx.fori_loop to JIT-compile the ENTIRE epoch,
    eliminating Python loop overhead for maximum performance (100-500x speedup).

    The step counter increments inside the fori_loop, enabling proper KL annealing
    and other step-dependent schedules to work correctly.

    Args:
        model: NNX model to train (mutated in place)
        optimizer: NNX optimizer (mutated in place)
        data: Pre-staged data array, shape (N, ...)
        batch_size: Batch size
        rng: Epoch RNG key
        loss_fn: From trainer.create_loss_fn() - signature:
            (model, batch, rng, step) -> (loss, metrics)
        base_step: Starting step number (for scheduling)
        data_key: Key to use in batch dict (default: "image")

    Returns:
        (final_step, averaged_metrics)

    Example:
        # Works with ANY existing trainer
        vae_trainer = VAETrainer(config)
        loss_fn = vae_trainer.create_loss_fn(loss_type="bce")
        step, metrics = train_epoch_staged(model, opt, data, 128, rng, loss_fn)

        diffusion_trainer = DiffusionTrainer(noise_schedule, config)
        loss_fn = diffusion_trainer.create_loss_fn()
        step, metrics = train_epoch_staged(model, opt, data, 64, rng, loss_fn)
    """
    num_samples = data.shape[0]
    num_batches = num_samples // batch_size

    if num_batches == 0:
        return base_step, {}

    # Get cached epoch runner (probes only on cache miss)
    cached = _get_or_create_epoch_runner(
        loss_fn, batch_size, num_batches, data_key, model, data, rng
    )

    # Initialize zero accumulators matching the cached metrics structure
    init_metrics = {key: jnp.array(0.0) for key in cached.metric_keys}

    # Execute JIT-compiled epoch with base_step for proper scheduling
    total_metrics = cached.run_epoch(
        model, optimizer, data, rng, init_metrics, jnp.array(base_step)
    )

    # Compute average metrics
    final_step = base_step + num_batches
    avg_metrics = {key: float(val) / num_batches for key, val in total_metrics.items()}

    return final_step, avg_metrics
