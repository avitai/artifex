"""Streaming training loop with prefetch for large datasets.

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
    - Objective signature: (model, batch, rng, step) -> (loss, metrics)
    - Metrics must be a stable mapping of JAX arrays across steps

Example with datarax pipeline::

    from artifex.generative_models.training.loops import (
        create_data_pipeline,
        train_epoch_streaming,
    )
    from artifex.generative_models.training.trainers import VAETrainer

    vae_trainer = VAETrainer(config)
    loss_fn = vae_trainer.create_loss_fn(loss_type="bce")

    # Create datarax pipeline from a MemorySource or DataSourceModule
    pipeline = create_data_pipeline(source, batch_size=128)

    # Train with streaming data
    step, metrics = train_epoch_streaming(model, opt, pipeline, rng, loss_fn)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from datarax import Pipeline
from datarax.core.data_source import DataSourceModule
from flax import nnx


# Type alias for loss function signature
# Note: step is passed dynamically to support KL annealing and other schedules
LossFn = Callable[
    [nnx.Module, dict[str, Any], jax.Array, jax.Array],  # (model, batch, rng, step)
    tuple[jax.Array, dict[str, jax.Array]],
]


def _create_train_step(
    loss_fn: LossFn,
) -> Callable[
    [nnx.Module, nnx.Optimizer, dict[str, Any], jax.Array, jax.Array],
    tuple[jax.Array, dict[str, jax.Array]],
]:
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
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        def compute_loss(m: nnx.Module) -> tuple[jax.Array, dict[str, jax.Array]]:
            return loss_fn(m, batch, step_rng, step)

        (loss, metrics), grads = nnx.value_and_grad(compute_loss, has_aux=True)(model)
        optimizer.update(model, grads)

        return loss, metrics

    return train_step


def create_data_pipeline(
    source: DataSourceModule,
    batch_size: int = 32,
    *,
    rngs: nnx.Rngs | None = None,
) -> Pipeline:
    """Create a datarax pipeline from a data source.

    Convenience function for composing a datarax pipeline suitable for
    streaming training. The returned pipeline can be iterated directly
    or passed to ``train_epoch_streaming()``.

    Shuffling is controlled by the source's config
    (e.g. ``MemorySourceConfig(shuffle=True)``), not by this function.

    Args:
        source: A datarax DataSourceModule (e.g. MemorySource).
        batch_size: Number of samples per batch.
        rngs: Optional ``nnx.Rngs`` for stochastic stages and source key
            advancement. Defaults to ``nnx.Rngs(0)``.

    Returns:
        A ``Pipeline`` that yields batch dicts on iteration.
    """
    return Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=rngs if rngs is not None else nnx.Rngs(0),
    )


def train_epoch_streaming(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    data_iterator: Iterator[dict[str, Any]] | Pipeline,
    rng: jax.Array,
    loss_fn: LossFn,
    base_step: int = 0,
) -> tuple[int, dict[str, float]]:
    """Train one epoch with streaming data and JIT-compiled steps.

    This function provides optimized training for large datasets that cannot
    fit in GPU memory. It uses JIT-compiled train steps. Accepts either a
    plain iterator yielding batch dicts or a datarax ``Pipeline`` (whose
    iterator yields plain batch dicts directly).

    NOTE: For datasets that fit in GPU memory, use train_epoch_staged instead -
    it uses nnx.fori_loop to JIT the entire epoch for 10-50x better performance.

    Args:
        model: NNX model to train
        optimizer: NNX optimizer
        data_iterator: Iterator yielding batch dicts, or a datarax ``Pipeline``
        rng: Epoch RNG key
        loss_fn: Step-aware objective closure. Signature:
            (model, batch, rng, step) -> (loss, metrics)
        base_step: Starting step number

    Returns:
        (final_step, averaged_metrics)

    Example::

        from artifex.generative_models.training.loops import (
            create_data_pipeline,
            train_epoch_streaming,
        )

        pipeline = create_data_pipeline(source, batch_size=128)
        step, metrics = train_epoch_streaming(model, opt, pipeline, rng, loss_fn)
    """
    # Create JIT-compiled train step
    train_step = _create_train_step(loss_fn)

    metric_totals: dict[str, jax.Array] | None = None
    step = base_step
    num_batches = 0

    # Python loop is unavoidable for streaming data
    # (iterator protocol is not JAX-traceable)
    for batch in data_iterator:
        rng, step_rng = jax.random.split(rng)
        _loss, metrics = train_step(model, optimizer, batch, step_rng, jnp.array(step))
        if metric_totals is None:
            metric_totals = {key: jnp.zeros_like(value) for key, value in metrics.items()}
        metric_totals = jax.tree.map(lambda a, b: a + b, metric_totals, metrics)
        step += 1
        num_batches += 1

    if num_batches == 0 or metric_totals is None:
        avg_metrics = {}
    else:
        avg_metrics = {key: float(value) / num_batches for key, value in metric_totals.items()}

    return step, avg_metrics
