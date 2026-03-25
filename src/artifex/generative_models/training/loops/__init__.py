"""High-performance training loops for generative models.

This module provides optimized training loops for explicit step-aware objective
closures with signature `(model, batch, rng, step) -> (loss, metrics)`.

Two strategies are provided:

1. **Staged (nnx.fori_loop)**: For datasets that fit in GPU memory (~10% VRAM)
   - 100-500x speedup by eliminating Python loop overhead
   - Data must be pre-staged on GPU with jax.device_put()
   - Callbacks work at epoch-level only

2. **Streaming (JIT + prefetch)**: For large datasets that must stream from CPU
   - 5-20x speedup via JIT-compiled steps and prefetch
   - Works with any data iterator/generator or datarax DAGExecutor pipeline
   - Supports batch-level callbacks

3. **create_data_pipeline**: Convenience function to create a datarax pipeline
   from a DataSourceModule (e.g. MemorySource).

Example:
    ```python
    from artifex.generative_models.training import (
        create_data_pipeline,
        train_epoch_staged,
        train_epoch_streaming,
    )
    from artifex.generative_models.training.trainers import VAETrainer

    # Create trainer and loss function
    trainer = VAETrainer(config)
    loss_fn = trainer.create_loss_fn(loss_type="bce")

    # Option 1: Staged training (small datasets, maximum speed)
    data = jax.device_put(train_images)
    step, metrics = train_epoch_staged(model, opt, data, 128, rng, loss_fn)

    # Option 2: Streaming with datarax pipeline (large datasets)
    pipeline = create_data_pipeline(source, batch_size=128)
    step, metrics = train_epoch_streaming(model, opt, pipeline, rng, loss_fn)
    ```
"""

from artifex.generative_models.training.loops.staged import train_epoch_staged
from artifex.generative_models.training.loops.streaming import (
    create_data_pipeline,
    train_epoch_streaming,
)


__all__ = [
    "create_data_pipeline",
    "train_epoch_staged",
    "train_epoch_streaming",
]
