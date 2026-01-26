"""High-performance training loops for generative models.

This module provides optimized training loops that work with ANY existing trainer
via the `create_loss_fn()` pattern. All 6 trainers (VAE, GAN, Diffusion, Flow,
Energy, Autoregressive) already implement this pattern.

Two strategies are provided:

1. **Staged (nnx.fori_loop)**: For datasets that fit in GPU memory (~10% VRAM)
   - 100-500x speedup by eliminating Python loop overhead
   - Data must be pre-staged on GPU with jax.device_put()
   - Callbacks work at epoch-level only

2. **Streaming (JIT + prefetch)**: For large datasets that must stream from CPU
   - 5-20x speedup via JIT-compiled steps and prefetch
   - Works with any data iterator/generator
   - Supports batch-level callbacks

Example:
    ```python
    from artifex.generative_models.training import (
        train_epoch_staged,
        train_epoch_streaming,
    )
    from artifex.generative_models.training.trainers import VAETrainer

    # Create trainer and loss function (works with ANY trainer)
    trainer = VAETrainer(config)
    loss_fn = trainer.create_loss_fn(step=0, loss_type="bce")

    # Option 1: Staged training (small datasets, maximum speed)
    data = jax.device_put(train_images)
    step, metrics = train_epoch_staged(model, opt, data, 128, rng, loss_fn)

    # Option 2: Streaming training (large datasets)
    step, metrics = train_epoch_streaming(model, opt, data_iter, rng, loss_fn)
    ```
"""

from artifex.generative_models.training.loops.staged import train_epoch_staged
from artifex.generative_models.training.loops.streaming import train_epoch_streaming


__all__ = [
    "train_epoch_staged",
    "train_epoch_streaming",
]
