# TensorBoard Integration

Visualize training metrics, generated samples, and model architecture using TensorBoard with Artifex.

## Overview

TensorBoard provides powerful visualization tools for machine learning experiments. Artifex integrates with TensorBoard to log metrics, visualize generated samples, and track training progress.

!!! tip "TensorBoard Benefits"
    - **Real-time Monitoring**: Watch training progress live
    - **Visualization**: Interactive charts and image galleries
    - **Lightweight**: No external services required
    - **Integration**: Works seamlessly with JAX/Flax

<div class="grid cards" markdown>

- :material-chart-line:{ .lg .middle } **Metrics Logging**

    ---

    Track scalars, histograms, and custom metrics

    [:octicons-arrow-right-24: Logging Guide](#logging-patterns)

- :material-image-multiple:{ .lg .middle } **Visualization**

    ---

    Visualize samples, latent spaces, and attention

    [:octicons-arrow-right-24: Visualization Guide](#visualization)

- :material-cog:{ .lg .middle } **Training Integration**

    ---

    Integrate TensorBoard with Artifex training

    [:octicons-arrow-right-24: Integration Guide](#integration-with-training)

</div>

---

## Quick Start with Built-in Callback

For most use cases, use the built-in `TensorBoardLoggerCallback`:

```python
from artifex.generative_models.training.callbacks import (
    TensorBoardLoggerCallback,
    TensorBoardLoggerConfig,
)

callback = TensorBoardLoggerCallback(TensorBoardLoggerConfig(
    log_dir="logs/experiment-1",
    flush_secs=60,
    log_every_n_steps=10,
))

trainer.fit(callbacks=[callback])

# View with: tensorboard --logdir logs
```

See [Logging Callbacks](../../training/logging.md) for full documentation.

### Profiling Traces

Use `JAXProfiler` to capture performance traces viewable in TensorBoard's Profile tab:

```python
from artifex.generative_models.training.callbacks import (
    JAXProfiler,
    ProfilingConfig,
)

profiler = JAXProfiler(ProfilingConfig(
    log_dir="logs/profiles",
    start_step=10,  # Skip JIT warmup
    end_step=20,    # Profile 10 steps
))

trainer.fit(callbacks=[profiler])

# View traces: tensorboard --logdir logs/profiles
```

The Profile tab shows:

- XLA compilation times
- Device (GPU/TPU) execution times
- Memory allocation patterns
- Kernel execution traces

See [Profiling Callbacks](../../training/profiling.md) for complete documentation.

The sections below cover advanced TensorBoard features not available through the callback.

---

## Prerequisites

```bash
# Install TensorBoard
pip install tensorboard tensorboardX

# Or using uv
uv pip install tensorboard tensorboardX
```

---

## Logging Patterns

### Basic Scalar Logging

Track training metrics over time.

```python
from torch.utils.tensorboard import SummaryWriter
import jax.numpy as jnp
import numpy as np

class TensorBoardLogger:
    """TensorBoard logging for Artifex models."""

    def __init__(self, log_dir: str = "./runs/experiment"):
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def log_scalars(self, metrics: dict, step: int = None):
        """Log scalar metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step (uses internal counter if None)
        """
        step = step if step is not None else self.step

        for name, value in metrics.items():
            # Convert JAX arrays to Python scalars
            if isinstance(value, jnp.ndarray):
                value = float(value)

            self.writer.add_scalar(name, value, step)

        if step is None:
            self.step += 1

    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        step: int,
    ):
        """Log training step metrics."""
        self.log_scalars({
            "train/loss": loss,
            "train/learning_rate": learning_rate,
        }, step=step)

    def log_validation(
        self,
        val_loss: float,
        metrics: dict,
        epoch: int,
    ):
        """Log validation metrics."""
        self.log_scalars({
            "val/loss": val_loss,
            **{f"val/{k}": v for k, v in metrics.items()}
        }, step=epoch)

    def close(self):
        """Close the writer."""
        self.writer.close()
```

### Image Logging

Visualize generated samples.

```python
class ImageLogger:
    """Log images to TensorBoard."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def log_images(
        self,
        images: jax.Array,
        tag: str,
        step: int,
        max_images: int = 16,
    ):
        """Log image batch.

        Args:
            images: Image batch (B, H, W, C) or (B, C, H, W)
            tag: Tag for the images
            step: Global step
            max_images: Maximum number of images to log
        """
        # Convert to numpy and limit number
        images_np = np.array(images[:max_images])

        # Denormalize from [-1, 1] to [0, 1]
        images_np = (images_np + 1) / 2
        images_np = np.clip(images_np, 0, 1)

        # Ensure channel-first format (C, H, W)
        if images_np.shape[-1] in [1, 3]:  # Channel-last
            images_np = np.transpose(images_np, (0, 3, 1, 2))

        # Log as image grid
        self.writer.add_images(tag, images_np, step)

    def log_image_comparison(
        self,
        real_images: jax.Array,
        generated_images: jax.Array,
        step: int,
    ):
        """Log real vs generated comparison."""
        self.log_images(real_images, "comparison/real", step)
        self.log_images(generated_images, "comparison/generated", step)
```

### Histogram Logging

Track parameter distributions.

```python
from flax import nnx

class HistogramLogger:
    """Log parameter histograms."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def log_model_parameters(
        self,
        model,
        step: int,
    ):
        """Log all model parameter histograms.

        Args:
            model: Flax NNX model
            step: Global step
        """
        state = nnx.state(model)

        for name, param in state.items():
            if isinstance(param, jnp.ndarray):
                # Convert to numpy
                param_np = np.array(param)

                # Log histogram
                self.writer.add_histogram(
                    f"parameters/{name}",
                    param_np,
                    step
                )

    def log_gradients(
        self,
        grads: dict,
        step: int,
    ):
        """Log gradient histograms."""
        for name, grad in grads.items():
            if isinstance(grad, jnp.ndarray):
                grad_np = np.array(grad)
                self.writer.add_histogram(
                    f"gradients/{name}",
                    grad_np,
                    step
                )
```

---

## Visualization

### Training Curves

Monitor loss and metrics over time.

```python
class TrainingVisualizer:
    """Visualize training progress."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_loss_components(
        self,
        total_loss: float,
        reconstruction_loss: float,
        kl_loss: float,
        step: int,
    ):
        """Log VAE loss components."""
        self.writer.add_scalars("loss_components", {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "kl_divergence": kl_loss,
        }, step)

    def log_gan_losses(
        self,
        g_loss: float,
        d_loss: float,
        d_real: float,
        d_fake: float,
        step: int,
    ):
        """Log GAN training metrics."""
        self.writer.add_scalars("gan/losses", {
            "generator": g_loss,
            "discriminator": d_loss,
        }, step)

        self.writer.add_scalars("gan/discriminator", {
            "real_score": d_real,
            "fake_score": d_fake,
        }, step)
```

### Sample Galleries

Create grids of generated samples.

```python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

def create_sample_grid(images: np.ndarray, nrow: int = 8) -> np.ndarray:
    """Create image grid for visualization.

    Args:
        images: Batch of images (B, H, W, C)
        nrow: Number of images per row

    Returns:
        Grid image as numpy array
    """
    batch_size, h, w, c = images.shape
    nrow = min(nrow, batch_size)
    ncol = (batch_size + nrow - 1) // nrow

    # Create figure
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    axes = axes.flatten() if batch_size > 1 else [axes]

    for idx, (ax, img) in enumerate(zip(axes, images)):
        if c == 1:  # Grayscale
            ax.imshow(img.squeeze(), cmap='gray')
        else:  # RGB
            ax.imshow(img)
        ax.axis('off')

    # Hide extra subplots
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Convert to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    grid = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    grid = grid.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return grid


def log_sample_gallery(
    logger: TensorBoardLogger,
    model,
    num_samples: int,
    step: int,
    rngs,
):
    """Log gallery of generated samples."""
    # Generate samples
    samples = model.sample(num_samples=num_samples, rngs=rngs)

    # Convert and denormalize
    samples_np = np.array(samples)
    samples_np = ((samples_np + 1) / 2 * 255).astype(np.uint8)

    # Create grid
    grid = create_sample_grid(samples_np)

    # Log to TensorBoard
    logger.writer.add_image(
        "samples/generated",
        grid,
        step,
        dataformats='HWC'
    )
```

### Latent Space Visualization

Visualize learned latent representations.

```python
def log_latent_space(
    logger: TensorBoardLogger,
    model,
    test_data: jax.Array,
    labels: jax.Array,
    step: int,
):
    """Log latent space embedding.

    Args:
        logger: TensorBoard logger
        model: Trained model with encoder
        test_data: Test images
        labels: Image labels
        step: Global step
    """
    # Encode to latent space
    latents, _ = model.encode(test_data)
    latents_np = np.array(latents)
    labels_np = np.array(labels)

    # Log embedding
    logger.writer.add_embedding(
        latents_np,
        metadata=labels_np.tolist(),
        label_img=test_data,
        global_step=step,
        tag="latent_space"
    )
```

---

## Integration with Training

### TensorBoard Callback

Integrate with Artifex training loop.

```python
from artifex.generative_models.training import Trainer

class TensorBoardTrainer(Trainer):
    """Trainer with TensorBoard logging."""

    def __init__(
        self,
        model,
        config: dict,
        log_dir: str = "./runs/experiment",
        **kwargs
    ):
        super().__init__(model, config, **kwargs)

        # Initialize TensorBoard
        self.tb_logger = TensorBoardLogger(log_dir)
        self.log_frequency = config.get("tb_log_frequency", 100)

    def on_train_step_end(self, step: int, loss: float, metrics: dict):
        """Log after each training step."""
        if step % self.log_frequency == 0:
            self.tb_logger.log_scalars({
                "train/loss": loss,
                **{f"train/{k}": v for k, v in metrics.items()}
            }, step=step)

    def on_validation_end(self, epoch: int, val_metrics: dict):
        """Log after validation."""
        self.tb_logger.log_scalars({
            f"val/{k}": v for k, v in val_metrics.items()
        }, step=epoch)

        # Log generated samples
        samples = self.model.sample(num_samples=16, rngs=self.rngs)
        image_logger = ImageLogger(self.tb_logger.writer)
        image_logger.log_images(samples, "samples/generated", epoch)

    def on_training_end(self):
        """Close TensorBoard on training end."""
        self.tb_logger.close()
```

### Complete Example

Full training example with TensorBoard.

```python
from flax import nnx
import jax
from artifex.generative_models.models.vae import VAE
from artifex.generative_models.training import Trainer

# Create model
model = VAE(
    latent_dim=128,
    image_shape=(28, 28, 1),
    rngs=nnx.Rngs(0),
)

# Training configuration
config = {
    "learning_rate": 1e-4,
    "batch_size": 128,
    "num_epochs": 50,
    "tb_log_frequency": 100,
}

# Create trainer with TensorBoard
trainer = TensorBoardTrainer(
    model=model,
    config=config,
    log_dir="./runs/vae_experiment",
)

# Train
trainer.train(train_data, val_data)

# View results
print("To view TensorBoard, run:")
print("tensorboard --logdir=./runs")
```

---

## Launching TensorBoard

### Basic Launch

```bash
# Launch TensorBoard
tensorboard --logdir=./runs

# Custom port
tensorboard --logdir=./runs --port=6007

# Multiple experiments
tensorboard --logdir=./runs --reload_interval=5
```

### Comparing Experiments

```bash
# Compare multiple experiments
tensorboard --logdir_spec=baseline:./runs/baseline,improved:./runs/improved
```

---

## Best Practices

### DO

!!! success "Recommended Practices"
    ✅ **Organize logs** by experiment in separate directories

    ✅ **Log periodically** (every 100-1000 steps)

    ✅ **Use meaningful tags** for metrics and images

    ✅ **Log validation samples** to track generation quality

    ✅ **Close writer** when training completes

### DON'T

!!! danger "Avoid These Mistakes"
    ❌ **Don't log every step** (creates huge files)

    ❌ **Don't log high-res images** frequently (use max_images)

    ❌ **Don't forget to flush** the writer periodically

    ❌ **Don't reuse log directories** without clearing

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **TensorBoard not showing data** | Data not flushed | Call `writer.flush()` or close writer |
| **Large log files** | Logging too frequently | Reduce logging frequency |
| **Images not appearing** | Wrong format | Ensure channel-first format (C, H, W) |
| **Port already in use** | TensorBoard running | Use different port with `--port` |
| **Slow performance** | Too many logs | Reduce log frequency or clear old runs |

---

## Summary

TensorBoard provides essential visualization:

- **Real-time Monitoring**: Track training progress live
- **Scalar Metrics**: Loss curves and validation metrics
- **Image Galleries**: Visualize generated samples
- **Histograms**: Monitor parameter distributions
- **Embeddings**: Explore latent spaces
- **Profiling Traces**: Analyze XLA compilation and device execution

Start visualizing your training today!

---

## Next Steps

<div class="grid cards" markdown>

- :material-chart-line:{ .lg .middle } **Weights & Biases**

    ---

    Advanced experiment tracking and sweeps

    [:octicons-arrow-right-24: W&B Integration](wandb.md)

- :material-cloud-upload:{ .lg .middle } **HuggingFace Hub**

    ---

    Share models with the community

    [:octicons-arrow-right-24: HuggingFace Integration](huggingface.md)

- :material-cog:{ .lg .middle } **Training Guide**

    ---

    Master the training system

    [:octicons-arrow-right-24: Training Guide](../training/training-guide.md)

- :material-chart-box:{ .lg .middle } **Benchmarking**

    ---

    Evaluate models comprehensively

    [:octicons-arrow-right-24: Evaluation Framework](../../benchmarks/index.md)

</div>
