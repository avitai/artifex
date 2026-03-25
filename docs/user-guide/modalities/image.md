# Image Modality Guide

This guide covers working with image data in Artifex, including image representations, datasets, preprocessing, and best practices for image-based generative models.

## Overview

Artifex's image modality provides a unified interface for working with different image formats and resolutions. It supports RGB, RGBA, and grayscale images with configurable preprocessing and a small retained augmentation helper surface.

<div class="grid cards" markdown>

- :material-palette:{ .lg .middle } **Multiple Representations**

    ---

    Support for RGB, RGBA, and grayscale images with automatic channel handling

- :material-resize:{ .lg .middle } **Flexible Resolutions**

    ---

    Work with any image size from 28x28 to 512x512 and beyond

- :material-tune:{ .lg .middle } **Preprocessing Pipeline**

    ---

    Built-in normalization, resizing, and validation

- :material-image-multiple:{ .lg .middle } **Synthetic Datasets**

    ---

    Ready-to-use synthetic datasets for testing and development

- :material-auto-fix:{ .lg .middle } **Basic Augmentation**

    ---

    Horizontal flip and brightness jitter through the retained augmentation helper

- :material-speedometer:{ .lg .middle } **JAX-Native**

    ---

    Full JAX compatibility with JIT compilation and GPU acceleration

</div>

## Image Representations

### Supported Formats

Artifex supports three image representations:

```python
from artifex.generative_models.modalities.image.base import ImageRepresentation

# RGB images (3 channels)
ImageRepresentation.RGB

# RGBA images (4 channels with alpha)
ImageRepresentation.RGBA

# Grayscale images (1 channel)
ImageRepresentation.GRAYSCALE
```

### Configuring Image Modality

```python
from artifex.generative_models.modalities import ImageModality
from artifex.generative_models.modalities.image.base import (
    ImageModalityConfig,
    ImageRepresentation
)
from flax import nnx

# Initialize RNG
rngs = nnx.Rngs(0)

# RGB configuration (64x64)
rgb_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=64,
    width=64,
    channels=3,  # Auto-determined from representation if None
    normalize=True,  # Normalize to [0, 1]
    augmentation=False,
    resize_method="bilinear"
)

# Create modality
rgb_modality = ImageModality(config=rgb_config, rngs=rngs)

# RGBA configuration
rgba_config = ImageModalityConfig(
    representation=ImageRepresentation.RGBA,
    height=128,
    width=128,
    channels=4  # Alpha channel included
)

rgba_modality = ImageModality(config=rgba_config, rngs=rngs)

# Grayscale configuration
grayscale_config = ImageModalityConfig(
    representation=ImageRepresentation.GRAYSCALE,
    height=28,
    width=28,
    channels=1
)

grayscale_modality = ImageModality(config=grayscale_config, rngs=rngs)
```

### Image Shape Properties

```python
# Access image dimensions
print(f"Image shape: {rgb_modality.image_shape}")  # (64, 64, 3)
print(f"Output shape: {rgb_modality.output_shape}")  # (64, 64, 3)

# For MNIST-like
print(f"Grayscale shape: {grayscale_modality.image_shape}")  # (28, 28, 1)
```

## Image Datasets

### Synthetic Image Datasets

Artifex provides several synthetic dataset types for testing and development.
All datasets are created via the `create_image_dataset()` factory, which returns
a `MemorySource` instance from datarax:

#### Random Patterns

```python
from artifex.generative_models.modalities.image.datasets import create_image_dataset

# Random noise patterns
random_dataset = create_image_dataset(
    "synthetic",
    config=rgb_config,
    rngs=rngs,
    dataset_size=10000,
    pattern_type="random",
)

# Get batch (stateful — advances internal index)
batch = random_dataset.get_batch(32)
print(batch["images"].shape)  # (32, 64, 64, 3)

# Each image is filled with uniform random noise
```

#### Gradient Patterns

```python
# Linear gradients with varying directions
gradient_dataset = create_image_dataset(
    "synthetic",
    config=rgb_config,
    rngs=rngs,
    dataset_size=10000,
    pattern_type="gradient",
)

# Gradients have:
# - Random directions
# - Smooth color transitions (for RGB)
# - Sinusoidal variations for visual interest
```

#### Checkerboard Patterns

```python
# Checkerboard patterns with random sizes
checkerboard_dataset = create_image_dataset(
    "synthetic",
    config=rgb_config,
    rngs=rngs,
    dataset_size=10000,
    pattern_type="checkerboard",
)

# Checkerboards have:
# - Random tile sizes (4-16 pixels)
# - Binary black/white pattern
# - Repeated across color channels
```

#### Circular Patterns

```python
# Circular patterns with random positions and radii
circles_dataset = create_image_dataset(
    "synthetic",
    config=rgb_config,
    rngs=rngs,
    dataset_size=10000,
    pattern_type="circles",
)

# Circles have:
# - Random center positions
# - Random radii
# - Gaussian noise for variation
```

### MNIST-Like Datasets

For digit-like pattern recognition:

```python
from artifex.generative_models.modalities.image.datasets import create_image_dataset

# Configure for MNIST-like images (28x28 grayscale)
mnist_config = ImageModalityConfig(
    representation=ImageRepresentation.GRAYSCALE,
    height=28,
    width=28,
    channels=1,
    normalize=True
)

# Create MNIST-like dataset (returns MemorySource)
mnist_dataset = create_image_dataset(
    "mnist_like",
    config=mnist_config,
    rngs=rngs,
    dataset_size=60000,
    num_classes=10,
)

# Get labeled batch
batch = mnist_dataset.get_batch(128)
print(batch["images"].shape)  # (128, 28, 28, 1)
print(batch["labels"].shape)  # (128,)

# Iterate over individual samples
for sample in mnist_dataset:
    image = sample["images"]  # (28, 28, 1)
    label = sample["labels"]  # Scalar label
    print(f"Label: {label}, Image shape: {image.shape}")
    break
```

**Generated patterns:**

- Class 0: Circle (hollow)
- Class 1: Vertical line
- Class 2: Horizontal line
- Additional classes follow similar geometric patterns

### Shuffled Datasets

Enable shuffling so each epoch sees a different ordering:

```python
shuffled_dataset = create_image_dataset(
    "synthetic",
    config=rgb_config,
    rngs=rngs,
    shuffle=True,
    dataset_size=5000,
    pattern_type="gradient",
)

# Iteration order is randomised per epoch
for sample in shuffled_dataset:
    print(sample["images"].shape)  # (64, 64, 3)
    break
```

## Image Preprocessing

### Normalization

```python
import jax.numpy as jnp

# Images in [0, 255] → [0, 1]
def normalize_uint8_images(images):
    """Normalize uint8 images to [0, 1]."""
    return images.astype(jnp.float32) / 255.0

# Images in [0, 1] → [-1, 1]
def normalize_to_symmetric(images):
    """Normalize to [-1, 1] range."""
    return images * 2.0 - 1.0

# Standardization (mean=0, std=1)
def standardize_images(images):
    """Standardize images to zero mean, unit variance."""
    mean = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    std = jnp.std(images, axis=(1, 2, 3), keepdims=True)
    return (images - mean) / (std + 1e-8)

# Usage
raw_images = jnp.array([...])  # Raw pixel values
normalized = normalize_uint8_images(raw_images)
standardized = standardize_images(normalized)
```

### Resizing

```python
import jax
from jax import image as jax_image

def resize_images(images, target_height, target_width, method="bilinear"):
    """Resize images to target dimensions.

    Args:
        images: Input images (N, H, W, C)
        target_height: Target height
        target_width: Target width
        method: Resize method ("bilinear" or "nearest")

    Returns:
        Resized images (N, target_height, target_width, C)
    """
    batch_size = images.shape[0]
    channels = images.shape[3]

    if method == "bilinear":
        # Use JAX's resize function
        resized = jax_image.resize(
            images,
            shape=(batch_size, target_height, target_width, channels),
            method="bilinear"
        )
    elif method == "nearest":
        resized = jax_image.resize(
            images,
            shape=(batch_size, target_height, target_width, channels),
            method="nearest"
        )
    else:
        raise ValueError(f"Unknown resize method: {method}")

    return resized

# Usage
images = jnp.array([...])  # (N, 32, 32, 3)
resized = resize_images(images, 64, 64, method="bilinear")
print(resized.shape)  # (N, 64, 64, 3)
```

### Using Modality Processor

```python
# Create modality with preprocessing
config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=64,
    width=64,
    normalize=True
)

modality = ImageModality(config=config, rngs=rngs)

# Process raw images
raw_images = jnp.array([...])  # Any shape
processed = modality.process(raw_images)

# Processed images:
# - Resized to (64, 64)
# - Normalized to [0, 1]
# - Batch dimension handled automatically
```

## Image Augmentation

The retained image helper layer exposes one small augmentation surface through
`AugmentationProcessor`: horizontal flips plus brightness jitter.

Rotations, contrast/saturation/hue transforms, crops, zoom, and noise
augmentations are not part of
`artifex.generative_models.modalities.image`.

```python
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities.image import (
    AugmentationProcessor,
    ImageModalityConfig,
    ImageRepresentation,
)

rngs = nnx.Rngs(0)
config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=64,
    width=64,
    normalize=True,
    augmentation=True,
)

augmenter = AugmentationProcessor(config=config, brightness_range=0.1, rngs=rngs)
images = jnp.ones((8, 64, 64, 3), dtype=jnp.float32) * 0.5
augmented = augmenter.augment_batch(images)

print(augmented.shape)  # (8, 64, 64, 3)
```

If you need richer augmentation pipelines, keep them in your dataset/training
stack rather than teaching them as modality-owned helpers.

## Image Helper Metrics

`compute_image_metrics(...)` provides modality-local helper metrics such as
`mse`, `psnr`, `ssim`, `ms_ssim`, `vendi_score`, and
`perceptual_distance`.

Benchmark-grade metrics such as FID, Inception Score, and LPIPS are owned by
`artifex.benchmarks.metrics.image`, not by this modality helper layer.

```python
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities.image import compute_image_metrics

rngs = nnx.Rngs(0)
generated = jnp.zeros((4, 64, 64, 3), dtype=jnp.float32)
reference = jnp.ones((4, 64, 64, 3), dtype=jnp.float32) * 0.25

metrics = compute_image_metrics(
    generated_images=generated,
    reference_images=reference,
    metrics=["mse", "psnr", "ssim"],
    rngs=rngs,
)

print(metrics.keys())  # dict_keys(["mse", "psnr", "ssim"])
```

## Working with Different Image Sizes

### Common Image Sizes

```python
# MNIST-like (28x28 grayscale)
mnist_config = ImageModalityConfig(
    representation=ImageRepresentation.GRAYSCALE,
    height=28,
    width=28
)

# CIFAR-like (32x32 RGB)
cifar_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=32,
    width=32
)

# Standard (64x64 RGB)
standard_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=64,
    width=64
)

# High-res (128x128 RGB)
highres_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=128,
    width=128
)

# Very high-res (256x256 RGB)
veryhighres_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=256,
    width=256
)
```

### Handling Non-Square Images

```python
# Wide images (16:9 aspect ratio)
wide_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=360,
    width=640
)

# Portrait images (9:16 aspect ratio)
portrait_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=640,
    width=360
)

# Custom aspect ratio
custom_config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=224,
    width=448  # 2:1 aspect ratio
)
```

## Complete Examples

### Example 1: Training with Augmentation

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities import ImageModality
from artifex.generative_models.modalities.image import (
    AugmentationProcessor,
    ImageModalityConfig,
    ImageRepresentation,
    create_image_dataset,
)

# Setup
rngs = nnx.Rngs(0)

config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=64,
    width=64,
    normalize=True
)

modality = ImageModality(config=config, rngs=rngs)
augmenter = AugmentationProcessor(config=config, brightness_range=0.1, rngs=rngs)

# Create datasets (MemorySource instances)
train_dataset = create_image_dataset(
    "synthetic",
    config=config,
    rngs=rngs,
    shuffle=True,
    dataset_size=10000,
    pattern_type="gradient",
)

val_dataset = create_image_dataset(
    "synthetic",
    config=config,
    rngs=rngs,
    dataset_size=1000,
    pattern_type="gradient",
)

# Training loop with augmentation
batch_size = 128
num_epochs = 10
key = jax.random.key(42)

for epoch in range(num_epochs):
    # Training
    num_batches = len(train_dataset) // batch_size

    for i in range(num_batches):
        # Get batch
        batch = train_dataset.get_batch(batch_size)

        # Apply retained modality-local augmentation helpers
        augmented = augmenter.augment_batch(batch["images"])

        # Training step (placeholder)
        # loss = train_step(model, augmented)

    # Reset for next epoch
    train_dataset.reset()

    # Validation (no augmentation)
    val_batches = len(val_dataset) // batch_size
    for i in range(val_batches):
        val_batch = val_dataset.get_batch(batch_size)
        # Validation step
        # val_loss = validate_step(model, val_batch["images"])

    val_dataset.reset()

    print(f"Epoch {epoch + 1}/{num_epochs} complete")
```

### Example 2: Multi-Resolution Training

```python
from artifex.generative_models.modalities.image import create_image_dataset

# Create datasets at multiple resolutions
resolutions = [32, 64, 128]
datasets = {}

for res in resolutions:
    res_config = ImageModalityConfig(
        representation=ImageRepresentation.RGB,
        height=res,
        width=res,
    )

    datasets[res] = create_image_dataset(
        "synthetic",
        config=res_config,
        rngs=rngs,
        dataset_size=5000,
        pattern_type="random",
    )

# Progressive training
for resolution in resolutions:
    print(f"Training at {resolution}x{resolution}")

    dataset = datasets[resolution]

    for epoch in range(5):
        for i in range(len(dataset) // 32):
            batch = dataset.get_batch(32)
            # Train at this resolution
            # loss = train_step(model, batch["images"], resolution)

        dataset.reset()
        print(f"  Epoch {epoch + 1}/5 at {resolution}x{resolution}")
```

### Example 3: Custom Image Dataset via MemorySource

```python
import jax
import jax.numpy as jnp
from flax import nnx
from datarax.sources import MemorySource, MemorySourceConfig

# Load your images into arrays (use PIL, OpenCV, etc.)
# For demonstration, generate placeholder data
num_images = 500
height, width, channels = 64, 64, 3

images = jnp.stack([
    jax.random.uniform(
        jax.random.key(i),
        (height, width, channels),
    )
    for i in range(num_images)
])

labels = jnp.array([i % 10 for i in range(num_images)])

# Wrap in a MemorySource for batching, shuffling, and iteration
source_config = MemorySourceConfig(shuffle=True)
custom_dataset = MemorySource(
    source_config,
    {"images": images, "labels": labels},
    rngs=nnx.Rngs(0),
)

# Get a batch (stateful — advances internal index)
batch = custom_dataset.get_batch(32)
print(batch["images"].shape)  # (32, 64, 64, 3)
print(batch["labels"].shape)  # (32,)

# Iterate over individual samples
for sample in custom_dataset:
    print(sample["images"].shape)  # (64, 64, 3)
    print(sample["labels"])
    break

# Reset internal state for the next epoch
custom_dataset.reset()
```

## Best Practices

### DO

!!! tip "Image Loading"
    - Use appropriate image resolution for your task
    - Normalize images to [0, 1] or [-1, 1] consistently
    - Choose representation that matches your data (RGB vs grayscale)
    - Validate image shapes before training
    - Cache preprocessed images when possible
    - Use synthetic datasets for testing pipelines

!!! tip "Augmentation"
    - Apply augmentation only during training, not validation
    - Keep modality-local augmentation expectations limited to horizontal flip and brightness jitter
    - Balance augmentation strength with training stability
    - Use vectorized batch augmentation through `AugmentationProcessor.augment_batch(...)`
    - Test retained augmentations visually before training

!!! tip "Performance"
    - Resize images to target resolution once
    - Use JAX's native image operations for GPU acceleration
    - Batch operations when possible
    - Clear image cache periodically for long runs
    - Profile image loading to identify bottlenecks
    - Consider mixed precision (float16) for memory savings

### DON'T

!!! danger "Common Mistakes"
    - Mix different image resolutions in same batch
    - Forget to normalize images
    - Apply augmentation during validation/testing
    - Teach rotations, crops, zoom, or noise helpers as if they lived in `artifex.generative_models.modalities.image`
    - Use non-JAX operations in data pipeline
    - Load full-resolution images if working with downscaled versions
    - Ignore color space (RGB vs BGR)
    - Use excessive augmentation that destroys image structure

!!! danger "Performance Issues"
    - Load images from disk in training loop
    - Use Python loops for image processing
    - Keep multiple copies of images in memory
    - Use very large batch sizes on limited GPU memory

!!! danger "Quality Issues"
    - Over-augment images (too much distortion)
    - Use inappropriate resize methods (nearest for photos)
    - Mix normalized and unnormalized images
    - Ignore aspect ratio when resizing
    - Assume benchmark metrics such as FID or LPIPS come from the modality helper layer

## Summary

This guide covered:

- **Image representations** - RGB, RGBA, and grayscale configurations
- **Image datasets** - Synthetic datasets with various patterns
- **Preprocessing** - Normalization, resizing, and validation
- **Augmentation** - Retained horizontal-flip and brightness-jitter helpers
- **Helper metrics** - Modality-local metrics plus the benchmark-package boundary
- **Different sizes** - Working with various image resolutions
- **Complete examples** - Training with retained helper augmentation, multi-resolution, custom datasets
- **Best practices** - DOs and DON'Ts for image data

## Next Steps

<div class="grid cards" markdown>

- :material-text:{ .lg .middle } **[Text Modality Guide](text.md)**

    ---

    Learn about text tokenization, vocabulary management, and sequence handling

- :material-volume-high:{ .lg .middle } **[Audio Modality Guide](audio.md)**

    ---

    Audio waveform processing, spectrograms, and audio augmentation

- :material-layers-triple:{ .lg .middle } **[Multi-modal Guide](multimodal.md)**

    ---

    Working with multiple modalities and aligned multi-modal datasets

- :material-api:{ .lg .middle } **[Data API Reference](../../api/data/loaders.md)**

    ---

    Complete API documentation for dataset classes and functions

</div>
