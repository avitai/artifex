# Image Modality Guide

This guide covers working with image data in Artifex, including image representations, datasets, preprocessing, and best practices for image-based generative models.

## Overview

Artifex's image modality provides a unified interface for working with different image formats and resolutions. It supports RGB, RGBA, and grayscale images with configurable preprocessing and augmentation.

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

- :material-auto-fix:{ .lg .middle } **Augmentation**

    ---

    Common image augmentation techniques (flip, rotate, brightness, contrast)

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

Artifex provides several synthetic dataset types for testing and development:

#### Random Patterns

```python
from artifex.generative_models.modalities.image.datasets import SyntheticImageDataset

# Random noise patterns
random_dataset = SyntheticImageDataset(
    config=rgb_config,
    dataset_size=10000,
    pattern_type="random",
    split="train",
    rngs=rngs
)

# Get batch
batch = random_dataset.get_batch(batch_size=32)
print(batch["images"].shape)  # (32, 64, 64, 3)

# Each image is filled with uniform random noise
```

#### Gradient Patterns

```python
# Linear gradients with varying directions
gradient_dataset = SyntheticImageDataset(
    config=rgb_config,
    dataset_size=10000,
    pattern_type="gradient",
    split="train",
    rngs=rngs
)

# Gradients have:
# - Random directions
# - Smooth color transitions (for RGB)
# - Sinusoidal variations for visual interest
```

#### Checkerboard Patterns

```python
# Checkerboard patterns with random sizes
checkerboard_dataset = SyntheticImageDataset(
    config=rgb_config,
    dataset_size=10000,
    pattern_type="checkerboard",
    split="train",
    rngs=rngs
)

# Checkerboards have:
# - Random tile sizes (4-16 pixels)
# - Binary black/white pattern
# - Repeated across color channels
```

#### Circular Patterns

```python
# Circular patterns with random positions and radii
circles_dataset = SyntheticImageDataset(
    config=rgb_config,
    dataset_size=10000,
    pattern_type="circles",
    split="train",
    rngs=rngs
)

# Circles have:
# - Random center positions
# - Random radii
# - Gaussian noise for variation
```

### MNIST-Like Datasets

For digit-like pattern recognition:

```python
from artifex.generative_models.modalities.image.datasets import MNISTLikeDataset

# Configure for MNIST-like images (28x28 grayscale)
mnist_config = ImageModalityConfig(
    representation=ImageRepresentation.GRAYSCALE,
    height=28,
    width=28,
    channels=1,
    normalize=True
)

# Create MNIST-like dataset
mnist_dataset = MNISTLikeDataset(
    config=mnist_config,
    dataset_size=60000,
    num_classes=10,
    split="train",
    rngs=rngs
)

# Get labeled batch
batch = mnist_dataset.get_batch(batch_size=128)
print(batch["images"].shape)  # (128, 28, 28, 1)
print(batch["labels"].shape)  # (128,)

# Iterate with labels
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

### Factory Function

```python
from artifex.generative_models.modalities.image.datasets import create_image_dataset

# Create dataset using factory
dataset = create_image_dataset(
    dataset_type="synthetic",  # or "mnist_like"
    config=rgb_config,
    pattern_type="gradient",
    dataset_size=5000,
    rngs=rngs
)

# MNIST-like via factory
mnist = create_image_dataset(
    dataset_type="mnist_like",
    config=mnist_config,
    dataset_size=60000,
    num_classes=10,
    rngs=rngs
)
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

### Basic Augmentations

```python
import jax
import jax.numpy as jnp

def random_horizontal_flip(image, key, prob=0.5):
    """Randomly flip image horizontally.

    Args:
        image: Input image (H, W, C)
        key: Random key
        prob: Probability of flipping

    Returns:
        Flipped or original image
    """
    flip = jax.random.bernoulli(key, prob)
    return jax.lax.cond(
        flip,
        lambda img: jnp.flip(img, axis=1),
        lambda img: img,
        image
    )

def random_vertical_flip(image, key, prob=0.5):
    """Randomly flip image vertically."""
    flip = jax.random.bernoulli(key, prob)
    return jax.lax.cond(
        flip,
        lambda img: jnp.flip(img, axis=0),
        lambda img: img,
        image
    )

def random_rotation(image, key):
    """Randomly rotate image by 0, 90, 180, or 270 degrees.

    Args:
        image: Input image (H, W, C)
        key: Random key

    Returns:
        Rotated image
    """
    k = jax.random.randint(key, (), 0, 4)
    return jnp.rot90(image, k=int(k), axes=(0, 1))

# Usage
key = jax.random.key(0)
keys = jax.random.split(key, 3)

image = jnp.array([...])  # (H, W, C)
image = random_horizontal_flip(image, keys[0])
image = random_vertical_flip(image, keys[1])
image = random_rotation(image, keys[2])
```

### Color Augmentations

```python
def random_brightness(image, key, delta=0.2):
    """Randomly adjust brightness.

    Args:
        image: Input image (H, W, C)
        key: Random key
        delta: Maximum brightness change

    Returns:
        Brightness-adjusted image
    """
    factor = jax.random.uniform(key, minval=1-delta, maxval=1+delta)
    return jnp.clip(image * factor, 0, 1)

def random_contrast(image, key, delta=0.2):
    """Randomly adjust contrast.

    Args:
        image: Input image (H, W, C)
        key: Random key
        delta: Maximum contrast change

    Returns:
        Contrast-adjusted image
    """
    factor = jax.random.uniform(key, minval=1-delta, maxval=1+delta)
    mean = jnp.mean(image)
    return jnp.clip((image - mean) * factor + mean, 0, 1)

def random_saturation(image, key, delta=0.2):
    """Randomly adjust saturation (RGB only).

    Args:
        image: Input RGB image (H, W, 3)
        key: Random key
        delta: Maximum saturation change

    Returns:
        Saturation-adjusted image
    """
    factor = jax.random.uniform(key, minval=1-delta, maxval=1+delta)

    # Convert to grayscale
    gray = jnp.mean(image, axis=-1, keepdims=True)

    # Interpolate between gray and original
    adjusted = gray + factor * (image - gray)

    return jnp.clip(adjusted, 0, 1)

def random_hue(image, key, delta=0.1):
    """Randomly adjust hue (RGB only).

    Args:
        image: Input RGB image (H, W, 3)
        key: Random key
        delta: Maximum hue change

    Returns:
        Hue-adjusted image
    """
    factor = jax.random.uniform(key, minval=-delta, maxval=delta)

    # Simple hue rotation by channel shifting
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    # Rotate through channels
    shifted = jnp.stack([
        r + factor * (g - r),
        g + factor * (b - g),
        b + factor * (r - b)
    ], axis=-1)

    return jnp.clip(shifted, 0, 1)

# Usage
key = jax.random.key(0)
keys = jax.random.split(key, 4)

rgb_image = jnp.array([...])  # (H, W, 3)
rgb_image = random_brightness(rgb_image, keys[0])
rgb_image = random_contrast(rgb_image, keys[1])
rgb_image = random_saturation(rgb_image, keys[2])
rgb_image = random_hue(rgb_image, keys[3])
```

### Noise Augmentations

```python
def add_gaussian_noise(image, key, std=0.05):
    """Add Gaussian noise to image.

    Args:
        image: Input image (H, W, C)
        key: Random key
        std: Standard deviation of noise

    Returns:
        Noisy image
    """
    noise = std * jax.random.normal(key, image.shape)
    return jnp.clip(image + noise, 0, 1)

def add_salt_pepper_noise(image, key, prob=0.01):
    """Add salt and pepper noise.

    Args:
        image: Input image (H, W, C)
        key: Random key
        prob: Probability of noise per pixel

    Returns:
        Noisy image
    """
    keys = jax.random.split(key, 2)

    # Salt (white pixels)
    salt_mask = jax.random.bernoulli(keys[0], prob, image.shape)
    image = jnp.where(salt_mask, 1.0, image)

    # Pepper (black pixels)
    pepper_mask = jax.random.bernoulli(keys[1], prob, image.shape)
    image = jnp.where(pepper_mask, 0.0, image)

    return image

def add_speckle_noise(image, key, std=0.1):
    """Add multiplicative speckle noise.

    Args:
        image: Input image (H, W, C)
        key: Random key
        std: Standard deviation of noise

    Returns:
        Noisy image
    """
    noise = 1 + std * jax.random.normal(key, image.shape)
    return jnp.clip(image * noise, 0, 1)

# Usage
key = jax.random.key(0)
keys = jax.random.split(key, 3)

image = jnp.array([...])  # (H, W, C)
noisy1 = add_gaussian_noise(image, keys[0], std=0.05)
noisy2 = add_salt_pepper_noise(image, keys[1], prob=0.01)
noisy3 = add_speckle_noise(image, keys[2], std=0.1)
```

### Geometric Augmentations

```python
def random_crop(image, key, crop_height, crop_width):
    """Randomly crop image.

    Args:
        image: Input image (H, W, C)
        key: Random key
        crop_height: Height of crop
        crop_width: Width of crop

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]

    # Random starting position
    top = jax.random.randint(key, (), 0, h - crop_height + 1)
    left = jax.random.randint(jax.random.fold_in(key, 1), (), 0, w - crop_width + 1)

    return image[top:top+crop_height, left:left+crop_width]

def center_crop(image, crop_height, crop_width):
    """Center crop image.

    Args:
        image: Input image (H, W, C)
        crop_height: Height of crop
        crop_width: Width of crop

    Returns:
        Center-cropped image
    """
    h, w = image.shape[:2]

    top = (h - crop_height) // 2
    left = (w - crop_width) // 2

    return image[top:top+crop_height, left:left+crop_width]

def random_zoom(image, key, zoom_range=(0.8, 1.2)):
    """Randomly zoom image.

    Args:
        image: Input image (H, W, C)
        key: Random key
        zoom_range: (min_zoom, max_zoom)

    Returns:
        Zoomed image
    """
    h, w, c = image.shape
    zoom_factor = jax.random.uniform(key, minval=zoom_range[0], maxval=zoom_range[1])

    # Calculate new size
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)

    # Resize
    from jax import image as jax_image
    zoomed = jax_image.resize(
        image[jnp.newaxis, ...],
        shape=(1, new_h, new_w, c),
        method="bilinear"
    )[0]

    # Crop or pad to original size
    if zoom_factor > 1.0:
        # Crop
        zoomed = center_crop(zoomed, h, w)
    else:
        # Pad
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        zoomed = jnp.pad(
            zoomed,
            ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w), (0, 0)),
            mode='constant'
        )

    return zoomed

# Usage
key = jax.random.key(0)
keys = jax.random.split(key, 2)

image = jnp.array([...])  # (64, 64, 3)
cropped = random_crop(image, keys[0], 48, 48)
zoomed = random_zoom(image, keys[1], zoom_range=(0.9, 1.1))
```

### Complete Augmentation Pipeline

```python
@jax.jit
def augment_image(image, key):
    """Apply comprehensive augmentation pipeline.

    Args:
        image: Input image (H, W, C)
        key: Random key

    Returns:
        Augmented image
    """
    keys = jax.random.split(key, 8)

    # Geometric augmentations
    image = random_horizontal_flip(image, keys[0], prob=0.5)
    image = random_rotation(image, keys[1])

    # Color augmentations
    image = random_brightness(image, keys[2], delta=0.2)
    image = random_contrast(image, keys[3], delta=0.2)

    # RGB-specific
    if image.shape[-1] == 3:
        image = random_saturation(image, keys[4], delta=0.2)
        image = random_hue(image, keys[5], delta=0.1)

    # Noise
    image = add_gaussian_noise(image, keys[6], std=0.02)

    return image

# Batch augmentation
def augment_batch(images, key):
    """Augment batch of images.

    Args:
        images: Batch of images (N, H, W, C)
        key: Random key

    Returns:
        Augmented batch
    """
    batch_size = images.shape[0]
    keys = jax.random.split(key, batch_size)

    # Vectorize over batch
    augmented = jax.vmap(augment_image)(images, keys)

    return augmented

# Usage in training
key = jax.random.key(0)
for batch in data_loader:
    key, subkey = jax.random.split(key)
    augmented_batch = augment_batch(batch["images"], subkey)
    # Use augmented_batch for training
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
    ImageModalityConfig,
    ImageRepresentation,
    SyntheticImageDataset
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

# Create datasets
train_dataset = SyntheticImageDataset(
    config=config,
    dataset_size=10000,
    pattern_type="gradient",
    split="train",
    rngs=rngs
)

val_dataset = SyntheticImageDataset(
    config=config,
    dataset_size=1000,
    pattern_type="gradient",
    split="val",
    rngs=rngs
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

        # Apply augmentation
        key, subkey = jax.random.split(key)
        augmented = augment_batch(batch["images"], subkey)

        # Training step (placeholder)
        # loss = train_step(model, augmented)

    # Validation (no augmentation)
    val_batches = len(val_dataset) // batch_size
    for i in range(val_batches):
        val_batch = val_dataset.get_batch(batch_size)
        # Validation step
        # val_loss = validate_step(model, val_batch["images"])

    print(f"Epoch {epoch + 1}/{num_epochs} complete")
```

### Example 2: Multi-Resolution Training

```python
# Create datasets at multiple resolutions
resolutions = [32, 64, 128]
datasets = {}

for res in resolutions:
    config = ImageModalityConfig(
        representation=ImageRepresentation.RGB,
        height=res,
        width=res
    )

    datasets[res] = SyntheticImageDataset(
        config=config,
        dataset_size=5000,
        pattern_type="random",
        rngs=rngs
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

        print(f"  Epoch {epoch + 1}/5 at {resolution}x{resolution}")
```

### Example 3: Custom Image Dataset

```python
from typing import Iterator
from artifex.generative_models.modalities.base import BaseDataset

class CustomImageDataset(BaseDataset):
    """Custom dataset loading images from file paths."""

    def __init__(
        self,
        config: ImageModalityConfig,
        image_paths: list[str],
        labels: list[int] = None,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, split, rngs=rngs)
        self.image_paths = image_paths
        self.labels = labels

        # Load and preprocess images
        self.images = self._load_images()

    def _load_images(self):
        """Load images from paths."""
        images = []
        for path in self.image_paths:
            # In practice, use PIL, OpenCV, etc.
            # For demo, generate synthetic
            img = jax.random.uniform(
                jax.random.key(hash(path)),
                (self.config.height, self.config.width, self.config.channels)
            )
            images.append(img)
        return images

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        for i, image in enumerate(self.images):
            sample = {"images": image, "index": jnp.array(i)}
            if self.labels:
                sample["labels"] = jnp.array(self.labels[i])
            yield sample

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        key = self.rngs.sample() if "sample" in self.rngs else jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self))

        batch_images = [self.images[int(idx)] for idx in indices]
        batch = {"images": jnp.stack(batch_images), "indices": indices}

        if self.labels:
            batch_labels = [self.labels[int(idx)] for idx in indices]
            batch["labels"] = jnp.array(batch_labels)

        return batch

# Usage
image_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg", ...]
labels = [0, 1, 0, 2, ...]  # Optional labels

custom_dataset = CustomImageDataset(
    config=config,
    image_paths=image_paths,
    labels=labels,
    rngs=rngs
)
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
    - Use JIT compilation for augmentation pipelines
    - Balance augmentation strength with training stability
    - Apply geometric augmentations before color augmentations
    - Use vectorized operations for batch augmentation
    - Test augmentations visually before training

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
    - Use non-JAX operations in data pipeline
    - Load full-resolution images if working with downscaled versions
    - Ignore color space (RGB vs BGR)
    - Use excessive augmentation that destroys image structure

!!! danger "Performance Issues"
    - Load images from disk in training loop
    - Use Python loops for image processing
    - Apply expensive augmentations without JIT
    - Keep multiple copies of images in memory
    - Use very large batch sizes on limited GPU memory

!!! danger "Quality Issues"
    - Over-augment images (too much distortion)
    - Use inappropriate resize methods (nearest for photos)
    - Mix normalized and unnormalized images
    - Ignore aspect ratio when resizing
    - Apply same augmentation to all images in batch

## Summary

This guide covered:

- **Image representations** - RGB, RGBA, and grayscale configurations
- **Image datasets** - Synthetic datasets with various patterns
- **Preprocessing** - Normalization, resizing, and validation
- **Augmentation** - Geometric, color, and noise augmentations
- **Different sizes** - Working with various image resolutions
- **Complete examples** - Training with augmentation, multi-resolution, custom datasets
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
