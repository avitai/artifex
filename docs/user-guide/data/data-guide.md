# Data Loading Guide

This guide provides practical instructions for loading and preprocessing data in Artifex, including custom datasets, augmentation strategies, and performance optimization.

## Quick Start

Here's a minimal example to get you started with data loading:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities.image import (
    ImageModalityConfig,
    ImageRepresentation
)
from artifex.generative_models.modalities.image.datasets import SyntheticImageDataset

# Initialize RNG
rngs = nnx.Rngs(0)

# Configure and create dataset
config = ImageModalityConfig(
    representation=ImageRepresentation.RGB,
    height=32,
    width=32,
    normalize=True
)

dataset = SyntheticImageDataset(
    config=config,
    dataset_size=1000,
    pattern_type="random",
    rngs=rngs
)

# Load a batch
batch = dataset.get_batch(batch_size=32)
print(f"Batch shape: {batch['images'].shape}")  # (32, 32, 32, 3)
```

## Loading Different Data Types

### Loading Image Data

#### From Synthetic Datasets

Artifex provides several synthetic dataset types for testing and development:

```python
from artifex.generative_models.modalities.image.datasets import (
    SyntheticImageDataset,
    MNISTLikeDataset,
    create_image_dataset
)

# Random patterns
random_dataset = SyntheticImageDataset(
    config=config,
    dataset_size=5000,
    pattern_type="random",
    rngs=rngs
)

# Gradient patterns
gradient_dataset = SyntheticImageDataset(
    config=config,
    dataset_size=5000,
    pattern_type="gradient",
    rngs=rngs
)

# Checkerboard patterns
checkerboard_dataset = SyntheticImageDataset(
    config=config,
    dataset_size=5000,
    pattern_type="checkerboard",
    rngs=rngs
)

# Circle patterns
circles_dataset = SyntheticImageDataset(
    config=config,
    dataset_size=5000,
    pattern_type="circles",
    rngs=rngs
)

# MNIST-like digit patterns
mnist_config = ImageModalityConfig(
    representation=ImageRepresentation.GRAYSCALE,
    height=28,
    width=28,
    normalize=True
)

mnist_dataset = MNISTLikeDataset(
    config=mnist_config,
    dataset_size=60000,
    num_classes=10,
    rngs=rngs
)

# Using factory function
dataset = create_image_dataset(
    dataset_type="synthetic",
    config=config,
    pattern_type="gradient",
    dataset_size=10000,
    rngs=rngs
)
```

#### Iterating Over Datasets

```python
# Iterate over individual samples
for sample in dataset:
    image = sample["images"]
    print(f"Image shape: {image.shape}")  # (32, 32, 3)
    break  # Process first sample

# Batch iteration
batch_size = 128
num_batches = len(dataset) // batch_size

for i in range(num_batches):
    batch = dataset.get_batch(batch_size)
    images = batch["images"]
    # Process batch
    print(f"Batch {i}: {images.shape}")
```

#### Loading Images from Arrays

```python
import jax.numpy as jnp
from artifex.generative_models.modalities.image.base import ImageModality

# Create modality
modality = ImageModality(config=config, rngs=rngs)

# Load from NumPy arrays (e.g., from PIL, OpenCV)
raw_images = jnp.array([...])  # Shape: (N, H, W, C)

# Process through modality
processed_images = modality.process(raw_images)

# Images are now:
# - Resized to config dimensions
# - Normalized to [0, 1]
# - Ready for model input
```

### Loading Text Data

#### From Synthetic Datasets

```python
from artifex.generative_models.modalities.text.datasets import (
    SyntheticTextDataset,
    SimpleTextDataset,
    create_text_dataset
)
from artifex.generative_models.core.configuration import ModalityConfiguration

# Configure text modality
text_config = ModalityConfiguration(
    name="text",
    modality_type="text",
    metadata={
        "text_params": {
            "vocab_size": 10000,
            "max_length": 512,
            "pad_token_id": 0,
            "unk_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "case_sensitive": False
        }
    }
)

# Random sentences
random_text = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="random_sentences",
    rngs=rngs
)

# Repeated phrases
repeated_text = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="repeated_phrases",
    rngs=rngs
)

# Numerical sequences
sequences = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="sequences",
    rngs=rngs
)

# Palindromes
palindromes = SyntheticTextDataset(
    config=text_config,
    dataset_size=5000,
    pattern_type="palindromes",
    rngs=rngs
)
```

#### From Text Strings

```python
# Create dataset from list of strings
texts = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming industries",
    "Deep neural networks can learn complex patterns",
    "Natural language processing enables text understanding",
    # ... more texts
]

text_dataset = SimpleTextDataset(
    config=text_config,
    texts=texts,
    split="train",
    rngs=rngs
)

# Get batch
batch = text_dataset.get_batch(batch_size=32)
print(batch["text_tokens"].shape)  # (32, 512)
print(len(batch["texts"]))  # 32 - list of original strings
```

#### Accessing Text Data

```python
# Iterate over samples
for sample in text_dataset:
    tokens = sample["text_tokens"]  # JAX array of token IDs
    text = sample["text"]  # Original text string
    index = sample["index"]  # Sample index

    print(f"Text: {text}")
    print(f"Tokens: {tokens.shape}")

# Get specific sample
sample_text = text_dataset.get_sample_text(0)
print(f"Sample 0: {sample_text}")

# Get vocabulary statistics
stats = text_dataset.get_vocab_stats()
print(stats)
# {
#     'unique_tokens': 1523,
#     'vocab_coverage': 0.1523,
#     'total_sequences': 5000,
#     'max_length': 512
# }
```

### Loading Audio Data

#### From Synthetic Datasets

```python
from artifex.generative_models.modalities.audio.datasets import (
    SyntheticAudioDataset,
    create_audio_dataset
)
from artifex.generative_models.modalities.audio.base import AudioModalityConfig

# Configure audio modality
audio_config = AudioModalityConfig(
    sample_rate=16000,
    duration=1.0,
    n_mels=80,
    hop_length=512,
    normalize=True
)

# Create synthetic audio dataset
audio_dataset = SyntheticAudioDataset(
    config=audio_config,
    n_samples=1000,
    audio_types=["sine", "noise", "chirp"]
)

# Access samples
sample = audio_dataset[0]
print(sample["audio"].shape)  # (16000,) - 1 second at 16kHz
print(sample["audio_type"])  # "sine"
print(sample["sample_rate"])  # 16000
print(sample["duration"])  # 1.0

# Using factory function
audio_dataset = create_audio_dataset(
    dataset_type="synthetic",
    config=audio_config,
    n_samples=5000,
    audio_types=["sine", "noise"]
)
```

#### Batching Audio Data

```python
# Get batch of audio samples
batch = audio_dataset.collate_fn([
    audio_dataset[0],
    audio_dataset[1],
    audio_dataset[2],
    audio_dataset[3]
])

print(batch["audio"].shape)  # (4, 16000)
print(len(batch["audio_type"]))  # 4
```

### Loading Multi-modal Data

#### Creating Aligned Multi-modal Datasets

```python
from artifex.generative_models.modalities.multi_modal.datasets import (
    MultiModalDataset,
    create_synthetic_multi_modal_dataset
)

# Create aligned multi-modal dataset
multi_dataset = create_synthetic_multi_modal_dataset(
    modalities=["image", "text", "audio"],
    num_samples=1000,
    alignment_strength=0.8,  # 0.0 = random, 1.0 = perfectly aligned
    image_shape=(32, 32, 3),
    text_vocab_size=1000,
    text_sequence_length=50,
    audio_sample_rate=16000,
    audio_duration=1.0,
    rngs=rngs
)

# Access multi-modal samples
sample = multi_dataset[0]
print(sample.keys())
# dict_keys(['image', 'text', 'audio', 'alignment_score', 'latent'])

print(sample["image"].shape)  # (32, 32, 3)
print(sample["text"].shape)  # (50,)
print(sample["audio"].shape)  # (16000,)
print(sample["alignment_score"])  # 0.8
print(sample["latent"].shape)  # (32,) - shared latent
```

#### Creating Paired Datasets

```python
from artifex.generative_models.modalities.multi_modal.datasets import (
    MultiModalPairedDataset
)

# Prepare paired data
image_data = jnp.array([...])  # (N, H, W, C)
text_data = jnp.array([...])  # (N, max_length)
audio_data = jnp.array([...])  # (N, n_samples)

# Define pairs
pairs = [
    ("image", "text"),
    ("image", "audio"),
    ("text", "audio")
]

# Create paired dataset
paired_dataset = MultiModalPairedDataset(
    pairs=pairs,
    data={
        "image": image_data,
        "text": text_data,
        "audio": audio_data
    },
    alignments=jnp.ones((len(image_data),))  # Optional alignment scores
)

# Access paired sample
sample = paired_dataset[0]
print(sample["image"].shape)  # (H, W, C)
print(sample["text"].shape)  # (max_length,)
print(sample["audio"].shape)  # (n_samples,)
print(sample["alignment_scores"])  # 1.0
```

## Custom Datasets

### Creating a Custom Dataset

To create a custom dataset, extend `BaseDataset` and implement the required methods:

```python
from typing import Iterator
import jax
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities.base import BaseDataset
from artifex.generative_models.core.protocols.configuration import BaseModalityConfig

class CustomImageDataset(BaseDataset):
    """Custom image dataset loading from files."""

    def __init__(
        self,
        config: BaseModalityConfig,
        image_paths: list[str],
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize custom dataset.

        Args:
            config: Modality configuration
            image_paths: List of paths to image files
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config, split, rngs=rngs)
        self.image_paths = image_paths
        self.images = self._load_images()

    def _load_images(self):
        """Load images from disk."""
        images = []
        for path in self.image_paths:
            # Load image using PIL, OpenCV, etc.
            # For example with PIL:
            # from PIL import Image
            # img = Image.open(path)
            # img_array = jnp.array(img)

            # For demonstration, create synthetic data
            img_array = jax.random.uniform(
                jax.random.key(hash(path)),
                (32, 32, 3)
            )
            images.append(img_array)

        return images

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        for i, image in enumerate(self.images):
            yield {
                "images": image,
                "index": jnp.array(i),
                "path": self.image_paths[i]
            }

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary
        """
        # Random sampling
        key = self.rngs.sample() if "sample" in self.rngs else jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self))

        # Gather samples
        batch_images = [self.images[int(idx)] for idx in indices]
        batch_paths = [self.image_paths[int(idx)] for idx in indices]

        return {
            "images": jnp.stack(batch_images),
            "indices": indices,
            "paths": batch_paths
        }

# Usage
image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg", ...]

custom_dataset = CustomImageDataset(
    config=config,
    image_paths=image_paths,
    split="train",
    rngs=rngs
)
```

### Custom Dataset with Caching

For expensive preprocessing, implement caching:

```python
class CachedDataset(BaseDataset):
    """Dataset with cached preprocessing."""

    def __init__(
        self,
        config: BaseModalityConfig,
        data_source: list,
        cache_dir: str = "/tmp/cache",
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, split, rngs=rngs)
        self.data_source = data_source
        self.cache_dir = cache_dir
        self.cache = {}

        # Create cache directory
        import os
        os.makedirs(cache_dir, exist_ok=True)

    def _load_and_preprocess(self, index: int):
        """Load and preprocess single sample with caching."""
        # Check cache first
        if index in self.cache:
            return self.cache[index]

        # Load raw data
        raw_data = self.data_source[index]

        # Expensive preprocessing
        processed = self._expensive_preprocessing(raw_data)

        # Cache result
        self.cache[index] = processed

        return processed

    def _expensive_preprocessing(self, data):
        """Expensive preprocessing operation."""
        # Example: compute features, embeddings, etc.
        return data

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        for i in range(len(self)):
            processed = self._load_and_preprocess(i)
            yield {"data": processed, "index": jnp.array(i)}

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        key = self.rngs.sample() if "sample" in self.rngs else jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self))

        batch_data = [
            self._load_and_preprocess(int(idx))
            for idx in indices
        ]

        return {
            "data": jnp.stack(batch_data),
            "indices": indices
        }
```

### Custom Dataset with Transformations

```python
class TransformDataset(BaseDataset):
    """Dataset with configurable transformations."""

    def __init__(
        self,
        config: BaseModalityConfig,
        base_dataset: BaseDataset,
        transforms: list = None,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize transform dataset.

        Args:
            config: Modality configuration
            base_dataset: Base dataset to transform
            transforms: List of transformation functions
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config, split, rngs=rngs)
        self.base_dataset = base_dataset
        self.transforms = transforms or []

    def _apply_transforms(self, sample: dict) -> dict:
        """Apply transformations to sample."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        for sample in self.base_dataset:
            yield self._apply_transforms(sample)

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        batch = self.base_dataset.get_batch(batch_size)
        return self._apply_transforms(batch)

# Define transformation functions
def normalize_images(sample):
    """Normalize images to [0, 1]."""
    if "images" in sample:
        images = sample["images"]
        sample["images"] = (images - images.min()) / (images.max() - images.min() + 1e-8)
    return sample

def add_noise(sample, noise_level=0.1):
    """Add Gaussian noise to images."""
    if "images" in sample:
        key = jax.random.key(0)
        noise = noise_level * jax.random.normal(key, sample["images"].shape)
        sample["images"] = jnp.clip(sample["images"] + noise, 0, 1)
    return sample

def random_flip(sample):
    """Randomly flip images horizontally."""
    if "images" in sample:
        key = jax.random.key(0)
        flip = jax.random.bernoulli(key, 0.5)
        if flip:
            sample["images"] = jnp.flip(sample["images"], axis=-2)
    return sample

# Usage
transforms = [normalize_images, add_noise, random_flip]

transformed_dataset = TransformDataset(
    config=config,
    base_dataset=base_dataset,
    transforms=transforms,
    rngs=rngs
)
```

## Data Augmentation

### Image Augmentation

```python
import jax
import jax.numpy as jnp

def augment_image(image, key):
    """Apply random augmentations to image.

    Args:
        image: Input image (H, W, C)
        key: Random key

    Returns:
        Augmented image
    """
    keys = jax.random.split(key, 5)

    # Random horizontal flip
    if jax.random.bernoulli(keys[0], 0.5):
        image = jnp.flip(image, axis=1)

    # Random rotation (90, 180, 270 degrees)
    k = jax.random.randint(keys[1], (), 0, 4)
    image = jnp.rot90(image, k=int(k), axes=(0, 1))

    # Random brightness adjustment
    brightness_factor = jax.random.uniform(keys[2], minval=0.8, maxval=1.2)
    image = jnp.clip(image * brightness_factor, 0, 1)

    # Random contrast adjustment
    contrast_factor = jax.random.uniform(keys[3], minval=0.8, maxval=1.2)
    mean = jnp.mean(image)
    image = jnp.clip((image - mean) * contrast_factor + mean, 0, 1)

    # Random Gaussian noise
    noise_level = jax.random.uniform(keys[4], minval=0, maxval=0.05)
    noise = noise_level * jax.random.normal(keys[4], image.shape)
    image = jnp.clip(image + noise, 0, 1)

    return image

# Apply to batch
def augment_batch(batch, key):
    """Apply augmentation to batch of images."""
    images = batch["images"]
    batch_size = images.shape[0]

    keys = jax.random.split(key, batch_size)

    augmented_images = []
    for i in range(batch_size):
        aug_image = augment_image(images[i], keys[i])
        augmented_images.append(aug_image)

    batch["images"] = jnp.stack(augmented_images)
    return batch

# JIT-compiled version for performance
@jax.jit
def augment_batch_jit(images, key):
    """JIT-compiled batch augmentation."""
    batch_size = images.shape[0]
    keys = jax.random.split(key, batch_size)

    def augment_single(carry, x):
        image, key = x
        augmented = augment_image(image, key)
        return carry, augmented

    _, augmented_images = jax.lax.scan(
        augment_single,
        None,
        (images, keys)
    )

    return augmented_images

# Usage in training
key = jax.random.key(0)
for batch in data_loader:
    key, subkey = jax.random.split(key)
    augmented_batch = augment_batch(batch, subkey)
    # Use augmented_batch for training
```

### Text Augmentation

```python
def augment_text_tokens(tokens, vocab_size, key):
    """Apply augmentation to text tokens.

    Args:
        tokens: Token sequence (seq_len,)
        vocab_size: Vocabulary size
        key: Random key

    Returns:
        Augmented tokens
    """
    keys = jax.random.split(key, 3)

    # Random token masking (15% of tokens)
    mask_prob = 0.15
    mask = jax.random.bernoulli(keys[0], mask_prob, tokens.shape)
    mask_token_id = 1  # UNK token
    tokens = jnp.where(mask, mask_token_id, tokens)

    # Random token replacement (5% of tokens)
    replace_prob = 0.05
    replace_mask = jax.random.bernoulli(keys[1], replace_prob, tokens.shape)
    random_tokens = jax.random.randint(keys[2], tokens.shape, 4, vocab_size)
    tokens = jnp.where(replace_mask, random_tokens, tokens)

    return tokens

def augment_text_batch(batch, vocab_size, key):
    """Apply augmentation to batch of text."""
    tokens = batch["text_tokens"]
    batch_size = tokens.shape[0]

    keys = jax.random.split(key, batch_size)

    augmented_tokens = []
    for i in range(batch_size):
        aug_tokens = augment_text_tokens(tokens[i], vocab_size, keys[i])
        augmented_tokens.append(aug_tokens)

    batch["text_tokens"] = jnp.stack(augmented_tokens)
    return batch
```

### Audio Augmentation

```python
def augment_audio(audio, sample_rate, key):
    """Apply augmentation to audio waveform.

    Args:
        audio: Audio waveform (n_samples,)
        sample_rate: Sample rate in Hz
        key: Random key

    Returns:
        Augmented audio
    """
    keys = jax.random.split(key, 4)

    # Random amplitude scaling
    scale_factor = jax.random.uniform(keys[0], minval=0.7, maxval=1.3)
    audio = audio * scale_factor

    # Random time shift
    max_shift = int(0.1 * len(audio))  # 10% shift
    shift = jax.random.randint(keys[1], (), -max_shift, max_shift)
    audio = jnp.roll(audio, int(shift))

    # Random Gaussian noise
    noise_level = jax.random.uniform(keys[2], minval=0, maxval=0.05)
    noise = noise_level * jax.random.normal(keys[3], audio.shape)
    audio = audio + noise

    # Normalize
    max_val = jnp.max(jnp.abs(audio))
    audio = jnp.where(max_val > 0, audio / max_val, audio)

    return audio

def augment_audio_batch(batch, sample_rate, key):
    """Apply augmentation to batch of audio."""
    audio = batch["audio"]
    batch_size = audio.shape[0]

    keys = jax.random.split(key, batch_size)

    augmented_audio = []
    for i in range(batch_size):
        aug_audio = augment_audio(audio[i], sample_rate, keys[i])
        augmented_audio.append(aug_audio)

    batch["audio"] = jnp.stack(augmented_audio)
    return batch
```

## Data Loaders

### Basic Data Loader

```python
def create_simple_data_loader(
    dataset: BaseDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False
):
    """Create a simple data loader.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch

    Yields:
        Batches of data
    """
    num_samples = len(dataset)
    num_batches = num_samples // batch_size
    if not drop_last and num_samples % batch_size != 0:
        num_batches += 1

    for epoch in range(1):  # Single epoch
        if shuffle:
            key = jax.random.key(epoch)
            indices = jax.random.permutation(key, num_samples)
        else:
            indices = jnp.arange(num_samples)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            actual_batch_size = end_idx - start_idx

            if actual_batch_size < batch_size and drop_last:
                continue

            batch = dataset.get_batch(actual_batch_size)
            yield batch

# Usage
data_loader = create_simple_data_loader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True
)

for batch in data_loader:
    # Training step
    pass
```

### Multi-Epoch Data Loader

```python
def create_multi_epoch_data_loader(
    dataset: BaseDataset,
    batch_size: int,
    num_epochs: int,
    shuffle: bool = True,
    drop_last: bool = False
):
    """Create a multi-epoch data loader.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        num_epochs: Number of epochs
        shuffle: Whether to shuffle data each epoch
        drop_last: Whether to drop last incomplete batch

    Yields:
        Tuples of (epoch, batch)
    """
    num_samples = len(dataset)

    for epoch in range(num_epochs):
        if shuffle:
            key = jax.random.key(epoch)
            indices = jax.random.permutation(key, num_samples)
        else:
            indices = jnp.arange(num_samples)

        num_batches = num_samples // batch_size
        if not drop_last and num_samples % batch_size != 0:
            num_batches += 1

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            actual_batch_size = end_idx - start_idx

            if actual_batch_size < batch_size and drop_last:
                continue

            batch = dataset.get_batch(actual_batch_size)
            yield epoch, batch

# Usage
data_loader = create_multi_epoch_data_loader(
    dataset=train_dataset,
    batch_size=128,
    num_epochs=10,
    shuffle=True
)

for epoch, batch in data_loader:
    print(f"Epoch {epoch}, batch shape: {batch['images'].shape}")
    # Training step
```

### Prefetching Data Loader

For better performance, implement prefetching:

```python
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

class PrefetchDataLoader:
    """Data loader with prefetching for better performance."""

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 2,
        prefetch_size: int = 2
    ):
        """Initialize prefetching data loader.

        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker threads
            prefetch_size: Number of batches to prefetch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.queue = Queue(maxsize=prefetch_size)

    def __iter__(self):
        """Iterate over prefetched batches."""
        num_samples = len(self.dataset)
        num_batches = num_samples // self.batch_size

        if self.shuffle:
            key = jax.random.key(0)
            indices = jax.random.permutation(key, num_samples)
        else:
            indices = jnp.arange(num_samples)

        def load_batch(batch_idx):
            """Load single batch."""
            batch = self.dataset.get_batch(self.batch_size)
            return batch

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit initial batches
            futures = []
            for i in range(min(self.prefetch_size, num_batches)):
                future = executor.submit(load_batch, i)
                futures.append(future)

            # Yield results and submit new batches
            for i in range(num_batches):
                # Wait for batch to be ready
                batch = futures[i % len(futures)].result()
                yield batch

                # Submit next batch
                next_idx = i + self.prefetch_size
                if next_idx < num_batches:
                    future = executor.submit(load_batch, next_idx)
                    futures.append(future)

# Usage
prefetch_loader = PrefetchDataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    prefetch_size=2
)

for batch in prefetch_loader:
    # Training step with prefetched data
    pass
```

## Performance Optimization

### JIT-Compiled Preprocessing

```python
import jax
import jax.numpy as jnp

@jax.jit
def preprocess_batch(batch):
    """JIT-compiled preprocessing for faster execution.

    Args:
        batch: Raw batch dictionary

    Returns:
        Preprocessed batch
    """
    images = batch["images"]

    # Normalize to [0, 1]
    images = (images - images.min()) / (images.max() - images.min() + 1e-8)

    # Standardize (mean=0, std=1)
    mean = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    std = jnp.std(images, axis=(1, 2, 3), keepdims=True)
    images = (images - mean) / (std + 1e-8)

    batch["images"] = images
    return batch

# Usage in training loop
for batch in data_loader:
    preprocessed_batch = preprocess_batch(batch)
    # Training step
```

### Vectorized Operations

```python
@jax.jit
def vectorized_preprocessing(images):
    """Vectorized preprocessing using jax.vmap.

    Args:
        images: Batch of images (N, H, W, C)

    Returns:
        Preprocessed images
    """
    def preprocess_single(image):
        """Preprocess single image."""
        # Normalize
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        # Random flip
        # ... more operations
        return image

    # Vectorize over batch dimension
    preprocessed = jax.vmap(preprocess_single)(images)
    return preprocessed

# Usage
images = batch["images"]
preprocessed = vectorized_preprocessing(images)
```

### Memory-Efficient Loading

```python
class MemoryEfficientDataset(BaseDataset):
    """Dataset that loads data on-demand to save memory."""

    def __init__(
        self,
        config: BaseModalityConfig,
        data_paths: list[str],
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize memory-efficient dataset.

        Args:
            config: Modality configuration
            data_paths: List of paths to data files
            split: Dataset split
            rngs: Random number generators
        """
        super().__init__(config, split, rngs=rngs)
        self.data_paths = data_paths
        # Don't load all data into memory

    def _load_sample(self, index: int):
        """Load single sample on-demand."""
        path = self.data_paths[index]
        # Load from disk
        # data = load_from_disk(path)
        # For demonstration:
        data = jax.random.uniform(jax.random.key(index), (32, 32, 3))
        return data

    def __len__(self) -> int:
        return len(self.data_paths)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        for i in range(len(self)):
            data = self._load_sample(i)
            yield {"data": data, "index": jnp.array(i)}

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        key = self.rngs.sample() if "sample" in self.rngs else jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self))

        # Load samples on-demand
        batch_data = [self._load_sample(int(idx)) for idx in indices]

        return {
            "data": jnp.stack(batch_data),
            "indices": indices
        }
```

### Caching Frequently Used Data

```python
from functools import lru_cache

class CachedLoader:
    """Data loader with LRU caching."""

    def __init__(self, dataset: BaseDataset, cache_size: int = 128):
        """Initialize cached loader.

        Args:
            dataset: Dataset to load from
            cache_size: Maximum number of samples to cache
        """
        self.dataset = dataset
        self.cache_size = cache_size

    @lru_cache(maxsize=128)
    def _get_cached_sample(self, index: int):
        """Get sample with caching."""
        # Access dataset through iterator
        for i, sample in enumerate(self.dataset):
            if i == index:
                return sample
        return None

    def get_batch(self, batch_size: int):
        """Get batch using cached samples."""
        # Generate random indices
        key = jax.random.key(0)
        indices = jax.random.randint(key, (batch_size,), 0, len(self.dataset))

        # Load samples (potentially from cache)
        samples = [self._get_cached_sample(int(idx)) for idx in indices]

        # Stack into batch
        batch = {}
        for key in samples[0].keys():
            batch[key] = jnp.stack([s[key] for s in samples])

        return batch

# Usage
cached_loader = CachedLoader(dataset, cache_size=256)
batch = cached_loader.get_batch(32)
```

## Common Patterns

### Training/Validation Split

```python
def split_dataset(
    full_dataset: BaseDataset,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """Split dataset into training and validation sets.

    Args:
        full_dataset: Full dataset to split
        train_ratio: Ratio of training data
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    num_samples = len(full_dataset)
    num_train = int(num_samples * train_ratio)

    # Generate random permutation
    key = jax.random.key(seed)
    indices = jax.random.permutation(key, num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Create subset datasets
    class SubsetDataset(BaseDataset):
        def __init__(self, dataset, indices, **kwargs):
            super().__init__(dataset.config, dataset.split, rngs=dataset.rngs)
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __iter__(self):
            for idx in self.indices:
                # Get sample from base dataset
                for i, sample in enumerate(self.dataset):
                    if i == int(idx):
                        yield sample
                        break

        def get_batch(self, batch_size):
            # Sample from subset indices
            key = jax.random.key(0)
            batch_indices = jax.random.choice(
                key,
                self.indices,
                shape=(batch_size,),
                replace=False
            )
            return self.dataset.get_batch(batch_size)

    train_dataset = SubsetDataset(full_dataset, train_indices)
    val_dataset = SubsetDataset(full_dataset, val_indices)

    return train_dataset, val_dataset

# Usage
train_dataset, val_dataset = split_dataset(
    full_dataset,
    train_ratio=0.8,
    seed=42
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
```

### Data Pipeline with Transformations

```python
def create_training_pipeline(
    dataset: BaseDataset,
    batch_size: int,
    augment: bool = True,
    normalize: bool = True,
    shuffle: bool = True
):
    """Create complete training data pipeline.

    Args:
        dataset: Base dataset
        batch_size: Batch size
        augment: Whether to apply augmentation
        normalize: Whether to normalize
        shuffle: Whether to shuffle

    Yields:
        Preprocessed and augmented batches
    """
    num_samples = len(dataset)
    num_batches = num_samples // batch_size

    for epoch in range(1):
        if shuffle:
            key = jax.random.key(epoch)
            indices = jax.random.permutation(key, num_samples)
        else:
            indices = jnp.arange(num_samples)

        for i in range(num_batches):
            # Get batch
            batch = dataset.get_batch(batch_size)

            # Normalize
            if normalize:
                images = batch["images"]
                images = (images - images.min()) / (images.max() - images.min() + 1e-8)
                batch["images"] = images

            # Augment
            if augment:
                key = jax.random.key(i)
                batch = augment_batch(batch, key)

            yield batch

# Usage
pipeline = create_training_pipeline(
    dataset=train_dataset,
    batch_size=128,
    augment=True,
    normalize=True,
    shuffle=True
)

for batch in pipeline:
    # Training step with preprocessed batch
    pass
```

### Multi-Modal Data Loading

```python
def create_multi_modal_loader(
    image_dataset: BaseDataset,
    text_dataset: BaseDataset,
    batch_size: int,
    align: bool = True
):
    """Create multi-modal data loader.

    Args:
        image_dataset: Image dataset
        text_dataset: Text dataset
        batch_size: Batch size
        align: Whether to align samples (use same indices)

    Yields:
        Multi-modal batches
    """
    assert len(image_dataset) == len(text_dataset), \
        "Datasets must have same length for alignment"

    num_samples = len(image_dataset)
    num_batches = num_samples // batch_size

    for i in range(num_batches):
        if align:
            # Use same indices for both modalities
            key = jax.random.key(i)
            indices = jax.random.randint(key, (batch_size,), 0, num_samples)

            # Would need to implement index-based loading
            # For now, use get_batch which samples randomly
            image_batch = image_dataset.get_batch(batch_size)
            text_batch = text_dataset.get_batch(batch_size)
        else:
            # Independent sampling
            image_batch = image_dataset.get_batch(batch_size)
            text_batch = text_dataset.get_batch(batch_size)

        # Combine batches
        multi_modal_batch = {
            "images": image_batch["images"],
            "text_tokens": text_batch["text_tokens"]
        }

        yield multi_modal_batch

# Usage
multi_modal_loader = create_multi_modal_loader(
    image_dataset=image_dataset,
    text_dataset=text_dataset,
    batch_size=64,
    align=True
)

for batch in multi_modal_loader:
    print(f"Images: {batch['images'].shape}")
    print(f"Text: {batch['text_tokens'].shape}")
```

## Troubleshooting

### Out of Memory Errors

**Problem:** Dataset too large to fit in memory.

**Solutions:**

```python
# 1. Load data on-demand
class OnDemandDataset(BaseDataset):
    """Load data on-demand instead of all at once."""

    def __init__(self, data_paths, **kwargs):
        super().__init__(**kwargs)
        self.data_paths = data_paths
        # Don't load data in __init__

    def get_batch(self, batch_size):
        # Load only requested samples
        pass

# 2. Use smaller batch sizes
batch_size = 32  # Instead of 128

# 3. Clear cache periodically
import gc
import jax

for i, batch in enumerate(data_loader):
    # Training step
    if i % 100 == 0:
        # Clear Python and JAX caches
        gc.collect()
        jax.clear_caches()

# 4. Use float16 instead of float32
def convert_to_float16(batch):
    """Convert batch to float16 to save memory."""
    batch["images"] = batch["images"].astype(jnp.float16)
    return batch
```

### Slow Data Loading

**Problem:** Data loading is the bottleneck.

**Solutions:**

```python
# 1. Use JIT compilation for preprocessing
@jax.jit
def fast_preprocess(batch):
    # JIT-compiled preprocessing
    return batch

# 2. Implement prefetching
prefetch_loader = PrefetchDataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    prefetch_size=2
)

# 3. Cache preprocessed data
cache = {}
def get_cached_batch(dataset, batch_size):
    cache_key = (dataset, batch_size)
    if cache_key not in cache:
        cache[cache_key] = dataset.get_batch(batch_size)
    return cache[cache_key]

# 4. Reduce preprocessing complexity
# Remove expensive operations from training loop
# Preprocess once and save to disk
```

### Inconsistent Batch Sizes

**Problem:** Last batch has different size, causing shape mismatches.

**Solutions:**

```python
# 1. Drop last incomplete batch
data_loader = create_simple_data_loader(
    dataset,
    batch_size=128,
    drop_last=True  # Drop last batch if smaller
)

# 2. Pad last batch
def pad_batch(batch, target_size):
    """Pad batch to target size."""
    current_size = batch["images"].shape[0]
    if current_size < target_size:
        padding_size = target_size - current_size
        # Repeat last samples
        padding = jnp.repeat(
            batch["images"][-1:],
            padding_size,
            axis=0
        )
        batch["images"] = jnp.concatenate([batch["images"], padding])
    return batch

# 3. Handle variable batch sizes in model
def train_step(model, batch):
    # Get actual batch size
    batch_size = batch["images"].shape[0]
    # Use batch_size in computations
    pass
```

### Data Corruption

**Problem:** Some samples are corrupted or invalid.

**Solutions:**

```python
def validate_sample(sample):
    """Validate sample data."""
    # Check for NaN values
    if jnp.any(jnp.isnan(sample["images"])):
        return False

    # Check value range
    if jnp.any(sample["images"] < 0) or jnp.any(sample["images"] > 1):
        return False

    # Check shape
    if sample["images"].shape != (32, 32, 3):
        return False

    return True

class ValidatedDataset(BaseDataset):
    """Dataset with validation."""

    def __init__(self, base_dataset, **kwargs):
        super().__init__(**kwargs)
        self.base_dataset = base_dataset

    def get_batch(self, batch_size):
        batch = self.base_dataset.get_batch(batch_size)

        # Validate all samples
        valid_samples = []
        for i in range(batch["images"].shape[0]):
            sample = {"images": batch["images"][i]}
            if validate_sample(sample):
                valid_samples.append(sample)

        # Rebuild batch with valid samples only
        if valid_samples:
            batch["images"] = jnp.stack([s["images"] for s in valid_samples])

        return batch

# Usage
validated_dataset = ValidatedDataset(raw_dataset, **kwargs)
```

## Best Practices

### DO

!!! tip "Dataset Design"
    - Implement all required methods (`__len__`, `__iter__`, `get_batch`)
    - Return dictionaries with descriptive keys
    - Use JAX arrays for all numeric data
    - Validate data shapes and types
    - Provide proper RNG handling for reproducibility
    - Cache preprocessed data when appropriate
    - Document expected data formats
    - Use protocol-based interfaces

!!! tip "Performance"
    - Use JIT compilation for preprocessing
    - Implement prefetching for I/O-bound operations
    - Vectorize operations with `jax.vmap`
    - Load data on-demand for large datasets
    - Clear caches periodically for long training runs
    - Profile data loading to identify bottlenecks
    - Use appropriate batch sizes for hardware

!!! tip "Data Quality"
    - Validate input data
    - Handle missing or corrupted samples gracefully
    - Normalize data to expected ranges
    - Apply augmentation only during training
    - Use consistent preprocessing across splits
    - Document data statistics and distributions

### DON'T

!!! danger "Common Mistakes"
    - Use PyTorch or TensorFlow tensors
    - Load entire large dataset into memory
    - Apply random augmentation during validation
    - Ignore RNG seeding
    - Mix different data types in batches
    - Perform heavy I/O in tight loops
    - Use non-deterministic operations without RNG
    - Forget to handle edge cases (empty batches, etc.)

!!! danger "Performance Pitfalls"
    - Recompute expensive preprocessing every batch
    - Use Python loops for array operations
    - Ignore batch size effects on memory
    - Skip JIT compilation for repeated operations
    - Load data synchronously without prefetching

!!! danger "Data Issues"
    - Skip data validation
    - Mix training and validation data
    - Apply inconsistent preprocessing
    - Ignore data distribution shifts
    - Use invalid or out-of-range values

## Summary

This guide covered:

- **Loading different data types** - Images, text, audio, and multi-modal data
- **Custom datasets** - Implementing custom dataset classes with caching and transformations
- **Data augmentation** - Augmentation strategies for images, text, and audio
- **Data loaders** - Simple, multi-epoch, and prefetching data loaders
- **Performance optimization** - JIT compilation, vectorization, and memory efficiency
- **Common patterns** - Training/validation splits, pipelines, and multi-modal loading
- **Troubleshooting** - Solutions for memory, speed, and data quality issues
- **Best practices** - DOs and DON'Ts for dataset design and implementation

## Next Steps

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } **[Image Modality Guide](../modalities/image.md)**

    ---

    Deep dive into image datasets, preprocessing, and augmentation techniques

- :material-text:{ .lg .middle } **[Text Modality Guide](../modalities/text.md)**

    ---

    Learn about tokenization, vocabulary management, and text processing

- :material-volume-high:{ .lg .middle } **[Audio Modality Guide](../modalities/audio.md)**

    ---

    Audio processing, spectrograms, and audio-specific augmentation

- :material-api:{ .lg .middle } **[Data API Reference](../../api/data/loaders.md)**

    ---

    Complete API documentation for all dataset classes and functions

</div>
