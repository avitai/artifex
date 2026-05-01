# Data Loading Guide

This guide covers loading and preprocessing data in Artifex. All dataset APIs follow the same two-layer pattern: pure data generation functions that return `dict[str, jnp.ndarray]`, and factory functions that wrap those dicts in a `datarax.sources.MemorySource` for pipeline integration.

## Quick Start

```python
import jax.numpy as jnp
from flax import nnx
from artifex.generative_models.modalities.image.datasets import create_image_dataset

rngs = nnx.Rngs(0)

# Factory function returns a MemorySource
source = create_image_dataset("synthetic", rngs=rngs, height=32, width=32, channels=3, dataset_size=1000)

# Random-access indexing
sample = source[0]
print(sample["images"].shape)  # (32, 32, 3)

# Batch retrieval
batch = source.get_batch(32)
print(batch["images"].shape)  # (32, 32, 32, 3)

# Length
print(len(source))  # 1000
```

## Core Concepts

### Two-Layer API

Every modality exposes the same structure:

1. **Data generation functions** -- pure functions that accept scalar parameters and return a `dict[str, jnp.ndarray]`. They have no side effects and do not depend on external state.

2. **Factory functions** -- thin wrappers that call a generation function and pack the result into a `MemorySource`. They accept an `rngs: nnx.Rngs` argument for shuffling support.

```python
from artifex.generative_models.modalities.image.datasets import (
    generate_synthetic_images,   # data generation (pure)
    create_image_dataset,        # factory (returns MemorySource)
)
```

### MemorySource

`datarax.sources.MemorySource` is the standard data container. It wraps a `dict[str, array]` and provides:

| Method / property | Description |
|---|---|
| `__len__()` | Total number of samples |
| `__getitem__(index)` | Single-sample random access; returns a dict of per-key slices |
| `__iter__()` | Iterate over samples one at a time |
| `get_batch(batch_size)` | Stateful batch retrieval with internal index tracking |

All factory functions return a `MemorySource`, so the access patterns above work uniformly across modalities.

### Pipeline Integration with datarax

To feed a `MemorySource` into a processing pipeline, use `datarax.Pipeline`:

```python
from datarax import Pipeline
from flax import nnx

source = create_image_dataset("synthetic", rngs=rngs, dataset_size=500, height=64, width=64)

pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))

for batch in pipeline:
    images = batch["images"]  # shape: (32, 64, 64, 3)
    # ... train step
```

`Pipeline` accepts additional options:

```python
pipeline = Pipeline(
    source=source,
    batch_size=64,
    stages=[],                 # ordered nnx.Module stages applied per batch
    rngs=nnx.Rngs(0),          # RNG state for stochastic stages and source
)
```

## Custom Datasets

If your data already lives in memory or comes from a preprocessing step outside
Artifex, the canonical integration path is to wrap it in a
`datarax.sources.MemorySource`.

```python
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

data = {
    "features": jnp.ones((100, 32)),
    "labels": jnp.zeros((100,)),
}

source = MemorySource(
    MemorySourceConfig(shuffle=True),
    data,
    rngs=nnx.Rngs(0),
)

sample = source[0]
batch = source.get_batch(16)
```

Use this when you need:

- custom preprocessing before dataset construction
- synthetic or programmatically generated data
- a thin bridge from an external data source into a datarax pipeline

---

## Image Data

### Synthetic Images

Generate images with configurable patterns (`random`, `gradient`, `checkerboard`, `circles`):

```python
from artifex.generative_models.modalities.image.datasets import (
    generate_synthetic_images,
    create_image_dataset,
)
from flax import nnx

# Pure generation -- returns dict with 'images' key
data = generate_synthetic_images(
    500,
    height=64,
    width=64,
    channels=3,
    pattern_type="gradient",
)
print(data["images"].shape)  # (500, 64, 64, 3)

# Factory -- returns MemorySource
rngs = nnx.Rngs(42)
source = create_image_dataset(
    "synthetic",
    rngs=rngs,
    height=64,
    width=64,
    channels=3,
    dataset_size=500,
    pattern_type="gradient",
)
```

Available pattern types:

| Pattern | Description |
|---|---|
| `random` | Uniform random pixel values |
| `gradient` | Smooth directional gradients with random angle |
| `checkerboard` | Alternating square grid pattern |
| `circles` | Circle shapes with slight noise |

### MNIST-Like Images

Generate labeled digit-like patterns useful for classification experiments:

```python
from artifex.generative_models.modalities.image.datasets import (
    generate_mnist_like_images,
    create_image_dataset,
)

# Pure generation -- returns dict with 'images' and 'labels'
data = generate_mnist_like_images(
    1000,
    height=28,
    width=28,
    channels=1,
    num_classes=10,
)
print(data["images"].shape)   # (1000, 28, 28, 1)
print(data["labels"].shape)   # (1000,)

# Via factory
source = create_image_dataset(
    "mnist_like",
    rngs=rngs,
    height=28,
    width=28,
    channels=1,
    dataset_size=1000,
    num_classes=10,
)
```

### Using ImageModalityConfig

When an `ImageModalityConfig` is available (e.g., from a model pipeline), the factory extracts height, width, and channels from it automatically:

```python
from artifex.generative_models.modalities.image import ImageModalityConfig

config = ImageModalityConfig(height=32, width=32, channels=3)

source = create_image_dataset(
    "synthetic",
    config=config,
    rngs=rngs,
    dataset_size=2000,
    pattern_type="circles",
)
```

---

## Text Data

### Synthetic Text

Generate tokenized text sequences from several pattern types (`random_sentences`, `repeated_phrases`, `sequences`, `palindromes`):

```python
from artifex.generative_models.modalities.text.datasets import (
    generate_synthetic_text_data,
    create_text_dataset,
)

# Pure generation -- returns dict with 'text_tokens' and 'index'
data = generate_synthetic_text_data(
    200,
    vocab_size=10000,
    max_length=128,
    pattern_type="random_sentences",
)
print(data["text_tokens"].shape)  # (200, 128)

# Via factory
source = create_text_dataset(
    "synthetic",
    rngs=rngs,
    dataset_size=200,
    vocab_size=10000,
    max_length=128,
    pattern_type="random_sentences",
)
```

### Text from Custom Strings

Tokenize your own text strings into a `MemorySource`:

```python
from artifex.generative_models.modalities.text.datasets import (
    generate_text_from_strings,
    create_text_dataset,
)

# Pure generation
texts = ["the cat runs quickly", "a dog jumps slowly"]
data = generate_text_from_strings(
    texts,
    vocab_size=5000,
    max_length=64,
)
print(data["text_tokens"].shape)  # (2, 64)

# Via factory (dataset_type="simple")
source = create_text_dataset(
    "simple",
    rngs=rngs,
    texts=texts,
    vocab_size=5000,
    max_length=64,
)
```

### Standalone Tokenization

The `simple_tokenize` function converts a single string to a padded token array:

```python
from artifex.generative_models.modalities.text.datasets import simple_tokenize

tokens = simple_tokenize(
    "hello world of generative models",
    vocab_size=10000,
    max_length=32,
)
print(tokens.shape)  # (32,)
```

---

## Audio Data

### Synthetic Audio

Generate waveforms from sine, noise, and chirp patterns:

```python
from artifex.generative_models.modalities.audio.datasets import (
    generate_synthetic_audio,
    create_audio_dataset,
)

# Pure generation -- returns dict with 'audio' key
data = generate_synthetic_audio(
    100,
    sample_rate=16000,
    duration=2.0,
    normalize=True,
    audio_types=("sine", "noise", "chirp"),
)
print(data["audio"].shape)  # (100, 32000)

# Via factory
source = create_audio_dataset(
    "synthetic",
    rngs=rngs,
    n_samples=100,
    sample_rate=16000,
    duration=2.0,
)
```

### Using AudioModalityConfig

```python
from artifex.generative_models.modalities.audio import AudioModalityConfig

config = AudioModalityConfig(sample_rate=22050, duration=1.0, normalize=True)
source = create_audio_dataset("synthetic", config=config, rngs=rngs, n_samples=500)
```

---

## Multi-Modal Data

### Synthetic Aligned Data

Generate correlated data across image, text, and audio modalities using a shared latent representation:

```python
from artifex.generative_models.modalities.multi_modal.datasets import (
    generate_multi_modal_data,
    create_synthetic_multi_modal_dataset,
)

# Pure generation
data = generate_multi_modal_data(
    ("image", "text"),
    num_samples=500,
    alignment_strength=0.8,
    image_shape=(32, 32, 3),
    text_vocab_size=1000,
    text_sequence_length=50,
)
print(data["image"].shape)            # (500, 32, 32, 3)
print(data["text"].shape)             # (500, 50)
print(data["alignment_score"].shape)  # (500,)
print(data["latent"].shape)           # (500, 32)

# Via factory
source = create_synthetic_multi_modal_dataset(
    ("image", "text", "audio"),
    num_samples=500,
    alignment_strength=0.9,
    rngs=rngs,
    shuffle=True,
    image_shape=(64, 64, 3),
    audio_sample_rate=16000,
    audio_duration=1.0,
)
```

### Paired Data from Existing Arrays

Wrap pre-existing modality arrays into a single `MemorySource`:

```python
import jax.numpy as jnp
from artifex.generative_models.modalities.multi_modal.datasets import (
    create_paired_multi_modal_dataset,
)

images = jnp.ones((100, 32, 32, 3))
captions = jnp.zeros((100, 50), dtype=jnp.int32)

source = create_paired_multi_modal_dataset(
    {"image": images, "text": captions},
    rngs=rngs,
    shuffle=False,
)
```

All modality arrays must have the same first dimension; a `ValueError` is raised otherwise.

### Aligned Dataset from Source Data

Generate additional aligned modalities from existing source data:

```python
from artifex.generative_models.modalities.multi_modal.datasets import create_aligned_dataset

source_data = {"image": jnp.ones((100, 32, 32, 3))}
source = create_aligned_dataset(
    source_data,
    target_modalities=["text", "audio"],
    rngs=rngs,
)
```

---

## Tabular Data

### Synthetic Tabular Data

Generate mixed-type features (numerical, categorical, ordinal, binary):

```python
from artifex.generative_models.modalities.tabular.datasets import (
    create_synthetic_tabular_dataset,
    create_simple_tabular_dataset,
)

# Full control over feature distribution
source, config = create_synthetic_tabular_dataset(
    num_features=10,
    num_samples=1000,
    numerical_ratio=0.4,
    categorical_ratio=0.3,
    ordinal_ratio=0.2,
    binary_ratio=0.1,
    max_categorical_cardinality=10,
    rngs=rngs,
)

# Quick dataset with sensible defaults (age, income, category, education, is_member)
source, config = create_simple_tabular_dataset(
    num_samples=500,
    rngs=rngs,
)
```

Both tabular factory functions return a tuple of `(MemorySource, TabularModalityConfig)` so that the config is available for downstream processing.

### Feature Statistics

Inspect generated tabular data with `compute_feature_statistics`:

```python
from artifex.generative_models.modalities.tabular.datasets import (
    generate_synthetic_tabular_data,
    compute_feature_statistics,
)

data = generate_synthetic_tabular_data(config, num_samples=1000)
stats = compute_feature_statistics(data, config, num_samples=1000)
# stats["age"]["mean"], stats["category"]["frequencies"], etc.
```

---

## Timeseries Data

### Synthetic Time Series

Generate sequences with configurable temporal patterns:

```python
from artifex.generative_models.modalities.timeseries.datasets import (
    generate_synthetic_timeseries,
    create_synthetic_timeseries_dataset,
    create_simple_timeseries_dataset,
)

# Pure generation -- returns dict with 'timeseries' key
data = generate_synthetic_timeseries(
    200,
    sequence_length=100,
    num_features=3,
    pattern_type="sinusoidal",
    noise_level=0.1,
    trend_strength=0.5,
)
print(data["timeseries"].shape)  # (200, 100, 3)

# Via factory
source = create_synthetic_timeseries_dataset(
    sequence_length=100,
    num_features=3,
    num_samples=200,
    pattern_type="mixed",
    noise_level=0.1,
    rngs=rngs,
    shuffle=True,
    trend_strength=0.5,
    seasonal_period=25,
)

# Quick dataset for testing
source = create_simple_timeseries_dataset(
    sequence_length=50,
    num_samples=100,
    rngs=rngs,
)
```

Available pattern types:

| Pattern | Description |
|---|---|
| `sinusoidal` | Sine waves with random frequencies and phases |
| `random_walk` | Cumulative sum of Gaussian steps |
| `ar` | AR(1) autoregressive process |
| `seasonal` | Periodic seasonal component with harmonics |
| `mixed` | Combination of sinusoidal, seasonal, and random walk |

---

## Working with MemorySource

All factory functions return a `MemorySource` (tabular factories return a tuple). Here are common usage patterns.

### Random Access

```python
sample = source[0]        # first sample
sample = source[len(source) - 1]  # last sample

# Each sample is a dict with the same keys as the underlying data
print(sample.keys())  # e.g. dict_keys(['images'])
```

### Iteration

```python
for sample in source:
    process(sample)
```

### Batch Retrieval

`get_batch` uses an internal index that advances automatically:

```python
batch1 = source.get_batch(32)  # samples 0-31
batch2 = source.get_batch(32)  # samples 32-63
```

### Shuffling

Enable shuffling at construction time:

```python
from datarax.sources import MemorySource, MemorySourceConfig

config = MemorySourceConfig(shuffle=True)
source = MemorySource(config, data, rngs=rngs)
```

Or use the `shuffle` parameter in any factory function:

```python
source = create_image_dataset("synthetic", rngs=rngs, shuffle=True, dataset_size=1000)
```

---

## Custom Data

To load your own data into the same pipeline, create a `MemorySource` directly:

```python
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

# Prepare data as a dict of arrays with matching first dimensions
my_data = {
    "features": jnp.ones((500, 128)),
    "labels": jnp.zeros((500,), dtype=jnp.int32),
}

config = MemorySourceConfig(shuffle=True)
source = MemorySource(config, my_data, rngs=nnx.Rngs(0))

# Use like any other source
print(len(source))             # 500
print(source[0]["features"].shape)  # (128,)

batch = source.get_batch(64)
print(batch["features"].shape)  # (64, 128)
```

### Feeding into a Pipeline

```python
from datarax import Pipeline
from flax import nnx

pipeline = Pipeline(source=source, stages=[], batch_size=64, rngs=nnx.Rngs(0))

for batch in pipeline:
    features = batch["features"]
    labels = batch["labels"]
    # ... training step
```

---

## Performance Tips

- **Batch size** -- choose a batch size that divides your dataset evenly; `Pipeline` always emits fixed-shape batches by reading from the source's stateless `get_batch_at(start, size, key)` interface.
- **Shuffling** -- enable `shuffle=True` in the factory or `MemorySourceConfig` for training. Disable it for deterministic evaluation.
- **JIT compilation** -- `Pipeline.step` is `@nnx.jit`-decorated; for whole-epoch JIT use `pipeline.scan(step_fn, length=n_batches, modules=(model, optimizer))`.
- **Device transfer** -- use `from datarax.distributed import prefetch_to_device` to asynchronously transfer batches to the accelerator.

---

## API Reference Summary

### Image

| Function | Returns |
|---|---|
| `generate_synthetic_images(n, *, height, width, channels, pattern_type)` | `dict` with `"images"` |
| `generate_mnist_like_images(n, *, height, width, channels, num_classes)` | `dict` with `"images"`, `"labels"` |
| `create_image_dataset(type, config?, *, rngs, shuffle, **kw)` | `MemorySource` |

### Text

| Function | Returns |
|---|---|
| `generate_synthetic_text_data(n, *, vocab_size, max_length, pattern_type, ...)` | `dict` with `"text_tokens"`, `"index"` |
| `generate_text_from_strings(texts, *, vocab_size, max_length, ...)` | `dict` with `"text_tokens"`, `"index"` |
| `simple_tokenize(text, *, vocab_size, max_length, ...)` | `jnp.ndarray` of shape `(max_length,)` |
| `create_text_dataset(type, *, rngs, shuffle, **kw)` | `MemorySource` |

### Audio

| Function | Returns |
|---|---|
| `generate_synthetic_audio(n, *, sample_rate, duration, normalize, audio_types)` | `dict` with `"audio"` |
| `create_audio_dataset(type, config?, *, rngs, shuffle, **kw)` | `MemorySource` |

### Multi-Modal

| Function | Returns |
|---|---|
| `generate_multi_modal_data(modalities, n, *, alignment_strength, ...)` | `dict` with per-modality keys + `"alignment_score"`, `"latent"` |
| `create_synthetic_multi_modal_dataset(modalities, n, strength, *, rngs, ...)` | `MemorySource` |
| `create_paired_multi_modal_dataset(data, alignments?, *, rngs, shuffle)` | `MemorySource` |
| `create_aligned_dataset(source_data, targets, model?, *, rngs)` | `MemorySource` |

### Tabular

| Function | Returns |
|---|---|
| `generate_synthetic_tabular_data(config, n, *, key)` | `dict` with per-feature keys |
| `compute_feature_statistics(data, config, n)` | `dict` of stats per feature |
| `create_synthetic_tabular_dataset(num_features, n, ratios..., *, rngs)` | `(MemorySource, TabularModalityConfig)` |
| `create_simple_tabular_dataset(n, split?, *, rngs, shuffle)` | `(MemorySource, TabularModalityConfig)` |

### Timeseries

| Function | Returns |
|---|---|
| `generate_synthetic_timeseries(n, *, seq_len, num_features, pattern, noise, ...)` | `dict` with `"timeseries"` |
| `create_synthetic_timeseries_dataset(seq_len, features, n, pattern, noise, *, rngs)` | `MemorySource` |
| `create_simple_timeseries_dataset(seq_len, n, *, rngs, shuffle)` | `MemorySource` |
