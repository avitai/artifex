# Data API Reference

API reference for Artifex's data loading system. All datasets use datarax `MemorySource` as the data container. Pure data generation functions produce dictionaries of arrays, and factory functions wrap those dictionaries in `MemorySource` instances for pipeline integration.

## Design Overview

The dataset API follows a two-layer design:

1. **Pure generation functions** produce `dict[str, jnp.ndarray]` from parameters. They have no side effects and do not depend on datarax.
2. **Factory functions** call the generation functions and wrap the result in a `MemorySource`. They accept `rngs: nnx.Rngs` and return a `MemorySource` (or a tuple when additional config is needed).

`MemorySource` provides `__len__`, `__getitem__`, `__iter__`, and `get_batch`. To build a training pipeline, pass the source to `datarax.from_source`:

```python
import datarax
from flax import nnx

source = create_image_dataset(rngs=nnx.Rngs(0))
pipeline = datarax.from_source(source, batch_size=32)
```

## Core Protocols

### `Modality`

Protocol defining the interface for data modalities.

```python
@runtime_checkable
class Modality(Protocol):
    """Protocol defining interface for data modalities."""

    name: str

    def get_extensions(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs
    ) -> dict[str, ModelExtension]:
        """Get modality-specific extensions.

        Args:
            config: Extension configuration
            rngs: Random number generators

        Returns:
            Dictionary mapping extension names to extension instances
        """
        ...

    def get_adapter(
        self,
        model_cls: type[GenerativeModel]
    ) -> ModelAdapter:
        """Get an adapter for the specified model class.

        Args:
            model_cls: The model class to adapt

        Returns:
            A model adapter for the specified model class
        """
        ...
```

### `ModelAdapter`

Protocol for model adapters that adapt generic models to specific modalities.

```python
@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol defining interface for model adapters."""

    def create(
        self,
        config: Any,
        *,
        rngs: nnx.Rngs,
        **kwargs: Any
    ) -> GenerativeModel:
        """Create a model with modality-specific adaptations.

        Args:
            config: Typed model configuration accepted by the factory surface
            rngs: Random number generator keys
            **kwargs: Additional keyword arguments for model creation

        Returns:
            An initialized model instance
        """
        ...
```

---

## Image Modality

Module: `artifex.generative_models.modalities.image.datasets`

### `generate_synthetic_images`

Generate synthetic image data as a plain dictionary.

```python
def generate_synthetic_images(
    num_samples: int,
    *,
    height: int = 64,
    width: int = 64,
    channels: int = 3,
    pattern_type: str = "random",
) -> dict[str, jnp.ndarray]:
    """Generate synthetic image data.

    Args:
        num_samples: Number of images to generate.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        pattern_type: Pattern for generation
            ('random', 'gradient', 'checkerboard', 'circles').

    Returns:
        Dictionary with 'images' array of shape (num_samples, H, W, C).

    Raises:
        ValueError: If height, width, or channels is non-positive.
        ValueError: If pattern_type is unknown.
    """
```

**Pattern Types:**

| Pattern | Description |
|---------|-------------|
| `"random"` | Random noise patterns |
| `"gradient"` | Linear gradients with varying directions |
| `"checkerboard"` | Checkerboard patterns with random sizes |
| `"circles"` | Circular patterns with random positions and radii |

### `generate_mnist_like_images`

Generate MNIST-like digit pattern data as a plain dictionary.

```python
def generate_mnist_like_images(
    num_samples: int,
    *,
    height: int = 28,
    width: int = 28,
    channels: int = 1,
    num_classes: int = 10,
) -> dict[str, jnp.ndarray]:
    """Generate MNIST-like digit pattern data.

    Args:
        num_samples: Number of images to generate.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        num_classes: Number of digit classes.

    Returns:
        Dictionary with 'images' and 'labels' arrays.

    Raises:
        ValueError: If height, width, or channels is non-positive.
    """
```

### `create_image_dataset`

Factory function returning a `MemorySource` of image data.

```python
def create_image_dataset(
    dataset_type: str = "synthetic",
    config: ImageModalityConfig | None = None,
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
    **kwargs: Any,
) -> MemorySource:
    """Create an image dataset as a MemorySource.

    Args:
        dataset_type: Type of dataset ('synthetic', 'mnist_like').
        config: Optional modality configuration. If provided,
            height/width/channels are extracted from it.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional generation parameters
            (height, width, channels, dataset_size, pattern_type, num_classes).

    Returns:
        MemorySource backed by generated image data.

    Raises:
        ValueError: If dataset_type is unknown.
    """
```

**Usage:**

```python
from flax import nnx
from artifex.generative_models.modalities.image.datasets import create_image_dataset

rngs = nnx.Rngs(0)

# Synthetic random images
source = create_image_dataset(
    dataset_type="synthetic",
    rngs=rngs,
    height=32,
    width=32,
    channels=3,
    dataset_size=500,
    pattern_type="gradient",
)

# MNIST-like digit patterns
source = create_image_dataset(
    dataset_type="mnist_like",
    rngs=rngs,
    height=28,
    width=28,
    channels=1,
    num_classes=10,
    dataset_size=1000,
)

batch = source.get_batch(32)
print(batch["images"].shape)  # (32, 28, 28, 1)
```

---

## Text Modality

Module: `artifex.generative_models.modalities.text.datasets`

### `simple_tokenize`

Hash-based tokenization of a text string into a fixed-length token array.

```python
def simple_tokenize(
    text: str,
    *,
    vocab_size: int = 10000,
    max_length: int = 512,
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    case_sensitive: bool = False,
) -> jnp.ndarray:
    """Simple hash-based tokenization.

    Args:
        text: Input text string.
        vocab_size: Size of the vocabulary.
        max_length: Maximum sequence length.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        case_sensitive: Whether tokenization is case-sensitive.

    Returns:
        Token IDs as JAX array of shape (max_length,).
    """
```

### `generate_synthetic_text_data`

Generate synthetic text data with token sequences.

```python
def generate_synthetic_text_data(
    num_samples: int,
    *,
    vocab_size: int = 10000,
    max_length: int = 512,
    pattern_type: str = "random_sentences",
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Generate synthetic text data with token sequences.

    Args:
        num_samples: Number of text samples.
        vocab_size: Size of the vocabulary.
        max_length: Maximum sequence length.
        pattern_type: Text generation pattern
            ('random_sentences', 'repeated_phrases', 'sequences', 'palindromes').
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        case_sensitive: Whether tokenization is case-sensitive.

    Returns:
        Dictionary with 'text_tokens' array of shape (num_samples, max_length)
        and 'index' array of shape (num_samples,).
    """
```

**Pattern Types:**

| Pattern | Description |
|---------|-------------|
| `"random_sentences"` | Simple subject-verb-adverb sentences |
| `"repeated_phrases"` | Repeated phrases for pattern testing |
| `"sequences"` | Numerical sequences |
| `"palindromes"` | Palindromic text patterns |

### `generate_text_from_strings`

Generate token data from a list of text strings.

```python
def generate_text_from_strings(
    texts: list[str],
    *,
    vocab_size: int = 10000,
    max_length: int = 512,
    pad_token_id: int = 0,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    case_sensitive: bool = False,
) -> dict[str, jnp.ndarray]:
    """Generate token data from a list of text strings.

    Args:
        texts: List of text strings.
        vocab_size: Size of the vocabulary.
        max_length: Maximum sequence length.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.
        case_sensitive: Whether tokenization is case-sensitive.

    Returns:
        Dictionary with 'text_tokens' and 'index' arrays.
    """
```

### `create_text_dataset`

Factory function returning a `MemorySource` of text data.

```python
def create_text_dataset(
    dataset_type: str = "synthetic",
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
    **kwargs: Any,
) -> MemorySource:
    """Create a text dataset as a MemorySource.

    Args:
        dataset_type: Type of dataset ('synthetic', 'simple').
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional generation parameters.
            For 'synthetic': dataset_size, vocab_size, max_length, pattern_type, etc.
            For 'simple': texts (list[str]), vocab_size, max_length, etc.

    Returns:
        MemorySource backed by generated text data.

    Raises:
        ValueError: If dataset_type is unknown.
    """
```

**Usage:**

```python
from flax import nnx
from artifex.generative_models.modalities.text.datasets import create_text_dataset

rngs = nnx.Rngs(0)

# Synthetic text
source = create_text_dataset(
    dataset_type="synthetic",
    rngs=rngs,
    dataset_size=500,
    vocab_size=5000,
    max_length=128,
    pattern_type="random_sentences",
)

# From raw strings
source = create_text_dataset(
    dataset_type="simple",
    rngs=rngs,
    texts=["hello world", "machine learning", "deep learning"],
    vocab_size=5000,
    max_length=64,
)

batch = source.get_batch(16)
print(batch["text_tokens"].shape)  # (16, 64)
```

---

## Audio Modality

Module: `artifex.generative_models.modalities.audio.datasets`

### `generate_synthetic_audio`

Generate synthetic audio waveform data as a plain dictionary.

```python
def generate_synthetic_audio(
    num_samples: int,
    *,
    sample_rate: int = 16000,
    duration: float = 2.0,
    normalize: bool = True,
    audio_types: tuple[str, ...] = ("sine", "noise", "chirp"),
) -> dict[str, jnp.ndarray]:
    """Generate synthetic audio data.

    Args:
        num_samples: Number of audio clips to generate.
        sample_rate: Audio sample rate in Hz.
        duration: Audio duration in seconds.
        normalize: Whether to normalize audio values.
        audio_types: Tuple of audio types to cycle through.

    Returns:
        Dictionary with 'audio' array of shape (num_samples, n_time_steps).
    """
```

**Audio Types:**

| Type | Description |
|------|-------------|
| `"sine"` | Sine waves with random frequencies (200--800 Hz) |
| `"noise"` | White Gaussian noise |
| `"chirp"` | Linear frequency sweeps (200--800 Hz) |

### `create_audio_dataset`

Factory function returning a `MemorySource` of audio data.

```python
def create_audio_dataset(
    dataset_type: str = "synthetic",
    config: AudioModalityConfig | None = None,
    *,
    rngs: nnx.Rngs | None = None,
    shuffle: bool = False,
    **kwargs: Any,
) -> MemorySource:
    """Create an audio dataset as a MemorySource.

    Args:
        dataset_type: Type of dataset to create ('synthetic').
        config: Optional modality configuration. If provided,
            sample_rate/duration/normalize are extracted from it.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional generation parameters
            (n_samples, sample_rate, duration, normalize, audio_types).

    Returns:
        MemorySource backed by generated audio data.

    Raises:
        ValueError: If dataset_type is unknown.
    """
```

**Usage:**

```python
from flax import nnx
from artifex.generative_models.modalities.audio.datasets import create_audio_dataset

rngs = nnx.Rngs(0)

source = create_audio_dataset(
    dataset_type="synthetic",
    rngs=rngs,
    n_samples=200,
    sample_rate=16000,
    duration=1.0,
    audio_types=("sine", "noise"),
)

batch = source.get_batch(8)
print(batch["audio"].shape)  # (8, 16000)
```

---

## Multi-Modal

Module: `artifex.generative_models.modalities.multi_modal.datasets`

### `generate_multi_modal_data`

Generate synthetic aligned multi-modal data as a plain dictionary.

```python
def generate_multi_modal_data(
    modalities: tuple[str, ...],
    num_samples: int,
    *,
    alignment_strength: float = 0.8,
    image_shape: tuple[int, int, int] = (32, 32, 3),
    text_vocab_size: int = 1000,
    text_sequence_length: int = 50,
    audio_sample_rate: int = 16000,
    audio_duration: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic aligned multi-modal data.

    Creates data across specified modalities using a shared latent
    representation to ensure cross-modal alignment.

    Args:
        modalities: Modality names to generate (e.g. "image", "text", "audio").
        num_samples: Number of samples to generate.
        alignment_strength: How strongly modalities are correlated (0-1).
        image_shape: Shape of image data (H, W, C).
        text_vocab_size: Vocabulary size for text token sampling.
        text_sequence_length: Length of text sequences.
        audio_sample_rate: Audio sampling rate in Hz.
        audio_duration: Audio clip duration in seconds.

    Returns:
        Dictionary mapping modality names to arrays of shape (num_samples, ...).
        Also includes 'alignment_score' and 'latent' arrays.
    """
```

### `create_synthetic_multi_modal_dataset`

Factory function returning a `MemorySource` of aligned multi-modal data.

```python
def create_synthetic_multi_modal_dataset(
    modalities: tuple[str, ...] | list[str],
    num_samples: int = 1000,
    alignment_strength: float = 0.8,
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
    **kwargs: Any,
) -> MemorySource:
    """Create a synthetic multi-modal dataset as a MemorySource.

    Args:
        modalities: Modality names to include.
        num_samples: Number of samples to generate.
        alignment_strength: How strongly modalities are aligned (0-1).
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional generation parameters (image_shape, etc.).

    Returns:
        MemorySource backed by generated multi-modal data.
    """
```

### `create_paired_multi_modal_dataset`

Wrap pre-existing paired multi-modal data in a `MemorySource`.

```python
def create_paired_multi_modal_dataset(
    data: dict[str, jax.Array],
    alignments: jax.Array | None = None,
    *,
    rngs: nnx.Rngs | None = None,
    shuffle: bool = False,
) -> MemorySource:
    """Create a paired multi-modal dataset from pre-existing data.

    Args:
        data: Dictionary mapping modality names to data arrays.
            All arrays must have the same first dimension.
        alignments: Optional alignment scores array.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.

    Returns:
        MemorySource backed by paired data.

    Raises:
        ValueError: If modalities have different sample counts.
    """
```

### `create_aligned_dataset`

Create an aligned multi-modal dataset from source data, generating missing modalities.

```python
def create_aligned_dataset(
    source_data: dict[str, jax.Array],
    target_modalities: list[str],
    alignment_model: nnx.Module | None = None,
    *,
    rngs: nnx.Rngs,
) -> MemorySource:
    """Create an aligned multi-modal dataset from source data.

    Takes existing modality data and generates additional aligned
    modalities, then wraps everything in a MemorySource.

    Args:
        source_data: Source modality data arrays.
        target_modalities: Target modalities to generate.
        alignment_model: Optional model for alignment (unused placeholder).
        rngs: Random number generators.

    Returns:
        MemorySource with source + generated modality data.
    """
```

**Usage:**

```python
from flax import nnx
from artifex.generative_models.modalities.multi_modal.datasets import (
    create_synthetic_multi_modal_dataset,
    create_paired_multi_modal_dataset,
)

rngs = nnx.Rngs(0)

# Generate aligned image + text data
source = create_synthetic_multi_modal_dataset(
    modalities=("image", "text"),
    num_samples=500,
    alignment_strength=0.9,
    rngs=rngs,
    image_shape=(32, 32, 3),
    text_vocab_size=1000,
    text_sequence_length=50,
)

batch = source.get_batch(16)
print(batch["image"].shape)  # (16, 32, 32, 3)
print(batch["text"].shape)   # (16, 50)

# Wrap existing paired arrays
import jax.numpy as jnp
paired_source = create_paired_multi_modal_dataset(
    data={
        "image": jnp.ones((100, 32, 32, 3)),
        "text": jnp.ones((100, 50), dtype=jnp.int32),
    },
    rngs=rngs,
)
```

---

## Tabular Modality

Module: `artifex.generative_models.modalities.tabular.datasets`

### `generate_synthetic_tabular_data`

Generate synthetic tabular data with mixed feature types.

```python
def generate_synthetic_tabular_data(
    modality_config: TabularModalityConfig,
    num_samples: int,
    *,
    key: jax.Array | None = None,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic tabular data with mixed feature types.

    Args:
        modality_config: Tabular modality configuration with feature definitions.
        num_samples: Number of samples to generate.
        key: Optional RNG key. If None, uses jax.random.key(0).

    Returns:
        Dictionary mapping feature names to data arrays.
    """
```

### `compute_feature_statistics`

Compute statistics about tabular features.

```python
def compute_feature_statistics(
    data: dict[str, jnp.ndarray],
    modality_config: TabularModalityConfig,
    num_samples: int,
) -> dict[str, dict[str, Any]]:
    """Compute statistics about tabular features.

    Args:
        data: Dictionary mapping feature names to data arrays.
        modality_config: Tabular modality configuration.
        num_samples: Number of samples in the dataset.

    Returns:
        Dictionary mapping feature names to their statistics.
    """
```

### `create_synthetic_tabular_dataset`

Factory function creating a `MemorySource` with configurable feature ratios.

```python
def create_synthetic_tabular_dataset(
    num_features: int = 10,
    num_samples: int = 1000,
    numerical_ratio: float = 0.4,
    categorical_ratio: float = 0.3,
    ordinal_ratio: float = 0.2,
    binary_ratio: float = 0.1,
    max_categorical_cardinality: int = 10,
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
) -> tuple[MemorySource, TabularModalityConfig]:
    """Create a synthetic tabular dataset with mixed feature types.

    Args:
        num_features: Total number of features.
        num_samples: Number of samples to generate.
        numerical_ratio: Proportion of numerical features.
        categorical_ratio: Proportion of categorical features.
        ordinal_ratio: Proportion of ordinal features.
        binary_ratio: Proportion of binary features.
        max_categorical_cardinality: Maximum vocabulary size for categorical.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.

    Returns:
        Tuple of (MemorySource, TabularModalityConfig).

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
```

### `create_simple_tabular_dataset`

Factory function creating a simple 5-feature tabular dataset for testing.

```python
def create_simple_tabular_dataset(
    num_samples: int = 500,
    split: str = "train",
    *,
    rngs: nnx.Rngs,
    shuffle: bool = False,
) -> tuple[MemorySource, TabularModalityConfig]:
    """Create a simple tabular dataset for testing.

    Args:
        num_samples: Number of samples to generate.
        split: Dataset split (unused, kept for API compatibility).
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.

    Returns:
        Tuple of (MemorySource, TabularModalityConfig).
    """
```

**Usage:**

```python
from flax import nnx
from artifex.generative_models.modalities.tabular.datasets import (
    create_synthetic_tabular_dataset,
    create_simple_tabular_dataset,
)

rngs = nnx.Rngs(0)

# Full synthetic tabular dataset
source, config = create_synthetic_tabular_dataset(
    num_features=10,
    num_samples=1000,
    numerical_ratio=0.4,
    categorical_ratio=0.3,
    ordinal_ratio=0.2,
    binary_ratio=0.1,
    rngs=rngs,
)

# Simple 5-feature dataset for quick testing
source, config = create_simple_tabular_dataset(
    num_samples=200,
    rngs=rngs,
)

batch = source.get_batch(32)
print(list(batch.keys()))  # ['age', 'income', 'category', 'education', 'is_member']
```

---

## Timeseries Modality

Module: `artifex.generative_models.modalities.timeseries.datasets`

### `generate_synthetic_timeseries`

Generate synthetic timeseries data as a plain dictionary.

```python
def generate_synthetic_timeseries(
    num_samples: int,
    *,
    sequence_length: int = 100,
    num_features: int = 1,
    pattern_type: str = "sinusoidal",
    noise_level: float = 0.1,
    trend_strength: float = 0.0,
    seasonal_period: int | None = None,
    key: jax.Array | None = None,
) -> dict[str, jnp.ndarray]:
    """Generate synthetic timeseries data.

    Args:
        num_samples: Number of time series to generate.
        sequence_length: Length of each time series.
        num_features: Number of features per timestep.
        pattern_type: Type of pattern ('sinusoidal', 'random_walk',
            'ar', 'seasonal', 'mixed').
        noise_level: Standard deviation of noise to add.
        trend_strength: Strength of linear trend component.
        seasonal_period: Period for seasonal patterns.
        key: Optional RNG key. If None, uses jax.random.key(0).

    Returns:
        Dictionary with 'timeseries' array of shape
        (num_samples, sequence_length, num_features).

    Raises:
        ValueError: If sequence_length, num_features, or num_samples is non-positive.
        ValueError: If noise_level is negative.
        ValueError: If pattern_type is unknown.
    """
```

**Pattern Types:**

| Pattern | Description |
|---------|-------------|
| `"sinusoidal"` | Sine waves with random frequencies and phases |
| `"random_walk"` | Cumulative random steps |
| `"ar"` | AR(1) autoregressive process |
| `"seasonal"` | Seasonal patterns with harmonics |
| `"mixed"` | Combination of sinusoidal, seasonal, and random walk |

### `create_synthetic_timeseries_dataset`

Factory function returning a `MemorySource` of timeseries data.

```python
def create_synthetic_timeseries_dataset(
    sequence_length: int = 100,
    num_features: int = 1,
    num_samples: int = 1000,
    pattern_type: str = "sinusoidal",
    noise_level: float = 0.1,
    *,
    rngs: nnx.Rngs | None = None,
    shuffle: bool = False,
    **kwargs: Any,
) -> MemorySource:
    """Create a synthetic timeseries dataset as a MemorySource.

    Args:
        sequence_length: Length of each time series.
        num_features: Number of features per timestep.
        num_samples: Number of time series to generate.
        pattern_type: Type of pattern to generate.
        noise_level: Level of noise to add.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional parameters (trend_strength, seasonal_period).

    Returns:
        MemorySource backed by generated timeseries data.
    """
```

### `create_simple_timeseries_dataset`

Convenience wrapper creating a small single-feature sinusoidal dataset for testing.

```python
def create_simple_timeseries_dataset(
    sequence_length: int = 50,
    num_samples: int = 100,
    *,
    rngs: nnx.Rngs | None = None,
    shuffle: bool = False,
    **kwargs: Any,
) -> MemorySource:
    """Create a simple timeseries dataset for testing.

    Args:
        sequence_length: Length of each time series.
        num_samples: Number of time series to generate.
        rngs: Random number generators.
        shuffle: Whether to shuffle data on iteration.
        **kwargs: Additional parameters.

    Returns:
        MemorySource backed by generated timeseries data.
    """
```

**Usage:**

```python
from flax import nnx
from artifex.generative_models.modalities.timeseries.datasets import (
    create_synthetic_timeseries_dataset,
    create_simple_timeseries_dataset,
)

rngs = nnx.Rngs(0)

# Multi-feature seasonal timeseries
source = create_synthetic_timeseries_dataset(
    sequence_length=200,
    num_features=3,
    num_samples=500,
    pattern_type="seasonal",
    noise_level=0.05,
    rngs=rngs,
    seasonal_period=50,
)

# Simple sinusoidal for quick tests
source = create_simple_timeseries_dataset(
    sequence_length=50,
    num_samples=100,
    rngs=rngs,
)

batch = source.get_batch(16)
print(batch["timeseries"].shape)  # (16, 50, 1)
```

---

## MemorySource Interface

`MemorySource` (from `datarax.sources`) wraps a `dict[str, jnp.ndarray]` and exposes:

| Method / Property | Description |
|-------------------|-------------|
| `__len__()` | Total number of samples |
| `__getitem__(idx)` | Get a single sample as a dict |
| `__iter__()` | Iterate over samples one at a time |
| `get_batch(batch_size)` | Get a batch of samples as a dict of stacked arrays |

To build a full data pipeline with shuffling, batching, and prefetching:

```python
import datarax
from flax import nnx
from artifex.generative_models.modalities.image.datasets import create_image_dataset

source = create_image_dataset(rngs=nnx.Rngs(0), dataset_size=1000)
pipeline = datarax.from_source(source, batch_size=32)

for batch in pipeline:
    images = batch["images"]
    # ... training step
```

---

## Type Aliases

Common type aliases used throughout the data API:

```python
ModalityData = jax.Array
ModalityBatch = dict[str, jax.Array]
EvaluationMetrics = dict[str, float]
```

## See Also

- [Data Loading Overview](../../user-guide/data/overview.md) - System overview
- [Data Loading Guide](../../user-guide/data/data-guide.md) - Practical usage guide
- [Image Modality Guide](../../user-guide/modalities/image.md) - Image-specific guide
- [Text Modality Guide](../../user-guide/modalities/text.md) - Text-specific guide
- [Audio Modality Guide](../../user-guide/modalities/audio.md) - Audio-specific guide
- [Multi-modal Guide](../../user-guide/modalities/multimodal.md) - Multi-modal guide
