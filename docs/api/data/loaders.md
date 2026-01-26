# Data API Reference

Complete API reference for Artifex's data loading system, including datasets, modalities, and utility functions.

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
            config: Model configuration (must be ModelConfig)
            rngs: Random number generator keys
            **kwargs: Additional keyword arguments for model creation

        Returns:
            An initialized model instance
        """
        ...
```

### `BaseDataset`

Abstract base class for all datasets.

```python
class BaseDataset(nnx.Module, ABC):
    """Abstract base class for modality datasets."""

    def __init__(
        self,
        config: BaseModalityConfig,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize dataset.

        Args:
            config: Modality configuration
            split: Dataset split ('train', 'val', 'test')
            rngs: Random number generators
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        ...

    @abstractmethod
    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with modality-specific data
        """
        ...

    def get_sample(self, index: int) -> dict[str, jax.Array]:
        """Get a single sample by index.

        Args:
            index: Sample index

        Returns:
            Sample data

        Raises:
            IndexError: If index is out of range
        """
        ...

    def get_data_statistics(self) -> dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        ...
```

## Image Modality

### `ImageModalityConfig`

Configuration for image modality processing.

```python
@dataclass
class ImageModalityConfig:
    """Configuration for image modality processing."""

    representation: ImageRepresentation = ImageRepresentation.RGB
    height: int = 64
    width: int | None = None
    channels: int | None = None
    normalize: bool = True
    augmentation: bool = False
    resize_method: str = "bilinear"

    def __post_init__(self):
        """Set defaults and validate configuration."""
        ...
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `representation` | `ImageRepresentation` | `RGB` | Image representation format (RGB, RGBA, GRAYSCALE) |
| `height` | `int` | `64` | Image height in pixels |
| `width` | `int \| None` | `None` | Image width in pixels (defaults to height for square images) |
| `channels` | `int \| None` | `None` | Number of channels (auto-determined if None) |
| `normalize` | `bool` | `True` | Whether to normalize pixel values to [0, 1] |
| `augmentation` | `bool` | `False` | Whether to enable data augmentation |
| `resize_method` | `str` | `"bilinear"` | Method for resizing ('bilinear', 'nearest') |

### `ImageModality`

Base image modality class providing unified interface for image generation.

```python
class ImageModality(GenerativeModel):
    """Base image modality class."""

    name = "image"

    def __init__(
        self,
        config: ImageModalityConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize image modality.

        Args:
            config: Image modality configuration
            rngs: Random number generators
        """
        ...

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Image shape (height, width, channels)."""
        ...

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Output shape for generated images."""
        ...

    def generate(
        self,
        n_samples: int = 1,
        height: int | None = None,
        width: int | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jax.Array:
        """Generate image samples.

        Args:
            n_samples: Number of image samples to generate
            height: Height override (uses config default if None)
            width: Width override (uses config default if None)
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated image array of shape (n_samples, height, width, channels)
        """
        ...

    def process(self, data: jax.Array, **kwargs) -> jax.Array:
        """Process image data for multi-modal fusion.

        Args:
            data: Image data with shape (height, width, channels) or
                  (batch, height, width, channels)
            **kwargs: Additional processing arguments

        Returns:
            Processed image features as flattened array
        """
        ...
```

### `SyntheticImageDataset`

Synthetic image dataset for testing and development.

```python
class SyntheticImageDataset(ImageDataset):
    """Synthetic image dataset."""

    def __init__(
        self,
        config: ImageModalityConfig,
        dataset_size: int = 1000,
        pattern_type: str = "random",
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize synthetic image dataset.

        Args:
            config: Image modality configuration
            dataset_size: Number of synthetic samples
            pattern_type: Type of pattern to generate
                ('random', 'gradient', 'checkerboard', 'circles')
            split: Dataset split
            rngs: Random number generators
        """
        ...

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Generate a batch of synthetic images.

        Args:
            batch_size: Number of images to generate

        Returns:
            Batch dictionary with 'images' key
        """
        ...
```

**Pattern Types:**

- `"random"`: Random noise patterns
- `"gradient"`: Linear gradients with varying directions
- `"checkerboard"`: Checkerboard patterns with random sizes
- `"circles"`: Circular patterns with random positions/radii

### `MNISTLikeDataset`

MNIST-like synthetic dataset for digit-like patterns.

```python
class MNISTLikeDataset(ImageDataset):
    """MNIST-like synthetic dataset."""

    def __init__(
        self,
        config: ImageModalityConfig,
        dataset_size: int = 1000,
        num_classes: int = 10,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize MNIST-like dataset.

        Args:
            config: Image modality configuration (should be grayscale, 28x28)
            dataset_size: Number of synthetic samples
            num_classes: Number of classes to generate
            split: Dataset split
            rngs: Random number generators
        """
        ...

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Generate a batch of digit-like images with labels.

        Args:
            batch_size: Number of images to generate

        Returns:
            Batch dictionary with 'images' and 'labels' keys
        """
        ...
```

### `create_image_dataset`

Factory function to create image datasets.

```python
def create_image_dataset(
    dataset_type: str = "synthetic",
    config: ImageModalityConfig | None = None,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> ImageDataset:
    """Factory function to create image datasets.

    Args:
        dataset_type: Type of dataset ('synthetic', 'mnist_like')
        config: Image modality configuration
        rngs: Random number generators
        **kwargs: Additional dataset parameters

    Returns:
        Created dataset instance

    Raises:
        ValueError: If dataset_type is unknown
    """
    ...
```

## Text Modality

### `TextDataset`

Base class for text datasets.

```python
class TextDataset(nnx.Module):
    """Base class for text datasets."""

    def __init__(
        self,
        config: ModalityConfiguration,
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize text dataset.

        Args:
            config: Text modality configuration (ModalityConfiguration)
            split: Dataset split ('train', 'val', 'test')
            rngs: Random number generators
        """
        ...

    def __len__(self) -> int:
        """Return dataset size."""
        ...

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        ...

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with 'text_tokens' and potentially 'labels'
        """
        ...
```

**Text Parameters (in config.metadata["text_params"]):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | `10000` | Vocabulary size |
| `max_length` | `int` | `512` | Maximum sequence length |
| `pad_token_id` | `int` | `0` | Padding token ID |
| `unk_token_id` | `int` | `1` | Unknown token ID |
| `bos_token_id` | `int` | `2` | Beginning-of-sequence token ID |
| `eos_token_id` | `int` | `3` | End-of-sequence token ID |
| `case_sensitive` | `bool` | `False` | Whether to preserve case |

### `SyntheticTextDataset`

Synthetic text dataset for testing and development.

```python
class SyntheticTextDataset(TextDataset):
    """Synthetic text dataset."""

    def __init__(
        self,
        config: ModalityConfiguration,
        dataset_size: int = 1000,
        pattern_type: str = "random_sentences",
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize synthetic text dataset.

        Args:
            config: Text modality configuration (ModalityConfiguration)
            dataset_size: Number of synthetic samples
            pattern_type: Type of pattern to generate
                ('random_sentences', 'repeated_phrases', 'sequences', 'palindromes')
            split: Dataset split
            rngs: Random number generators
        """
        ...

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Batch dictionary with text data
        """
        ...

    def get_vocab_stats(self) -> dict[str, int]:
        """Get vocabulary statistics.

        Returns:
            Dictionary with vocabulary statistics
        """
        ...
```

**Pattern Types:**

- `"random_sentences"`: Simple subject-verb-adverb sentences
- `"repeated_phrases"`: Repeated phrases for pattern testing
- `"sequences"`: Numerical sequences
- `"palindromes"`: Palindromic text patterns

### `SimpleTextDataset`

Simple text dataset from list of strings.

```python
class SimpleTextDataset(TextDataset):
    """Simple text dataset from list of strings."""

    def __init__(
        self,
        config: ModalityConfiguration,
        texts: list[str],
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize simple text dataset.

        Args:
            config: Text modality configuration (ModalityConfiguration)
            texts: List of text strings
            split: Dataset split
            rngs: Random number generators
        """
        ...
```

### `create_text_dataset`

Factory function to create text datasets.

```python
def create_text_dataset(
    config: ModalityConfiguration,
    dataset_type: str = "synthetic",
    split: str = "train",
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> TextDataset:
    """Factory function to create text datasets.

    Args:
        config: Text modality configuration
        dataset_type: Type of dataset ('synthetic', 'simple')
        split: Dataset split
        rngs: Random number generators
        **kwargs: Additional arguments for specific dataset types

    Returns:
        Text dataset instance

    Raises:
        ValueError: If dataset_type is unknown
    """
    ...
```

## Audio Modality

### `AudioModalityConfig`

Configuration for audio modality processing.

```python
@dataclass
class AudioModalityConfig:
    """Configuration for audio modality processing."""

    representation: AudioRepresentation = AudioRepresentation.RAW_WAVEFORM
    sample_rate: int = 16000
    n_mel_channels: int = 80
    hop_length: int = 256
    n_fft: int = 1024
    duration: float = 2.0
    normalize: bool = True
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `representation` | `AudioRepresentation` | `RAW_WAVEFORM` | Audio representation format |
| `sample_rate` | `int` | `16000` | Audio sample rate in Hz |
| `n_mel_channels` | `int` | `80` | Number of mel-spectrogram channels |
| `hop_length` | `int` | `256` | Hop length for STFT/mel-spectrogram |
| `n_fft` | `int` | `1024` | FFT size for spectral representations |
| `duration` | `float` | `2.0` | Default audio duration in seconds |
| `normalize` | `bool` | `True` | Whether to normalize audio values |

### `AudioModality`

Base audio modality class providing unified interface for audio generation.

```python
class AudioModality(GenerativeModel):
    """Base audio modality class."""

    def __init__(
        self,
        config: AudioModalityConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize audio modality.

        Args:
            config: Audio modality configuration
            rngs: Random number generators
        """
        ...

    @property
    def n_time_steps(self) -> int:
        """Number of time steps for raw waveform."""
        ...

    @property
    def n_time_frames(self) -> int:
        """Number of time frames for spectral representations."""
        ...

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Output shape for generated audio."""
        ...

    def generate(
        self,
        n_samples: int = 1,
        duration: float | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Generate audio samples.

        Args:
            n_samples: Number of audio samples to generate
            duration: Duration override (uses config default if None)
            rngs: Random number generators
            **kwargs: Additional generation parameters

        Returns:
            Generated audio array
        """
        ...
```

### `SyntheticAudioDataset`

Synthetic audio dataset for testing and benchmarking.

```python
class SyntheticAudioDataset(AudioDataset):
    """Synthetic audio dataset."""

    def __init__(
        self,
        config: AudioModalityConfig,
        n_samples: int = 1000,
        audio_types: list | None = None,
        name: str = "SyntheticAudioDataset",
    ):
        """Initialize synthetic audio dataset.

        Args:
            config: Audio modality configuration
            n_samples: Number of synthetic samples to generate
            audio_types: Types of audio to generate ["sine", "noise", "chirp"]
            name: Dataset name
        """
        ...

    def __getitem__(self, idx: int) -> dict[str, jax.Array]:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary containing 'audio' and optional metadata
        """
        ...

    def collate_fn(
        self,
        batch: list[dict[str, jax.Array]]
    ) -> dict[str, jax.Array]:
        """Collate function for batching.

        Args:
            batch: List of dataset items

        Returns:
            Batched data dictionary
        """
        ...
```

**Audio Types:**

- `"sine"`: Sine waves with random frequencies (200-800 Hz)
- `"noise"`: White Gaussian noise
- `"chirp"`: Linear frequency sweeps

### `create_audio_dataset`

Factory function to create audio datasets.

```python
def create_audio_dataset(
    dataset_type: str = "synthetic",
    config: AudioModalityConfig | None = None,
    **kwargs
) -> AudioDataset:
    """Factory function to create audio datasets.

    Args:
        dataset_type: Type of dataset to create ("synthetic")
        config: Audio modality configuration
        **kwargs: Additional dataset-specific parameters

    Returns:
        Audio dataset instance

    Raises:
        ValueError: If dataset_type is unknown
    """
    ...
```

## Multi-Modal

### `MultiModalDataset`

Dataset containing multiple aligned modalities.

```python
class MultiModalDataset(BaseDataset):
    """Dataset containing multiple aligned modalities."""

    def __init__(
        self,
        modalities: list[str],
        num_samples: int,
        image_shape: tuple[int, int, int] = (32, 32, 3),
        text_vocab_size: int = 1000,
        text_sequence_length: int = 50,
        audio_sample_rate: int = 16000,
        audio_duration: float = 1.0,
        alignment_strength: float = 0.8,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize multi-modal dataset.

        Args:
            modalities: List of modality names to include
            num_samples: Number of samples in the dataset
            image_shape: Shape of image data
            text_vocab_size: Vocabulary size for text
            text_sequence_length: Length of text sequences
            audio_sample_rate: Audio sampling rate
            audio_duration: Audio clip duration in seconds
            alignment_strength: How strongly modalities are aligned (0-1)
            rngs: Random number generators
        """
        ...

    def __getitem__(self, idx: int) -> dict[str, jax.Array]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing data for each modality

        Raises:
            IndexError: If index is out of range
        """
        ...

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Batch size

        Returns:
            Batch of multi-modal data
        """
        ...
```

### `MultiModalPairedDataset`

Dataset with explicitly paired multi-modal data.

```python
class MultiModalPairedDataset(BaseDataset):
    """Dataset with explicitly paired multi-modal data."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        data: dict[str, jax.Array],
        alignments: jax.Array | None = None,
    ):
        """Initialize paired multi-modal dataset.

        Args:
            pairs: List of modality pairs
            data: Dictionary of modality data
            alignments: Optional alignment scores for pairs
        """
        ...

    def __getitem__(self, idx: int) -> dict[str, jax.Array | float]:
        """Get a paired sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with paired data
        """
        ...
```

### `create_synthetic_multi_modal_dataset`

Create a synthetic multi-modal dataset.

```python
def create_synthetic_multi_modal_dataset(
    modalities: list[str],
    num_samples: int = 1000,
    alignment_strength: float = 0.8,
    *,
    rngs: nnx.Rngs,
    **kwargs,
) -> MultiModalDataset:
    """Create a synthetic multi-modal dataset.

    Args:
        modalities: List of modality names
        num_samples: Number of samples
        alignment_strength: How strongly modalities are aligned
        rngs: Random number generators
        **kwargs: Additional arguments for dataset

    Returns:
        Multi-modal dataset
    """
    ...
```

## Utility Functions

### `validate_modality_interface`

Validate that an instance implements the Modality protocol.

```python
def validate_modality_interface(modality_instance: Any) -> bool:
    """Validate that an instance implements the Modality protocol.

    Args:
        modality_instance: Instance to validate

    Returns:
        True if instance implements Modality protocol
    """
    ...
```

### `create_modality_factory`

Create a factory function for a modality.

```python
def create_modality_factory(
    modality_class: type,
    default_config: BaseModalityConfig,
):
    """Create a factory function for a modality.

    Args:
        modality_class: The modality class to instantiate
        default_config: Default configuration

    Returns:
        Factory function
    """
    ...
```

## Registry Functions

### `register_modality`

Register a modality in the global registry.

```python
def register_modality(name: str, modality_class: type):
    """Register a modality.

    Args:
        name: Modality name
        modality_class: Modality class
    """
    ...
```

### `get_modality`

Get a modality class by name.

```python
def get_modality(name: str) -> type:
    """Get a modality by name.

    Args:
        name: Modality name

    Returns:
        Modality class

    Raises:
        KeyError: If modality not found
    """
    ...
```

### `list_modalities`

List all registered modalities.

```python
def list_modalities() -> list[str]:
    """List all registered modalities.

    Returns:
        List of modality names
    """
    ...
```

## Type Aliases

Common type aliases used throughout the data API:

```python
# Modality data types
ModalityData = jax.Array
ModalityBatch = dict[str, jax.Array]
ModalityConfig = BaseModalityConfig
EvaluationMetrics = dict[str, float]
```

## Examples

### Creating a Custom Dataset

```python
from typing import Iterator
import jax.numpy as jnp
from artifex.generative_models.modalities.base import BaseDataset

class MyCustomDataset(BaseDataset):
    """Custom dataset implementation."""

    def __init__(
        self,
        config: BaseModalityConfig,
        data_paths: list[str],
        split: str = "train",
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(config, split, rngs=rngs)
        self.data_paths = data_paths
        self.data = self._load_data()

    def _load_data(self):
        # Implement data loading logic
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        for sample in self.data:
            yield sample

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        # Implement batch sampling logic
        pass
```

### Using Factory Functions

```python
from flax import nnx

# Create datasets using factories
rngs = nnx.Rngs(0)

# Image dataset
image_dataset = create_image_dataset(
    dataset_type="synthetic",
    config=image_config,
    pattern_type="gradient",
    dataset_size=1000,
    rngs=rngs
)

# Text dataset
text_dataset = create_text_dataset(
    config=text_config,
    dataset_type="synthetic",
    pattern_type="random_sentences",
    dataset_size=1000,
    rngs=rngs
)

# Audio dataset
audio_dataset = create_audio_dataset(
    dataset_type="synthetic",
    config=audio_config,
    n_samples=1000,
    audio_types=["sine", "noise"]
)
```

## See Also

- [Data Loading Overview](../../user-guide/data/overview.md) - System overview
- [Data Loading Guide](../../user-guide/data/data-guide.md) - Practical usage guide
- [Image Modality Guide](../../user-guide/modalities/image.md) - Image-specific guide
- [Text Modality Guide](../../user-guide/modalities/text.md) - Text-specific guide
- [Audio Modality Guide](../../user-guide/modalities/audio.md) - Audio-specific guide
- [Multi-modal Guide](../../user-guide/modalities/multimodal.md) - Multi-modal guide
