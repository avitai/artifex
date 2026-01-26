"""Audio dataset implementations for audio modality."""

import jax
import jax.numpy as jnp

from .base import AudioModalityConfig


class AudioDataset:
    """Base class for audio datasets."""

    def __init__(
        self,
        config: AudioModalityConfig,
        name: str = "AudioDataset",
    ):
        """Initialize audio dataset.

        Args:
            config: Audio modality configuration
            name: Dataset name
        """
        self.config = config
        self.name = name
        self._data: list[dict[str, jax.Array]] | None = None

    def __len__(self) -> int:
        """Get dataset size."""
        if self._data is None:
            self._initialize_data()
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, jax.Array]:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary containing 'audio' and optional metadata
        """
        if self._data is None:
            self._initialize_data()
        return self._data[idx]

    def _initialize_data(self):
        """Initialize dataset data. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_data")

    def collate_fn(self, batch: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
        """Collate function for batching.

        Args:
            batch: List of dataset items

        Returns:
            Batched data dictionary
        """
        # Stack audio samples
        audio_batch = jnp.stack([item["audio"] for item in batch])

        result = {"audio": audio_batch}

        # Handle additional keys if present
        for key in batch[0].keys():
            if key != "audio":
                values = [item[key] for item in batch]
                if isinstance(values[0], jax.Array):
                    result[key] = jnp.stack(values)
                else:
                    result[key] = values

        return result


class SyntheticAudioDataset(AudioDataset):
    """Synthetic audio dataset for testing and benchmarking."""

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
        super().__init__(config, name)
        self.n_samples = n_samples
        self.audio_types = audio_types or ["sine", "noise", "chirp"]

    def _initialize_data(self):
        """Generate synthetic audio data."""
        self._data = []

        sample_rate = self.config.sample_rate
        duration = self.config.duration
        n_time_steps = int(sample_rate * duration)

        key = jax.random.key(42)  # Fixed seed for reproducibility

        for i in range(self.n_samples):
            key, subkey = jax.random.split(key)

            # Choose audio type
            audio_type = self.audio_types[i % len(self.audio_types)]

            if audio_type == "sine":
                # Generate sine wave with random frequency
                freq = jax.random.uniform(subkey, minval=200.0, maxval=800.0)
                t = jnp.linspace(0, duration, n_time_steps)
                audio = 0.5 * jnp.sin(2 * jnp.pi * freq * t)

            elif audio_type == "noise":
                # Generate white noise
                audio = 0.3 * jax.random.normal(subkey, shape=(n_time_steps,))

            elif audio_type == "chirp":
                # Generate chirp signal
                t = jnp.linspace(0, duration, n_time_steps)
                f0, f1 = 200.0, 800.0
                freq_t = f0 + (f1 - f0) * t / duration
                audio = 0.5 * jnp.sin(2 * jnp.pi * freq_t * t)

            else:
                # Default to silence
                audio = jnp.zeros(n_time_steps)

            # Normalize if required
            if self.config.normalize:
                max_val = jnp.max(jnp.abs(audio))
                audio = jnp.where(max_val > 0, audio / max_val, audio)

            item = {
                "audio": audio,
                "audio_type": audio_type,
                "sample_rate": sample_rate,
                "duration": duration,
                "index": i,
            }

            self._data.append(item)


def create_audio_dataset(
    dataset_type: str = "synthetic", config: AudioModalityConfig | None = None, **kwargs
) -> AudioDataset:
    """Factory function to create audio datasets.

    Args:
        dataset_type: Type of dataset to create ("synthetic")
        config: Audio modality configuration
        **kwargs: Additional dataset-specific parameters

    Returns:
        Audio dataset instance
    """
    if config is None:
        config = AudioModalityConfig()

    if dataset_type == "synthetic":
        return SyntheticAudioDataset(config, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
