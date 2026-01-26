"""Multi-modal datasets for training and evaluation.

This module provides datasets that contain multiple modalities with
alignment information.
"""

from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.modalities.base import BaseDataset


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
        # Create a simple config for BaseDataset
        from artifex.generative_models.core.protocols.configuration import BaseModalityConfig

        config = BaseModalityConfig()
        super().__init__(config=config, split="train", rngs=rngs)
        self.modalities = modalities
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.text_vocab_size = text_vocab_size
        self.text_sequence_length = text_sequence_length
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.alignment_strength = alignment_strength
        self.rngs = rngs

        # Generate synthetic aligned data
        self._generate_data()

    def _generate_data(self):
        """Generate synthetic aligned multi-modal data."""
        self.data = []

        for i in range(self.num_samples):
            sample = {}

            # Generate a shared latent representation
            if "params" in self.rngs:
                key = self.rngs.params()
            else:
                key = jax.random.key(i)

            latent_dim = 32
            shared_latent = jax.random.normal(key, (latent_dim,))

            # Generate modality-specific data from shared latent
            if "image" in self.modalities:
                sample["image"] = self._generate_image_from_latent(
                    shared_latent, jax.random.fold_in(key, 1)
                )

            if "text" in self.modalities:
                sample["text"] = self._generate_text_from_latent(
                    shared_latent, jax.random.fold_in(key, 2)
                )

            if "audio" in self.modalities:
                sample["audio"] = self._generate_audio_from_latent(
                    shared_latent, jax.random.fold_in(key, 3)
                )

            # Add alignment score
            sample["alignment_score"] = self.alignment_strength
            sample["latent"] = shared_latent

            self.data.append(sample)

    def _generate_image_from_latent(self, latent: jax.Array, key: jax.Array) -> jax.Array:
        """Generate image from latent representation.

        Args:
            latent: Shared latent representation
            key: Random key

        Returns:
            Generated image
        """
        # Simple generation: use latent to modulate random noise
        noise = jax.random.normal(key, self.image_shape)

        # Use latent to create spatial patterns
        h, w, c = self.image_shape
        x = jnp.linspace(-1, 1, w)
        y = jnp.linspace(-1, 1, h)
        xx, yy = jnp.meshgrid(x, y)

        # Create patterns based on latent
        pattern = jnp.zeros((h, w))
        for i in range(min(len(latent), 8)):
            freq = 2 + i
            phase = latent[i] * jnp.pi
            amplitude = jnp.abs(latent[i])
            pattern += amplitude * jnp.sin(freq * xx + phase) * jnp.cos(freq * yy + phase)

        # Normalize and expand to color channels
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
        pattern = jnp.stack([pattern] * c, axis=-1)

        # Mix pattern with noise based on alignment strength
        image = self.alignment_strength * pattern + (1 - self.alignment_strength) * noise

        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        return image

    def _generate_text_from_latent(self, latent: jax.Array, key: jax.Array) -> jax.Array:
        """Generate text sequence from latent representation.

        Args:
            latent: Shared latent representation
            key: Random key

        Returns:
            Generated text sequence (token indices)
        """
        # Use latent to bias token selection
        # Create token probabilities based on latent
        latent_expanded = jnp.tile(latent, (self.text_vocab_size // len(latent) + 1))
        latent_expanded = latent_expanded[: self.text_vocab_size]

        # Convert to probabilities
        token_logits = latent_expanded * self.alignment_strength
        token_probs = jax.nn.softmax(token_logits)

        # Sample tokens
        tokens = []
        for i in range(self.text_sequence_length):
            token_key = jax.random.fold_in(key, i)
            token = jax.random.choice(token_key, self.text_vocab_size, p=token_probs)
            tokens.append(token)

        return jnp.array(tokens)

    def _generate_audio_from_latent(self, latent: jax.Array, key: jax.Array) -> jax.Array:
        """Generate audio waveform from latent representation.

        Args:
            latent: Shared latent representation
            key: Random key

        Returns:
            Generated audio waveform
        """
        num_samples = int(self.audio_sample_rate * self.audio_duration)
        t = jnp.linspace(0, self.audio_duration, num_samples)

        # Use latent to create audio as sum of sinusoids
        waveform = jnp.zeros(num_samples)

        for i in range(min(len(latent), 10)):
            # Frequency between 100-2000 Hz
            freq = 100 + 1900 * (jnp.abs(latent[i]) % 1)
            phase = latent[i] * 2 * jnp.pi
            amplitude = jnp.abs(latent[i]) * 0.1

            waveform += amplitude * jnp.sin(2 * jnp.pi * freq * t + phase)

        # Add noise based on alignment strength
        noise = jax.random.normal(key, (num_samples,)) * 0.1
        waveform = self.alignment_strength * waveform + (1 - self.alignment_strength) * noise

        # Normalize
        waveform = waveform / (jnp.max(jnp.abs(waveform)) + 1e-8)

        return waveform

    def __len__(self) -> int:
        """Get dataset length."""
        return self.num_samples

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over dataset samples."""
        for i in range(len(self)):
            yield self.data[i]

    def __getitem__(self, idx: int) -> dict[str, jax.Array]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing data for each modality
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        return self.data[idx]

    def get_batch(self, batch_size: int) -> dict[str, jax.Array]:
        """Get a batch of samples.

        Args:
            batch_size: Batch size

        Returns:
            Batch of multi-modal data
        """
        if "batch" in self.rngs:
            key = self.rngs.batch()
        else:
            key = jax.random.key(0)

        indices = jax.random.choice(key, self.num_samples, shape=(batch_size,), replace=False)

        batch = {}
        for i, idx in enumerate(indices):
            sample = self.data[int(idx)]
            for key, value in sample.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        # Stack arrays
        for key in batch:
            batch[key] = jnp.stack(batch[key])

        return batch


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
    return MultiModalDataset(
        modalities=modalities,
        num_samples=num_samples,
        alignment_strength=alignment_strength,
        rngs=rngs,
        **kwargs,
    )


def create_aligned_dataset(
    source_data: dict[str, jax.Array],
    target_modalities: list[str],
    alignment_model: nnx.Module | None = None,
    *,
    rngs: nnx.Rngs,
) -> MultiModalDataset:
    """Create an aligned multi-modal dataset from source data.

    Args:
        source_data: Source modality data
        target_modalities: Target modalities to generate
        alignment_model: Optional model for alignment
        rngs: Random number generators

    Returns:
        Aligned multi-modal dataset
    """
    # This is a placeholder for more sophisticated alignment
    # In practice, this would use the alignment model to generate
    # aligned data in target modalities

    num_samples = len(next(iter(source_data.values())))

    dataset = MultiModalDataset(
        modalities=list(source_data.keys()) + target_modalities,
        num_samples=num_samples,
        rngs=rngs,
    )

    # Override with provided source data
    for i in range(num_samples):
        for modality, data in source_data.items():
            dataset.data[i][modality] = data[i]

    return dataset


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
        # Create a simple config for BaseDataset
        from artifex.generative_models.core.protocols.configuration import BaseModalityConfig

        config = BaseModalityConfig()
        # Use a default Rngs if not provided
        super().__init__(config=config, split="train", rngs=nnx.Rngs())
        self.pairs = pairs
        self.data = data
        self.alignments = alignments

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate that all paired modalities have same number of samples."""
        num_samples = None

        for modality in self.data:
            if num_samples is None:
                num_samples = len(self.data[modality])
            else:
                if len(self.data[modality]) != num_samples:
                    raise ValueError(
                        f"Modality '{modality}' has {len(self.data[modality])} samples, "
                        f"expected {num_samples}"
                    )

        self.num_samples = num_samples

    def __len__(self) -> int:
        """Get dataset length."""
        return self.num_samples

    def __iter__(self) -> Iterator[dict[str, jax.Array | float]]:
        """Iterate over dataset samples."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, jax.Array | float]:
        """Get a paired sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with paired data
        """
        sample: dict[str, jax.Array | float] = {}

        # Add data for each modality
        for modality in self.data:
            sample[modality] = self.data[modality][idx]

        # Add alignment scores if available
        if self.alignments is not None:
            sample["alignment_scores"] = self.alignments[idx]

        # Add pair information
        sample["pairs"] = self.pairs

        return sample
