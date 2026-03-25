"""Multi-modal datasets backed by datarax MemorySource.

Provides factory functions for creating synthetic multi-modal data sources
with alignment information across modalities. Uses datarax's MemorySource
as the data container rather than custom DataSourceModule subclasses.
"""

from typing import Any

import jax
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

from .base import validate_multi_modal_helper_modalities


# ---------------------------------------------------------------------------
# Data generation (pure functions, no class needed)
# ---------------------------------------------------------------------------


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
    """
    modalities = validate_multi_modal_helper_modalities(modalities)

    all_images: list[jax.Array] = []
    all_text: list[jax.Array] = []
    all_audio: list[jax.Array] = []
    all_latent: list[jax.Array] = []

    for i in range(num_samples):
        key = jax.random.key(i)
        latent_dim = 32
        shared_latent = jax.random.normal(key, (latent_dim,))
        all_latent.append(shared_latent)

        if "image" in modalities:
            all_images.append(
                _generate_image_from_latent(
                    shared_latent,
                    jax.random.fold_in(key, 1),
                    image_shape=image_shape,
                    alignment_strength=alignment_strength,
                )
            )

        if "text" in modalities:
            all_text.append(
                _generate_text_from_latent(
                    shared_latent,
                    jax.random.fold_in(key, 2),
                    vocab_size=text_vocab_size,
                    sequence_length=text_sequence_length,
                    alignment_strength=alignment_strength,
                )
            )

        if "audio" in modalities:
            all_audio.append(
                _generate_audio_from_latent(
                    shared_latent,
                    jax.random.fold_in(key, 3),
                    sample_rate=audio_sample_rate,
                    duration=audio_duration,
                    alignment_strength=alignment_strength,
                )
            )

    data: dict[str, jnp.ndarray] = {}

    if "image" in modalities:
        data["image"] = jnp.stack(all_images)
    if "text" in modalities:
        data["text"] = jnp.stack(all_text)
    if "audio" in modalities:
        data["audio"] = jnp.stack(all_audio)

    data["alignment_score"] = jnp.full((num_samples,), alignment_strength)
    data["latent"] = jnp.stack(all_latent)

    return data


def _generate_image_from_latent(
    latent: jax.Array,
    key: jax.Array,
    *,
    image_shape: tuple[int, int, int],
    alignment_strength: float,
) -> jax.Array:
    """Generate image from latent representation."""
    noise = jax.random.normal(key, image_shape)
    h, w, c = image_shape
    x = jnp.linspace(-1, 1, w)
    y = jnp.linspace(-1, 1, h)
    xx, yy = jnp.meshgrid(x, y)

    pattern = jnp.zeros((h, w))
    for i in range(min(len(latent), 8)):
        freq = 2 + i
        phase = latent[i] * jnp.pi
        amplitude = jnp.abs(latent[i])
        pattern += amplitude * jnp.sin(freq * xx + phase) * jnp.cos(freq * yy + phase)

    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
    pattern = jnp.stack([pattern] * c, axis=-1)

    image = alignment_strength * pattern + (1 - alignment_strength) * noise
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def _generate_text_from_latent(
    latent: jax.Array,
    key: jax.Array,
    *,
    vocab_size: int,
    sequence_length: int,
    alignment_strength: float,
) -> jax.Array:
    """Generate text sequence from latent representation."""
    latent_expanded = jnp.tile(latent, (vocab_size // len(latent) + 1))
    latent_expanded = latent_expanded[:vocab_size]

    token_logits = latent_expanded * alignment_strength
    token_probs = nnx.softmax(token_logits)

    tokens = []
    for i in range(sequence_length):
        token_key = jax.random.fold_in(key, i)
        token = jax.random.choice(token_key, vocab_size, p=token_probs)
        tokens.append(token)

    return jnp.array(tokens)


def _generate_audio_from_latent(
    latent: jax.Array,
    key: jax.Array,
    *,
    sample_rate: int,
    duration: float,
    alignment_strength: float,
) -> jax.Array:
    """Generate audio waveform from latent representation."""
    num_audio_samples = int(sample_rate * duration)
    t = jnp.linspace(0, duration, num_audio_samples)

    waveform = jnp.zeros(num_audio_samples)
    for i in range(min(len(latent), 10)):
        freq = 100 + 1900 * (jnp.abs(latent[i]) % 1)
        phase = latent[i] * 2 * jnp.pi
        amplitude = jnp.abs(latent[i]) * 0.1
        waveform += amplitude * jnp.sin(2 * jnp.pi * freq * t + phase)

    noise = jax.random.normal(key, (num_audio_samples,)) * 0.1
    waveform = alignment_strength * waveform + (1 - alignment_strength) * noise
    waveform = waveform / (jnp.max(jnp.abs(waveform)) + 1e-8)
    return waveform


# ---------------------------------------------------------------------------
# Factory functions — return MemorySource instances
# ---------------------------------------------------------------------------


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

    Generates aligned multi-modal data and wraps it in a datarax
    MemorySource for pipeline integration.

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
    if isinstance(modalities, list):
        modalities = tuple(modalities)

    modalities = validate_multi_modal_helper_modalities(modalities)

    data = generate_multi_modal_data(
        modalities,
        num_samples,
        alignment_strength=alignment_strength,
        **kwargs,
    )

    config = MemorySourceConfig(shuffle=shuffle)
    return MemorySource(config, data, rngs=rngs)


def create_paired_multi_modal_dataset(
    data: dict[str, jax.Array],
    alignments: jax.Array | None = None,
    *,
    rngs: nnx.Rngs | None = None,
    shuffle: bool = False,
) -> MemorySource:
    """Create a paired multi-modal dataset from pre-existing data.

    Wraps explicitly paired multi-modal data arrays in a MemorySource.

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
    # Validate consistent lengths
    num_samples = None
    for modality, mod_data in data.items():
        if num_samples is None:
            num_samples = len(mod_data)
        elif len(mod_data) != num_samples:
            raise ValueError(
                f"Modality '{modality}' has {len(mod_data)} samples, expected {num_samples}"
            )

    # Include alignments in the data dict if provided
    source_data = dict(data)
    if alignments is not None:
        source_data["alignment_scores"] = alignments

    config = MemorySourceConfig(shuffle=shuffle)
    return MemorySource(config, source_data, rngs=rngs)


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
    num_samples = len(next(iter(source_data.values())))
    all_modalities = tuple(source_data.keys()) + tuple(target_modalities)

    # Generate all modality data
    generated = generate_multi_modal_data(
        all_modalities,
        num_samples,
    )

    # Override generated data with provided source data
    for modality, mod_array in source_data.items():
        generated[modality] = mod_array

    config = MemorySourceConfig(shuffle=False)
    return MemorySource(config, generated, rngs=rngs)
