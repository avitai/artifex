"""Audio datasets backed by datarax MemorySource.

Provides pure data generation functions and factory functions that wrap
generated data in datarax MemorySource for pipeline integration.
"""

from typing import Any

import jax
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

from .base import AudioModalityConfig


# ---------------------------------------------------------------------------
# Data generation (pure functions)
# ---------------------------------------------------------------------------


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
    n_time_steps = int(sample_rate * duration)
    audios = []
    for i in range(num_samples):
        key = jax.random.key(i)
        audio_type = audio_types[i % len(audio_types)]
        audio = _generate_audio(key, audio_type, n_time_steps, duration)
        if normalize:
            max_val = jnp.max(jnp.abs(audio))
            audio = jnp.where(max_val > 0, audio / max_val, audio)
        audios.append(audio)

    return {"audio": jnp.stack(audios)}


def _generate_audio(
    key: jax.Array,
    audio_type: str,
    n_time_steps: int,
    duration: float,
) -> jax.Array:
    """Generate audio waveform for the given type.

    Args:
        key: RNG key for randomization.
        audio_type: Type of audio ('sine', 'noise', 'chirp').
        n_time_steps: Number of time steps.
        duration: Audio duration in seconds.

    Returns:
        Audio waveform array of shape (n_time_steps,).
    """
    if audio_type == "sine":
        freq = jax.random.uniform(key, minval=200.0, maxval=800.0)
        t = jnp.linspace(0, duration, n_time_steps)
        return 0.5 * jnp.sin(2 * jnp.pi * freq * t)
    elif audio_type == "noise":
        return 0.3 * jax.random.normal(key, shape=(n_time_steps,))
    elif audio_type == "chirp":
        t = jnp.linspace(0, duration, n_time_steps)
        f0, f1 = 200.0, 800.0
        freq_t = f0 + (f1 - f0) * t / duration
        return 0.5 * jnp.sin(2 * jnp.pi * freq_t * t)
    else:
        return jnp.zeros(n_time_steps)


# ---------------------------------------------------------------------------
# Factory functions — return MemorySource instances
# ---------------------------------------------------------------------------


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
    if config is not None:
        kwargs.setdefault("sample_rate", config.sample_rate)
        kwargs.setdefault("duration", config.duration)
        kwargs.setdefault("normalize", config.normalize)

    num_samples = kwargs.pop("n_samples", kwargs.pop("num_samples", 1000))

    if dataset_type == "synthetic":
        data = generate_synthetic_audio(num_samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    source_config = MemorySourceConfig(shuffle=shuffle)
    return MemorySource(source_config, data, rngs=rngs)
