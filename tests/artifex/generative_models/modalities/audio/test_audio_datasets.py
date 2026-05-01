"""Tests for audio datasets backed by datarax MemorySource."""

import jax.numpy as jnp
import pytest
from datarax import Pipeline
from datarax.core.data_source import DataSourceModule
from datarax.sources import MemorySource
from flax import nnx

from artifex.generative_models.modalities.audio.datasets import (
    create_audio_dataset,
    generate_synthetic_audio,
)


@pytest.fixture
def rngs():
    """Random number generators for testing."""
    return nnx.Rngs(42)


# --- Data generation functions ---


class TestGenerateSyntheticAudio:
    """Test pure audio generation function."""

    def test_basic_generation(self) -> None:
        data = generate_synthetic_audio(10, sample_rate=8000, duration=0.5)
        assert "audio" in data
        assert data["audio"].shape == (10, 4000)  # 8000 * 0.5

    def test_audio_types(self) -> None:
        for audio_type in ["sine", "noise", "chirp"]:
            data = generate_synthetic_audio(
                2, sample_rate=8000, duration=0.5, audio_types=(audio_type,)
            )
            assert data["audio"].shape == (2, 4000)

    def test_normalization(self) -> None:
        data = generate_synthetic_audio(5, sample_rate=8000, duration=0.5, normalize=True)
        max_val = jnp.max(jnp.abs(data["audio"]))
        assert max_val <= 1.0 + 1e-6

    def test_no_normalization(self) -> None:
        data = generate_synthetic_audio(5, sample_rate=8000, duration=0.5, normalize=False)
        assert "audio" in data

    def test_values_are_finite(self) -> None:
        data = generate_synthetic_audio(5, sample_rate=8000, duration=0.5)
        assert jnp.all(jnp.isfinite(data["audio"]))


# --- MemorySource factory ---


class TestCreateAudioDataset:
    """Test factory function returns MemorySource."""

    def test_returns_memory_source(self, rngs) -> None:
        source = create_audio_dataset("synthetic", rngs=rngs, n_samples=5)
        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)

    def test_correct_length(self, rngs) -> None:
        source = create_audio_dataset(
            "synthetic", rngs=rngs, n_samples=10, sample_rate=8000, duration=0.5
        )
        assert len(source) == 10

    def test_getitem(self, rngs) -> None:
        source = create_audio_dataset(
            "synthetic", rngs=rngs, n_samples=5, sample_rate=8000, duration=0.5
        )
        item = source[0]
        assert "audio" in item
        assert item["audio"].shape == (4000,)

    def test_negative_index(self, rngs) -> None:
        source = create_audio_dataset(
            "synthetic", rngs=rngs, n_samples=5, sample_rate=8000, duration=0.5
        )
        item = source[-1]
        assert "audio" in item

    def test_out_of_bounds_raises(self, rngs) -> None:
        source = create_audio_dataset(
            "synthetic", rngs=rngs, n_samples=5, sample_rate=8000, duration=0.5
        )
        with pytest.raises(IndexError):
            source[5]

    def test_iteration(self, rngs) -> None:
        source = create_audio_dataset(
            "synthetic", rngs=rngs, n_samples=3, sample_rate=8000, duration=0.5
        )
        elements = list(source)
        assert len(elements) == 3
        for el in elements:
            assert "audio" in el

    def test_get_batch(self, rngs) -> None:
        source = create_audio_dataset(
            "synthetic", rngs=rngs, n_samples=10, sample_rate=8000, duration=0.5
        )
        batch = source.get_batch(4)
        assert "audio" in batch
        assert batch["audio"].shape == (4, 4000)

    def test_unknown_type_raises(self, rngs) -> None:
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_audio_dataset("unknown", rngs=rngs)


# --- Pipeline integration ---


class TestAudioPipeline:
    """Test datarax pipeline integration."""

    def test_batched_pipeline(self, rngs) -> None:
        source = create_audio_dataset(
            "synthetic", rngs=rngs, n_samples=6, sample_rate=8000, duration=0.5
        )
        pipeline = Pipeline(source=source, stages=[], batch_size=3, rngs=nnx.Rngs(0))
        batch = next(iter(pipeline))
        assert "audio" in batch
        assert batch["audio"].shape[0] == 3
