"""Tests for multi-modal dataset MemorySource integration."""

import jax.numpy as jnp
import pytest
from datarax import build_source_pipeline
from datarax.core.data_source import DataSourceModule
from datarax.sources import MemorySource
from flax import nnx

from artifex.generative_models.modalities.multi_modal.datasets import (
    create_paired_multi_modal_dataset,
    create_synthetic_multi_modal_dataset,
    generate_multi_modal_data,
)


@pytest.fixture
def rngs():
    """Standard RNG fixture."""
    return nnx.Rngs(42)


class TestGenerateMultiModalData:
    """Test raw data generation functions."""

    def test_rejects_unsupported_modalities(self):
        """Unsupported modalities should fail explicitly instead of being dropped."""
        with pytest.raises(ValueError, match="Unsupported multi-modal helper modality"):
            generate_multi_modal_data(
                modalities=("image", "protein"),
                num_samples=2,
            )

    def test_generates_image_and_text(self):
        """Test generating image+text data dict."""
        data = generate_multi_modal_data(
            modalities=("image", "text"),
            num_samples=10,
            image_shape=(8, 8, 3),
            text_sequence_length=5,
        )

        assert "image" in data
        assert "text" in data
        assert "alignment_score" in data
        assert "latent" in data
        assert data["image"].shape == (10, 8, 8, 3)
        assert data["text"].shape == (10, 5)

    def test_generates_audio(self):
        """Test generating data with audio modality."""
        data = generate_multi_modal_data(
            modalities=("audio",),
            num_samples=5,
            audio_sample_rate=8000,
            audio_duration=0.5,
        )

        assert "audio" in data
        expected_len = int(8000 * 0.5)
        assert data["audio"].shape == (5, expected_len)

    def test_image_only(self):
        """Test generating image-only data."""
        data = generate_multi_modal_data(
            modalities=("image",),
            num_samples=5,
            image_shape=(8, 8, 1),
        )

        assert "image" in data
        assert "text" not in data
        assert data["image"].shape == (5, 8, 8, 1)

    def test_alignment_score_matches_strength(self):
        """Test alignment scores reflect the configured strength."""
        data = generate_multi_modal_data(
            modalities=("image",),
            num_samples=5,
            alignment_strength=0.5,
        )
        assert jnp.allclose(data["alignment_score"], 0.5)


class TestCreateSyntheticMultiModalDataset:
    """Test the main factory function."""

    def test_rejects_unsupported_modalities(self, rngs):
        """Factory should preserve the same supported modality boundary."""
        with pytest.raises(ValueError, match="Unsupported multi-modal helper modality"):
            create_synthetic_multi_modal_dataset(
                modalities=("image", "protein"),
                num_samples=2,
                rngs=rngs,
            )

    def test_returns_memory_source(self, rngs):
        """Test that factory returns a MemorySource (which is a DataSourceModule)."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image", "text"),
            num_samples=10,
            rngs=rngs,
            image_shape=(8, 8, 3),
            text_sequence_length=5,
        )

        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)

    def test_correct_length(self, rngs):
        """Test dataset has correct number of samples."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image", "text"),
            num_samples=20,
            rngs=rngs,
            image_shape=(8, 8, 3),
        )
        assert len(source) == 20

    def test_getitem_returns_dict(self, rngs):
        """Test __getitem__ returns dict with modality keys."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image", "text"),
            num_samples=10,
            rngs=rngs,
            image_shape=(8, 8, 3),
            text_sequence_length=5,
        )
        sample = source[0]

        assert isinstance(sample, dict)
        assert "image" in sample
        assert "text" in sample
        assert "alignment_score" in sample
        assert "latent" in sample
        assert sample["image"].shape == (8, 8, 3)
        assert sample["text"].shape == (5,)

    def test_negative_index(self, rngs):
        """Test negative indexing works."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image",),
            num_samples=10,
            rngs=rngs,
            image_shape=(8, 8, 3),
        )
        sample = source[-1]
        assert "image" in sample

    def test_out_of_bounds_raises(self, rngs):
        """Test IndexError on out-of-bounds access."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image",),
            num_samples=10,
            rngs=rngs,
            image_shape=(8, 8, 3),
        )
        with pytest.raises(IndexError):
            source[10]

    def test_iteration(self, rngs):
        """Test __iter__ yields all samples."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image",),
            num_samples=10,
            rngs=rngs,
            image_shape=(8, 8, 3),
        )
        samples = list(source)
        assert len(samples) == 10

    def test_get_batch(self, rngs):
        """Test get_batch returns stacked arrays."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image", "text"),
            num_samples=20,
            rngs=rngs,
            image_shape=(8, 8, 3),
            text_sequence_length=5,
        )
        batch = source.get_batch(5)

        assert "image" in batch
        assert "text" in batch
        assert batch["image"].shape == (5, 8, 8, 3)
        assert batch["text"].shape == (5, 5)

    def test_with_audio(self, rngs):
        """Test dataset with all three modalities."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image", "text", "audio"),
            num_samples=10,
            rngs=rngs,
            image_shape=(8, 8, 3),
            audio_sample_rate=8000,
            audio_duration=0.5,
        )
        sample = source[0]

        assert "audio" in sample
        assert sample["audio"].shape == (int(8000 * 0.5),)

    def test_accepts_list_modalities(self, rngs):
        """Test that list modalities are accepted (converted to tuple)."""
        source = create_synthetic_multi_modal_dataset(
            modalities=["image", "text"],
            num_samples=5,
            rngs=rngs,
            image_shape=(8, 8, 3),
        )
        assert len(source) == 5


class TestCreatePairedMultiModalDataset:
    """Test paired dataset factory."""

    def test_returns_memory_source(self):
        """Test that factory returns a MemorySource."""
        data = {
            "image": jnp.ones((10, 8, 8, 3)),
            "text": jnp.ones((10, 5), dtype=jnp.int32),
        }
        source = create_paired_multi_modal_dataset(data)

        assert isinstance(source, MemorySource)
        assert len(source) == 10

    def test_getitem(self):
        """Test sample access on paired data."""
        data = {
            "image": jnp.ones((5, 8, 8, 3)),
            "text": jnp.ones((5, 10), dtype=jnp.int32),
        }
        source = create_paired_multi_modal_dataset(data)
        sample = source[0]

        assert "image" in sample
        assert "text" in sample

    def test_with_alignments(self):
        """Test paired data with alignment scores."""
        data = {
            "image": jnp.ones((5, 8, 8, 3)),
            "text": jnp.ones((5, 10), dtype=jnp.int32),
        }
        alignments = jnp.array([0.9, 0.8, 0.7, 0.6, 0.5])
        source = create_paired_multi_modal_dataset(data, alignments=alignments)

        sample = source[0]
        assert "alignment_scores" in sample

    def test_mismatched_samples_raises(self):
        """Test that mismatched sample counts raise ValueError."""
        data = {
            "image": jnp.ones((10, 8, 8, 3)),
            "text": jnp.ones((5, 10)),
        }
        with pytest.raises(ValueError, match="samples"):
            create_paired_multi_modal_dataset(data)


class TestPipelineIntegration:
    """Test datarax pipeline integration."""

    def test_batched_pipeline(self, rngs):
        """Test that MemorySource works with the datarax batched pipeline."""
        source = create_synthetic_multi_modal_dataset(
            modalities=("image", "text"),
            num_samples=20,
            rngs=rngs,
            image_shape=(8, 8, 3),
            text_sequence_length=5,
        )
        pipeline = build_source_pipeline(source, batch_size=5)
        batch = next(iter(pipeline)).get_data()

        assert "image" in batch
        assert batch["image"].shape[0] == 5
