"""Tests for text datasets backed by datarax MemorySource."""

import jax.numpy as jnp
import pytest
from datarax import build_source_pipeline
from datarax.core.data_source import DataSourceModule
from datarax.sources import MemorySource
from flax import nnx

from artifex.generative_models.modalities.text.datasets import (
    create_text_dataset,
    generate_synthetic_text_data,
    generate_text_from_strings,
    simple_tokenize,
)


@pytest.fixture
def rngs():
    """Random number generators for testing."""
    return nnx.Rngs(42)


# --- Tokenization ---


class TestSimpleTokenize:
    """Test standalone tokenization function."""

    def test_basic_tokenization(self) -> None:
        tokens = simple_tokenize("hello world", max_length=16)
        assert tokens.shape == (16,)
        assert tokens.dtype == jnp.int32

    def test_bos_eos_tokens(self) -> None:
        tokens = simple_tokenize("hello", max_length=16, bos_token_id=2, eos_token_id=3)
        assert int(tokens[0]) == 2  # BOS
        # EOS should be somewhere in there
        assert 3 in tokens.tolist()

    def test_padding(self) -> None:
        tokens = simple_tokenize("hi", max_length=16, pad_token_id=0)
        # Most of the sequence should be padding
        assert int(tokens[-1]) == 0

    def test_truncation(self) -> None:
        long_text = " ".join(f"word{i}" for i in range(100))
        tokens = simple_tokenize(long_text, max_length=16)
        assert tokens.shape == (16,)


# --- Data generation functions ---


class TestGenerateSyntheticTextData:
    """Test pure text generation function."""

    def test_basic_generation(self) -> None:
        data = generate_synthetic_text_data(10, max_length=16, vocab_size=100)
        assert "text_tokens" in data
        assert "index" in data
        assert data["text_tokens"].shape == (10, 16)
        assert data["index"].shape == (10,)

    def test_pattern_types(self) -> None:
        for pattern in ["random_sentences", "repeated_phrases", "sequences", "palindromes"]:
            data = generate_synthetic_text_data(
                3, max_length=32, vocab_size=100, pattern_type=pattern
            )
            assert data["text_tokens"].shape == (3, 32)

    def test_token_dtype(self) -> None:
        data = generate_synthetic_text_data(5, max_length=16)
        assert data["text_tokens"].dtype == jnp.int32


class TestGenerateTextFromStrings:
    """Test string-to-token generation function."""

    def test_basic_generation(self) -> None:
        texts = ["hello world", "machine learning"]
        data = generate_text_from_strings(texts, max_length=16, vocab_size=100)
        assert "text_tokens" in data
        assert "index" in data
        assert data["text_tokens"].shape == (2, 16)
        assert data["index"].shape == (2,)

    def test_preserves_count(self) -> None:
        texts = ["a", "b", "c", "d"]
        data = generate_text_from_strings(texts, max_length=16)
        assert data["text_tokens"].shape[0] == 4


# --- MemorySource factory ---


class TestCreateTextDataset:
    """Test factory function returns MemorySource."""

    def test_returns_memory_source(self, rngs) -> None:
        source = create_text_dataset("synthetic", rngs=rngs, dataset_size=5, max_length=16)
        assert isinstance(source, MemorySource)
        assert isinstance(source, DataSourceModule)

    def test_correct_length(self, rngs) -> None:
        source = create_text_dataset("synthetic", rngs=rngs, dataset_size=10, max_length=16)
        assert len(source) == 10

    def test_getitem(self, rngs) -> None:
        source = create_text_dataset(
            "synthetic", rngs=rngs, dataset_size=5, max_length=16, vocab_size=100
        )
        item = source[0]
        assert "text_tokens" in item
        assert "index" in item
        assert item["text_tokens"].shape == (16,)

    def test_negative_index(self, rngs) -> None:
        source = create_text_dataset("synthetic", rngs=rngs, dataset_size=5, max_length=16)
        item = source[-1]
        assert "text_tokens" in item

    def test_out_of_bounds_raises(self, rngs) -> None:
        source = create_text_dataset("synthetic", rngs=rngs, dataset_size=5, max_length=16)
        with pytest.raises(IndexError):
            source[5]

    def test_iteration(self, rngs) -> None:
        source = create_text_dataset(
            "synthetic", rngs=rngs, dataset_size=3, max_length=16, vocab_size=100
        )
        elements = list(source)
        assert len(elements) == 3
        for el in elements:
            assert "text_tokens" in el
            assert el["text_tokens"].shape == (16,)

    def test_get_batch(self, rngs) -> None:
        source = create_text_dataset(
            "synthetic", rngs=rngs, dataset_size=10, max_length=16, vocab_size=100
        )
        batch = source.get_batch(4)
        assert "text_tokens" in batch
        assert batch["text_tokens"].shape == (4, 16)

    def test_pattern_types(self, rngs) -> None:
        for pattern in ["random_sentences", "repeated_phrases", "sequences", "palindromes"]:
            source = create_text_dataset(
                "synthetic",
                rngs=rngs,
                dataset_size=3,
                max_length=32,
                vocab_size=100,
                pattern_type=pattern,
            )
            batch = source.get_batch(2)
            assert batch["text_tokens"].shape == (2, 32)

    def test_simple_dataset(self, rngs) -> None:
        source = create_text_dataset(
            "simple",
            rngs=rngs,
            texts=["hello", "world"],
            vocab_size=100,
            max_length=16,
        )
        assert isinstance(source, MemorySource)
        assert len(source) == 2
        item = source[0]
        assert "text_tokens" in item
        assert item["text_tokens"].shape == (16,)

    def test_simple_get_batch(self, rngs) -> None:
        source = create_text_dataset(
            "simple",
            rngs=rngs,
            texts=["a", "b", "c", "d"],
            vocab_size=100,
            max_length=16,
        )
        batch = source.get_batch(2)
        assert "text_tokens" in batch
        assert batch["text_tokens"].shape == (2, 16)

    def test_unknown_type_raises(self, rngs) -> None:
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_text_dataset("unknown", rngs=rngs)


# --- Pipeline integration ---


class TestTextPipeline:
    """Test datarax pipeline integration."""

    def test_batched_pipeline(self, rngs) -> None:
        source = create_text_dataset(
            "synthetic", rngs=rngs, dataset_size=6, max_length=16, vocab_size=100
        )
        pipeline = build_source_pipeline(source, batch_size=3)
        batch = next(iter(pipeline)).get_data()
        assert "text_tokens" in batch
        assert batch["text_tokens"].shape[0] == 3
