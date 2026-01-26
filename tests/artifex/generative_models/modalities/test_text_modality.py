"""Tests for text modality implementation.

This module provides comprehensive tests for the text modality,
covering all components and functionality.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ModalityConfig
from artifex.generative_models.modalities.text import (
    compute_text_metrics,
    create_text_dataset,
    create_text_modality,
    create_text_processor,
    PositionEncodingProcessor,
    SequenceAugmentationProcessor,
    SimpleTextDataset,
    SyntheticTextDataset,
    TextEvaluationSuite,
    TextMetrics,
    TextModality,
    TextProcessor,
    TokenizationProcessor,
)


def get_text_params(config: ModalityConfig) -> dict:
    """Extract text parameters from config for easier access in tests."""
    return config.metadata.get("text_params", {})


@pytest.fixture
def rngs():
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def config():
    """Standard text modality configuration."""
    cfg = ModalityConfig(
        name="test_text_modality",
        modality_name="text",
        supported_models=["transformer", "vae", "diffusion"],
        default_metrics=["perplexity", "bleu", "rouge"],
        preprocessing_steps=[
            {"type": "tokenize", "strategy": "simple"},
            {"type": "pad", "max_length": 128},
        ],
        metadata={
            "text_params": {
                "vocab_size": 1000,
                "max_length": 128,
                "min_length": 1,
                "representation": "word_level",
                "tokenization_strategy": "simple",
                "pad_token_id": 0,
                "unk_token_id": 1,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "special_tokens": [],
                "handle_oov": "unk",
                "case_sensitive": True,
            },
        },
    )
    return cfg


@pytest.fixture
def simple_config():
    """Simple configuration for basic tests."""
    cfg = ModalityConfig(
        name="simple_text_modality",
        modality_name="text",
        supported_models=["transformer"],
        default_metrics=["perplexity"],
        preprocessing_steps=[
            {"type": "tokenize", "strategy": "simple"},
            {"type": "pad", "max_length": 32},
        ],
        metadata={
            "text_params": {
                "vocab_size": 100,
                "max_length": 32,
                "min_length": 1,
                "representation": "word_level",
                "pad_token_id": 0,
                "unk_token_id": 1,
                "bos_token_id": 2,
                "eos_token_id": 3,
            },
        },
    )
    return cfg


# TestTextModalityConfig removed - using ModalityConfig instead


class TestTextModality:
    """Test text modality core functionality."""

    def test_initialization(self, config, rngs):
        """Test text modality initialization."""
        modality = TextModality(config=config, rngs=rngs)
        assert modality.config == config
        assert modality.name == "text"

    def test_initialization_default_config(self, rngs):
        """Test initialization with default config."""
        modality = TextModality(rngs=rngs)
        assert modality.config is not None
        assert modality.vocab_size == 10000

    def test_get_extensions(self, config, rngs):
        """Test extension configuration."""
        modality = TextModality(config=config, rngs=rngs)

        # Default extensions - use the modality's config
        extensions = modality.get_extensions(config)
        assert "position_encoding" in extensions
        assert extensions["position_encoding"]["type"] == "sinusoidal"

        # Custom extensions - create a new ModalityConfig with custom metadata
        custom_config = ModalityConfig(
            name="custom_text_modality",
            modality_name="text",
            supported_models=["transformer"],
            default_metrics=["perplexity"],
            preprocessing_steps=[],
            metadata={
                "text_params": {
                    "vocab_size": 1000,
                    "max_length": 128,
                    "pad_token_id": 0,
                },
                "use_position_encoding": True,
                "position_encoding_type": "learned",
                "use_attention_monitoring": True,
                "track_head_importance": False,
                "use_custom_tokenizer": True,
                "tokenization_strategy": "bpe",
            },
        )
        extensions = modality.get_extensions(custom_config)
        assert extensions["position_encoding"]["type"] == "learned"
        assert "attention_monitoring" in extensions
        assert extensions["attention_monitoring"]["track_head_importance"] is False
        assert "tokenization" in extensions
        assert extensions["tokenization"]["strategy"] == "bpe"

    def test_preprocess_text_single(self, config, rngs):
        """Test preprocessing single text."""
        modality = TextModality(config=config, rngs=rngs)

        text = "hello world"
        tokens = modality.preprocess_text(text)

        assert tokens.shape == (1, modality.max_length)
        assert tokens.dtype == jnp.int32
        assert tokens[0, 0] == modality.bos_token_id  # BOS token
        assert tokens[0, -1] == modality.pad_token_id  # Should be padded

    def test_preprocess_text_batch(self, config, rngs):
        """Test preprocessing batch of texts."""
        modality = TextModality(config=config, rngs=rngs)

        texts = ["hello world", "machine learning", "deep neural networks"]
        tokens = modality.preprocess_text(texts)

        assert tokens.shape == (3, modality.max_length)
        assert tokens.dtype == jnp.int32

        # Check BOS tokens
        for i in range(3):
            assert tokens[i, 0] == modality.bos_token_id

    def test_case_sensitivity(self, rngs):
        """Test case sensitivity handling."""
        # Case insensitive (default)
        config_insensitive = ModalityConfig(
            name="case_insensitive_text",
            modality_name="text",
            supported_models=["transformer"],
            default_metrics=["perplexity"],
            preprocessing_steps=[
                {"type": "tokenize", "strategy": "simple"},
                {"type": "pad", "max_length": 32},
            ],
            metadata={
                "text_params": {
                    "case_sensitive": False,
                    "max_length": 32,
                    "vocab_size": 1000,
                },
            },
        )
        modality_insensitive = TextModality(config=config_insensitive, rngs=rngs)

        tokens1 = modality_insensitive.preprocess_text("Hello World")
        tokens2 = modality_insensitive.preprocess_text("hello world")

        # Should be the same when case insensitive
        assert jnp.array_equal(tokens1, tokens2)

        # Case sensitive
        config_sensitive = ModalityConfig(
            name="case_sensitive_text",
            modality_name="text",
            supported_models=["transformer"],
            default_metrics=["perplexity"],
            preprocessing_steps=[
                {"type": "tokenize", "strategy": "simple"},
                {"type": "pad", "max_length": 32},
            ],
            metadata={
                "text_params": {
                    "case_sensitive": True,
                    "max_length": 32,
                    "vocab_size": 1000,
                },
            },
        )
        modality_sensitive = TextModality(config=config_sensitive, rngs=rngs)

        tokens3 = modality_sensitive.preprocess_text("Hello World")
        tokens4 = modality_sensitive.preprocess_text("hello world")

        # Should be different when case sensitive
        assert not jnp.array_equal(tokens3, tokens4)

    def test_postprocess_tokens(self, config, rngs):
        """Test token postprocessing."""
        modality = TextModality(config=config, rngs=rngs)

        # Create sample tokens
        tokens = jnp.array(
            [
                [modality.bos_token_id, 10, 20, 30, modality.eos_token_id]
                + [modality.pad_token_id] * (modality.max_length - 5)
            ]
        )

        texts = modality.postprocess_tokens(tokens)
        assert len(texts) == 1
        assert isinstance(texts[0], str)
        assert len(texts[0]) > 0  # Should have some content

    def test_factory_function(self, config, rngs):
        """Test factory function."""
        modality = create_text_modality(config=config, rngs=rngs)
        assert isinstance(modality, TextModality)
        assert modality.config == config

        # Test with default config
        modality_default = create_text_modality(rngs=rngs)
        assert isinstance(modality_default, TextModality)
        assert modality_default.config is not None


class TestSyntheticTextDataset:
    """Test synthetic text dataset."""

    def test_initialization(self, config, rngs):
        """Test dataset initialization."""
        dataset = SyntheticTextDataset(
            config=config,
            dataset_size=100,
            pattern_type="random_sentences",
            rngs=rngs,
        )
        assert len(dataset) == 100
        assert dataset.pattern_type == "random_sentences"

    def test_different_patterns(self, simple_config, rngs):
        """Test different pattern types."""
        patterns = ["random_sentences", "repeated_phrases", "sequences", "palindromes"]

        for pattern in patterns:
            dataset = SyntheticTextDataset(
                config=simple_config,
                dataset_size=10,
                pattern_type=pattern,
                rngs=rngs,
            )
            assert len(dataset) == 10

            # Test data generation
            sample = next(iter(dataset))
            assert "text_tokens" in sample
            assert "text" in sample
            assert sample["text_tokens"].shape == (get_text_params(simple_config)["max_length"],)

    def test_iteration(self, simple_config, rngs):
        """Test dataset iteration."""
        dataset = SyntheticTextDataset(
            config=simple_config,
            dataset_size=5,
            rngs=rngs,
        )

        samples = list(dataset)
        assert len(samples) == 5

        for sample in samples:
            assert "text_tokens" in sample
            assert "text" in sample
            assert "index" in sample
            assert sample["text_tokens"].dtype == jnp.int32

    def test_get_batch(self, simple_config, rngs):
        """Test batch generation."""
        dataset = SyntheticTextDataset(
            config=simple_config,
            dataset_size=20,
            rngs=rngs,
        )

        batch = dataset.get_batch(batch_size=4)
        assert "text_tokens" in batch
        assert "texts" in batch
        assert "indices" in batch
        assert batch["text_tokens"].shape == (4, get_text_params(simple_config)["max_length"])
        assert len(batch["texts"]) == 4

    def test_vocab_stats(self, simple_config, rngs):
        """Test vocabulary statistics."""
        dataset = SyntheticTextDataset(
            config=simple_config,
            dataset_size=10,
            rngs=rngs,
        )

        stats = dataset.get_vocab_stats()
        assert "unique_tokens" in stats
        assert "vocab_coverage" in stats
        assert "total_sequences" in stats
        assert stats["total_sequences"] == 10
        assert stats["unique_tokens"] > 0

    def test_get_sample_text(self, simple_config, rngs):
        """Test getting sample text."""
        dataset = SyntheticTextDataset(
            config=simple_config,
            dataset_size=5,
            rngs=rngs,
        )

        text = dataset.get_sample_text(0)
        assert isinstance(text, str)
        assert len(text) > 0

        # Test invalid index
        with pytest.raises(IndexError):
            dataset.get_sample_text(10)


class TestSimpleTextDataset:
    """Test simple text dataset."""

    def test_initialization(self, config, rngs):
        """Test dataset initialization."""
        texts = ["hello world", "machine learning", "deep learning"]
        dataset = SimpleTextDataset(
            config=config,
            texts=texts,
            rngs=rngs,
        )
        assert len(dataset) == 3
        assert dataset.texts == texts

    def test_iteration(self, simple_config, rngs):
        """Test dataset iteration."""
        texts = ["hello", "world"]
        dataset = SimpleTextDataset(
            config=simple_config,
            texts=texts,
            rngs=rngs,
        )

        samples = list(dataset)
        assert len(samples) == 2

        for i, sample in enumerate(samples):
            assert sample["text"] == texts[i]
            assert sample["text_tokens"].shape == (get_text_params(simple_config)["max_length"],)

    def test_get_batch(self, simple_config, rngs):
        """Test batch generation."""
        texts = ["hello world", "machine learning", "deep neural networks"]
        dataset = SimpleTextDataset(
            config=simple_config,
            texts=texts,
            rngs=rngs,
        )

        batch = dataset.get_batch(batch_size=2)
        assert batch["text_tokens"].shape == (2, get_text_params(simple_config)["max_length"])
        assert len(batch["texts"]) == 2


class TestTextEvaluationSuite:
    """Test text evaluation metrics."""

    def test_initialization(self, config, rngs):
        """Test evaluation suite initialization."""
        evaluator = TextEvaluationSuite(config=config, rngs=rngs)
        assert evaluator.config == config

    def test_bleu_score(self, simple_config, rngs):
        """Test BLEU score computation."""
        evaluator = TextEvaluationSuite(config=simple_config, rngs=rngs)

        # Create sample sequences
        generated = jnp.array(
            [
                [2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
                [2, 10, 25, 35, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
            ]
        )
        reference = jnp.array(
            [
                [2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
                [2, 15, 25, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
            ]
        )

        bleu_scores = evaluator.compute_bleu_score(generated, reference, n=1)
        assert bleu_scores.shape == (2,)
        assert 0.0 <= bleu_scores[0] <= 1.0
        assert 0.0 <= bleu_scores[1] <= 1.0

    def test_rouge_l(self, simple_config, rngs):
        """Test ROUGE-L score computation."""
        evaluator = TextEvaluationSuite(config=simple_config, rngs=rngs)

        generated = jnp.array(
            [[2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )
        reference = jnp.array(
            [[2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )

        rouge_scores = evaluator.compute_rouge_l(generated, reference)
        assert rouge_scores.shape == (1,)
        assert rouge_scores[0] == 1.0  # Perfect match

    def test_distinct_ngrams(self, simple_config, rngs):
        """Test distinct n-gram computation."""
        evaluator = TextEvaluationSuite(config=simple_config, rngs=rngs)

        # Repetitive sequence
        repetitive = jnp.array(
            [[2, 10, 10, 10, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )

        distinct_1 = evaluator.compute_distinct_ngrams(repetitive, n=1)
        assert 0.0 <= distinct_1[0] <= 1.0

        # Diverse sequence
        diverse = jnp.array(
            [[2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )

        distinct_1_diverse = evaluator.compute_distinct_ngrams(diverse, n=1)
        assert distinct_1_diverse[0] >= distinct_1[0]  # More diverse

    def test_repetition_rate(self, simple_config, rngs):
        """Test repetition rate computation."""
        evaluator = TextEvaluationSuite(config=simple_config, rngs=rngs)

        # Repetitive sequence
        repetitive = jnp.array(
            [[2, 10, 10, 20, 20, 3] + [0] * (get_text_params(simple_config)["max_length"] - 6)]
        )

        rep_rates = evaluator.compute_repetition_rate(repetitive)
        assert rep_rates.shape == (1,)
        assert rep_rates[0] > 0.0  # Should have some repetition

    def test_vocabulary_coverage(self, simple_config, rngs):
        """Test vocabulary coverage computation."""
        evaluator = TextEvaluationSuite(config=simple_config, rngs=rngs)

        # Create sequences with diverse tokens
        diverse_tokens = jnp.array(
            [
                [2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
                [2, 40, 50, 60, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
            ]
        )

        coverage = evaluator.compute_vocabulary_coverage(diverse_tokens)
        assert 0.0 <= coverage <= 1.0

    def test_evaluate_batch(self, simple_config, rngs):
        """Test batch evaluation."""
        evaluator = TextEvaluationSuite(config=simple_config, rngs=rngs)

        generated = jnp.array(
            [
                [2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
                [2, 15, 25, 35, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5),
            ]
        )

        metrics = evaluator.evaluate_batch(generated)
        assert isinstance(metrics, TextMetrics)
        assert 0.0 <= metrics.distinct_1 <= 1.0
        assert 0.0 <= metrics.distinct_2 <= 1.0
        assert 0.0 <= metrics.repetition_rate <= 1.0
        assert 0.0 <= metrics.vocab_coverage <= 1.0
        assert metrics.avg_length > 0

    def test_evaluate_batch_with_reference(self, simple_config, rngs):
        """Test batch evaluation with reference sequences."""
        evaluator = TextEvaluationSuite(config=simple_config, rngs=rngs)

        generated = jnp.array(
            [[2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )
        reference = jnp.array(
            [[2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )

        metrics = evaluator.evaluate_batch(generated, reference_tokens=reference)
        assert metrics.bleu_1 > 0.0
        assert metrics.rouge_l > 0.0


class TestTextProcessors:
    """Test text processing utilities."""

    def test_text_processor(self, config, rngs):
        """Test basic text processor."""
        processor = TextProcessor(config=config, rngs=rngs)

        sequences = jnp.array([[2, 10, 20, 3] + [0] * (get_text_params(config)["max_length"] - 4)])

        result = processor.process_sequences(sequences)
        assert "sequences" in result
        assert "mask" in result
        assert "lengths" in result
        assert result["batch_size"] == 1

    def test_causal_mask(self, config, rngs):
        """Test causal mask creation."""
        processor = TextProcessor(config=config, rngs=rngs)

        mask = processor.create_causal_mask(5)
        assert mask.shape == (5, 5)
        assert mask[0, 1] == 0  # Upper triangular should be 0
        assert mask[1, 0] == 1  # Lower triangular should be 1

    def test_tokenization_processor(self, simple_config, rngs):
        """Test tokenization processor."""
        processor = TokenizationProcessor(config=simple_config, rngs=rngs)

        # Test tokenization
        text = "hello world"
        tokens = processor.tokenize_text(text)
        assert tokens.shape == (get_text_params(simple_config)["max_length"],)
        assert tokens[0] == get_text_params(simple_config)["bos_token_id"]

        # Test detokenization
        reconstructed = processor.detokenize_sequence(tokens)
        assert isinstance(reconstructed, str)

        # Test batch processing
        texts = ["hello", "world"]
        batch_tokens = processor.encode_batch(texts)
        assert batch_tokens.shape == (2, get_text_params(simple_config)["max_length"])

        batch_texts = processor.decode_batch(batch_tokens)
        assert len(batch_texts) == 2

    def test_position_encoding_processor(self, simple_config, rngs):
        """Test position encoding processor."""
        # Sinusoidal encoding
        processor_sin = PositionEncodingProcessor(
            config=simple_config,
            embedding_dim=64,
            encoding_type="sinusoidal",
            rngs=rngs,
        )

        embeddings = jnp.ones((2, 10, 64))
        encoded = processor_sin.apply_position_encoding(embeddings)
        assert encoded.shape == embeddings.shape

        # Learned encoding
        processor_learned = PositionEncodingProcessor(
            config=simple_config,
            embedding_dim=64,
            encoding_type="learned",
            rngs=rngs,
        )

        encoded_learned = processor_learned.apply_position_encoding(embeddings)
        assert encoded_learned.shape == embeddings.shape

    def test_augmentation_processor(self, simple_config, rngs):
        """Test sequence augmentation processor."""
        processor = SequenceAugmentationProcessor(
            config=simple_config,
            dropout_rate=0.1,
            rngs=rngs,
        )

        sequences = jnp.array(
            [[2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )

        # Test token dropout
        augmented = processor.apply_token_dropout(sequences, deterministic=False)
        assert augmented.shape == sequences.shape

        # Test deterministic mode
        deterministic_result = processor.apply_token_dropout(sequences, deterministic=True)
        assert jnp.array_equal(deterministic_result, sequences)

    def test_processor_factory(self, simple_config, rngs):
        """Test processor factory function."""
        # Basic processor
        basic = create_text_processor(simple_config, "basic", rngs=rngs)
        assert isinstance(basic, TextProcessor)

        # Tokenization processor
        tokenizer = create_text_processor(simple_config, "tokenization", rngs=rngs)
        assert isinstance(tokenizer, TokenizationProcessor)

        # Position encoding processor
        position = create_text_processor(simple_config, "position", rngs=rngs, embedding_dim=128)
        assert isinstance(position, PositionEncodingProcessor)

        # Augmentation processor
        augment = create_text_processor(simple_config, "augmentation", rngs=rngs, dropout_rate=0.2)
        assert isinstance(augment, SequenceAugmentationProcessor)


class TestTextDatasetFactory:
    """Test text dataset factory functions."""

    def test_create_synthetic_dataset(self, simple_config, rngs):
        """Test synthetic dataset creation."""
        dataset = create_text_dataset(
            config=simple_config,
            dataset_type="synthetic",
            rngs=rngs,
            dataset_size=10,
            pattern_type="random_sentences",
        )
        assert isinstance(dataset, SyntheticTextDataset)
        assert len(dataset) == 10

    def test_create_simple_dataset(self, simple_config, rngs):
        """Test simple dataset creation."""
        texts = ["hello world", "machine learning"]
        dataset = create_text_dataset(
            config=simple_config,
            dataset_type="simple",
            rngs=rngs,
            texts=texts,
        )
        assert isinstance(dataset, SimpleTextDataset)
        assert len(dataset) == 2

    def test_invalid_dataset_type(self, simple_config, rngs):
        """Test invalid dataset type."""
        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_text_dataset(
                config=simple_config,
                dataset_type="invalid",
                rngs=rngs,
            )


class TestTextMetricsFactory:
    """Test text metrics factory function."""

    def test_compute_text_metrics(self, simple_config, rngs):
        """Test text metrics computation."""
        generated = jnp.array(
            [[2, 10, 20, 30, 3] + [0] * (get_text_params(simple_config)["max_length"] - 5)]
        )

        metrics = compute_text_metrics(
            generated_tokens=generated,
            config=simple_config,
            rngs=rngs,
        )
        assert isinstance(metrics, TextMetrics)
        assert metrics.avg_length > 0


class TestTextModalityIntegration:
    """Integration tests for text modality components."""

    def test_end_to_end_workflow(self, simple_config, rngs):
        """Test complete end-to-end workflow."""
        # Create modality
        modality = create_text_modality(config=simple_config, rngs=rngs)

        # Create dataset
        dataset = create_text_dataset(
            config=simple_config,
            dataset_type="synthetic",
            rngs=rngs,
            dataset_size=5,
        )

        # Get batch
        batch = dataset.get_batch(batch_size=2)
        generated_tokens = batch["text_tokens"]

        # Compute metrics
        metrics = compute_text_metrics(
            generated_tokens=generated_tokens,
            config=simple_config,
            rngs=rngs,
        )

        # Verify everything works
        assert isinstance(modality, TextModality)
        assert generated_tokens.shape == (2, get_text_params(simple_config)["max_length"])
        assert isinstance(metrics, TextMetrics)
        assert metrics.avg_length > 0

    def test_modality_with_processors(self, simple_config, rngs):
        """Test modality integration with processors."""
        modality = create_text_modality(config=simple_config, rngs=rngs)

        # Create tokenization processor
        tokenizer = create_text_processor(simple_config, "tokenization", rngs=rngs)

        # Test text processing
        texts = ["hello world", "machine learning"]
        tokens = tokenizer.encode_batch(texts)
        reconstructed = tokenizer.decode_batch(tokens)

        assert tokens.shape == (2, get_text_params(simple_config)["max_length"])
        assert len(reconstructed) == 2
        assert all(isinstance(text, str) for text in reconstructed)

        # Use modality to avoid unused variable warning
        assert isinstance(modality, TextModality)

    def test_configuration_flexibility(self, rngs):
        """Test configuration flexibility."""
        # Small config
        small_config = ModalityConfig(
            name="small_text",
            modality_name="text",
            supported_models=["transformer"],
            default_metrics=["perplexity"],
            preprocessing_steps=[
                {"type": "tokenize", "strategy": "simple"},
                {"type": "pad", "max_length": 16},
            ],
            metadata={
                "text_params": {
                    "vocab_size": 50,
                    "max_length": 16,
                },
            },
        )
        small_modality = create_text_modality(config=small_config, rngs=rngs)

        # Large config
        large_config = ModalityConfig(
            name="large_text",
            modality_name="text",
            supported_models=["transformer"],
            default_metrics=["perplexity"],
            preprocessing_steps=[
                {"type": "tokenize", "strategy": "simple"},
                {"type": "pad", "max_length": 256},
            ],
            metadata={
                "text_params": {
                    "vocab_size": 5000,
                    "max_length": 256,
                },
            },
        )
        large_modality = create_text_modality(config=large_config, rngs=rngs)

        # Both should work
        assert small_modality.vocab_size == 50
        assert large_modality.vocab_size == 5000

        # Test with different representations
        char_config = ModalityConfig(
            name="char_text",
            modality_name="text",
            supported_models=["transformer"],
            default_metrics=["perplexity"],
            preprocessing_steps=[
                {"type": "tokenize", "strategy": "character"},
            ],
            metadata={
                "text_params": {
                    "representation": "character",
                    "vocab_size": 256,
                    "max_length": 64,
                },
            },
        )
        char_modality = create_text_modality(config=char_config, rngs=rngs)
        assert char_modality.config.metadata["text_params"]["representation"] == "character"
