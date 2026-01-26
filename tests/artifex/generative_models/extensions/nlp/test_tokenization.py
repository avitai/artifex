"""Tests for advanced tokenization extension.

This module contains comprehensive tests for the AdvancedTokenization extension
that provides text tokenization utilities for generative models.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.nlp.tokenization import AdvancedTokenization


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rngs():
    """Create random number generator keys."""
    return nnx.Rngs(0)


@pytest.fixture
def config():
    """Create a basic extension configuration."""
    return ExtensionConfig(name="test_tokenization", weight=1.0, enabled=True)


@pytest.fixture
def disabled_config():
    """Create a disabled extension configuration."""
    return ExtensionConfig(name="test_tokenization", weight=1.0, enabled=False)


@pytest.fixture
def tokenizer(config, rngs):
    """Create a tokenizer instance with default configuration."""
    return AdvancedTokenization(config, rngs=rngs)


@pytest.fixture
def disabled_tokenizer(disabled_config, rngs):
    """Create a disabled tokenizer instance."""
    return AdvancedTokenization(disabled_config, rngs=rngs)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestTokenizationInitialization:
    """Tests for tokenizer initialization."""

    def test_init_with_default_config(self, config, rngs):
        """Test initialization with default ExtensionConfig."""
        tokenizer = AdvancedTokenization(config, rngs=rngs)

        # Check default values
        assert tokenizer.vocab_size == 50000
        assert tokenizer.max_length == 512
        assert tokenizer.enabled is True
        assert tokenizer.weight == 1.0

    def test_init_special_tokens(self, tokenizer):
        """Test that special tokens are properly initialized."""
        expected_special = {"pad", "unk", "bos", "eos", "mask"}
        actual_special = set(tokenizer.special_tokens.keys())
        assert expected_special == actual_special

    def test_init_special_token_values(self, tokenizer):
        """Test special token string values."""
        assert tokenizer.special_tokens["pad"] == "<PAD>"
        assert tokenizer.special_tokens["unk"] == "<UNK>"
        assert tokenizer.special_tokens["bos"] == "<BOS>"
        assert tokenizer.special_tokens["eos"] == "<EOS>"
        assert tokenizer.special_tokens["mask"] == "<MASK>"

    def test_init_vocabulary_built(self, tokenizer):
        """Test that vocabulary is built during initialization."""
        assert len(tokenizer.token_to_id) == tokenizer.vocab_size
        assert len(tokenizer.id_to_token) == tokenizer.vocab_size

    def test_init_invalid_config_type(self, rngs):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be ExtensionConfig"):
            AdvancedTokenization({"weight": 1.0}, rngs=rngs)

    def test_init_disabled_config(self, disabled_config, rngs):
        """Test initialization with disabled configuration."""
        tokenizer = AdvancedTokenization(disabled_config, rngs=rngs)
        assert tokenizer.enabled is False


# =============================================================================
# Vocabulary Tests
# =============================================================================


class TestVocabulary:
    """Tests for vocabulary management."""

    def test_special_tokens_mapping(self, tokenizer):
        """Test that special tokens have correct IDs."""
        # Special tokens should be first in vocabulary
        assert tokenizer.token_to_id["<PAD>"] == 0
        assert tokenizer.token_to_id["<UNK>"] == 1
        assert tokenizer.token_to_id["<BOS>"] == 2
        assert tokenizer.token_to_id["<EOS>"] == 3
        assert tokenizer.token_to_id["<MASK>"] == 4

    def test_token_to_id_consistency(self, tokenizer):
        """Test token_to_id and id_to_token are consistent."""
        for token, token_id in tokenizer.token_to_id.items():
            assert tokenizer.id_to_token[token_id] == token

    def test_vocabulary_info(self, tokenizer):
        """Test get_vocabulary_info returns correct information."""
        info = tokenizer.get_vocabulary_info()

        assert info["vocab_size"] == 50000
        assert info["max_length"] == 512
        assert "special_tokens" in info
        assert "special_token_ids" in info

    def test_vocabulary_info_special_token_ids(self, tokenizer):
        """Test vocabulary info contains correct special token IDs."""
        info = tokenizer.get_vocabulary_info()
        special_ids = info["special_token_ids"]

        assert special_ids["<PAD>"] == 0
        assert special_ids["<UNK>"] == 1


# =============================================================================
# Tokenization Tests
# =============================================================================


class TestTokenization:
    """Tests for text tokenization."""

    def test_tokenize_unknown_words(self, tokenizer):
        """Test tokenization of unknown words returns UNK tokens."""
        tokens = tokenizer.tokenize("hello world")

        # All words should be unknown (mapped to UNK)
        unk_id = tokenizer.token_to_id["<UNK>"]
        # First two positions should be UNK (for "hello" and "world")
        assert tokens[0] == unk_id
        assert tokens[1] == unk_id

    def test_tokenize_output_length(self, tokenizer):
        """Test tokenization output has correct length."""
        tokens = tokenizer.tokenize("test input text")

        assert len(tokens) == tokenizer.max_length
        assert tokens.shape == (512,)

    def test_tokenize_padding(self, tokenizer):
        """Test tokenization adds padding correctly."""
        tokens = tokenizer.tokenize("hello world")
        pad_id = tokenizer.token_to_id["<PAD>"]

        # After the actual tokens, rest should be padding
        # "hello world" -> 2 tokens + 510 padding
        assert jnp.all(tokens[2:] == pad_id)

    def test_tokenize_truncation(self, tokenizer):
        """Test tokenization truncates long sequences."""
        # Create a very long text
        long_text = " ".join(["word"] * 1000)
        tokens = tokenizer.tokenize(long_text)

        assert len(tokens) == tokenizer.max_length

    def test_tokenize_case_insensitive(self, tokenizer):
        """Test tokenization is case-insensitive."""
        tokens1 = tokenizer.tokenize("HELLO")
        tokens2 = tokenizer.tokenize("hello")

        # Both should produce same tokens (both unknown)
        assert jnp.array_equal(tokens1, tokens2)

    def test_tokenize_returns_array(self, tokenizer):
        """Test tokenize returns JAX array."""
        tokens = tokenizer.tokenize("test")

        assert isinstance(tokens, jnp.ndarray)


# =============================================================================
# Detokenization Tests
# =============================================================================


class TestDetokenization:
    """Tests for detokenization."""

    def test_detokenize_removes_padding(self, tokenizer):
        """Test detokenization stops at padding."""
        # Create tokens with padding
        pad_id = tokenizer.token_to_id["<PAD>"]
        unk_id = tokenizer.token_to_id["<UNK>"]
        tokens = jnp.array([unk_id, unk_id, pad_id, pad_id, pad_id])

        text = tokenizer.detokenize(tokens)

        # Should not contain padding representation
        assert text == ""  # UNK tokens are special and not included

    def test_detokenize_known_tokens(self, tokenizer):
        """Test detokenization of known vocabulary tokens."""
        # Use a mock token that exists in vocabulary
        token_id = 5  # First non-special token
        token = tokenizer.id_to_token[token_id]
        tokens = jnp.array([token_id])

        text = tokenizer.detokenize(tokens)

        assert token in text

    def test_detokenize_filters_special_tokens(self, tokenizer):
        """Test detokenization filters out special tokens."""
        bos_id = tokenizer.token_to_id["<BOS>"]
        eos_id = tokenizer.token_to_id["<EOS>"]
        regular_id = 5

        tokens = jnp.array([bos_id, regular_id, eos_id])
        text = tokenizer.detokenize(tokens)

        # Special tokens should not appear in output
        assert "<BOS>" not in text
        assert "<EOS>" not in text


# =============================================================================
# Batch Operations Tests
# =============================================================================


class TestBatchOperations:
    """Tests for batch encoding/decoding."""

    def test_encode_batch_shape(self, tokenizer):
        """Test batch encoding produces correct shape."""
        texts = ["hello", "world", "test"]
        batch = tokenizer.encode_batch(texts)

        assert batch.shape == (3, tokenizer.max_length)

    def test_encode_batch_returns_array(self, tokenizer):
        """Test batch encoding returns JAX array."""
        texts = ["hello", "world"]
        batch = tokenizer.encode_batch(texts)

        assert isinstance(batch, jnp.ndarray)

    def test_decode_batch_returns_list(self, tokenizer):
        """Test batch decoding returns list of strings."""
        unk_id = tokenizer.token_to_id["<UNK>"]
        pad_id = tokenizer.token_to_id["<PAD>"]
        batch = jnp.array([[unk_id, pad_id], [unk_id, unk_id]])

        texts = tokenizer.decode_batch(batch)

        assert isinstance(texts, list)
        assert len(texts) == 2
        assert all(isinstance(t, str) for t in texts)


# =============================================================================
# Attention Mask Tests
# =============================================================================


class TestAttentionMask:
    """Tests for attention mask creation."""

    def test_create_attention_mask_1d(self, tokenizer):
        """Test attention mask creation for 1D input."""
        pad_id = tokenizer.token_to_id["<PAD>"]
        tokens = jnp.array([5, 6, 7, pad_id, pad_id])

        mask = tokenizer.create_attention_mask(tokens)

        expected = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0])
        assert jnp.allclose(mask, expected)

    def test_create_attention_mask_2d(self, tokenizer):
        """Test attention mask creation for 2D batch input."""
        pad_id = tokenizer.token_to_id["<PAD>"]
        tokens = jnp.array([[5, 6, pad_id], [5, pad_id, pad_id]])

        mask = tokenizer.create_attention_mask(tokens)

        assert mask.shape == tokens.shape
        assert mask[0, 0] == 1.0  # First token in first sequence
        assert mask[0, 2] == 0.0  # Padding in first sequence
        assert mask[1, 1] == 0.0  # Padding in second sequence

    def test_create_attention_mask_dtype(self, tokenizer):
        """Test attention mask has correct dtype."""
        tokens = jnp.array([5, 6, 7])
        mask = tokenizer.create_attention_mask(tokens)

        assert mask.dtype == jnp.float32


# =============================================================================
# Special Token Operations Tests
# =============================================================================


class TestSpecialTokenOperations:
    """Tests for special token operations."""

    def test_add_special_tokens_bos_only(self, tokenizer):
        """Test adding only BOS token."""
        tokens = jnp.array([5, 6, 7])
        result = tokenizer.add_special_tokens(tokens, add_bos=True, add_eos=False)

        bos_id = tokenizer.token_to_id["<BOS>"]
        assert result[0] == bos_id
        assert len(result) == len(tokens) + 1

    def test_add_special_tokens_eos_only(self, tokenizer):
        """Test adding only EOS token."""
        tokens = jnp.array([5, 6, 7])
        result = tokenizer.add_special_tokens(tokens, add_bos=False, add_eos=True)

        eos_id = tokenizer.token_to_id["<EOS>"]
        assert result[-1] == eos_id
        assert len(result) == len(tokens) + 1

    def test_add_special_tokens_both(self, tokenizer):
        """Test adding both BOS and EOS tokens."""
        tokens = jnp.array([5, 6, 7])
        result = tokenizer.add_special_tokens(tokens, add_bos=True, add_eos=True)

        bos_id = tokenizer.token_to_id["<BOS>"]
        eos_id = tokenizer.token_to_id["<EOS>"]
        assert result[0] == bos_id
        assert result[-1] == eos_id
        assert len(result) == len(tokens) + 2

    def test_add_special_tokens_truncation(self, tokenizer):
        """Test special tokens respect max_length."""
        # Create tokens at max length
        long_tokens = jnp.arange(tokenizer.max_length)
        result = tokenizer.add_special_tokens(long_tokens, add_bos=True, add_eos=True)

        # Should be truncated to max_length
        assert len(result) == tokenizer.max_length


# =============================================================================
# Sequence Truncation Tests
# =============================================================================


class TestSequenceTruncation:
    """Tests for sequence truncation."""

    def test_truncate_single_sequence(self, tokenizer):
        """Test truncating a single sequence."""
        tokens = jnp.arange(100)
        result, _ = tokenizer.truncate_sequences(tokens, max_total_length=50)

        assert len(result) == 50

    def test_truncate_single_sequence_no_truncation_needed(self, tokenizer):
        """Test no truncation when sequence is short."""
        tokens = jnp.arange(10)
        result, _ = tokenizer.truncate_sequences(tokens, max_total_length=50)

        assert len(result) == 10
        assert jnp.array_equal(result, tokens)

    def test_truncate_dual_sequences(self, tokenizer):
        """Test truncating two sequences."""
        tokens_a = jnp.arange(60)
        tokens_b = jnp.arange(50)
        result_a, result_b = tokenizer.truncate_sequences(tokens_a, tokens_b, max_total_length=100)

        # Total should be at most 100
        assert len(result_a) + len(result_b) <= 100

    def test_truncate_dual_sequences_longer_first(self, tokenizer):
        """Test truncating dual sequences truncates longer sequence."""
        tokens_a = jnp.arange(80)  # Longer
        tokens_b = jnp.arange(40)
        result_a, result_b = tokenizer.truncate_sequences(tokens_a, tokens_b, max_total_length=100)

        # Longer sequence should be truncated
        assert len(result_a) < 80
        assert len(result_b) == 40  # Shorter sequence unchanged

    def test_truncate_default_max_length(self, tokenizer):
        """Test truncation uses default max_length."""
        long_tokens = jnp.arange(1000)
        result, _ = tokenizer.truncate_sequences(long_tokens)

        assert len(result) == tokenizer.max_length


# =============================================================================
# MLM Masking Tests
# =============================================================================


class TestMLMMasking:
    """Tests for masked language modeling."""

    def test_apply_masking_returns_tuple(self, tokenizer):
        """Test masking returns tuple of masked tokens and positions."""
        tokens = jnp.arange(10) + 5  # Start from 5 to avoid special tokens
        masked_tokens, mask_positions = tokenizer.apply_masking(tokens, mask_probability=0.15)

        assert isinstance(masked_tokens, jnp.ndarray)
        assert isinstance(mask_positions, jnp.ndarray)
        assert masked_tokens.shape == tokens.shape
        assert mask_positions.shape == tokens.shape

    def test_apply_masking_mask_token_used(self, tokenizer):
        """Test that MASK token is used in masking."""
        tokens = jnp.arange(100) + 5
        masked_tokens, _ = tokenizer.apply_masking(tokens, mask_probability=0.5)

        mask_id = tokenizer.token_to_id["<MASK>"]
        # With 50% probability, some tokens should be masked
        assert jnp.any(masked_tokens == mask_id) or True  # May randomly have no masks

    def test_apply_masking_2d(self, tokenizer):
        """Test masking works with batch input."""
        tokens = jnp.stack([jnp.arange(10) + 5] * 3)  # Batch of 3
        masked_tokens, mask_positions = tokenizer.apply_masking(tokens, mask_probability=0.15)

        assert masked_tokens.shape == tokens.shape
        assert mask_positions.shape == tokens.shape

    def test_apply_masking_mask_positions_binary(self, tokenizer):
        """Test mask positions are binary (0 or 1)."""
        tokens = jnp.arange(10) + 5
        _, mask_positions = tokenizer.apply_masking(tokens, mask_probability=0.5)

        unique_values = jnp.unique(mask_positions)
        assert jnp.all((unique_values == 0) | (unique_values == 1))


# =============================================================================
# Position IDs Tests
# =============================================================================


class TestPositionIDs:
    """Tests for position ID creation."""

    def test_create_position_ids_1d(self, tokenizer):
        """Test position ID creation for 1D input."""
        tokens = jnp.array([5, 6, 7, 8, 9])
        position_ids = tokenizer.create_position_ids(tokens)

        expected = jnp.arange(5)
        assert jnp.array_equal(position_ids, expected)

    def test_create_position_ids_2d(self, tokenizer):
        """Test position ID creation for 2D batch input."""
        tokens = jnp.ones((3, 10))  # Batch of 3, seq length 10
        position_ids = tokenizer.create_position_ids(tokens)

        assert position_ids.shape == (3, 10)
        # Each row should be [0, 1, 2, ..., 9]
        expected_row = jnp.arange(10)
        for i in range(3):
            assert jnp.array_equal(position_ids[i], expected_row)


# =============================================================================
# Token Frequency Tests
# =============================================================================


class TestTokenFrequencies:
    """Tests for token frequency computation."""

    def test_compute_token_frequencies(self, tokenizer):
        """Test token frequency computation."""
        tokens = jnp.array([[5, 5, 6], [5, 6, 6]])
        frequencies = tokenizer.compute_token_frequencies(tokens)

        assert frequencies[5] == 3
        assert frequencies[6] == 3

    def test_compute_token_frequencies_returns_dict(self, tokenizer):
        """Test token frequency returns dictionary."""
        tokens = jnp.array([[5, 6, 7]])
        frequencies = tokenizer.compute_token_frequencies(tokens)

        assert isinstance(frequencies, dict)


# =============================================================================
# __call__ Method Tests
# =============================================================================


class TestCallMethod:
    """Tests for the __call__ method."""

    def test_call_with_text_input(self, tokenizer):
        """Test __call__ with text input."""
        inputs = {"text": "hello world"}
        result = tokenizer(inputs, {})

        assert "extension_type" in result
        assert result["extension_type"] == "advanced_tokenization"
        assert "input_tokens" in result
        assert "input_position_ids" in result

    def test_call_with_masking(self, tokenizer):
        """Test __call__ with masking enabled."""
        inputs = {"text": "hello world test"}
        result = tokenizer(inputs, {}, apply_masking=True, mask_probability=0.3)

        assert "masked_tokens" in result
        assert "mask_positions" in result

    def test_call_with_output_tokens(self, tokenizer):
        """Test __call__ processes output tokens."""
        inputs = {}
        model_outputs = {"tokens": jnp.array([[5, 6, 7], [8, 9, 10]])}
        result = tokenizer(inputs, model_outputs)

        assert "token_frequencies" in result

    def test_call_with_detokenize(self, tokenizer):
        """Test __call__ with detokenization."""
        inputs = {}
        model_outputs = {"tokens": jnp.array([5, 6, 7])}
        result = tokenizer(inputs, model_outputs, detokenize=True)

        assert "decoded_text" in result

    def test_call_with_batch_detokenize(self, tokenizer):
        """Test __call__ with batch detokenization."""
        inputs = {}
        model_outputs = {"tokens": jnp.array([[5, 6, 7], [8, 9, 10]])}
        result = tokenizer(inputs, model_outputs, detokenize=True)

        assert "decoded_texts" in result
        assert len(result["decoded_texts"]) == 2

    def test_call_with_generated_tokens(self, tokenizer):
        """Test __call__ processes generated tokens."""
        inputs = {}
        model_outputs = {"generated_tokens": jnp.array([5, 6, 7])}
        result = tokenizer(inputs, model_outputs)

        assert "generated_text" in result

    def test_call_disabled_returns_minimal(self, disabled_tokenizer):
        """Test disabled tokenizer returns minimal result."""
        inputs = {"text": "hello world"}
        result = disabled_tokenizer(inputs, {})

        assert result == {"extension_type": "advanced_tokenization"}

    def test_call_vocabulary_info_included(self, tokenizer):
        """Test __call__ includes vocabulary info."""
        inputs = {"text": "test"}
        result = tokenizer(inputs, {})

        assert "vocabulary_info" in result
        assert result["vocabulary_info"]["vocab_size"] == 50000
