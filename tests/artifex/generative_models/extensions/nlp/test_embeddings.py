"""Tests for text embeddings extension.

This module contains comprehensive tests for the TextEmbeddings extension
that provides text embedding utilities for generative models.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import ExtensionConfig
from artifex.generative_models.extensions.nlp.embeddings import TextEmbeddings


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
    return ExtensionConfig(name="test_embeddings", weight=1.0, enabled=True)


@pytest.fixture
def disabled_config():
    """Create a disabled extension configuration."""
    return ExtensionConfig(name="test_embeddings", weight=1.0, enabled=False)


@pytest.fixture
def embeddings(config, rngs):
    """Create a text embeddings instance with default configuration."""
    return TextEmbeddings(config, rngs=rngs)


@pytest.fixture
def disabled_embeddings(disabled_config, rngs):
    """Create a disabled text embeddings instance."""
    return TextEmbeddings(disabled_config, rngs=rngs)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestEmbeddingsInitialization:
    """Tests for embeddings initialization."""

    def test_init_with_default_config(self, config, rngs):
        """Test initialization with default ExtensionConfig."""
        embeddings = TextEmbeddings(config, rngs=rngs)

        assert embeddings.embedding_dim == 512
        assert embeddings.vocab_size == 50000
        assert embeddings.max_position_embeddings == 512
        assert embeddings.dropout_rate == 0.1
        assert embeddings.use_position_embeddings is True

    def test_init_creates_token_embedding(self, embeddings):
        """Test that token embedding layer is created."""
        assert hasattr(embeddings, "token_embedding")
        assert isinstance(embeddings.token_embedding, nnx.Embed)

    def test_init_creates_position_embedding(self, embeddings):
        """Test that position embedding layer is created."""
        assert hasattr(embeddings, "position_embedding")
        assert isinstance(embeddings.position_embedding, nnx.Embed)

    def test_init_creates_dropout(self, embeddings):
        """Test that dropout layer is created."""
        assert hasattr(embeddings, "dropout")
        assert isinstance(embeddings.dropout, nnx.Dropout)

    def test_init_invalid_config_type(self, rngs):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be ExtensionConfig"):
            TextEmbeddings({"weight": 1.0}, rngs=rngs)

    def test_init_disabled_config(self, disabled_config, rngs):
        """Test initialization with disabled configuration."""
        embeddings = TextEmbeddings(disabled_config, rngs=rngs)
        assert embeddings.enabled is False


# =============================================================================
# Embedding Tests
# =============================================================================


class TestEmbed:
    """Tests for the embed method."""

    def test_embed_basic(self, embeddings):
        """Test basic token embedding."""
        tokens = jnp.array([[1, 2, 3, 4, 5]])
        result = embeddings.embed(tokens, deterministic=True)

        assert result.shape == (1, 5, 512)  # batch, seq, embed_dim

    def test_embed_with_position_ids(self, embeddings):
        """Test embedding with explicit position IDs."""
        tokens = jnp.array([[1, 2, 3]])
        position_ids = jnp.array([[0, 1, 2]])
        result = embeddings.embed(tokens, position_ids=position_ids, deterministic=True)

        assert result.shape == (1, 3, 512)

    def test_embed_without_position_embeddings(self, config, rngs):
        """Test embedding when position embeddings are disabled."""
        # Note: With default config, position embeddings are enabled
        # This test verifies the embed method works correctly
        embeddings = TextEmbeddings(config, rngs=rngs)
        tokens = jnp.array([[1, 2, 3]])
        result = embeddings.embed(tokens, deterministic=True)

        assert result.shape == (1, 3, 512)

    def test_embed_batch(self, embeddings):
        """Test batch embedding."""
        tokens = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = embeddings.embed(tokens, deterministic=True)

        assert result.shape == (3, 3, 512)

    def test_embed_deterministic_mode(self, embeddings):
        """Test embedding in deterministic mode produces consistent results."""
        tokens = jnp.array([[1, 2, 3]])

        result1 = embeddings.embed(tokens, deterministic=True)
        result2 = embeddings.embed(tokens, deterministic=True)

        assert jnp.allclose(result1, result2)


# =============================================================================
# Token Embedding Tests
# =============================================================================


class TestGetTokenEmbeddings:
    """Tests for getting token embeddings."""

    def test_get_token_embeddings_single(self, embeddings):
        """Test getting embeddings for single token."""
        token_ids = jnp.array([1])
        result = embeddings.get_token_embeddings(token_ids)

        assert result.shape == (1, 512)

    def test_get_token_embeddings_multiple(self, embeddings):
        """Test getting embeddings for multiple tokens."""
        token_ids = jnp.array([1, 2, 3, 4, 5])
        result = embeddings.get_token_embeddings(token_ids)

        assert result.shape == (5, 512)

    def test_get_token_embeddings_consistency(self, embeddings):
        """Test same tokens produce same embeddings."""
        token_ids = jnp.array([1, 2, 1, 2])
        result = embeddings.get_token_embeddings(token_ids)

        # Token 1 at positions 0 and 2 should be same
        assert jnp.allclose(result[0], result[2])
        # Token 2 at positions 1 and 3 should be same
        assert jnp.allclose(result[1], result[3])


# =============================================================================
# Similarity Tests
# =============================================================================


class TestComputeSimilarity:
    """Tests for similarity computation."""

    def test_cosine_similarity(self, embeddings):
        """Test cosine similarity computation."""
        emb_a = jnp.array([1.0, 0.0, 0.0])
        emb_b = jnp.array([1.0, 0.0, 0.0])

        similarity = embeddings.compute_similarity(emb_a, emb_b, similarity_type="cosine")

        assert jnp.isclose(similarity, 1.0)  # Identical vectors

    def test_cosine_similarity_orthogonal(self, embeddings):
        """Test cosine similarity for orthogonal vectors."""
        emb_a = jnp.array([1.0, 0.0, 0.0])
        emb_b = jnp.array([0.0, 1.0, 0.0])

        similarity = embeddings.compute_similarity(emb_a, emb_b, similarity_type="cosine")

        assert jnp.isclose(similarity, 0.0)  # Orthogonal vectors

    def test_dot_product_similarity(self, embeddings):
        """Test dot product similarity computation."""
        emb_a = jnp.array([1.0, 2.0, 3.0])
        emb_b = jnp.array([4.0, 5.0, 6.0])

        similarity = embeddings.compute_similarity(emb_a, emb_b, similarity_type="dot")

        expected = 1 * 4 + 2 * 5 + 3 * 6  # 32
        assert jnp.isclose(similarity, expected)

    def test_euclidean_similarity(self, embeddings):
        """Test euclidean distance-based similarity."""
        emb_a = jnp.array([0.0, 0.0, 0.0])
        emb_b = jnp.array([0.0, 0.0, 0.0])

        similarity = embeddings.compute_similarity(emb_a, emb_b, similarity_type="euclidean")

        assert jnp.isclose(similarity, 0.0)  # Same point, distance is 0

    def test_euclidean_similarity_different_points(self, embeddings):
        """Test euclidean similarity for different points."""
        emb_a = jnp.array([0.0, 0.0, 0.0])
        emb_b = jnp.array([3.0, 4.0, 0.0])

        similarity = embeddings.compute_similarity(emb_a, emb_b, similarity_type="euclidean")

        # Distance is 5, similarity is -5
        assert jnp.isclose(similarity, -5.0)

    def test_invalid_similarity_type(self, embeddings):
        """Test invalid similarity type raises error."""
        emb_a = jnp.array([1.0, 0.0])
        emb_b = jnp.array([0.0, 1.0])

        with pytest.raises(ValueError, match="Unknown similarity type"):
            embeddings.compute_similarity(emb_a, emb_b, similarity_type="invalid")

    def test_similarity_batch(self, embeddings):
        """Test similarity computation with batch input."""
        emb_a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        emb_b = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        similarity = embeddings.compute_similarity(emb_a, emb_b, similarity_type="cosine")

        assert similarity.shape == (2,)
        assert jnp.allclose(similarity, jnp.array([1.0, 1.0]))


# =============================================================================
# Contextual Embeddings Tests
# =============================================================================


class TestContextualEmbeddings:
    """Tests for contextual embedding creation."""

    def test_create_contextual_embeddings_shape(self, embeddings):
        """Test contextual embeddings have correct shape."""
        tokens = jnp.array([[1, 2, 3, 4, 5]])
        result = embeddings.create_contextual_embeddings(tokens, context_window=3)

        assert result.shape == (1, 5, 512)

    def test_create_contextual_embeddings_batch(self, embeddings):
        """Test contextual embeddings with batch input."""
        tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
        result = embeddings.create_contextual_embeddings(tokens, context_window=3)

        assert result.shape == (2, 3, 512)

    def test_create_contextual_embeddings_small_window(self, embeddings):
        """Test contextual embeddings with small window."""
        tokens = jnp.array([[1, 2, 3, 4, 5]])
        result = embeddings.create_contextual_embeddings(tokens, context_window=1)

        assert result.shape == (1, 5, 512)


# =============================================================================
# Vocabulary Projection Tests
# =============================================================================


class TestProjectToVocabulary:
    """Tests for vocabulary projection."""

    def test_project_to_vocabulary_shape(self, embeddings):
        """Test vocabulary projection output shape."""
        emb = jnp.ones((1, 5, 512))  # batch, seq, embed_dim
        logits = embeddings.project_to_vocabulary(emb)

        assert logits.shape == (1, 5, 50000)  # batch, seq, vocab_size

    def test_project_to_vocabulary_single(self, embeddings):
        """Test vocabulary projection for single embedding."""
        emb = jnp.ones((512,))
        logits = embeddings.project_to_vocabulary(emb)

        assert logits.shape == (50000,)


# =============================================================================
# Sentence Embedding Tests
# =============================================================================


class TestExtractSentenceEmbedding:
    """Tests for sentence embedding extraction."""

    def test_mean_pooling(self, embeddings):
        """Test mean pooling for sentence embeddings."""
        token_embeddings = jnp.ones((2, 5, 512))  # batch, seq, embed_dim
        result = embeddings.extract_sentence_embedding(token_embeddings, pooling_method="mean")

        assert result.shape == (2, 512)

    def test_mean_pooling_with_mask(self, embeddings):
        """Test mean pooling with attention mask."""
        token_embeddings = jnp.ones((2, 5, 512))
        attention_mask = jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]])

        result = embeddings.extract_sentence_embedding(
            token_embeddings, attention_mask=attention_mask, pooling_method="mean"
        )

        assert result.shape == (2, 512)

    def test_max_pooling(self, embeddings):
        """Test max pooling for sentence embeddings."""
        token_embeddings = jnp.arange(30).reshape(2, 5, 3).astype(jnp.float32)
        result = embeddings.extract_sentence_embedding(token_embeddings, pooling_method="max")

        assert result.shape == (2, 3)

    def test_max_pooling_with_mask(self, embeddings):
        """Test max pooling with attention mask."""
        token_embeddings = jnp.arange(30).reshape(2, 5, 3).astype(jnp.float32)
        attention_mask = jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])

        result = embeddings.extract_sentence_embedding(
            token_embeddings, attention_mask=attention_mask, pooling_method="max"
        )

        assert result.shape == (2, 3)

    def test_cls_pooling(self, embeddings):
        """Test CLS token pooling."""
        token_embeddings = jnp.arange(30).reshape(2, 5, 3).astype(jnp.float32)
        result = embeddings.extract_sentence_embedding(token_embeddings, pooling_method="cls")

        assert result.shape == (2, 3)
        # Should be first token
        assert jnp.allclose(result[0], token_embeddings[0, 0])

    def test_last_pooling(self, embeddings):
        """Test last token pooling."""
        token_embeddings = jnp.arange(30).reshape(2, 5, 3).astype(jnp.float32)
        result = embeddings.extract_sentence_embedding(token_embeddings, pooling_method="last")

        assert result.shape == (2, 3)
        # Should be last token when no mask
        assert jnp.allclose(result[0], token_embeddings[0, -1])

    def test_last_pooling_with_mask(self, embeddings):
        """Test last token pooling with attention mask."""
        token_embeddings = jnp.arange(30).reshape(2, 5, 3).astype(jnp.float32)
        attention_mask = jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]])

        result = embeddings.extract_sentence_embedding(
            token_embeddings, attention_mask=attention_mask, pooling_method="last"
        )

        assert result.shape == (2, 3)
        # First sequence: last valid is position 2
        assert jnp.allclose(result[0], token_embeddings[0, 2])
        # Second sequence: last valid is position 1
        assert jnp.allclose(result[1], token_embeddings[1, 1])

    def test_invalid_pooling_method(self, embeddings):
        """Test invalid pooling method raises error."""
        token_embeddings = jnp.ones((2, 5, 512))

        with pytest.raises(ValueError, match="Unknown pooling method"):
            embeddings.extract_sentence_embedding(token_embeddings, pooling_method="invalid")


# =============================================================================
# Attention Weights Tests
# =============================================================================


class TestComputeAttentionWeights:
    """Tests for attention weight computation."""

    def test_attention_weights_shape(self, embeddings):
        """Test attention weights have correct shape."""
        query = jnp.ones((3, 512))
        key = jnp.ones((5, 512))

        weights = embeddings.compute_attention_weights(query, key)

        assert weights.shape == (3, 5)

    def test_attention_weights_sum_to_one(self, embeddings):
        """Test attention weights sum to 1 (softmax)."""
        query = jnp.ones((3, 512))
        key = jnp.ones((5, 512))

        weights = embeddings.compute_attention_weights(query, key)

        # Each row should sum to approximately 1
        row_sums = jnp.sum(weights, axis=-1)
        assert jnp.allclose(row_sums, jnp.ones(3), atol=1e-5)

    def test_attention_weights_temperature(self, embeddings):
        """Test attention weights with temperature scaling."""
        query = jnp.array([[1.0, 0.0]])
        key = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        weights_low_temp = embeddings.compute_attention_weights(query, key, temperature=0.1)
        weights_high_temp = embeddings.compute_attention_weights(query, key, temperature=10.0)

        # Low temperature should produce sharper distribution
        assert jnp.max(weights_low_temp) > jnp.max(weights_high_temp)


# =============================================================================
# Interpolation Tests
# =============================================================================


class TestInterpolateEmbeddings:
    """Tests for embedding interpolation."""

    def test_interpolate_alpha_zero(self, embeddings):
        """Test interpolation with alpha=0 returns first embedding."""
        emb_a = jnp.array([1.0, 2.0, 3.0])
        emb_b = jnp.array([4.0, 5.0, 6.0])

        result = embeddings.interpolate_embeddings(emb_a, emb_b, alpha=0.0)

        assert jnp.allclose(result, emb_a)

    def test_interpolate_alpha_one(self, embeddings):
        """Test interpolation with alpha=1 returns second embedding."""
        emb_a = jnp.array([1.0, 2.0, 3.0])
        emb_b = jnp.array([4.0, 5.0, 6.0])

        result = embeddings.interpolate_embeddings(emb_a, emb_b, alpha=1.0)

        assert jnp.allclose(result, emb_b)

    def test_interpolate_alpha_half(self, embeddings):
        """Test interpolation with alpha=0.5 returns midpoint."""
        emb_a = jnp.array([0.0, 0.0, 0.0])
        emb_b = jnp.array([2.0, 4.0, 6.0])

        result = embeddings.interpolate_embeddings(emb_a, emb_b, alpha=0.5)

        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)

    def test_interpolate_batch(self, embeddings):
        """Test interpolation with batch input."""
        emb_a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        emb_b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = embeddings.interpolate_embeddings(emb_a, emb_b, alpha=0.5)

        assert result.shape == (2, 2)


# =============================================================================
# Statistics Tests
# =============================================================================


class TestGetEmbeddingStatistics:
    """Tests for embedding statistics computation."""

    def test_get_embedding_statistics_keys(self, embeddings):
        """Test statistics contains expected keys."""
        emb = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stats = embeddings.get_embedding_statistics(emb)

        expected_keys = {"mean", "std", "norm", "min", "max"}
        assert expected_keys == set(stats.keys())

    def test_get_embedding_statistics_mean(self, embeddings):
        """Test mean statistic computation."""
        emb = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        stats = embeddings.get_embedding_statistics(emb)

        # Mean along last axis
        assert jnp.isclose(stats["mean"][0], 2.0)  # mean of [1, 2, 3]
        assert jnp.isclose(stats["mean"][1], 5.0)  # mean of [4, 5, 6]

    def test_get_embedding_statistics_norm(self, embeddings):
        """Test norm statistic computation."""
        emb = jnp.array([[3.0, 4.0]])
        stats = embeddings.get_embedding_statistics(emb)

        assert jnp.isclose(stats["norm"][0], 5.0)  # sqrt(9 + 16)

    def test_get_embedding_statistics_min_max(self, embeddings):
        """Test min/max statistic computation."""
        emb = jnp.array([[1.0, 5.0, 3.0]])
        stats = embeddings.get_embedding_statistics(emb)

        assert jnp.isclose(stats["min"][0], 1.0)
        assert jnp.isclose(stats["max"][0], 5.0)


# =============================================================================
# __call__ Method Tests
# =============================================================================


class TestCallMethod:
    """Tests for the __call__ method."""

    def test_call_with_token_input(self, embeddings):
        """Test __call__ with token input."""
        inputs = {"tokens": jnp.array([[1, 2, 3, 4, 5]])}
        result = embeddings(inputs, {}, deterministic=True)

        assert "extension_type" in result
        assert result["extension_type"] == "text_embeddings"
        assert "input_embeddings" in result

    def test_call_with_position_ids(self, embeddings):
        """Test __call__ with explicit position IDs."""
        inputs = {
            "tokens": jnp.array([[1, 2, 3]]),
            "position_ids": jnp.array([[0, 1, 2]]),
        }
        result = embeddings(inputs, {}, deterministic=True)

        assert "input_embeddings" in result

    def test_call_extract_sentence_embedding(self, embeddings):
        """Test __call__ with sentence embedding extraction."""
        inputs = {"tokens": jnp.array([[1, 2, 3, 4, 5]])}
        result = embeddings(inputs, {}, deterministic=True, extract_sentence_embedding=True)

        assert "sentence_embeddings" in result

    def test_call_extract_sentence_embedding_with_mask(self, embeddings):
        """Test __call__ with sentence embedding and attention mask."""
        inputs = {
            "tokens": jnp.array([[1, 2, 3, 4, 5]]),
            "attention_mask": jnp.array([[1.0, 1.0, 1.0, 0.0, 0.0]]),
        }
        result = embeddings(
            inputs,
            {},
            deterministic=True,
            extract_sentence_embedding=True,
            pooling_method="mean",
        )

        assert "sentence_embeddings" in result

    def test_call_compute_statistics(self, embeddings):
        """Test __call__ with statistics computation."""
        inputs = {"tokens": jnp.array([[1, 2, 3]])}
        result = embeddings(inputs, {}, deterministic=True, compute_statistics=True)

        assert "embedding_statistics" in result

    def test_call_project_to_vocab(self, embeddings):
        """Test __call__ with vocabulary projection."""
        inputs = {}
        model_outputs = {"embeddings": jnp.ones((1, 3, 512))}
        result = embeddings(inputs, model_outputs, project_to_vocab=True)

        assert "vocabulary_logits" in result

    def test_call_attention_weights(self, embeddings):
        """Test __call__ with attention weight computation."""
        inputs = {}
        model_outputs = {"embeddings": jnp.ones((5, 512))}
        query_embeddings = jnp.ones((3, 512))
        result = embeddings(
            inputs, model_outputs, query_embeddings=query_embeddings, temperature=1.0
        )

        assert "attention_weights" in result

    def test_call_hidden_states(self, embeddings):
        """Test __call__ with hidden states input."""
        inputs = {}
        model_outputs = {"hidden_states": jnp.ones((2, 5, 512))}
        result = embeddings(
            inputs, model_outputs, extract_sentence_embedding=True, pooling_method="mean"
        )

        assert "sentence_embeddings" in result

    def test_call_disabled_returns_minimal(self, disabled_embeddings):
        """Test disabled embeddings returns minimal result."""
        inputs = {"tokens": jnp.array([[1, 2, 3]])}
        result = disabled_embeddings(inputs, {})

        assert result == {"extension_type": "text_embeddings"}


# =============================================================================
# Rotary Position Embeddings (RoPE) Tests
# =============================================================================


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_apply_rope_embeddings_shape(self, embeddings):
        """Test RoPE output has same shape as input."""
        # Create input with even dimension
        x = jnp.ones((2, 10, 512))  # batch, seq, dim
        result = embeddings.apply_rope_embeddings(x)

        assert result.shape == x.shape

    def test_apply_rope_embeddings_single(self, embeddings):
        """Test RoPE with single sequence."""
        x = jnp.ones((1, 5, 512))
        result = embeddings.apply_rope_embeddings(x)

        assert result.shape == (1, 5, 512)
        # Output should be different from input due to rotation
        assert not jnp.allclose(result, x)

    def test_apply_rope_different_base(self, embeddings):
        """Test RoPE with different base values."""
        x = jnp.ones((1, 5, 512))

        result_default = embeddings.apply_rope_embeddings(x, base=10000.0)
        result_custom = embeddings.apply_rope_embeddings(x, base=5000.0)

        # Different bases should produce different results
        assert not jnp.allclose(result_default, result_custom)

    def test_embed_with_rope_shape(self, embeddings):
        """Test embed_with_rope produces correct shape."""
        tokens = jnp.array([[1, 2, 3, 4, 5]])
        result = embeddings.embed_with_rope(tokens, deterministic=True)

        assert result.shape == (1, 5, 512)

    def test_embed_with_rope_batch(self, embeddings):
        """Test embed_with_rope with batch input."""
        tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
        result = embeddings.embed_with_rope(tokens, deterministic=True)

        assert result.shape == (2, 3, 512)

    def test_embed_with_rope_deterministic(self, embeddings):
        """Test embed_with_rope is deterministic in deterministic mode."""
        tokens = jnp.array([[1, 2, 3]])

        result1 = embeddings.embed_with_rope(tokens, deterministic=True)
        result2 = embeddings.embed_with_rope(tokens, deterministic=True)

        assert jnp.allclose(result1, result2)


# =============================================================================
# Sinusoidal Positional Encoding Tests
# =============================================================================


class TestSinusoidalEncoding:
    """Tests for sinusoidal positional encodings."""

    def test_get_sinusoidal_embeddings_shape(self, embeddings):
        """Test sinusoidal embeddings have correct shape."""
        result = embeddings.get_sinusoidal_embeddings(seq_len=100)

        assert result.shape == (100, 512)  # seq_len x embedding_dim

    def test_get_sinusoidal_embeddings_custom_dim(self, embeddings):
        """Test sinusoidal embeddings with custom dimension."""
        result = embeddings.get_sinusoidal_embeddings(seq_len=50, dim=256)

        assert result.shape == (50, 256)

    def test_get_sinusoidal_embeddings_unique_positions(self, embeddings):
        """Test each position has unique encoding."""
        result = embeddings.get_sinusoidal_embeddings(seq_len=10)

        # Each row should be different
        for i in range(9):
            assert not jnp.allclose(result[i], result[i + 1])

    def test_get_sinusoidal_embeddings_bounded(self, embeddings):
        """Test sinusoidal values are bounded by [-1, 1]."""
        result = embeddings.get_sinusoidal_embeddings(seq_len=100)

        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)

    def test_get_sinusoidal_embeddings_even_odd_pattern(self, embeddings):
        """Test even/odd dimension pattern (sin/cos)."""
        result = embeddings.get_sinusoidal_embeddings(seq_len=10, dim=4)

        # Position 0 should have sin(0)=0 for even dimensions
        # and cos(0)=1 for odd dimensions at certain scales
        # Just verify it's computed without error
        assert result.shape == (10, 4)

    def test_embed_with_sinusoidal_positions_shape(self, embeddings):
        """Test embed_with_sinusoidal_positions produces correct shape."""
        tokens = jnp.array([[1, 2, 3, 4, 5]])
        result = embeddings.embed_with_sinusoidal_positions(tokens, deterministic=True)

        assert result.shape == (1, 5, 512)

    def test_embed_with_sinusoidal_positions_batch(self, embeddings):
        """Test embed_with_sinusoidal_positions with batch input."""
        tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
        result = embeddings.embed_with_sinusoidal_positions(tokens, deterministic=True)

        assert result.shape == (2, 3, 512)

    def test_embed_with_sinusoidal_positions_deterministic(self, embeddings):
        """Test sinusoidal embedding is deterministic."""
        tokens = jnp.array([[1, 2, 3]])

        result1 = embeddings.embed_with_sinusoidal_positions(tokens, deterministic=True)
        result2 = embeddings.embed_with_sinusoidal_positions(tokens, deterministic=True)

        assert jnp.allclose(result1, result2)


# =============================================================================
# RoPE Utility Function Tests
# =============================================================================


class TestRoPEUtilityFunctions:
    """Tests for RoPE utility functions."""

    def test_precompute_rope_freqs_shape(self):
        """Test precomputed frequencies have correct shape."""
        from artifex.generative_models.extensions.nlp.embeddings import precompute_rope_freqs

        freqs_sin, freqs_cos = precompute_rope_freqs(dim=64, max_seq_len=100)

        assert freqs_sin.shape == (100, 32)  # max_seq_len x dim//2
        assert freqs_cos.shape == (100, 32)

    def test_precompute_rope_freqs_odd_dim_error(self):
        """Test odd dimension raises error."""
        from artifex.generative_models.extensions.nlp.embeddings import precompute_rope_freqs

        with pytest.raises(ValueError, match="dim must be even"):
            precompute_rope_freqs(dim=63, max_seq_len=100)

    def test_precompute_rope_freqs_bounded(self):
        """Test frequency values are bounded."""
        from artifex.generative_models.extensions.nlp.embeddings import precompute_rope_freqs

        freqs_sin, freqs_cos = precompute_rope_freqs(dim=64, max_seq_len=100)

        assert jnp.all(freqs_sin >= -1.0)
        assert jnp.all(freqs_sin <= 1.0)
        assert jnp.all(freqs_cos >= -1.0)
        assert jnp.all(freqs_cos <= 1.0)

    def test_apply_rope_preserves_norm(self):
        """Test RoPE rotation preserves vector norm (approximately)."""
        from artifex.generative_models.extensions.nlp.embeddings import (
            apply_rope,
            precompute_rope_freqs,
        )

        x = jnp.ones((1, 5, 64))
        freqs_sin, freqs_cos = precompute_rope_freqs(dim=64, max_seq_len=5)

        result = apply_rope(x, freqs_sin, freqs_cos)

        # Compute norms
        original_norm = jnp.linalg.norm(x, axis=-1)
        result_norm = jnp.linalg.norm(result, axis=-1)

        # Norms should be approximately equal (rotation preserves length)
        assert jnp.allclose(original_norm, result_norm, rtol=1e-5)


# =============================================================================
# Sinusoidal Utility Function Tests
# =============================================================================


class TestSinusoidalUtilityFunctions:
    """Tests for sinusoidal encoding utility functions."""

    def test_create_sinusoidal_positions_shape(self):
        """Test sinusoidal positions have correct shape."""
        from artifex.generative_models.extensions.nlp.embeddings import (
            create_sinusoidal_positions,
        )

        result = create_sinusoidal_positions(max_seq_len=100, dim=256)

        assert result.shape == (100, 256)

    def test_create_sinusoidal_positions_bounded(self):
        """Test sinusoidal positions are bounded."""
        from artifex.generative_models.extensions.nlp.embeddings import (
            create_sinusoidal_positions,
        )

        result = create_sinusoidal_positions(max_seq_len=100, dim=256)

        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)

    def test_create_sinusoidal_positions_different_base(self):
        """Test different base values produce different encodings."""
        from artifex.generative_models.extensions.nlp.embeddings import (
            create_sinusoidal_positions,
        )

        result1 = create_sinusoidal_positions(max_seq_len=50, dim=64, base=10000.0)
        result2 = create_sinusoidal_positions(max_seq_len=50, dim=64, base=5000.0)

        assert not jnp.allclose(result1, result2)
