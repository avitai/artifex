"""Tests for transformer implementations.

These tests verify the correctness of the transformer components including:
- Feed Forward Networks
- Attention Masks
- Encoder Blocks
- Decoder Blocks
- Full Encoder/Decoder architectures
- API features like QK normalization and broadcast dropout
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.layers.transformers import (
    create_attention_mask,
    create_transformer,
    FeedForwardNetwork,
    TransformerDecoder,
    TransformerDecoderBlock,
    TransformerEncoder,
    TransformerEncoderBlock,
)


@pytest.fixture
def rng_keys():
    """Fixture providing random keys for tests."""
    main_key = jax.random.key(42)
    keys = jax.random.split(main_key, 4)
    return {
        "params": keys[0],
        "dropout": keys[1],
        "attention": keys[2],
        "extra": keys[3],
    }


class TestFeedForwardNetwork:
    """Tests for the FeedForwardNetwork class."""

    def test_init_default(self, rng_keys):
        """Test initialization with default parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with default values
        ffn = FeedForwardNetwork(in_features=64, rngs=rngs)
        assert ffn.in_features == 64
        assert ffn.hidden_features == 64 * 4  # Default 4x multiplier
        assert ffn.out_features == 64
        assert ffn.dropout_rate == 0.0
        assert ffn.dropout is None  # No dropout by default

    def test_init_custom(self, rng_keys):
        """Test initialization with custom parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Test with custom values including dropout
        ffn = FeedForwardNetwork(
            in_features=32,
            hidden_features=128,
            out_features=16,
            dropout_rate=0.1,
            use_bias=False,
            rngs=rngs,
        )
        assert ffn.in_features == 32
        assert ffn.hidden_features == 128
        assert ffn.out_features == 16
        assert ffn.dropout_rate == 0.1
        assert ffn.dropout is not None

    def test_forward_basic(self, rng_keys):
        """Test basic forward pass."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create a module with no dropout for deterministic testing
        ffn = FeedForwardNetwork(in_features=4, hidden_features=8, out_features=4, rngs=rngs)

        # Create a sample input
        x = jnp.ones((2, 4))

        # Test forward pass
        y = ffn(x, deterministic=True)

        # Check shape
        assert y.shape == (2, 4)

        # The output should be deterministic given the same input
        y2 = ffn(x, deterministic=True)
        assert jnp.allclose(y, y2)

    def test_forward_with_rngs(self, rng_keys):
        """Test forward pass with explicit rngs parameter."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Create a module with dropout
        ffn = FeedForwardNetwork(in_features=4, dropout_rate=0.1, rngs=rngs)

        # Create a sample input
        x = jnp.ones((2, 4))

        # Test forward pass with explicit rngs
        y = ffn(x, deterministic=False, rngs=rngs)
        assert y.shape == (2, 4)

    def test_dropout_behavior(self, rng_keys):
        """Test dropout functionality."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Create a module with significant dropout for visible effect
        ffn = FeedForwardNetwork(in_features=8, dropout_rate=0.5, rngs=rngs)

        # Create a sample input with varied values
        key = jax.random.key(123)
        x = jax.random.normal(key, (3, 8))

        # Test without dropout (deterministic=True)
        y_no_dropout = ffn(x, deterministic=True, rngs=rngs)

        # Test with dropout (deterministic=False)
        y_with_dropout = ffn(x, deterministic=False, rngs=rngs)

        # Verify shapes are the same
        assert y_no_dropout.shape == y_with_dropout.shape

        # The outputs should be different due to dropout (probabilistic test)
        # This should pass almost always with 50% dropout rate
        assert not jnp.allclose(y_no_dropout, y_with_dropout, atol=1e-6)

    def test_runtime_errors(self, rng_keys):
        """Test runtime error handling."""
        rngs_params_only = nnx.Rngs(params=rng_keys["params"])

        # Test that dropout requires proper rngs when dropout_rate > 0
        ffn = FeedForwardNetwork(in_features=4, dropout_rate=0.1, rngs=rngs_params_only)

        x = jnp.ones((2, 4))

        # Should work fine in deterministic mode even without dropout rngs
        y = ffn(x, deterministic=True)
        assert y.shape == (2, 4)

        # Test that normal operation works with no dropout
        ffn_no_dropout = FeedForwardNetwork(in_features=4, dropout_rate=0.0, rngs=rngs_params_only)
        y = ffn_no_dropout(x, deterministic=False)
        assert y.shape == (2, 4)

        # Test parameter validation during initialization
        with pytest.raises(TypeError):
            # Should fail due to missing required rngs parameter
            FeedForwardNetwork(in_features=4, dropout_rate=0.1)


class TestCreateAttentionMask:
    """Tests for the create_attention_mask function."""

    def test_2d_mask_encoder(self):
        """Test creating attention mask from 2D input for encoder."""
        # Create a 2D mask [batch, length]
        batch_size = 2
        seq_length = 3
        num_heads = 4

        # Test with a mask where all positions are valid (1)
        mask = jnp.ones((batch_size, seq_length))
        attention_mask = create_attention_mask(mask, num_heads, is_decoder=False)

        # Expected shape: [batch, num_heads, seq_length, seq_length]
        expected_shape = (batch_size, num_heads, seq_length, seq_length)
        assert attention_mask.shape == expected_shape

        # All values should be 1 (no masking) since input was all ones
        assert jnp.all(attention_mask == 1.0)

        # Test with a mask where some positions are masked (0)
        mask = jnp.array([[1, 1, 0], [1, 0, 0]])
        attention_mask = create_attention_mask(mask, num_heads, is_decoder=False)

        # Check specific positions for symmetric encoder mask
        # First batch: positions involving index 2 should be 0
        assert jnp.all(attention_mask[0, :, 0, 2] == 0)
        assert jnp.all(attention_mask[0, :, 1, 2] == 0)
        assert jnp.all(attention_mask[0, :, 2, 0] == 0)
        assert jnp.all(attention_mask[0, :, 2, 1] == 0)
        assert jnp.all(attention_mask[0, :, 2, 2] == 0)

        # Valid positions should be 1
        assert jnp.all(attention_mask[0, :, 0, 0] == 1)
        assert jnp.all(attention_mask[0, :, 0, 1] == 1)
        assert jnp.all(attention_mask[0, :, 1, 0] == 1)
        assert jnp.all(attention_mask[0, :, 1, 1] == 1)

    def test_2d_mask_decoder_causal(self):
        """Test creating attention mask from 2D input for decoder (causal)."""
        batch_size = 2
        seq_length = 3
        num_heads = 4

        # Test with a mask where all positions are valid (1)
        mask = jnp.ones((batch_size, seq_length))
        attention_mask = create_attention_mask(mask, num_heads, is_decoder=True)

        # Expected shape: [batch, num_heads, seq_length, seq_length]
        expected_shape = (batch_size, num_heads, seq_length, seq_length)
        assert attention_mask.shape == expected_shape

        # Check causal pattern (upper triangular should be 0)
        # For position (i,j), if j > i, attention_mask should be 0
        for i in range(seq_length):
            for j in range(seq_length):
                if j > i:
                    assert jnp.all(attention_mask[:, :, i, j] == 0)
                else:
                    assert jnp.all(attention_mask[:, :, i, j] == 1)

    def test_3d_mask_cross_attention(self):
        """Test creating attention mask from 3D input (cross-attention)."""
        batch_size = 2
        query_len = 3
        key_len = 4
        num_heads = 4

        # Create a 3D mask [batch, query_len, key_len]
        mask = jnp.ones((batch_size, query_len, key_len))
        attention_mask = create_attention_mask(mask, num_heads)

        # Expected shape: [batch, num_heads, query_len, key_len]
        expected_shape = (batch_size, num_heads, query_len, key_len)
        assert attention_mask.shape == expected_shape

        # All values should be 1 (no masking) since input was all ones
        assert jnp.all(attention_mask == 1.0)

        # Test with partial masking
        mask = jnp.array(
            [
                [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],  # First batch
                [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]],  # Second batch
            ]
        )
        attention_mask = create_attention_mask(mask, num_heads)

        # Check that masking is preserved
        assert jnp.all(attention_mask[0, :, 0, 2] == 0)
        assert jnp.all(attention_mask[0, :, 0, 3] == 0)
        assert jnp.all(attention_mask[0, :, 0, 0] == 1)
        assert jnp.all(attention_mask[0, :, 0, 1] == 1)

    def test_head_broadcasting(self):
        """Test that masks are properly broadcast to all attention heads."""
        batch_size = 1
        seq_length = 2
        num_heads = 8

        mask = jnp.array([[1, 0]])  # Simple 2D mask
        attention_mask = create_attention_mask(mask, num_heads, is_decoder=False)

        # Should have correct shape
        assert attention_mask.shape == (batch_size, num_heads, seq_length, seq_length)

        # All heads should have identical masks
        for h in range(num_heads):
            assert jnp.array_equal(attention_mask[0, h], attention_mask[0, 0])


class TestTransformerEncoderBlock:
    """Tests for the TransformerEncoderBlock class."""

    def test_init_default(self, rng_keys):
        """Test initialization with default parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with default values
        block = TransformerEncoderBlock(hidden_dim=64, num_heads=4, rngs=rngs)
        assert block.hidden_dim == 64
        assert block.num_heads == 4
        assert hasattr(block, "attention")
        assert hasattr(block, "mlp")
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")
        assert block.dropout is None  # No dropout by default

    def test_init_with_new_features(self, rng_keys):
        """Test initialization with new API features."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Test with new features
        block = TransformerEncoderBlock(
            hidden_dim=32,
            num_heads=2,
            mlp_ratio=2.0,
            dropout_rate=0.1,
            attention_dropout_rate=0.2,
            normalize_qk=True,  # New feature
            broadcast_dropout=False,  # New feature
            use_bias=False,
            rngs=rngs,
        )
        assert block.hidden_dim == 32
        assert block.num_heads == 2
        assert block.dropout_rate == 0.1
        assert block.dropout is not None

        # Check that attention layer was created with new parameters
        assert hasattr(block.attention, "normalize_qk")

    def test_forward_basic(self, rng_keys):
        """Test basic forward pass."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create a module
        block = TransformerEncoderBlock(hidden_dim=8, num_heads=2, rngs=rngs)

        # Create a sample input
        batch_size = 2
        seq_len = 3
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))

        # Test forward pass without mask
        y = block(x, deterministic=True)

        # Check shape
        assert y.shape == (batch_size, seq_len, hidden_dim)

        # Test forward pass with mask
        mask = jnp.ones((batch_size, seq_len))
        y = block(x, mask=mask, deterministic=True)

        # Check shape
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_forward_with_rngs(self, rng_keys):
        """Test forward pass with explicit rngs."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Create a module with dropout
        block = TransformerEncoderBlock(hidden_dim=8, num_heads=2, dropout_rate=0.1, rngs=rngs)

        # Create a sample input
        batch_size = 2
        seq_len = 3
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))

        # Test forward pass with explicit rngs
        y = block(x, deterministic=False, rngs=rngs)
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_dropout_behavior(self, rng_keys):
        """Test that dropout is applied correctly."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Create a module with significant dropout
        block = TransformerEncoderBlock(
            hidden_dim=8,
            num_heads=2,
            dropout_rate=0.5,
            attention_dropout_rate=0.3,
            rngs=rngs,
        )

        # Create varied input
        batch_size = 2
        seq_len = 3
        hidden_dim = 8
        key = jax.random.key(456)
        x = jax.random.normal(key, shape=(batch_size, seq_len, hidden_dim))

        # Test without dropout
        y_no_dropout = block(x, deterministic=True, rngs=rngs)

        # Test with dropout
        y_with_dropout = block(x, deterministic=False, rngs=rngs)

        # Verify shapes are the same
        assert y_no_dropout.shape == y_with_dropout.shape

        # The outputs should be different due to dropout
        assert not jnp.allclose(y_no_dropout, y_with_dropout, atol=1e-6)

    def test_attention_mask_processing(self, rng_keys):
        """Test that attention masks are processed correctly."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = TransformerEncoderBlock(hidden_dim=8, num_heads=2, rngs=rngs)

        batch_size = 2
        seq_len = 3
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))

        # Test with 2D mask
        mask_2d = jnp.array([[1, 1, 0], [1, 0, 0]])
        y = block(x, mask=mask_2d, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

        # Test with pre-computed 4D mask
        mask_4d = jnp.ones((batch_size, 2, seq_len, seq_len))  # 2 heads
        y = block(x, mask=mask_4d, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)


class TestTransformerDecoderBlock:
    """Tests for the TransformerDecoderBlock class."""

    def test_init_default(self, rng_keys):
        """Test initialization with default parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with default values
        block = TransformerDecoderBlock(hidden_dim=64, num_heads=4, rngs=rngs)
        assert block.hidden_dim == 64
        assert block.num_heads == 4
        assert hasattr(block, "self_attention")
        assert hasattr(block, "cross_attention")
        assert hasattr(block, "mlp")
        assert hasattr(block, "norm1")
        assert hasattr(block, "norm2")
        assert hasattr(block, "norm3")
        assert block.dropout is None

    def test_init_with_new_features(self, rng_keys):
        """Test initialization with new API features."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Test with new features
        block = TransformerDecoderBlock(
            hidden_dim=32,
            num_heads=2,
            mlp_ratio=3.0,
            dropout_rate=0.15,
            attention_dropout_rate=0.25,
            normalize_qk=True,
            broadcast_dropout=False,
            rngs=rngs,
        )
        assert block.hidden_dim == 32
        assert block.num_heads == 2
        assert block.dropout_rate == 0.15
        assert block.dropout is not None

    def test_forward_basic(self, rng_keys):
        """Test basic forward pass."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create a module
        block = TransformerDecoderBlock(hidden_dim=8, num_heads=2, rngs=rngs)

        # Create sample inputs
        batch_size = 2
        seq_len = 3
        enc_len = 4
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))
        encoder_output = jnp.ones((batch_size, enc_len, hidden_dim))

        # Test forward pass without masks
        y = block(x, encoder_output, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_forward_with_masks(self, rng_keys):
        """Test forward pass with various mask configurations."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = TransformerDecoderBlock(hidden_dim=8, num_heads=2, rngs=rngs)

        batch_size = 2
        seq_len = 3
        enc_len = 4
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))
        encoder_output = jnp.ones((batch_size, enc_len, hidden_dim))

        # Test with self-attention mask only
        self_mask = jnp.ones((batch_size, seq_len))
        y = block(x, encoder_output, self_attention_mask=self_mask, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

        # Test with cross-attention mask only
        cross_mask = jnp.ones((batch_size, seq_len, enc_len))
        y = block(x, encoder_output, cross_attention_mask=cross_mask, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

        # Test with both masks
        y = block(
            x,
            encoder_output,
            self_attention_mask=self_mask,
            cross_attention_mask=cross_mask,
            deterministic=True,
        )
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_decode_mode(self, rng_keys):
        """Test decoder block in decode mode."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = TransformerDecoderBlock(hidden_dim=8, num_heads=2, rngs=rngs)

        batch_size = 1
        seq_len = 1  # Single token for autoregressive decoding
        enc_len = 4
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))
        encoder_output = jnp.ones((batch_size, enc_len, hidden_dim))

        # Initialize cache for self-attention before using decode=True
        block.self_attention.init_cache(x.shape)

        # Test with decode=True
        y = block(x, encoder_output, deterministic=True, decode=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_dropout_behavior(self, rng_keys):
        """Test dropout behavior in decoder block."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Create a module with dropout
        block = TransformerDecoderBlock(
            hidden_dim=8,
            num_heads=2,
            dropout_rate=0.4,
            attention_dropout_rate=0.3,
            rngs=rngs,
        )

        # Create varied inputs
        batch_size = 2
        seq_len = 3
        enc_len = 4
        hidden_dim = 8

        key1, key2 = jax.random.split(jax.random.key(789))
        x = jax.random.normal(key1, shape=(batch_size, seq_len, hidden_dim))
        encoder_output = jax.random.normal(key2, shape=(batch_size, enc_len, hidden_dim))

        # Test without dropout
        y_no_dropout = block(x, encoder_output, deterministic=True, rngs=rngs)

        # Test with dropout
        y_with_dropout = block(x, encoder_output, deterministic=False, rngs=rngs)

        # Verify shapes are the same
        assert y_no_dropout.shape == y_with_dropout.shape

        # The outputs should be different due to dropout
        assert not jnp.allclose(y_no_dropout, y_with_dropout, atol=1e-6)


class TestTransformerEncoder:
    """Tests for the TransformerEncoder class."""

    def test_init_default(self, rng_keys):
        """Test initialization with default parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with default values
        encoder = TransformerEncoder(num_layers=2, hidden_dim=64, num_heads=4, rngs=rngs)
        assert encoder.num_layers == 2
        assert encoder.hidden_dim == 64
        assert encoder.num_heads == 4
        assert len(encoder.layers) == 2
        assert hasattr(encoder, "norm")
        assert encoder.pos_encoding_type == "sinusoidal"
        assert encoder.pos_encoding is not None

    def test_init_with_new_features(self, rng_keys):
        """Test initialization with new API features."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Test with new features
        encoder = TransformerEncoder(
            num_layers=3,
            hidden_dim=32,
            num_heads=2,
            mlp_ratio=2.0,
            dropout_rate=0.1,
            attention_dropout_rate=0.2,
            normalize_qk=True,
            broadcast_dropout=False,
            pos_encoding_type="learned",
            max_len=512,
            rngs=rngs,
        )
        assert encoder.num_layers == 3
        assert encoder.hidden_dim == 32
        assert encoder.num_heads == 2
        assert len(encoder.layers) == 3
        assert encoder.pos_encoding_type == "learned"

    def test_forward_basic(self, rng_keys):
        """Test basic forward pass."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create a module
        encoder = TransformerEncoder(
            num_layers=2, hidden_dim=8, num_heads=2, pos_encoding_type="sinusoidal", rngs=rngs
        )

        # Create a sample input
        batch_size = 2
        seq_len = 4
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))

        # Test forward pass without mask
        y = encoder(x, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

        # Test forward pass with mask
        mask = jnp.ones((batch_size, seq_len))
        y = encoder(x, mask=mask, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_positional_encodings(self, rng_keys):
        """Test different positional encoding types."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create a sample input
        batch_size = 2
        seq_len = 4
        hidden_dim = 8  # Even for rotary
        x = jnp.ones((batch_size, seq_len, hidden_dim))

        # Test with different positional encoding types
        pos_types = ["sinusoidal", "learned", "rotary", "none"]
        for pos_type in pos_types:
            encoder = TransformerEncoder(
                num_layers=1,
                hidden_dim=hidden_dim,
                num_heads=2,
                pos_encoding_type=pos_type,
                max_len=16,
                rngs=rngs,
            )

            # Forward pass should work
            y = encoder(x, deterministic=True)
            assert y.shape == (batch_size, seq_len, hidden_dim)

            # Check pos_encoding attribute
            if pos_type == "none":
                assert not hasattr(encoder, "pos_encoding")
            else:
                assert hasattr(encoder, "pos_encoding")

    def test_sequence_length_validation(self, rng_keys):
        """Test sequence length validation with positional encodings."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create encoder with small max_len
        encoder = TransformerEncoder(
            num_layers=1,
            hidden_dim=8,
            num_heads=2,
            pos_encoding_type="sinusoidal",
            max_len=3,  # Small max length
            rngs=rngs,
        )

        # Input that exceeds max_len
        batch_size = 1
        seq_len = 5  # > max_len
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))

        # Should raise ValueError due to sequence length validation
        with pytest.raises(ValueError, match="Sequence length.*exceeds maximum length"):
            encoder(x, deterministic=True)

    def test_layer_consistency(self, rng_keys):
        """Test that all layers have consistent parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        encoder = TransformerEncoder(
            num_layers=3,
            hidden_dim=16,
            num_heads=4,
            normalize_qk=True,
            broadcast_dropout=False,
            rngs=rngs,
        )

        # All layers should have the same configuration
        for layer in encoder.layers:
            assert layer.hidden_dim == 16
            assert layer.num_heads == 4


class TestTransformerDecoder:
    """Tests for the TransformerDecoder class."""

    def test_init_default(self, rng_keys):
        """Test initialization with default parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with default values
        decoder = TransformerDecoder(num_layers=2, hidden_dim=64, num_heads=4, rngs=rngs)
        assert decoder.num_layers == 2
        assert decoder.hidden_dim == 64
        assert decoder.num_heads == 4
        assert len(decoder.layers) == 2
        assert hasattr(decoder, "norm")
        assert decoder.pos_encoding_type == "sinusoidal"
        assert decoder.pos_encoding is not None

    def test_init_with_new_features(self, rng_keys):
        """Test initialization with new API features."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Test with new features
        decoder = TransformerDecoder(
            num_layers=3,
            hidden_dim=32,
            num_heads=2,
            mlp_ratio=3.0,
            dropout_rate=0.15,
            attention_dropout_rate=0.25,
            normalize_qk=True,
            broadcast_dropout=False,
            pos_encoding_type="learned",
            rngs=rngs,
        )
        assert decoder.num_layers == 3
        assert decoder.hidden_dim == 32
        assert decoder.num_heads == 2
        assert len(decoder.layers) == 3
        assert decoder.pos_encoding_type == "learned"

    def test_forward_basic(self, rng_keys):
        """Test basic forward pass."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create a module
        decoder = TransformerDecoder(
            num_layers=2, hidden_dim=8, num_heads=2, pos_encoding_type="sinusoidal", rngs=rngs
        )

        # Create sample inputs
        batch_size = 2
        seq_len = 3
        enc_len = 4
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))
        encoder_output = jnp.ones((batch_size, enc_len, hidden_dim))

        # Test forward pass without masks
        y = decoder(x, encoder_output, deterministic=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_forward_with_masks(self, rng_keys):
        """Test forward pass with masks."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        decoder = TransformerDecoder(num_layers=1, hidden_dim=8, num_heads=2, rngs=rngs)

        batch_size = 2
        seq_len = 3
        enc_len = 4
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))
        encoder_output = jnp.ones((batch_size, enc_len, hidden_dim))

        # Test with masks
        self_mask = jnp.ones((batch_size, seq_len))
        cross_mask = jnp.ones((batch_size, seq_len, enc_len))
        y = decoder(
            x,
            encoder_output,
            self_attention_mask=self_mask,
            cross_attention_mask=cross_mask,
            deterministic=True,
        )
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_decode_mode(self, rng_keys):
        """Test decoder in autoregressive decode mode."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        decoder = TransformerDecoder(num_layers=1, hidden_dim=8, num_heads=2, rngs=rngs)

        batch_size = 1
        seq_len = 1  # Single token for autoregressive decoding
        enc_len = 4
        hidden_dim = 8
        x = jnp.ones((batch_size, seq_len, hidden_dim))
        encoder_output = jnp.ones((batch_size, enc_len, hidden_dim))

        # Test decode mode
        y = decoder(x, encoder_output, deterministic=True, decode=True)
        assert y.shape == (batch_size, seq_len, hidden_dim)

    def test_positional_encodings(self, rng_keys):
        """Test different positional encoding types."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create sample inputs
        batch_size = 2
        seq_len = 3
        enc_len = 4
        hidden_dim = 8  # Even for rotary
        x = jnp.ones((batch_size, seq_len, hidden_dim))
        encoder_output = jnp.ones((batch_size, enc_len, hidden_dim))

        # Test with different positional encoding types
        pos_types = ["sinusoidal", "learned", "rotary", "none"]
        for pos_type in pos_types:
            decoder = TransformerDecoder(
                num_layers=1,
                hidden_dim=hidden_dim,
                num_heads=2,
                pos_encoding_type=pos_type,
                rngs=rngs,
            )

            # Forward pass should work
            y = decoder(x, encoder_output, deterministic=True)
            assert y.shape == (batch_size, seq_len, hidden_dim)


class TestCreateTransformer:
    """Tests for the create_transformer function."""

    def test_basic_creation(self, rng_keys):
        """Test basic transformer creation."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with default values
        encoder, decoder = create_transformer(
            num_encoder_layers=2, num_decoder_layers=2, hidden_dim=64, num_heads=4, rngs=rngs
        )

        # Check encoder
        assert isinstance(encoder, TransformerEncoder)
        assert encoder.num_layers == 2
        assert encoder.hidden_dim == 64
        assert encoder.num_heads == 4
        assert encoder.pos_encoding_type == "sinusoidal"

        # Check decoder
        assert isinstance(decoder, TransformerDecoder)
        assert decoder.num_layers == 2
        assert decoder.hidden_dim == 64
        assert decoder.num_heads == 4
        assert decoder.pos_encoding_type == "sinusoidal"

    def test_new_api_features(self, rng_keys):
        """Test creation with new API features."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Test with new features
        encoder, decoder = create_transformer(
            num_encoder_layers=3,
            num_decoder_layers=2,
            hidden_dim=32,
            num_heads=4,
            normalize_qk=True,
            broadcast_dropout=False,
            dropout_rate=0.1,
            attention_dropout_rate=0.15,
            rngs=rngs,
        )

        # Both should have the new features
        assert encoder.num_layers == 3
        assert decoder.num_layers == 2
        assert encoder.hidden_dim == 32
        assert decoder.hidden_dim == 32

    def test_different_pos_encodings(self, rng_keys):
        """Test creating transformer with different positional encodings."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with different position encodings for encoder and decoder
        encoder, decoder = create_transformer(
            num_encoder_layers=1,
            num_decoder_layers=1,
            hidden_dim=64,
            num_heads=4,
            use_different_encoder_decoder_pos=True,
            encoder_pos_encoding_type="sinusoidal",
            decoder_pos_encoding_type="learned",
            rngs=rngs,
        )

        # Check encoder and decoder have different pos encoding types
        assert encoder.pos_encoding_type == "sinusoidal"
        assert decoder.pos_encoding_type == "learned"

    def test_validation_errors(self, rng_keys):
        """Test validation and error handling."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test error when use_different_encoder_decoder_pos but missing types
        with pytest.raises(ValueError, match="encoder_pos_encoding_type.*must be provided"):
            create_transformer(
                num_encoder_layers=1,
                num_decoder_layers=1,
                hidden_dim=64,
                num_heads=4,
                use_different_encoder_decoder_pos=True,
                rngs=rngs,
            )

        # Test error with odd hidden_dim for rotary encoding
        with pytest.raises(ValueError, match="Hidden dimension must be even for rotary"):
            create_transformer(
                num_encoder_layers=1,
                num_decoder_layers=1,
                hidden_dim=63,  # Odd number
                num_heads=3,
                pos_encoding_type="rotary",
                rngs=rngs,
            )

    def test_end_to_end_forward(self, rng_keys):
        """Test end-to-end forward pass through created transformer."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Create transformer with small dimensions for testing
        encoder, decoder = create_transformer(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=8,
            num_heads=2,
            dropout_rate=0.1,
            rngs=rngs,
        )

        # Create sample inputs
        batch_size = 2
        src_len = 4
        tgt_len = 3
        hidden_dim = 8

        src = jnp.ones((batch_size, src_len, hidden_dim))
        tgt = jnp.ones((batch_size, tgt_len, hidden_dim))

        # Test encoder forward pass
        enc_output = encoder(src, deterministic=True, rngs=rngs)
        assert enc_output.shape == (batch_size, src_len, hidden_dim)

        # Test decoder forward pass with encoder output
        dec_output = decoder(tgt, enc_output, deterministic=True, rngs=rngs)
        assert dec_output.shape == (batch_size, tgt_len, hidden_dim)

        # Test with masks
        src_mask = jnp.ones((batch_size, src_len))
        tgt_self_mask = jnp.ones((batch_size, tgt_len))
        tgt_cross_mask = jnp.ones((batch_size, tgt_len, src_len))

        enc_output = encoder(src, mask=src_mask, deterministic=True, rngs=rngs)
        dec_output = decoder(
            tgt,
            enc_output,
            self_attention_mask=tgt_self_mask,
            cross_attention_mask=tgt_cross_mask,
            deterministic=True,
            rngs=rngs,
        )
        assert dec_output.shape == (batch_size, tgt_len, hidden_dim)

    def test_parameter_consistency(self, rng_keys):
        """Test that encoder and decoder have consistent parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        encoder, decoder = create_transformer(
            num_encoder_layers=2,
            num_decoder_layers=3,
            hidden_dim=32,
            num_heads=4,
            normalize_qk=True,
            broadcast_dropout=False,
            rngs=rngs,
        )

        # Both should have same basic parameters
        assert encoder.hidden_dim == decoder.hidden_dim == 32
        assert encoder.num_heads == decoder.num_heads == 4


class TestIntegration:
    """Integration tests for the complete transformer system."""

    def test_training_simulation(self, rng_keys):
        """Simulate a basic training step to ensure everything works together."""
        rngs = nnx.Rngs(params=rng_keys["params"], dropout=rng_keys["dropout"])

        # Create a complete transformer
        encoder, decoder = create_transformer(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_dim=16,
            num_heads=4,
            dropout_rate=0.1,
            attention_dropout_rate=0.05,
            normalize_qk=True,
            pos_encoding_type="learned",
            rngs=rngs,
        )

        # Create realistic inputs
        batch_size = 4
        src_len = 8
        tgt_len = 6
        hidden_dim = 16

        key1, key2 = jax.random.split(jax.random.key(12345))
        src = jax.random.normal(key1, (batch_size, src_len, hidden_dim))
        tgt = jax.random.normal(key2, (batch_size, tgt_len, hidden_dim))

        # Create masks with some padding
        src_mask = jnp.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],  # 6 real tokens
                [1, 1, 1, 1, 1, 0, 0, 0],  # 5 real tokens
                [1, 1, 1, 1, 1, 1, 1, 0],  # 7 real tokens
                [1, 1, 1, 1, 1, 1, 1, 1],  # 8 real tokens
            ]
        )

        tgt_mask = jnp.array(
            [
                [1, 1, 1, 1, 0, 0],  # 4 real tokens
                [1, 1, 1, 0, 0, 0],  # 3 real tokens
                [1, 1, 1, 1, 1, 0],  # 5 real tokens
                [1, 1, 1, 1, 1, 1],  # 6 real tokens
            ]
        )

        # Forward pass (training mode)
        enc_output = encoder(src, mask=src_mask, deterministic=False, rngs=rngs)
        dec_output = decoder(
            tgt, enc_output, self_attention_mask=tgt_mask, deterministic=False, rngs=rngs
        )

        # Check outputs
        assert enc_output.shape == (batch_size, src_len, hidden_dim)
        assert dec_output.shape == (batch_size, tgt_len, hidden_dim)

        # Forward pass (evaluation mode)
        enc_output_eval = encoder(src, mask=src_mask, deterministic=True, rngs=rngs)
        dec_output_eval = decoder(
            tgt, enc_output_eval, self_attention_mask=tgt_mask, deterministic=True, rngs=rngs
        )

        # Shapes should be the same
        assert enc_output_eval.shape == enc_output.shape
        assert dec_output_eval.shape == dec_output.shape

        # Outputs should be different due to dropout
        assert not jnp.allclose(enc_output, enc_output_eval, atol=1e-6)
        assert not jnp.allclose(dec_output, dec_output_eval, atol=1e-6)

    def test_autoregressive_generation(self, rng_keys):
        """Test autoregressive generation pattern."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create transformer optimized for generation
        encoder, decoder = create_transformer(
            num_encoder_layers=1,
            num_decoder_layers=1,
            hidden_dim=8,
            num_heads=2,
            pos_encoding_type="sinusoidal",
            rngs=rngs,
        )

        # Encode source
        batch_size = 1
        src_len = 4
        hidden_dim = 8
        src = jnp.ones((batch_size, src_len, hidden_dim))
        enc_output = encoder(src, deterministic=True)

        # Test without decode mode (simpler approach)
        max_decode_len = 3
        generated = []

        for step in range(max_decode_len):
            # Build current target sequence
            if step == 0:
                # Start with initial token
                tgt = jnp.ones((batch_size, 1, hidden_dim))
            else:
                # Concatenate all previously generated tokens
                tgt = jnp.concatenate(generated, axis=1)

            # Decode without using cache (simpler for testing)
            dec_output = decoder(
                tgt,
                enc_output,
                deterministic=True,
                decode=False,  # Don't use caching for this test
            )

            # Take the last token's output
            next_token = dec_output[:, -1:, :]
            generated.append(next_token)

        # Final generated sequence
        final_output = jnp.concatenate(generated, axis=1)
        assert final_output.shape == (batch_size, max_decode_len, hidden_dim)

        # Test a simpler decode mode scenario
        # Initialize cache and test single token processing
        single_token = jnp.ones((batch_size, 1, hidden_dim))

        # Initialize caches
        for layer in decoder.layers:
            if hasattr(layer.self_attention, "init_cache"):
                layer.self_attention.init_cache(single_token.shape)

        # Process single token with decode=True
        dec_output = decoder(single_token, enc_output, deterministic=True, decode=True)
        assert dec_output.shape == (batch_size, 1, hidden_dim)


if __name__ == "__main__":
    # Manual test runner for quick verification
    print("Running manual transformer tests...")

    # Create test fixture
    rng_keys = {
        "params": jax.random.key(42),
        "dropout": jax.random.key(43),
        "attention": jax.random.key(44),
        "extra": jax.random.key(45),
    }

    # Test basic components
    test_ffn = TestFeedForwardNetwork()
    test_ffn.test_init_default(rng_keys)
    test_ffn.test_forward_basic(rng_keys)
    print("✓ FeedForwardNetwork tests passed")

    # Test attention mask creation
    test_mask = TestCreateAttentionMask()
    test_mask.test_2d_mask_encoder()
    test_mask.test_2d_mask_decoder_causal()
    print("✓ Attention mask tests passed")

    # Test encoder block
    test_enc_block = TestTransformerEncoderBlock()
    test_enc_block.test_init_default(rng_keys)
    test_enc_block.test_forward_basic(rng_keys)
    print("✓ TransformerEncoderBlock tests passed")

    # Test decoder block
    test_dec_block = TestTransformerDecoderBlock()
    test_dec_block.test_init_default(rng_keys)
    test_dec_block.test_forward_basic(rng_keys)
    print("✓ TransformerDecoderBlock tests passed")

    # Test full encoder/decoder
    test_encoder = TestTransformerEncoder()
    test_encoder.test_init_default(rng_keys)
    test_encoder.test_forward_basic(rng_keys)
    print("✓ TransformerEncoder tests passed")

    test_decoder = TestTransformerDecoder()
    test_decoder.test_init_default(rng_keys)
    test_decoder.test_forward_basic(rng_keys)
    print("✓ TransformerDecoder tests passed")

    # Test transformer creation
    test_create = TestCreateTransformer()
    test_create.test_basic_creation(rng_keys)
    test_create.test_end_to_end_forward(rng_keys)
    print("✓ create_transformer tests passed")

    # Test integration
    test_integration = TestIntegration()
    test_integration.test_training_simulation(rng_keys)
    test_integration.test_autoregressive_generation(rng_keys)
    print("✓ Integration tests passed")

    print("\nAll manual tests completed successfully!")
    print("Run 'pytest' for the complete test suite with detailed output.")
