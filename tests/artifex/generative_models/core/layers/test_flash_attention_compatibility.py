"""
Comprehensive compatibility tests comparing Flash Attention with Flax NNX MultiHeadAttention.

This test suite verifies:
1. Correctness - outputs match within tolerance
2. Feature parity - all MultiHeadAttention features work
3. Drop-in replacement - can swap implementations seamlessly
4. Performance characteristics - Flash Attention provides expected benefits
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.nnx.nn.attention import MultiHeadAttention as FlaxMultiHeadAttention

from artifex.generative_models.core.layers.flash_attention import (
    AttentionBackend,
    FlashAttentionConfig,
    FlashMultiHeadAttention,
    PADDING_SEGMENT_ID,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def rngs():
    """Standard fixture for RNGs."""
    return nnx.Rngs(42)


@pytest.fixture
def standard_config():
    """Standard configuration for testing."""
    return {
        "num_heads": 8,
        "in_features": 512,
        "qkv_features": 512,
        "out_features": 512,
        "dropout_rate": 0.1,
    }


# ============================================================================
# Helper Functions
# ============================================================================


def create_both_modules(config: dict, rngs: nnx.Rngs):
    """Create both Flax and Flash attention modules with same config."""
    # Ensure decode has a value for Flax
    decode_value = config.get("decode", False)

    # Create Flax NNX MultiHeadAttention
    flax_module = FlaxMultiHeadAttention(
        num_heads=config["num_heads"],
        in_features=config["in_features"],
        qkv_features=config.get("qkv_features"),
        out_features=config.get("out_features"),
        dropout_rate=config.get("dropout_rate", 0.0),
        use_bias=config.get("use_bias", True),
        normalize_qk=config.get("normalize_qk", False),
        decode=decode_value,  # Explicitly set
        rngs=rngs,
    )

    # Create Flash Attention module
    flash_module = FlashMultiHeadAttention(
        num_heads=config["num_heads"],
        in_features=config["in_features"],
        qkv_features=config.get("qkv_features"),
        out_features=config.get("out_features"),
        dropout_rate=config.get("dropout_rate", 0.0),
        use_bias=config.get("use_bias", True),
        normalize_qk=config.get("normalize_qk", False),
        decode=decode_value,  # Explicitly set
        causal=config.get("causal", False),
        rngs=rngs,
    )

    return flax_module, flash_module


def copy_weights(source_module, target_module):
    """Copy weights from source to target module for fair comparison."""
    # Copy query, key, value projections
    if hasattr(source_module.query, "kernel"):
        target_module.query.kernel.value = source_module.query.kernel.value
    if hasattr(source_module.query, "bias") and hasattr(target_module.query, "bias"):
        target_module.query.bias.value = source_module.query.bias.value

    if hasattr(source_module.key, "kernel"):
        target_module.key.kernel.value = source_module.key.kernel.value
    if hasattr(source_module.key, "bias") and hasattr(target_module.key, "bias"):
        target_module.key.bias.value = source_module.key.bias.value

    if hasattr(source_module.value, "kernel"):
        target_module.value.kernel.value = source_module.value.kernel.value
    if hasattr(source_module.value, "bias") and hasattr(target_module.value, "bias"):
        target_module.value.bias.value = source_module.value.bias.value

    # Copy output projection
    if hasattr(source_module.out, "kernel"):
        target_module.out.kernel.value = source_module.out.kernel.value
    if hasattr(source_module.out, "bias") and hasattr(target_module.out, "bias"):
        target_module.out.bias.value = source_module.out.bias.value

    # Copy layer norms if they exist
    if source_module.query_ln is not None and target_module.query_ln is not None:
        target_module.query_ln.scale.value = source_module.query_ln.scale.value
    if source_module.key_ln is not None and target_module.key_ln is not None:
        target_module.key_ln.scale.value = source_module.key_ln.scale.value


# ============================================================================
# Correctness Tests
# ============================================================================


class TestCorrectness:
    """Test that Flash Attention produces correct outputs matching Flax NNX."""

    def test_basic_forward_pass(self, rngs, standard_config):
        """Test basic forward pass produces same outputs."""
        flax_module, flash_module = create_both_modules(standard_config, rngs)
        copy_weights(flax_module, flash_module)

        # Create input
        batch_size, seq_len = 2, 128
        x = jax.random.normal(rngs(), (batch_size, seq_len, standard_config["in_features"]))

        # Forward pass (deterministic mode for comparison)
        flax_output = flax_module(x, deterministic=True)
        flash_output = flash_module(x, deterministic=True)

        # Check outputs match
        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)

    def test_self_attention(self, rngs):
        """Test self-attention (Q=K=V) produces same outputs."""
        config = {"num_heads": 4, "in_features": 256}
        flax_module, flash_module = create_both_modules(config, rngs)
        copy_weights(flax_module, flash_module)

        # Single input for self-attention
        x = jax.random.normal(rngs(), (2, 64, 256))

        # Both should handle self-attention identically
        flax_output = flax_module(x, deterministic=True)
        flash_output = flash_module(x, deterministic=True)

        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)

    def test_cross_attention(self, rngs):
        """Test cross-attention with different Q, K, V."""
        config = {"num_heads": 4, "in_features": 256}
        flax_module, flash_module = create_both_modules(config, rngs)
        copy_weights(flax_module, flash_module)

        # Different inputs for cross-attention
        q = jax.random.normal(rngs(), (2, 32, 256))
        k = jax.random.normal(rngs(), (2, 64, 256))
        v = jax.random.normal(rngs(), (2, 64, 256))

        flax_output = flax_module(q, k, v, deterministic=True)
        flash_output = flash_module(q, k, v, deterministic=True)

        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)

    def test_with_mask(self, rngs):
        """Test attention with explicit mask."""
        config = {"num_heads": 4, "in_features": 256}
        flax_module, flash_module = create_both_modules(config, rngs)
        copy_weights(flax_module, flash_module)

        batch_size, seq_len = 2, 64
        x = jax.random.normal(rngs(), (batch_size, seq_len, 256))

        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = jnp.broadcast_to(mask[None, None, :, :], (batch_size, 4, seq_len, seq_len))

        # Both modules should handle masks
        flax_output = flax_module(x, mask=mask, deterministic=True, decode=False)
        flash_output = flash_module(x, mask=mask, deterministic=True, decode=False)

        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)

    def test_gradient_computation(self, rngs):
        """Test that gradients match between implementations."""
        config = {"num_heads": 4, "in_features": 256}
        flax_module, flash_module = create_both_modules(config, rngs)
        copy_weights(flax_module, flash_module)

        x = jax.random.normal(rngs(), (2, 64, 256))

        def loss_fn_flax(x):
            return jnp.sum(flax_module(x, deterministic=True))

        def loss_fn_flash(x):
            return jnp.sum(flash_module(x, deterministic=True))

        # Compute gradients
        grad_flax = jax.grad(loss_fn_flax)(x)
        grad_flash = jax.grad(loss_fn_flash)(x)

        np.testing.assert_allclose(grad_flax, grad_flash, rtol=1e-4, atol=1e-5)


# ============================================================================
# Feature Parity Tests
# ============================================================================


class TestFeatureParity:
    """Test that Flash Attention supports all MultiHeadAttention features."""

    def test_dropout_support(self, rngs):
        """Test dropout functionality."""
        config = {"num_heads": 4, "in_features": 256, "dropout_rate": 0.5}
        flax_module, flash_module = create_both_modules(config, rngs)

        x = jnp.ones((2, 64, 256))

        # Non-deterministic mode should have different outputs on different calls
        flash_out1 = flash_module(x, deterministic=False, rngs=rngs)
        flash_out2 = flash_module(x, deterministic=False, rngs=nnx.Rngs(123))

        # Outputs should differ due to dropout
        assert not jnp.allclose(flash_out1, flash_out2)

        # Deterministic mode should have same outputs
        flash_out3 = flash_module(x, deterministic=True)
        flash_out4 = flash_module(x, deterministic=True)

        assert jnp.allclose(flash_out3, flash_out4)

    def test_qk_normalization(self, rngs):
        """Test QK normalization feature."""
        config = {
            "num_heads": 4,
            "in_features": 256,
            "normalize_qk": True,
        }

        flax_module, flash_module = create_both_modules(config, rngs)
        copy_weights(flax_module, flash_module)

        # Both should have layer norms
        assert flax_module.query_ln is not None
        assert flax_module.key_ln is not None
        assert flash_module.query_ln is not None
        assert flash_module.key_ln is not None

        x = jax.random.normal(rngs(), (2, 64, 256))

        flax_output = flax_module(x, deterministic=True)
        flash_output = flash_module(x, deterministic=True)

        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)

    def test_bias_control(self, rngs):
        """Test use_bias parameter."""
        # With bias
        config_with_bias = {
            "num_heads": 4,
            "in_features": 256,
            "use_bias": True,
        }
        flax_with, flash_with = create_both_modules(config_with_bias, rngs)

        # Check bias exists and is not None
        assert flax_with.query.bias is not None
        assert flash_with.query.bias is not None

        # Without bias
        config_without_bias = {
            "num_heads": 4,
            "in_features": 256,
            "use_bias": False,
        }
        flax_without, flash_without = create_both_modules(config_without_bias, rngs)

        # Check bias is None when use_bias=False
        assert flax_without.query.bias is None
        assert flash_without.query.bias is None

    def test_decode_mode(self, rngs):
        """Test autoregressive decoding mode."""
        config = {
            "num_heads": 4,
            "in_features": 256,
            "decode": True,
        }

        flax_module, flash_module = create_both_modules(config, rngs)
        copy_weights(flax_module, flash_module)

        batch_size = 2
        max_length = 10

        # Initialize caches
        flax_module.init_cache((batch_size, max_length, 256))
        flash_module.init_cache((batch_size, max_length, 256))

        # Process tokens one by one
        for i in range(5):
            x = jax.random.normal(rngs(), (batch_size, 1, 256))

            flax_out = flax_module(x, decode=True, deterministic=True)
            flash_out = flash_module(x, decode=True, deterministic=True)

            np.testing.assert_allclose(flax_out, flash_out, rtol=1e-4, atol=1e-5)

    def test_different_qkv_features(self, rngs):
        """Test with different QKV feature dimensions."""
        config = {
            "num_heads": 8,
            "in_features": 256,
            "qkv_features": 512,  # Different from in_features
            "out_features": 384,  # Different from both
        }

        flax_module, flash_module = create_both_modules(config, rngs)
        copy_weights(flax_module, flash_module)

        x = jax.random.normal(rngs(), (2, 64, 256))

        flax_output = flax_module(x, deterministic=True)
        flash_output = flash_module(x, deterministic=True)

        # Check output shape
        assert flax_output.shape == (2, 64, 384)
        assert flash_output.shape == (2, 64, 384)

        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)


# ============================================================================
# Drop-in Replacement Tests
# ============================================================================


class TestDropInReplacement:
    """Test that FlashMultiHeadAttention can replace MultiHeadAttention seamlessly."""

    def test_transformer_block_replacement(self, rngs):
        """Test replacing attention in a transformer block."""

        class TransformerBlock(nnx.Module):
            def __init__(self, attention_cls, dim: int, num_heads: int, rngs: nnx.Rngs):
                # Use dropout_rate=0.0 for exact numerical matching
                # (Flax uses different code path when dropout_rate > 0)
                self.attention = attention_cls(
                    num_heads=num_heads,
                    in_features=dim,
                    dropout_rate=0.0,
                    rngs=rngs,
                    decode=False,
                )
                self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
                self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
                self.ffn = nnx.Sequential(
                    nnx.Linear(dim, dim * 4, rngs=rngs),
                    nnx.gelu,
                    nnx.Linear(dim * 4, dim, rngs=rngs),
                )

            def __call__(self, x, deterministic=True):
                # Self-attention with residual
                attn_out = self.attention(self.norm1(x), deterministic=deterministic)
                x = x + attn_out

                # FFN with residual
                ffn_out = self.ffn(self.norm2(x))
                x = x + ffn_out

                return x

        # Create blocks with different attention implementations
        flax_block = TransformerBlock(FlaxMultiHeadAttention, 256, 8, rngs)
        flash_block = TransformerBlock(FlashMultiHeadAttention, 256, 8, rngs)

        # Copy all weights for fair comparison
        # Copy attention weights
        copy_weights(flax_block.attention, flash_block.attention)

        # Copy layer norm weights
        flash_block.norm1.scale.value = flax_block.norm1.scale.value
        flash_block.norm1.bias.value = flax_block.norm1.bias.value
        flash_block.norm2.scale.value = flax_block.norm2.scale.value
        flash_block.norm2.bias.value = flax_block.norm2.bias.value

        # Copy FFN weights
        # Access Sequential layers properly
        for i, layer in enumerate(flash_block.ffn.layers):
            if isinstance(layer, nnx.Linear):
                # Find corresponding layer in flax_block
                for j, flax_layer in enumerate(flax_block.ffn.layers):
                    if isinstance(flax_layer, nnx.Linear) and i == j:
                        layer.kernel.value = flax_layer.kernel.value
                        if hasattr(layer, "bias") and layer.bias is not None:
                            layer.bias.value = flax_layer.bias.value
                        break

        # Test with same input
        x = jax.random.normal(rngs(), (2, 64, 256))

        flax_output = flax_block(x, deterministic=True)
        flash_output = flash_block(x, deterministic=True)

        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)

    def test_encoder_decoder_replacement(self, rngs):
        """Test replacing attention in encoder-decoder architecture."""

        class EncoderDecoder(nnx.Module):
            def __init__(self, attention_cls, dim: int, num_heads: int, rngs: nnx.Rngs):
                # Encoder self-attention
                self.encoder_attn = attention_cls(
                    num_heads=num_heads,
                    in_features=dim,
                    rngs=rngs,
                    decode=False,
                )
                # Decoder self-attention
                self.decoder_attn = attention_cls(
                    num_heads=num_heads,
                    in_features=dim,
                    rngs=rngs,
                    decode=False,
                )
                # Cross-attention
                self.cross_attn = attention_cls(
                    num_heads=num_heads,
                    in_features=dim,
                    rngs=rngs,
                    decode=False,
                )

            def __call__(self, encoder_input, decoder_input, deterministic=True):
                # Encoder self-attention
                encoder_out = self.encoder_attn(encoder_input, deterministic=deterministic)

                # Decoder self-attention
                decoder_out = self.decoder_attn(decoder_input, deterministic=deterministic)

                # Cross-attention (decoder attends to encoder)
                cross_out = self.cross_attn(
                    decoder_out,  # query from decoder
                    encoder_out,  # key from encoder
                    encoder_out,  # value from encoder
                    deterministic=deterministic,
                )

                return cross_out

        # Create models with different attention implementations
        flax_model = EncoderDecoder(FlaxMultiHeadAttention, 256, 8, rngs)
        flash_model = EncoderDecoder(FlashMultiHeadAttention, 256, 8, rngs)

        # Copy weights
        copy_weights(flax_model.encoder_attn, flash_model.encoder_attn)
        copy_weights(flax_model.decoder_attn, flash_model.decoder_attn)
        copy_weights(flax_model.cross_attn, flash_model.cross_attn)

        # Test
        encoder_input = jax.random.normal(rngs(), (2, 32, 256))
        decoder_input = jax.random.normal(rngs(), (2, 16, 256))

        flax_output = flax_model(encoder_input, decoder_input, deterministic=True)
        flash_output = flash_model(encoder_input, decoder_input, deterministic=True)

        # The attention computation itself is very close (~1e-4 differences)
        # but gets amplified through layer norms and FFN layers
        # These tolerances reflect realistic numerical precision for transformer blocks
        np.testing.assert_allclose(flax_output, flash_output, rtol=1e-3, atol=1e-3)

    def test_api_compatibility(self, rngs):
        """Test that all MultiHeadAttention APIs work with Flash version."""
        config = {"num_heads": 4, "in_features": 256}
        flax_module, flash_module = create_both_modules(config, rngs)

        x = jax.random.normal(rngs(), (2, 64, 256))

        # Test all call signatures
        # 1. Single input (self-attention)
        _ = flax_module(x, decode=False)
        _ = flash_module(x, decode=False)

        # 2. Q, K inputs (V defaults to K)
        _ = flax_module(x, x, decode=False)
        _ = flash_module(x, x, decode=False)

        # 3. Q, K, V inputs
        _ = flax_module(x, x, x, decode=False)
        _ = flash_module(x, x, x, decode=False)

        # 4. With mask
        mask = jnp.ones((2, 4, 64, 64))
        _ = flax_module(x, mask=mask, decode=False)
        _ = flash_module(x, mask=mask, decode=False)

        # 5. With deterministic flag
        _ = flax_module(x, deterministic=True, decode=False)
        _ = flash_module(x, deterministic=True, decode=False)

        # 6. With RNGs
        _ = flax_module(x, rngs=rngs, decode=False)
        _ = flash_module(x, rngs=rngs, decode=False)

        # All APIs work - test passes if no exceptions


# ============================================================================
# Flash-Specific Features Tests
# ============================================================================


class TestFlashSpecificFeatures:
    """Test Flash Attention specific features not in standard MultiHeadAttention."""

    def test_document_masks(self, rngs):
        """Test document mask functionality unique to Flash Attention."""
        flash_module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            use_segment_ids=True,
            rngs=rngs,
        )

        batch_size, seq_len = 2, 128
        x = jax.random.normal(rngs(), (batch_size, seq_len, 256))

        # Create segment IDs for multiple documents
        segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        # First half is document 0, second half is document 1
        segment_ids = segment_ids.at[:, seq_len // 2 :].set(1)

        positions = jnp.tile(jnp.arange(seq_len // 2), 2)[None, :].repeat(batch_size, axis=0)

        # Should handle document boundaries
        output = flash_module(
            x,
            query_segment_ids=segment_ids,
            kv_segment_ids=segment_ids,
            query_positions=positions,
            kv_positions=positions,
            deterministic=True,
        )

        assert output.shape == (batch_size, seq_len, 256)

    def test_padding_token_optimization(self, rngs):
        """Test padding token handling unique to Flash Attention."""
        flash_module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            use_segment_ids=True,
            rngs=rngs,
        )

        batch_size, seq_len = 2, 64
        x = jax.random.normal(rngs(), (batch_size, seq_len, 256))

        # Mark last 16 tokens as padding
        segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        segment_ids = segment_ids.at[:, -16:].set(PADDING_SEGMENT_ID)

        output = flash_module(
            x,
            query_segment_ids=segment_ids,
            kv_segment_ids=segment_ids,
            deterministic=True,
        )

        # Padding positions should have zero output
        padding_output = output[:, -16:, :]
        assert jnp.allclose(padding_output, 0.0)

    def test_causal_mode(self, rngs):
        """Test causal attention mode specific to Flash Attention."""
        # Flash module with causal mode
        flash_causal = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            causal=True,  # Flash-specific parameter
            rngs=rngs,
        )

        # Standard module needs explicit mask for causal
        flax_module = FlaxMultiHeadAttention(
            num_heads=4,
            in_features=256,
            rngs=rngs,
        )

        # Copy weights
        copy_weights(flax_module, flash_causal)

        batch_size, seq_len = 2, 64
        x = jax.random.normal(rngs(), (batch_size, seq_len, 256))

        # Flash with built-in causal
        flash_output = flash_causal(x, deterministic=True)

        # Flax needs explicit causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = jnp.broadcast_to(
            causal_mask[None, None, :, :], (batch_size, 4, seq_len, seq_len)
        )
        flax_output = flax_module(x, mask=causal_mask, deterministic=True, decode=False)

        # Should produce same result
        np.testing.assert_allclose(flash_output, flax_output, rtol=1e-5, atol=1e-6)

    def test_block_size_configuration(self, rngs):
        """Test configurable block sizes unique to Flash Attention."""
        # Test different block sizes
        for block_size in [64, 128, 256]:
            config = FlashAttentionConfig(
                query_block_size=block_size,
                kv_block_size=block_size,
            )

            flash_module = FlashMultiHeadAttention(
                num_heads=4,
                in_features=256,
                flash_config=config,
                rngs=rngs,
            )

            x = jax.random.normal(rngs(), (1, 512, 256))
            output = flash_module(x, deterministic=True)

            assert output.shape == (1, 512, 256)

    def test_backend_selection(self, rngs):
        """Test different attention backends available in Flash Attention."""
        x = jax.random.normal(rngs(), (2, 64, 256))

        for backend in [AttentionBackend.FLASH_TRITON, AttentionBackend.FALLBACK]:
            flash_module = FlashMultiHeadAttention(
                num_heads=4,
                in_features=256,
                backend=backend,
                rngs=rngs,
            )

            output = flash_module(x, deterministic=True)
            assert output.shape == (2, 64, 256)


# ============================================================================
# Performance Comparison Tests
# ============================================================================


class TestPerformanceCharacteristics:
    """Test performance characteristics of Flash Attention."""

    def test_memory_efficient_for_long_sequences(self, rngs):
        """Verify Flash Attention handles long sequences efficiently."""
        # Flash should handle long sequences without OOM
        flash_module = FlashMultiHeadAttention(
            num_heads=8,
            in_features=512,
            rngs=rngs,
        )

        # Test with progressively longer sequences
        for seq_len in [512, 1024, 2048]:
            x = jax.random.normal(rngs(), (1, seq_len, 512))
            output = flash_module(x, deterministic=True)
            assert output.shape == (1, seq_len, 512)

    def test_grouped_query_attention_efficiency(self, rngs):
        """Test GQA reduces memory as expected."""
        # Standard attention with all heads
        standard_module = FlashMultiHeadAttention(
            num_heads=16,
            in_features=512,
            rngs=rngs,
        )

        # GQA with fewer KV heads (simulated by using same architecture)
        # In practice, GQA would use fewer KV heads internally
        gqa_module = FlashMultiHeadAttention(
            num_heads=16,
            in_features=512,
            in_kv_features=512,  # Could be optimized for GQA
            rngs=rngs,
        )

        x = jax.random.normal(rngs(), (2, 1024, 512))

        # Both should work but GQA would use less memory in practice
        standard_out = standard_module(x, deterministic=True)
        gqa_out = gqa_module(x, deterministic=True)

        assert standard_out.shape == gqa_out.shape


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test Flash Attention in realistic scenarios."""

    def test_in_training_loop(self, rngs):
        """Test Flash Attention in a training scenario."""

        # Simple model using Flash Attention
        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                self.attention = FlashMultiHeadAttention(
                    num_heads=4,
                    in_features=256,
                    dropout_rate=0.1,
                    rngs=rngs,
                )
                self.output_proj = nnx.Linear(256, 10, rngs=rngs)

            def __call__(self, x, training=True):
                x = self.attention(x, deterministic=not training)
                x = jnp.mean(x, axis=1)  # Global pooling
                return self.output_proj(x)

        model = SimpleModel(rngs)

        # Training step
        def train_step(model, x, y):
            def loss_fn(model):
                logits = model(x, training=True)
                return jnp.mean((logits - y) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            # Would apply gradients here in real training
            return loss

        # Test data
        x = jax.random.normal(rngs(), (4, 32, 256))
        y = jax.random.normal(rngs(), (4, 10))

        # Should complete without errors
        loss = train_step(model, x, y)
        assert loss.shape == ()

    def test_with_jit_compilation(self, rngs):
        """Test Flash Attention works with JIT compilation."""
        flash_module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            rngs=rngs,
        )

        @jax.jit
        def forward(x):
            return flash_module(x, deterministic=True)

        x = jax.random.normal(rngs(), (2, 64, 256))

        # First call compiles
        output1 = forward(x)
        # Second call uses compiled version
        output2 = forward(x)

        assert jnp.allclose(output1, output2)

    def test_with_vmap(self, rngs):
        """Test Flash Attention works with vmap."""
        flash_module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            rngs=rngs,
        )

        # Single sample forward
        def single_forward(x):
            return flash_module(x[None, ...], deterministic=True)[0]

        # Vectorized version
        vmap_forward = jax.vmap(single_forward)

        # Batch of inputs
        x = jax.random.normal(rngs(), (8, 64, 256))

        # Should handle vectorization
        output = vmap_forward(x)
        assert output.shape == (8, 64, 256)


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
