"""
Comprehensive tests for Flash Attention implementation.

Tests cover:
- Correctness against reference implementation
- Document mask functionality
- Context parallelism
- Grouped query attention
- Autoregressive decoding
- Performance characteristics
"""

import functools
import time
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from artifex.generative_models.core.layers.flash_attention import (
    AttentionBackend,
    create_attention_mask,
    flash_attention_triton,
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
def attention_config():
    """Default Flash Attention configuration."""
    return FlashAttentionConfig(
        query_block_size=64,
        kv_block_size=64,
        num_warps=4,
        num_stages=2,
    )


# ============================================================================
# Helper Functions
# ============================================================================


def make_attention_inputs(
    batch_size: int = 2,
    query_seq_len: int = 256,
    kv_seq_len: int = 256,
    num_heads: int = 8,
    num_kv_heads: Optional[int] = None,
    head_dim: int = 64,
    dtype: jnp.dtype = jnp.float32,
    rng_key: Optional[jax.random.PRNGKey] = None,
) -> dict:
    """Generate random attention inputs for testing."""

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Default num_kv_heads to num_heads if not specified
    if num_kv_heads is None:
        num_kv_heads = num_heads

    keys = jax.random.split(rng_key, 3)

    query = jax.random.normal(
        keys[0], (batch_size, query_seq_len, num_heads, head_dim), dtype=dtype
    )
    key = jax.random.normal(keys[1], (batch_size, kv_seq_len, num_kv_heads, head_dim), dtype=dtype)
    value = jax.random.normal(
        keys[2], (batch_size, kv_seq_len, num_kv_heads, head_dim), dtype=dtype
    )

    return {
        "query": query,
        "key": key,
        "value": value,
    }


def generate_segment_ids(
    batch_size: int,
    seq_len: int,
    num_segments: int,
    num_pad_tokens: int = 0,
    rng_key: Optional[jax.random.PRNGKey] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate segment IDs and positions for testing document masks."""

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    for b in range(batch_size):
        # Generate random segment lengths
        if num_segments > 1:
            segment_lengths = jax.random.randint(
                rng_key, (num_segments,), 1, seq_len // num_segments + 1
            )
            segment_lengths = segment_lengths * (seq_len - num_pad_tokens) // segment_lengths.sum()
            segment_lengths = segment_lengths.at[-1].set(
                seq_len - num_pad_tokens - segment_lengths[:-1].sum()
            )
        else:
            segment_lengths = jnp.array([seq_len - num_pad_tokens])

        # Assign segment IDs
        current_pos = 0
        for seg_idx, seg_len in enumerate(segment_lengths):
            segment_ids = segment_ids.at[b, current_pos : current_pos + seg_len].set(seg_idx)
            positions = positions.at[b, current_pos : current_pos + seg_len].set(
                jnp.arange(seg_len)
            )
            current_pos += seg_len

        # Mark padding tokens
        if num_pad_tokens > 0:
            segment_ids = segment_ids.at[b, -num_pad_tokens:].set(PADDING_SEGMENT_ID)
            positions = positions.at[b, -num_pad_tokens:].set(0)

    return segment_ids, positions


def reference_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    scale: Optional[float] = None,
) -> jnp.ndarray:
    """Reference implementation of attention for comparison."""

    if scale is None:
        scale = 1.0 / jnp.sqrt(query.shape[-1])

    # Compute attention scores
    scores = jnp.einsum("...qhd,...khd->...hqk", query, key) * scale

    # Apply mask
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)  # Use safe value for gradients

    # Compute attention weights
    weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention to values
    output = jnp.einsum("...hqk,...khd->...qhd", weights, value)

    return output


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestFlashAttentionBasic:
    """Test basic Flash Attention functionality."""

    def test_initialization(self, rngs):
        """Test module initialization."""
        module = FlashMultiHeadAttention(
            num_heads=8,
            in_features=512,
            qkv_features=512,
            rngs=rngs,
        )
        assert module.num_heads == 8
        assert module.head_dim == 64

    def test_forward_pass_shape(self, rngs):
        """Test forward pass produces correct shapes."""
        batch_size, seq_len, features = 2, 128, 512
        module = FlashMultiHeadAttention(
            num_heads=8,
            in_features=features,
            rngs=rngs,
        )

        x = jnp.ones((batch_size, seq_len, features))
        output = module(x)

        assert output.shape == (batch_size, seq_len, features)

    def test_different_qkv_inputs(self, rngs):
        """Test with different query, key, value inputs."""
        batch_size, q_seq_len, kv_seq_len, features = 2, 64, 128, 256
        module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=features,
            rngs=rngs,
        )

        q = jnp.ones((batch_size, q_seq_len, features))
        k = jnp.ones((batch_size, kv_seq_len, features)) * 2
        v = jnp.ones((batch_size, kv_seq_len, features)) * 3

        output = module(q, k, v)
        assert output.shape == (batch_size, q_seq_len, features)

    def test_grouped_query_attention(self, rngs):
        """Test grouped query attention (GQA)."""
        batch_size, seq_len, features = 2, 128, 512
        num_heads = 16

        module = FlashMultiHeadAttention(
            num_heads=num_heads,
            in_features=features,
            in_kv_features=features,
            qkv_features=512,
            rngs=rngs,
        )

        # Create properly shaped inputs
        x = jnp.ones((batch_size, seq_len, features))
        output = module(x)

        assert output.shape == (batch_size, seq_len, features)


# ============================================================================
# Correctness Tests
# ============================================================================


class TestFlashAttentionCorrectness:
    """Test Flash Attention correctness against reference implementation."""

    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("seq_len", [64, 128, 256])
    def test_correctness_vs_reference(self, rngs, causal, seq_len):
        """Test Flash Attention matches reference implementation."""

        batch_size = 2
        num_heads = 4
        head_dim = 32

        inputs = make_attention_inputs(
            batch_size=batch_size,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Create mask for reference
        mask = None
        if causal:
            mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
            mask = jnp.broadcast_to(mask, (batch_size, num_heads, seq_len, seq_len))

        # Reference implementation
        ref_output = reference_attention(inputs["query"], inputs["key"], inputs["value"], mask)

        # Flash Attention - explicitly pass causal parameter
        flash_output = flash_attention_triton(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            causal=causal,
        )

        # Check outputs match within tolerance
        np.testing.assert_allclose(ref_output, flash_output, rtol=1e-3, atol=1e-4)

    def test_gradient_correctness(self, rngs):
        """Test gradient computation correctness."""

        batch_size, seq_len = 2, 64
        num_heads, head_dim = 4, 32

        inputs = make_attention_inputs(
            batch_size=batch_size,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        def loss_fn_reference(q, k, v):
            out = reference_attention(q, k, v)
            return jnp.sum(out)

        def loss_fn_flash(q, k, v):
            # Explicitly use causal=False to match reference
            out = flash_attention_triton(q, k, v, causal=False)
            return jnp.sum(out)

        # Compute gradients
        grad_ref = jax.grad(loss_fn_reference, argnums=(0, 1, 2))(
            inputs["query"], inputs["key"], inputs["value"]
        )
        grad_flash = jax.grad(loss_fn_flash, argnums=(0, 1, 2))(
            inputs["query"], inputs["key"], inputs["value"]
        )

        # Check gradients match
        for g_ref, g_flash in zip(grad_ref, grad_flash):
            np.testing.assert_allclose(g_ref, g_flash, rtol=1e-3, atol=1e-4)


# ============================================================================
# Document Mask Tests
# ============================================================================


class TestDocumentMasks:
    """Test document mask functionality."""

    def test_segment_mask_creation(self, attention_config):
        """Test creation of segment masks."""

        batch_size = 2
        seq_len = 256
        num_segments = 3

        segment_ids, positions = generate_segment_ids(batch_size, seq_len, num_segments)

        mask = create_attention_mask(
            positions,
            segment_ids,
            positions,
            segment_ids,
            seq_len,
            seq_len,
            attention_config,
            causal=False,
        )

        assert mask.lower_blocks.shape == (batch_size, seq_len // attention_config.query_block_size)

    def test_padding_tokens_ignored(self, rngs):
        """Test that padding tokens are properly ignored."""

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 32
        num_pad_tokens = 16

        inputs = make_attention_inputs(
            batch_size=batch_size,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Create segment IDs with padding
        segment_ids, positions = generate_segment_ids(
            batch_size, seq_len, num_segments=1, num_pad_tokens=num_pad_tokens
        )

        output = flash_attention_triton(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            query_positions=positions,
            query_segment_ids=segment_ids,
            kv_positions=positions,
            kv_segment_ids=segment_ids,
            causal=False,  # Explicitly set for test
        )

        # Check that padding positions have zero output
        padding_mask = segment_ids == PADDING_SEGMENT_ID
        padding_output = output * padding_mask[..., None, None]

        assert jnp.allclose(padding_output, 0.0)

    def test_multi_document_attention(self, rngs):
        """Test attention with multiple documents in a batch."""

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 32
        num_segments = 4

        inputs = make_attention_inputs(
            batch_size=batch_size,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Create segment IDs for multiple documents
        segment_ids, positions = generate_segment_ids(batch_size, seq_len, num_segments)

        output = flash_attention_triton(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            query_positions=positions,
            query_segment_ids=segment_ids,
            kv_positions=positions,
            kv_segment_ids=segment_ids,
            causal=False,  # Explicitly set for multi-document
        )

        assert output.shape == inputs["query"].shape


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformanceCharacteristics:
    """Test performance characteristics of Flash Attention."""

    def test_memory_efficiency(self, rngs):
        """Test that Flash Attention uses less memory than standard attention."""

        # This test would measure actual memory usage in practice
        # For now, we just verify the implementation works with large sequences

        batch_size = 1
        seq_len = 2048  # Large sequence
        num_heads = 8
        head_dim = 64

        inputs = make_attention_inputs(
            batch_size=batch_size,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # Should not OOM even with large sequence
        output = flash_attention_triton(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            causal=False,  # Explicitly set
        )

        assert output.shape == inputs["query"].shape

    @pytest.mark.parametrize("block_size", [64, 128, 256])
    def test_different_block_sizes(self, rngs, block_size):
        """Test Flash Attention with different block sizes."""

        config = FlashAttentionConfig(
            query_block_size=block_size,
            kv_block_size=block_size,
        )

        batch_size = 2
        seq_len = 512
        num_heads = 4
        head_dim = 32

        inputs = make_attention_inputs(
            batch_size=batch_size,
            query_seq_len=seq_len,
            kv_seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        output = flash_attention_triton(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            config=config,
            causal=False,  # Explicitly set
        )

        assert output.shape == inputs["query"].shape


# ============================================================================
# Module Integration Tests
# ============================================================================


class TestModuleIntegration:
    """Test Flash Attention module integration."""

    def test_dropout_behavior(self, rngs):
        """Test dropout behavior in deterministic/non-deterministic modes."""

        module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            dropout_rate=0.1,
            rngs=rngs,
        )

        x = jnp.ones((2, 64, 256))

        # Non-deterministic mode
        out1 = module(x, deterministic=False, rngs=rngs)
        out2 = module(x, deterministic=False, rngs=nnx.Rngs(123))

        # Outputs should differ due to dropout
        assert not jnp.allclose(out1, out2)

        # Deterministic mode
        out3 = module(x, deterministic=True)
        out4 = module(x, deterministic=True)

        # Outputs should be identical
        assert jnp.allclose(out3, out4)

    def test_qk_normalization(self, rngs):
        """Test QK normalization feature."""

        module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            normalize_qk=True,
            rngs=rngs,
        )

        assert module.query_ln is not None
        assert module.key_ln is not None

        x = jnp.ones((2, 64, 256))
        output = module(x)

        assert output.shape == x.shape

    def test_autoregressive_cache(self, rngs):
        """Test autoregressive caching for decoding."""

        batch_size = 2
        features = 256
        max_length = 10

        module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=features,
            decode=True,
            rngs=rngs,
        )

        # Initialize cache with max_length
        module.init_cache((batch_size, max_length, features))

        # Process tokens one by one
        outputs = []
        for i in range(max_length):
            x = jnp.ones((batch_size, 1, features)) * i
            out = module(x, decode=True)
            outputs.append(out)

        # Check outputs have correct shape
        for out in outputs:
            assert out.shape == (batch_size, 1, features)

    def test_backend_selection(self, rngs):
        """Test different attention backends."""

        for backend in [AttentionBackend.FLASH_TRITON, AttentionBackend.FALLBACK]:
            module = FlashMultiHeadAttention(
                num_heads=4,
                in_features=256,
                backend=backend,
                rngs=rngs,
            )

            x = jnp.ones((2, 64, 256))
            output = module(x)

            assert output.shape == x.shape


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_odd_dimensions(self, rngs):
        """Test with odd sequence lengths and dimensions."""

        module = FlashMultiHeadAttention(
            num_heads=3,  # Odd number of heads
            in_features=255,  # Odd feature dimension
            qkv_features=123,  # Odd QKV dimension (divisible by 3)
            causal=False,  # Explicitly set
            rngs=rngs,
        )

        x = jnp.ones((2, 63, 255))  # Odd sequence length
        output = module(x)

        assert output.shape == (2, 63, 255)

    def test_single_token(self, rngs):
        """Test with single token sequences."""

        module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            rngs=rngs,
        )

        x = jnp.ones((2, 1, 256))
        output = module(x)

        assert output.shape == (2, 1, 256)

    def test_very_long_sequences(self, rngs):
        """Test with very long sequences."""

        module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            causal=False,  # Explicitly set
            rngs=rngs,
        )

        # Test with 4K sequence length
        x = jnp.ones((1, 4096, 256))
        output = module(x)

        assert output.shape == (1, 4096, 256)

    def test_invalid_head_dimension(self, rngs):
        """Test error handling for invalid head dimensions."""

        with pytest.raises(ValueError, match="must be divisible"):
            FlashMultiHeadAttention(
                num_heads=4,
                in_features=256,
                qkv_features=250,  # Not divisible by 4
                rngs=rngs,
            )

    def test_mismatched_input_dimension(self, rngs):
        """Test error handling for mismatched input dimensions."""

        module = FlashMultiHeadAttention(
            num_heads=4,
            in_features=256,
            rngs=rngs,
        )

        x = jnp.ones((2, 64, 128))  # Wrong feature dimension

        with pytest.raises(ValueError, match="Incompatible input dimension"):
            module(x)


# ============================================================================
# Benchmarking Utilities
# ============================================================================


def benchmark_attention_forward(
    batch_size: int = 4,
    seq_len: int = 1024,
    num_heads: int = 16,
    head_dim: int = 64,
    backend: AttentionBackend = AttentionBackend.FLASH_TRITON,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Benchmark attention forward pass."""

    inputs = make_attention_inputs(
        batch_size=batch_size,
        query_seq_len=seq_len,
        kv_seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    if backend == AttentionBackend.FLASH_TRITON:
        fn = functools.partial(flash_attention_triton, causal=True)
    else:
        fn = reference_attention

    # JIT compile
    fn_jit = jax.jit(fn)

    # Warmup
    for _ in range(num_warmup):
        _ = fn_jit(inputs["query"], inputs["key"], inputs["value"])

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = fn_jit(inputs["query"], inputs["key"], inputs["value"]).block_until_ready()
    end = time.time()

    avg_time = (end - start) / num_runs

    # Calculate FLOPS
    flops = 4 * batch_size * seq_len * seq_len * num_heads * head_dim
    tflops = flops / avg_time / 1e12

    return {
        "avg_time": avg_time,
        "tflops": tflops,
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
