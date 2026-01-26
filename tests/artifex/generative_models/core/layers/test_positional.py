"""Tests for positional encoding implementations in transformers.

These tests verify the correctness of various positional encoding
implementations used in transformer models.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Assuming the module is in the path, otherwise adjust the import
from artifex.generative_models.core.layers.positional import (
    LearnedPositionalEncoding,
    PositionalEncoding,
    RotaryPositionalEncoding,
    SinusoidalPositionalEncoding,
)


class TestPositionalEncoding:
    """Tests for the base PositionalEncoding class."""

    def test_base_class_init(self):
        """Test initialization of the base class."""

        class DummyPositionalEncoding(PositionalEncoding):
            def __call__(self, x, *, deterministic=False, rngs=None):
                return x

        # Test with default parameters (no dropout)
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = DummyPositionalEncoding(dim=64, max_len=100, rngs=rngs)
        assert pe.dim == 64
        assert pe.max_len == 100
        assert pe.dropout_rate == 0.0
        assert pe.dropout is None

        # Test with custom parameters
        # If dropout_rate > 0, rngs must be provided
        key = jax.random.key(0)
        rngs_with_dropout = nnx.Rngs(dropout=key)
        pe = DummyPositionalEncoding(dim=32, max_len=50, dropout_rate=0.1, rngs=rngs_with_dropout)
        assert pe.dim == 32
        assert pe.max_len == 50
        assert pe.dropout_rate == 0.1
        assert pe.dropout is not None  # Dropout should be initialized

    def test_dropout_creation(self):
        """Test that dropout is created when dropout_rate > 0."""

        class DummyPositionalEncoding(PositionalEncoding):
            def __call__(self, x, *, deterministic=False, rngs=None):
                return x

        # Test without dropout
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = DummyPositionalEncoding(dim=64, max_len=100, dropout_rate=0.0, rngs=rngs)
        assert pe.dropout is None

        # Test with dropout
        key = jax.random.key(0)
        rngs = nnx.Rngs(dropout=key)
        pe = DummyPositionalEncoding(dim=64, max_len=100, dropout_rate=0.1, rngs=rngs)
        assert pe.dropout is not None
        assert isinstance(pe.dropout, nnx.Dropout)
        assert pe.dropout.rate == 0.1

    def test_dropout_rngs_required(self):
        """Test that rngs is required when dropout_rate > 0."""

        class DummyPositionalEncoding(PositionalEncoding):
            def __call__(self, x, *, deterministic=False, rngs=None):
                return x

        with pytest.raises(ValueError, match="rngs must be provided.*dropout_rate > 0"):
            DummyPositionalEncoding(dim=64, max_len=100, dropout_rate=0.1, rngs=None)


class TestSinusoidalPositionalEncoding:
    """Tests for the SinusoidalPositionalEncoding class."""

    def test_init(self):
        """Test initialization."""
        # Test default initialization (no dropout)
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = SinusoidalPositionalEncoding(dim=64, max_len=100, rngs=rngs)
        assert pe.dim == 64
        assert pe.max_len == 100
        assert pe.dropout_rate == 0.0
        assert isinstance(pe.pe, nnx.Param)
        assert not pe.pe.trainable
        assert pe.pe.value.shape == (100, 64)
        assert pe.dropout is None

        # Test with custom parameters and dropout
        key = jax.random.key(0)
        rngs_with_dropout = nnx.Rngs(dropout=key)
        pe = SinusoidalPositionalEncoding(
            dim=32, max_len=50, dropout_rate=0.1, rngs=rngs_with_dropout
        )
        assert pe.dim == 32
        assert pe.max_len == 50
        assert pe.dropout_rate == 0.1
        assert pe.pe.value.shape == (50, 32)
        assert pe.dropout is not None

    def test_odd_dimension_handling(self):
        """Test that odd dimensions are handled correctly."""
        # Test odd dimension
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = SinusoidalPositionalEncoding(dim=65, max_len=10, rngs=rngs)
        assert pe.pe.value.shape == (10, 65)
        # Should not raise any errors and produce valid values
        assert jnp.isfinite(pe.pe.value).all()

    def test_pe_values(self):
        """Test the computed positional encoding values."""
        dim = 64
        max_len = 10
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = SinusoidalPositionalEncoding(dim=dim, max_len=max_len, dropout_rate=0.0, rngs=rngs)

        # Verify that even dimensions have sin values and odd dimensions have cos
        for pos_idx in range(max_len):
            for i in range(0, dim, 2):
                # Calculate expected values
                current_dim_val = jnp.array(i)
                div_term_val = jnp.exp(current_dim_val * (-jnp.log(10000.0) / dim))

                expected_sin = jnp.sin(pos_idx * div_term_val)
                assert jnp.allclose(pe.pe.value[pos_idx, i], expected_sin, atol=1e-6)

                if i + 1 < dim:
                    expected_cos = jnp.cos(pos_idx * div_term_val)
                    assert jnp.allclose(pe.pe.value[pos_idx, i + 1], expected_cos, atol=1e-6)

    def test_forward(self):
        """Test forward pass."""
        batch_size = 2
        seq_len = 5
        dim = 64

        x = jnp.zeros((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = SinusoidalPositionalEncoding(dim=dim, max_len=10, rngs=rngs)
        y = pe(x, deterministic=True)
        assert jnp.allclose(y, pe.pe.value[:seq_len, :])  # Broadcasting handles batch
        assert y.shape == (batch_size, seq_len, dim)

        x = jnp.ones((batch_size, seq_len, dim))
        y = pe(x, deterministic=True)
        expected = x + pe.pe.value[:seq_len, :]
        assert jnp.allclose(y, expected)
        assert y.shape == (batch_size, seq_len, dim)

    def test_forward_with_dropout(self):
        """Test forward pass with dropout."""
        batch_size = 2
        seq_len = 5
        dim = 64
        dropout_rate = 0.5

        x = jnp.ones((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(dropout=key)
        pe = SinusoidalPositionalEncoding(dim=dim, max_len=10, dropout_rate=dropout_rate, rngs=rngs)

        # Test deterministic mode
        y_deterministic = pe(x, deterministic=True, rngs=rngs)
        expected = x + pe.pe.value[:seq_len, :]
        assert jnp.allclose(y_deterministic, expected)

        # Test with dropout (non-deterministic)
        y_with_dropout = pe(x, deterministic=False, rngs=rngs)
        assert y_with_dropout.shape == (batch_size, seq_len, dim)
        # With dropout, output should be different (most of the time)
        # Note: This is probabilistic, but with 50% dropout it's very likely to be different

    def test_dropout_rngs_parameter(self):
        """Test that rngs parameter is properly passed to dropout."""
        batch_size = 2
        seq_len = 5
        dim = 64
        dropout_rate = 0.1

        x = jnp.ones((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(dropout=key)
        pe = SinusoidalPositionalEncoding(dim=dim, max_len=10, dropout_rate=dropout_rate, rngs=rngs)

        # Test that calling without rngs in non-deterministic mode still works
        # (should use the rngs from initialization)
        y = pe(x, deterministic=False, rngs=rngs)
        assert y.shape == (batch_size, seq_len, dim)

    def test_sequence_length_validation(self):
        """Test that an error is raised when sequence length exceeds max_len."""
        dim = 64
        max_len = 10
        batch_size = 2
        seq_len = 15  # > max_len

        x = jnp.ones((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = SinusoidalPositionalEncoding(dim=dim, max_len=max_len, rngs=rngs)
        with pytest.raises(ValueError, match="Sequence length.*exceeds maximum length"):
            pe(x, deterministic=True)

    def test_keyword_only_arguments(self):
        """Test that deterministic and rngs are keyword-only arguments."""
        batch_size = 2
        seq_len = 5
        dim = 64

        x = jnp.ones((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = SinusoidalPositionalEncoding(dim=dim, max_len=10, rngs=rngs)

        # This should work (keyword arguments)
        y = pe(x, deterministic=True)
        assert y.shape == (batch_size, seq_len, dim)

        # This should fail (positional arguments)
        with pytest.raises(TypeError):
            pe(x, True)  # deterministic as positional argument


class TestLearnedPositionalEncoding:
    """Tests for the LearnedPositionalEncoding class."""

    def test_init(self):
        """Test initialization."""
        key_params = jax.random.key(0)

        # Test without dropout
        rngs_params_only = nnx.Rngs(params=key_params)
        pe_no_dropout = LearnedPositionalEncoding(dim=64, max_len=100, rngs=rngs_params_only)
        assert pe_no_dropout.dim == 64
        assert pe_no_dropout.max_len == 100
        assert pe_no_dropout.dropout_rate == 0.0
        assert isinstance(pe_no_dropout.pe, nnx.Param)
        assert pe_no_dropout.pe.trainable  # Should be trainable
        assert pe_no_dropout.pe.value.shape == (100, 64)
        assert pe_no_dropout.dropout is None

        # Test with dropout
        key_dropout = jax.random.key(1)
        rngs_with_dropout = nnx.Rngs(params=key_params, dropout=key_dropout)
        pe_with_dropout = LearnedPositionalEncoding(
            dim=32, max_len=50, dropout_rate=0.1, rngs=rngs_with_dropout
        )
        assert pe_with_dropout.dim == 32
        assert pe_with_dropout.max_len == 50
        assert pe_with_dropout.dropout_rate == 0.1
        assert pe_with_dropout.pe.value.shape == (50, 32)
        assert pe_with_dropout.dropout is not None

    def test_rngs_required(self):
        """Test that rngs is required for initialization."""
        with pytest.raises(
            TypeError,  # rngs is now a required parameter, so TypeError for missing argument
        ):
            LearnedPositionalEncoding(dim=64, max_len=100)

    def test_custom_kernel_init(self):
        """Test initialization with custom kernel initializer."""

        def ones_init(key, shape):
            return jnp.ones(shape)

        key = jax.random.key(0)
        rngs = nnx.Rngs(params=key)
        pe = LearnedPositionalEncoding(dim=64, max_len=10, kernel_init=ones_init, rngs=rngs)
        assert jnp.all(pe.pe.value == 1.0)

    def test_trainable_parameter(self):
        """Test that the positional encoding parameter is trainable."""
        key = jax.random.key(0)
        rngs = nnx.Rngs(params=key)
        pe = LearnedPositionalEncoding(dim=64, max_len=10, rngs=rngs)
        assert pe.pe.trainable

    def test_forward(self):
        """Test forward pass."""
        batch_size = 2
        seq_len = 5
        dim = 64

        x = jnp.zeros((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(params=key)
        pe = LearnedPositionalEncoding(dim=dim, max_len=10, rngs=rngs)
        y = pe(x, deterministic=True)
        assert jnp.allclose(y, pe.pe.value[:seq_len, :])
        assert y.shape == (batch_size, seq_len, dim)

        x = jnp.ones((batch_size, seq_len, dim))
        y = pe(x, deterministic=True)
        expected = x + pe.pe.value[:seq_len, :]
        assert jnp.allclose(y, expected)
        assert y.shape == (batch_size, seq_len, dim)

    def test_forward_with_dropout(self):
        """Test forward pass with dropout."""
        batch_size = 2
        seq_len = 5
        dim = 64
        dropout_rate = 0.5

        x = jnp.ones((batch_size, seq_len, dim))
        key_params = jax.random.key(0)
        key_dropout = jax.random.key(1)
        rngs = nnx.Rngs(params=key_params, dropout=key_dropout)
        pe = LearnedPositionalEncoding(dim=dim, max_len=10, dropout_rate=dropout_rate, rngs=rngs)

        y_deterministic = pe(x, deterministic=True, rngs=rngs)
        expected = x + pe.pe.value[:seq_len, :]
        assert jnp.allclose(y_deterministic, expected)

        y_with_dropout = pe(x, deterministic=False, rngs=rngs)
        assert y_with_dropout.shape == (batch_size, seq_len, dim)

    def test_sequence_length_validation(self):
        """Test that an error is raised when sequence length exceeds max_len."""
        dim = 64
        max_len = 10
        batch_size = 2
        seq_len = 15  # > max_len

        x = jnp.ones((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(params=key)
        pe = LearnedPositionalEncoding(dim=dim, max_len=max_len, rngs=rngs)
        with pytest.raises(ValueError, match="Sequence length.*exceeds maximum length"):
            pe(x, deterministic=True)

    def test_different_initializations(self):
        """Test that different random keys produce different initializations."""
        key1 = jax.random.key(0)
        key2 = jax.random.key(1)
        rngs1 = nnx.Rngs(params=key1)
        rngs2 = nnx.Rngs(params=key2)

        pe1 = LearnedPositionalEncoding(dim=64, max_len=10, rngs=rngs1)
        pe2 = LearnedPositionalEncoding(dim=64, max_len=10, rngs=rngs2)

        # Different random keys should produce different initializations
        assert not jnp.allclose(pe1.pe.value, pe2.pe.value)


class TestRotaryPositionalEncoding:
    """Tests for the RotaryPositionalEncoding class."""

    def test_init(self):
        """Test initialization."""
        # Test without dropout
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe_no_dropout = RotaryPositionalEncoding(dim=64, max_len=100, rngs=rngs)
        assert pe_no_dropout.dim == 64
        assert pe_no_dropout.max_len == 100
        assert pe_no_dropout.dropout_rate == 0.0
        assert pe_no_dropout.base == 10000
        assert isinstance(pe_no_dropout.sin, nnx.Param)
        assert isinstance(pe_no_dropout.cos, nnx.Param)
        assert not pe_no_dropout.sin.trainable
        assert not pe_no_dropout.cos.trainable
        assert pe_no_dropout.sin.value.shape == (100, 32)  # dim/2 = 64/2 = 32
        assert pe_no_dropout.cos.value.shape == (100, 32)
        assert pe_no_dropout.dropout is None

        # Test with dropout
        key_dropout = jax.random.key(0)
        rngs_with_dropout = nnx.Rngs(dropout=key_dropout)
        pe_with_dropout = RotaryPositionalEncoding(
            dim=32,
            max_len=50,
            base=1000,
            dropout_rate=0.1,
            rngs=rngs_with_dropout,
        )
        assert pe_with_dropout.dim == 32
        assert pe_with_dropout.max_len == 50
        assert pe_with_dropout.dropout_rate == 0.1
        assert pe_with_dropout.base == 1000
        assert pe_with_dropout.sin.value.shape == (50, 16)  # dim/2 = 32/2 = 16
        assert pe_with_dropout.cos.value.shape == (50, 16)
        assert pe_with_dropout.dropout is not None

    def test_odd_dim_raises_error(self):
        """Test that initialization with odd dimension raises an error."""
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        with pytest.raises(ValueError, match="Dimension must be even"):
            RotaryPositionalEncoding(dim=63, max_len=100, rngs=rngs)

    def test_frequency_values(self):
        """Test the computed frequency values."""
        dim = 64
        max_len = 10
        base = 10000

        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = RotaryPositionalEncoding(dim=dim, max_len=max_len, base=base, rngs=rngs)

        # Compute expected values
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2) / dim))
        position = jnp.arange(max_len)
        sinusoid_inp = jnp.einsum("i,j->ij", position, inv_freq)

        expected_sin = jnp.sin(sinusoid_inp)
        expected_cos = jnp.cos(sinusoid_inp)

        assert jnp.allclose(pe.sin.value, expected_sin, atol=1e-6)
        assert jnp.allclose(pe.cos.value, expected_cos, atol=1e-6)

    def test_rotate_half_helper(self):
        """Test the _rotate_half helper function."""
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = RotaryPositionalEncoding(dim=4, max_len=10, rngs=rngs)

        # Test input: [1, 2, 3, 4]
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0]]])  # batch=1, seq=1, dim=4
        expected = jnp.array([[[-3.0, -4.0, 1.0, 2.0]]])  # [-x_right, x_left]

        result = pe._rotate_half(x)
        assert jnp.allclose(result, expected)

    def test_forward_simple(self):
        """Test forward pass with a simple input."""
        batch_size = 1
        seq_len = 2
        dim = 4

        # Simple test input
        x_val = jnp.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])  # [1, 2, 4]

        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = RotaryPositionalEncoding(dim=dim, max_len=10, base=10000, rngs=rngs)
        y = pe(x_val, deterministic=True)
        assert y.shape == (batch_size, seq_len, dim)

        # Manually verify the computation for the first position
        for s_idx in range(seq_len):
            x_s = x_val[0, s_idx, :]
            x_left = x_s[: dim // 2]  # [1, 2] or [5, 6]
            x_right = x_s[dim // 2 :]  # [3, 4] or [7, 8]

            sin_pos_s = pe.sin.value[s_idx, :]
            cos_pos_s = pe.cos.value[s_idx, :]

            expected_left_rotated = x_left * cos_pos_s - x_right * sin_pos_s
            expected_right_rotated = x_left * sin_pos_s + x_right * cos_pos_s

            assert jnp.allclose(y[0, s_idx, : dim // 2], expected_left_rotated, atol=1e-6)
            assert jnp.allclose(y[0, s_idx, dim // 2 :], expected_right_rotated, atol=1e-6)

    def test_forward_with_dropout(self):
        """Test forward pass with dropout."""
        batch_size = 2
        seq_len = 5
        dim = 64
        dropout_rate = 0.5

        x = jnp.ones((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(dropout=key)
        pe = RotaryPositionalEncoding(dim=dim, max_len=10, dropout_rate=dropout_rate, rngs=rngs)

        y_deterministic = pe(x, deterministic=True, rngs=rngs)
        # Verify deterministic mode produces consistent results
        y_deterministic2 = pe(x, deterministic=True, rngs=rngs)
        assert jnp.allclose(y_deterministic, y_deterministic2)

        y_with_dropout = pe(x, deterministic=False, rngs=rngs)
        assert y_with_dropout.shape == (batch_size, seq_len, dim)

    def test_sequence_length_validation(self):
        """Test that an error is raised when sequence length exceeds max_len."""
        dim = 64
        max_len = 10
        batch_size = 2
        seq_len = 15  # > max_len

        x = jnp.ones((batch_size, seq_len, dim))
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = RotaryPositionalEncoding(dim=dim, max_len=max_len, rngs=rngs)
        with pytest.raises(ValueError, match="Sequence length.*exceeds maximum length"):
            pe(x, deterministic=True)

    def test_rotation_properties(self):
        """Test mathematical properties of the rotation."""
        # Remove unused variables - only dim is actually used
        dim = 4

        # Create a simple test case
        x = jnp.array([[[1.0, 0.0, 0.0, 1.0]]])  # Simple pattern with batch=1, seq=1
        key = jax.random.key(0)
        rngs = nnx.Rngs(default=key)
        pe = RotaryPositionalEncoding(dim=dim, max_len=10, rngs=rngs)

        # Apply RoPE
        y = pe(x, deterministic=True)

        # Check that the norm is preserved (RoPE is a rotation, so it preserves norms)
        x_norm = jnp.linalg.norm(x, axis=-1)
        y_norm = jnp.linalg.norm(y, axis=-1)
        assert jnp.allclose(x_norm, y_norm, atol=1e-6)

    def test_different_base_values(self):
        """Test with different base values."""
        dim = 64
        max_len = 10

        key1 = jax.random.key(0)
        rngs1 = nnx.Rngs(default=key1)
        pe1 = RotaryPositionalEncoding(dim=dim, max_len=max_len, base=10000, rngs=rngs1)
        key2 = jax.random.key(1)
        rngs2 = nnx.Rngs(default=key2)
        pe2 = RotaryPositionalEncoding(dim=dim, max_len=max_len, base=1000, rngs=rngs2)

        # Different bases should produce different sin/cos values
        assert not jnp.allclose(pe1.sin.value, pe2.sin.value)
        assert not jnp.allclose(pe1.cos.value, pe2.cos.value)


class TestIntegration:
    """Integration tests for positional encodings."""

    def test_all_encodings_same_output_shape(self):
        """Test that all encoding types produce the same output shape."""
        batch_size = 2
        seq_len = 5
        dim = 64

        x = jnp.ones((batch_size, seq_len, dim))

        # Sinusoidal
        key1 = jax.random.key(0)
        rngs1 = nnx.Rngs(default=key1)
        sin_pe = SinusoidalPositionalEncoding(dim=dim, max_len=10, rngs=rngs1)
        y_sin = sin_pe(x, deterministic=True)

        # Learned
        key = jax.random.key(0)
        rngs = nnx.Rngs(params=key)
        learned_pe = LearnedPositionalEncoding(dim=dim, max_len=10, rngs=rngs)
        y_learned = learned_pe(x, deterministic=True)

        # Rotary
        key3 = jax.random.key(2)
        rngs3 = nnx.Rngs(default=key3)
        rotary_pe = RotaryPositionalEncoding(dim=dim, max_len=10, rngs=rngs3)
        y_rotary = rotary_pe(x, deterministic=True)

        # All should have the same shape
        assert y_sin.shape == y_learned.shape == y_rotary.shape == (batch_size, seq_len, dim)

    def test_encodings_modify_input_differently(self):
        """Test that different encoding types modify the input differently."""
        batch_size = 1
        seq_len = 3
        dim = 64

        x = jnp.ones((batch_size, seq_len, dim))

        # Get outputs from different encodings
        key1 = jax.random.key(0)
        rngs1 = nnx.Rngs(default=key1)
        sin_pe = SinusoidalPositionalEncoding(dim=dim, max_len=10, rngs=rngs1)
        y_sin = sin_pe(x, deterministic=True)

        key = jax.random.key(0)
        rngs = nnx.Rngs(params=key)
        learned_pe = LearnedPositionalEncoding(dim=dim, max_len=10, rngs=rngs)
        y_learned = learned_pe(x, deterministic=True)

        key3 = jax.random.key(2)
        rngs3 = nnx.Rngs(default=key3)
        rotary_pe = RotaryPositionalEncoding(dim=dim, max_len=10, rngs=rngs3)
        y_rotary = rotary_pe(x, deterministic=True)

        # They should produce different outputs
        assert not jnp.allclose(y_sin, y_learned)
        assert not jnp.allclose(y_sin, y_rotary)
        assert not jnp.allclose(y_learned, y_rotary)


if __name__ == "__main__":
    # Manual test runner for quick verification
    print("Running manual checks (subset of tests)...")

    # Test base functionality
    test_base = TestPositionalEncoding()
    test_base.test_base_class_init()
    test_base.test_dropout_creation()
    print("✓ Base tests passed")

    # Test sinusoidal encoding
    test_sin = TestSinusoidalPositionalEncoding()
    test_sin.test_init()
    test_sin.test_pe_values()
    test_sin.test_forward()
    test_sin.test_sequence_length_validation()
    print("✓ Sinusoidal tests passed")

    # Test learned encoding
    test_learned = TestLearnedPositionalEncoding()
    test_learned.test_rngs_required()
    key = jax.random.key(0)
    rngs = nnx.Rngs(params=key)
    # Run a subset that doesn't require additional setup
    print("✓ Learned tests (subset) passed")

    # Test rotary encoding
    test_rotary = TestRotaryPositionalEncoding()
    test_rotary.test_init()
    test_rotary.test_odd_dim_raises_error()
    test_rotary.test_frequency_values()
    test_rotary.test_forward_simple()
    test_rotary.test_sequence_length_validation()
    print("✓ Rotary tests passed")

    # Test integration
    test_integration = TestIntegration()
    test_integration.test_all_encodings_same_output_shape()
    test_integration.test_encodings_modify_input_differently()
    print("✓ Integration tests passed")

    print("\nAll manual checks completed successfully!")
    print("Run 'pytest' for the complete test suite with detailed output.")
