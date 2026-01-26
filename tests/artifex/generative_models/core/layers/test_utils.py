"""Tests for shared layer utility functions."""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.layers._utils import (
    apply_norm,
    create_norm_layer,
    normalize_size_param,
)


class TestNormalizeSizeParam:
    """Tests for normalize_size_param."""

    def test_int_to_1d(self) -> None:
        """Test integer expansion to 1D tuple."""
        assert normalize_size_param(3, 1, "kernel_size") == (3,)

    def test_int_to_2d(self) -> None:
        """Test integer expansion to 2D tuple."""
        assert normalize_size_param(5, 2, "kernel_size") == (5, 5)

    def test_int_to_3d(self) -> None:
        """Test integer expansion to 3D tuple."""
        assert normalize_size_param(7, 3, "stride") == (7, 7, 7)

    def test_tuple_passthrough_2d(self) -> None:
        """Test tuple of correct length passes through."""
        assert normalize_size_param((3, 5), 2, "dilation") == (3, 5)

    def test_list_passthrough_3d(self) -> None:
        """Test list of correct length passes through."""
        assert normalize_size_param([1, 2, 3], 3, "kernel_size") == (1, 2, 3)

    def test_wrong_length_raises(self) -> None:
        """Test that wrong-length sequence raises ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be"):
            normalize_size_param([1, 2, 3], 2, "kernel_size")

    def test_empty_sequence_raises(self) -> None:
        """Test that empty sequence raises ValueError."""
        with pytest.raises(ValueError, match="stride must be"):
            normalize_size_param([], 2, "stride")

    def test_float_values_cast_to_int(self) -> None:
        """Test that float sequence values are cast to int."""
        result = normalize_size_param([3.0, 5.0], 2, "kernel_size")
        assert result == (3, 5)
        assert all(isinstance(v, int) for v in result)


class TestCreateNormLayer:
    """Tests for create_norm_layer."""

    @pytest.fixture
    def rngs(self) -> nnx.Rngs:
        """Provide RNGs for layer creation."""
        return nnx.Rngs(0)

    def test_batch_norm(self, rngs: nnx.Rngs) -> None:
        """Test batch norm creation."""
        layer = create_norm_layer("batch", 64, rngs=rngs)
        assert isinstance(layer, nnx.BatchNorm)

    def test_layer_norm(self, rngs: nnx.Rngs) -> None:
        """Test layer norm creation."""
        layer = create_norm_layer("layer", 64, rngs=rngs)
        assert isinstance(layer, nnx.LayerNorm)

    def test_group_norm(self, rngs: nnx.Rngs) -> None:
        """Test group norm creation."""
        layer = create_norm_layer("group", 64, group_norm_num_groups=32, rngs=rngs)
        assert isinstance(layer, nnx.GroupNorm)

    def test_group_norm_indivisible_raises(self, rngs: nnx.Rngs) -> None:
        """Test that incompatible group count raises ValueError."""
        with pytest.raises(ValueError, match="Features .* must be divisible by"):
            create_norm_layer("group", 65, group_norm_num_groups=32, rngs=rngs)

    def test_unknown_norm_type_raises(self, rngs: nnx.Rngs) -> None:
        """Test that unknown norm type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown norm_type"):
            create_norm_layer("instance", 64, rngs=rngs)


class TestApplyNorm:
    """Tests for apply_norm."""

    @pytest.fixture
    def rngs(self) -> nnx.Rngs:
        """Provide RNGs for layer creation."""
        return nnx.Rngs(0)

    def test_none_layer_returns_input(self) -> None:
        """Test that None norm layer returns input unchanged."""
        x = jnp.ones((2, 8, 8, 64))
        result = apply_norm(x, None, "batch", deterministic=True)
        assert jnp.array_equal(result, x)

    def test_batch_norm_deterministic(self, rngs: nnx.Rngs) -> None:
        """Test batch norm in deterministic mode."""
        x = jnp.ones((2, 8, 8, 64))
        layer = create_norm_layer("batch", 64, rngs=rngs)
        result = apply_norm(x, layer, "batch", deterministic=True)
        assert result.shape == x.shape

    def test_layer_norm(self, rngs: nnx.Rngs) -> None:
        """Test layer norm application."""
        x = jnp.ones((2, 8, 8, 64))
        layer = create_norm_layer("layer", 64, rngs=rngs)
        result = apply_norm(x, layer, "layer", deterministic=True)
        assert result.shape == x.shape

    def test_group_norm(self, rngs: nnx.Rngs) -> None:
        """Test group norm application."""
        x = jnp.ones((2, 8, 8, 64))
        layer = create_norm_layer("group", 64, group_norm_num_groups=32, rngs=rngs)
        result = apply_norm(x, layer, "group", deterministic=True)
        assert result.shape == x.shape
