"""Tests for ResNet block implementations.

These tests verify the correctness of ResNet components including:
- Basic ResNet blocks
- Bottleneck blocks
- Factory functions
- Validation logic
- Different normalization types
"""

import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.layers.resnet import (
    BottleneckBlock,
    create_resnet_block,
    create_resnet_stage,
    ResNetBlock,
)


def should_run_gpu_intensive_tests():
    """Check if GPU-intensive tests should be run.

    Returns:
        bool: True if tests should run, False if they should be skipped

    Tests will run when:
    1. GPU is available and properly configured, OR
    2. JAX_PLATFORMS is set to 'cpu', OR
    3. RUN_RESNET_GPU_TESTS is explicitly set to '1'

    Note: With proper CUDA configuration (JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1),
    these tests should run successfully on GPU.
    """
    # Import here to avoid circular imports
    from tests.utils.gpu_test_utils import is_gpu_available

    # Check if JAX is configured to use CPU
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if platforms.lower() == "cpu":
        return True

    # Check if tests are explicitly enabled
    if os.environ.get("RUN_RESNET_GPU_TESTS", "") != "":
        return True

    # Check if GPU is properly available and configured
    if is_gpu_available():
        return True

    # Default to skipping if none of the above conditions are met
    return False


@pytest.fixture
def rng_keys():
    """Fixture providing random keys for tests."""
    main_key = jax.random.key(42)
    keys = jax.random.split(main_key, 4)
    return {
        "params": keys[0],
        "dropout": keys[1],
        "batch_stats": keys[2],
        "extra": keys[3],
    }


class TestResNetBlock:
    """Tests for the ResNetBlock class."""

    def test_init_default(self, rng_keys):
        """Test initialization of ResNetBlock with default parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test default initialization
        block = ResNetBlock(in_features=64, features=64, rngs=rngs)
        assert block.in_features == 64
        assert block.features == 64
        assert block.kernel_size_tuple == (3, 3)
        assert block.strides_tuple == (1, 1)
        assert block.padding == "SAME"
        assert block.use_bias is True
        assert block.use_norm is True
        assert block.norm_type == "batch"
        assert block.activation is nnx.relu

        # Check layer creation
        assert isinstance(block.conv1, nnx.Conv)
        assert isinstance(block.conv2, nnx.Conv)
        assert not hasattr(block, "skip_projection")  # Not needed: same features, stride=1
        assert isinstance(block.norm1, nnx.BatchNorm)
        assert isinstance(block.norm2, nnx.BatchNorm)
        assert block.norm_skip is None

    def test_init_custom_parameters(self, rng_keys):
        """Test initialization with custom parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with custom parameters
        block = ResNetBlock(
            in_features=32,
            features=64,
            kernel_size=5,
            stride=2,
            padding="VALID",
            use_bias=False,
            norm_type="layer",
            activation=nnx.gelu,
            rngs=rngs,
        )
        assert block.in_features == 32
        assert block.features == 64
        assert block.kernel_size_tuple == (5, 5)
        assert block.strides_tuple == (2, 2)
        assert block.padding == "VALID"
        assert block.use_bias is False
        assert block.norm_type == "layer"
        assert block.activation == nnx.gelu

        # Should create skip projection due to feature mismatch and stride
        assert isinstance(block.skip_projection, nnx.Conv)
        assert isinstance(block.norm1, nnx.LayerNorm)
        assert isinstance(block.norm2, nnx.LayerNorm)
        assert isinstance(block.norm_skip, nnx.LayerNorm)

    def test_init_with_stride_only(self, rng_keys):
        """Test initialization with stride but same features."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = ResNetBlock(in_features=64, features=64, stride=2, rngs=rngs)
        assert block.strides_tuple == (2, 2)
        assert isinstance(block.skip_projection, nnx.Conv)  # Should be created due to stride
        assert block.skip_projection.strides == (2, 2)

    def test_init_with_feature_change_only(self, rng_keys):
        """Test initialization with different features but no stride."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = ResNetBlock(in_features=32, features=64, stride=1, rngs=rngs)
        assert block.in_features == 32
        assert block.features == 64
        assert block.strides_tuple == (1, 1)
        assert isinstance(block.skip_projection, nnx.Conv)  # Created due to feature mismatch
        assert block.skip_projection.strides == (1, 1)

    def test_init_normalization_types(self, rng_keys):
        """Test initialization with different normalization types."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Batch norm
        block_bn = ResNetBlock(in_features=64, features=64, norm_type="batch", rngs=rngs)
        assert isinstance(block_bn.norm1, nnx.BatchNorm)
        assert isinstance(block_bn.norm2, nnx.BatchNorm)

        # Layer norm
        block_ln = ResNetBlock(in_features=64, features=64, norm_type="layer", rngs=rngs)
        assert isinstance(block_ln.norm1, nnx.LayerNorm)
        assert isinstance(block_ln.norm2, nnx.LayerNorm)

        # Group norm
        block_gn = ResNetBlock(
            in_features=64, features=64, norm_type="group", group_norm_num_groups=32, rngs=rngs
        )
        assert isinstance(block_gn.norm1, nnx.GroupNorm)
        assert isinstance(block_gn.norm2, nnx.GroupNorm)

        # No norm
        block_no_norm = ResNetBlock(in_features=64, features=64, use_norm=False, rngs=rngs)
        assert block_no_norm.norm1 is None
        assert block_no_norm.norm2 is None
        assert block_no_norm.norm_skip is None

    def test_parameter_validation(self, rng_keys):
        """Test parameter validation in initialization."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test invalid features
        with pytest.raises(ValueError, match="features must be positive"):
            ResNetBlock(in_features=64, features=0, rngs=rngs)

        with pytest.raises(ValueError, match="in_features must be positive"):
            ResNetBlock(in_features=-1, features=64, rngs=rngs)

        # Test invalid norm type
        with pytest.raises(ValueError, match="norm_type must be one of"):
            ResNetBlock(in_features=64, features=64, norm_type="invalid", rngs=rngs)

        # Test invalid padding
        with pytest.raises(ValueError, match="padding must be"):
            ResNetBlock(in_features=64, features=64, padding="INVALID", rngs=rngs)

        # Test invalid kernel_size
        with pytest.raises(ValueError, match="kernel_size must be"):
            ResNetBlock(in_features=64, features=64, kernel_size=[1, 2, 3], rngs=rngs)

        # Test invalid stride
        with pytest.raises(ValueError, match="stride must be"):
            ResNetBlock(in_features=64, features=64, stride=[1], rngs=rngs)

        # Test group norm with incompatible features
        with pytest.raises(ValueError, match="Features .* must be divisible by"):
            ResNetBlock(
                in_features=65, features=65, norm_type="group", group_norm_num_groups=32, rngs=rngs
            )

    def test_required_rngs(self):
        """Test that rngs parameter is required."""
        with pytest.raises(TypeError):
            ResNetBlock(in_features=64, features=64)

    def test_forward_basic(self, rng_keys):
        """Test basic forward pass."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = ResNetBlock(in_features=16, features=16, rngs=rngs)
        x = jnp.ones((2, 32, 32, 16))
        y = block(x, deterministic=True)
        assert y.shape == (2, 32, 32, 16)

    def test_forward_with_stride(self, rng_keys):
        """Test forward pass with stride."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = ResNetBlock(in_features=16, features=16, stride=2, rngs=rngs)
        x = jnp.ones((2, 32, 32, 16))
        y = block(x, deterministic=True)
        assert y.shape == (2, 16, 16, 16)  # Spatial dimensions halved

    def test_forward_with_feature_change(self, rng_keys):
        """Test forward pass with feature dimension change."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = ResNetBlock(in_features=16, features=32, stride=2, rngs=rngs)
        x = jnp.ones((2, 32, 32, 16))
        y = block(x, deterministic=True)
        assert y.shape == (2, 16, 16, 32)

    def test_forward_different_norm_types(self, rng_keys):
        """Test forward pass with different normalization types."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])
        x = jnp.ones((2, 32, 32, 16))

        # Test each normalization type
        for norm_type in ["batch", "layer", "group"]:
            block = ResNetBlock(
                in_features=16,
                features=16,
                norm_type=norm_type,
                group_norm_num_groups=16,  # Compatible with 16 features
                rngs=rngs,
            )
            y = block(x, deterministic=True)
            assert y.shape == (2, 32, 32, 16)

    def test_forward_no_norm(self, rng_keys):
        """Test forward pass without normalization."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = ResNetBlock(in_features=16, features=16, use_norm=False, rngs=rngs)
        x = jnp.ones((2, 32, 32, 16))
        y = block(x, deterministic=True)
        assert y.shape == (2, 32, 32, 16)

    def test_deterministic_vs_training_mode(self, rng_keys):
        """Test difference between deterministic and training modes."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        # Use batch norm to see difference between modes
        block = ResNetBlock(in_features=16, features=16, norm_type="batch", rngs=rngs)
        x = jax.random.normal(rng_keys["extra"], (2, 32, 32, 16))

        # Training mode
        y_train = block(x, deterministic=False)

        # Evaluation mode
        y_eval = block(x, deterministic=True)

        assert y_train.shape == y_eval.shape == (2, 32, 32, 16)
        # Results should be different due to batch norm behavior
        assert not jnp.allclose(y_train, y_eval, atol=1e-6)


class TestBottleneckBlock:
    """Tests for the BottleneckBlock class."""

    def test_init_default(self, rng_keys):
        """Test initialization with default parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = BottleneckBlock(in_features=256, out_features=256, rngs=rngs)
        assert block.in_features == 256
        assert block.out_features == 256
        assert block.bottleneck_channels == 256 // 4  # Default expansion ratio is 4
        assert block.kernel_size_tuple == (3, 3)
        assert block.strides_tuple == (1, 1)
        assert block.padding == "SAME"
        assert block.use_bias is True
        assert block.use_norm is True
        assert block.norm_type == "batch"
        assert block.activation == nnx.relu

        # Check layer creation
        assert isinstance(block.conv1, nnx.Conv)
        assert isinstance(block.conv2, nnx.Conv)
        assert isinstance(block.conv3, nnx.Conv)
        assert not hasattr(block, "skip_projection")  # Not needed: same features, stride=1
        assert isinstance(block.norm1, nnx.BatchNorm)
        assert isinstance(block.norm2, nnx.BatchNorm)
        assert isinstance(block.norm3, nnx.BatchNorm)

    def test_init_custom_parameters(self, rng_keys):
        """Test initialization with custom parameters."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = BottleneckBlock(
            in_features=128,
            out_features=256,
            bottleneck_expansion_ratio=2.0,
            kernel_size=5,
            stride=2,
            padding="VALID",
            use_bias=False,
            norm_type="layer",
            activation=nnx.gelu,
            rngs=rngs,
        )
        assert block.in_features == 128
        assert block.out_features == 256
        assert block.bottleneck_channels == 256 // 2  # Custom expansion ratio
        assert block.kernel_size_tuple == (5, 5)
        assert block.strides_tuple == (2, 2)
        assert block.padding == "VALID"
        assert block.use_bias is False
        assert block.norm_type == "layer"
        assert block.activation == nnx.gelu

        # Should create skip projection due to feature mismatch and stride
        assert isinstance(block.skip_projection, nnx.Conv)
        assert isinstance(block.norm1, nnx.LayerNorm)
        assert isinstance(block.norm2, nnx.LayerNorm)
        assert isinstance(block.norm3, nnx.LayerNorm)
        assert isinstance(block.norm_skip, nnx.LayerNorm)

    def test_bottleneck_expansion_calculations(self, rng_keys):
        """Test bottleneck channel calculations."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test different expansion ratios
        test_cases = [
            (256, 4.0, 64),  # Standard ResNet
            (256, 2.0, 128),  # Less compression
            (128, 8.0, 16),  # More compression
            (100, 3.0, 33),  # Non-power-of-2
        ]

        for out_features, ratio, expected_bottleneck in test_cases:
            block = BottleneckBlock(
                in_features=out_features,
                out_features=out_features,
                bottleneck_expansion_ratio=ratio,
                rngs=rngs,
            )
            assert block.bottleneck_channels == expected_bottleneck

    def test_parameter_validation(self, rng_keys):
        """Test parameter validation."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test invalid out_features
        with pytest.raises(ValueError, match="out_features must be positive"):
            BottleneckBlock(in_features=64, out_features=0, rngs=rngs)

        # Test invalid bottleneck_expansion_ratio
        with pytest.raises(ValueError, match="bottleneck_expansion_ratio must be positive"):
            BottleneckBlock(
                in_features=64, out_features=64, bottleneck_expansion_ratio=-1, rngs=rngs
            )

        # Test group norm validation
        with pytest.raises(ValueError, match="Bottleneck channels .* must be divisible by"):
            BottleneckBlock(
                in_features=64,
                out_features=65,  # Will create bottleneck_channels = 16 (65//4)
                norm_type="group",
                group_norm_num_groups=32,  # 16 not divisible by 32
                rngs=rngs,
            )

    def test_forward_basic(self, rng_keys):
        """Test basic forward pass."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = BottleneckBlock(in_features=64, out_features=64, rngs=rngs)
        x = jnp.ones((2, 32, 32, 64))
        y = block(x, deterministic=True)
        assert y.shape == (2, 32, 32, 64)

    def test_forward_with_stride(self, rng_keys):
        """Test forward pass with stride."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = BottleneckBlock(in_features=64, out_features=64, stride=2, rngs=rngs)
        x = jnp.ones((2, 32, 32, 64))
        y = block(x, deterministic=True)
        assert y.shape == (2, 16, 16, 64)

    def test_forward_with_feature_expansion(self, rng_keys):
        """Test forward pass with feature expansion."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = BottleneckBlock(in_features=64, out_features=128, stride=2, rngs=rngs)
        x = jnp.ones((2, 32, 32, 64))
        y = block(x, deterministic=True)
        assert y.shape == (2, 16, 16, 128)

    def test_forward_no_norm(self, rng_keys):
        """Test forward pass without normalization."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        block = BottleneckBlock(in_features=64, out_features=64, use_norm=False, rngs=rngs)
        x = jnp.ones((2, 32, 32, 64))
        y = block(x, deterministic=True)
        assert y.shape == (2, 32, 32, 64)
        assert block.norm1 is None
        assert block.norm2 is None
        assert block.norm3 is None


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_resnet_block_basic(self, rng_keys):
        """Test create_resnet_block with basic type."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = create_resnet_block(
            block_type="basic",
            in_features=64,
            out_features=128,
            stride=2,
            norm_type="layer",
            rngs=rngs,
        )

        assert isinstance(block, ResNetBlock)
        assert block.in_features == 64
        assert block.features == 128
        assert block.strides_tuple == (2, 2)
        assert block.norm_type == "layer"

    def test_create_resnet_block_bottleneck(self, rng_keys):
        """Test create_resnet_block with bottleneck type."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        block = create_resnet_block(
            block_type="bottleneck",
            in_features=128,
            out_features=256,
            stride=2,
            norm_type="batch",
            activation=nnx.gelu,
            rngs=rngs,
        )

        assert isinstance(block, BottleneckBlock)
        assert block.in_features == 128
        assert block.out_features == 256
        assert block.strides_tuple == (2, 2)
        assert block.norm_type == "batch"
        assert block.activation == nnx.gelu

    def test_create_resnet_block_invalid_type(self, rng_keys):
        """Test create_resnet_block with invalid block type."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        with pytest.raises(ValueError, match="Unsupported block_type"):
            create_resnet_block(
                block_type="invalid",
                in_features=64,
                out_features=64,
                rngs=rngs,
            )

    def test_create_resnet_stage_basic(self, rng_keys):
        """Test create_resnet_stage with basic blocks."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        stage = create_resnet_stage(
            block_type="basic",
            num_blocks=3,
            in_features=64,
            out_features=128,
            stride=2,
            norm_type="batch",
            rngs=rngs,
        )

        assert len(stage) == 3
        assert all(isinstance(block, ResNetBlock) for block in stage)

        # First block should have stride and feature change
        assert stage[0].in_features == 64
        assert stage[0].features == 128
        assert stage[0].strides_tuple == (2, 2)

        # Remaining blocks should have stride=1 and same features
        for block in stage[1:]:
            assert block.in_features == 128
            assert block.features == 128
            assert block.strides_tuple == (1, 1)

    def test_create_resnet_stage_bottleneck(self, rng_keys):
        """Test create_resnet_stage with bottleneck blocks."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        stage = create_resnet_stage(
            block_type="bottleneck",
            num_blocks=4,
            in_features=256,
            out_features=512,
            stride=2,
            norm_type="layer",
            activation=nnx.gelu,
            rngs=rngs,
        )

        assert len(stage) == 4
        assert all(isinstance(block, BottleneckBlock) for block in stage)

        # First block should have stride and feature change
        assert stage[0].in_features == 256
        assert stage[0].out_features == 512
        assert stage[0].strides_tuple == (2, 2)
        assert stage[0].activation == nnx.gelu

        # Remaining blocks should have stride=1 and same features
        for block in stage[1:]:
            assert block.in_features == 512
            assert block.out_features == 512
            assert block.strides_tuple == (1, 1)

    def test_create_resnet_stage_validation(self, rng_keys):
        """Test create_resnet_stage parameter validation."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test invalid num_blocks
        with pytest.raises(ValueError, match="num_blocks must be positive"):
            create_resnet_stage(
                block_type="basic",
                num_blocks=0,
                in_features=64,
                out_features=64,
                rngs=rngs,
            )

    def test_create_resnet_stage_with_kwargs(self, rng_keys):
        """Test create_resnet_stage with additional kwargs."""
        rngs = nnx.Rngs(params=rng_keys["params"])

        stage = create_resnet_stage(
            block_type="bottleneck",
            num_blocks=2,
            in_features=128,
            out_features=256,
            stride=1,
            norm_type="group",
            rngs=rngs,
            bottleneck_expansion_ratio=2.0,  # Additional kwarg
            group_norm_num_groups=32,
            use_bias=False,
        )

        assert len(stage) == 2
        for block in stage:
            assert block.norm_type == "group"
            assert block.use_bias is False
            assert block.bottleneck_channels == 256 // 2  # Custom expansion ratio


class TestHelperMethods:
    """Tests for shared utility functions used by ResNet blocks."""

    def test_normalize_size_param(self):
        """Test normalize_size_param utility function."""
        from artifex.generative_models.core.layers._utils import normalize_size_param

        # Test integer input
        result = normalize_size_param(3, 2, "kernel_size")
        assert result == (3, 3)

        # Test tuple input
        result = normalize_size_param((3, 5), 2, "stride")
        assert result == (3, 5)

        # Test list input
        result = normalize_size_param([2, 4], 2, "kernel_size")
        assert result == (2, 4)

        # Test invalid input
        with pytest.raises(ValueError, match="kernel_size must be"):
            normalize_size_param([1, 2, 3], 2, "kernel_size")

    def test_apply_norm(self, rng_keys):
        """Test apply_norm utility function."""
        from artifex.generative_models.core.layers._utils import (
            apply_norm,
            create_norm_layer,
        )

        rngs = nnx.Rngs(params=rng_keys["params"])

        # Test with no norm layer
        x = jnp.ones((2, 32, 32, 64))
        result = apply_norm(x, None, "batch", deterministic=True)
        assert jnp.array_equal(result, x)

        # Test with batch norm
        bn = create_norm_layer("batch", 64, rngs=rngs)
        result = apply_norm(x, bn, "batch", deterministic=True)
        assert result.shape == x.shape

        # Test with layer norm
        ln = create_norm_layer("layer", 64, rngs=rngs)
        result = apply_norm(x, ln, "layer", deterministic=True)
        assert result.shape == x.shape


class TestIntegration:
    """Integration tests for ResNet blocks."""

    def test_resnet_block_stage_pipeline(self, rng_keys):
        """Test a complete pipeline of ResNet blocks."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create a simple ResNet-like pipeline
        # Stage 1: 64 -> 64, stride=1
        stage1 = create_resnet_stage(
            block_type="basic",
            num_blocks=2,
            in_features=64,
            out_features=64,
            stride=1,
            rngs=rngs,
        )

        # Stage 2: 64 -> 128, stride=2
        stage2 = create_resnet_stage(
            block_type="basic",
            num_blocks=2,
            in_features=64,
            out_features=128,
            stride=2,
            rngs=rngs,
        )

        # Forward pass through pipeline
        x = jnp.ones((2, 32, 32, 64))

        # Stage 1
        for block in stage1:
            x = block(x, deterministic=True)
        assert x.shape == (2, 32, 32, 64)

        # Stage 2
        for block in stage2:
            x = block(x, deterministic=True)
        assert x.shape == (2, 16, 16, 128)

    def test_bottleneck_vs_basic_comparison(self, rng_keys):
        """Test comparison between basic and bottleneck blocks."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create comparable blocks
        basic_block = create_resnet_block(
            block_type="basic",
            in_features=64,
            out_features=128,
            stride=2,
            rngs=rngs,
        )

        bottleneck_block = create_resnet_block(
            block_type="bottleneck",
            in_features=64,
            out_features=128,
            stride=2,
            rngs=rngs,
        )

        # Same input
        x = jnp.ones((2, 32, 32, 64))

        # Both should produce same output shape
        y_basic = basic_block(x, deterministic=True)
        y_bottleneck = bottleneck_block(x, deterministic=True)

        assert y_basic.shape == y_bottleneck.shape == (2, 16, 16, 128)

    def test_mixed_normalization_pipeline(self, rng_keys):
        """Test pipeline with different normalization types."""
        if not should_run_gpu_intensive_tests():
            pytest.skip("Skipping GPU-intensive test. Set JAX_PLATFORMS=cpu to run safely.")

        rngs = nnx.Rngs(params=rng_keys["params"])

        # Create blocks with different norm types
        blocks = [
            create_resnet_block("basic", 64, 64, norm_type="batch", rngs=rngs),
            create_resnet_block("basic", 64, 64, norm_type="layer", rngs=rngs),
            create_resnet_block(
                "basic", 64, 64, norm_type="group", group_norm_num_groups=32, rngs=rngs
            ),
            create_resnet_block("basic", 64, 64, use_norm=False, rngs=rngs),
        ]

        # Forward pass through all blocks
        x = jnp.ones((2, 32, 32, 64))
        for block in blocks:
            x = block(x, deterministic=True)
            assert x.shape == (2, 32, 32, 64)


if __name__ == "__main__":
    # Manual test runner for quick verification
    print("Running manual ResNet tests...")

    # Create test fixture
    rng_keys = {
        "params": jax.random.key(42),
        "dropout": jax.random.key(43),
        "batch_stats": jax.random.key(44),
        "extra": jax.random.key(45),
    }

    # Test basic components
    test_resnet = TestResNetBlock()
    test_resnet.test_init_default(rng_keys)
    test_resnet.test_parameter_validation(rng_keys)
    print("✓ ResNetBlock tests passed")

    test_bottleneck = TestBottleneckBlock()
    test_bottleneck.test_init_default(rng_keys)
    test_bottleneck.test_parameter_validation(rng_keys)
    print("✓ BottleneckBlock tests passed")

    test_factory = TestFactoryFunctions()
    test_factory.test_create_resnet_block_basic(rng_keys)
    test_factory.test_create_resnet_stage_basic(rng_keys)
    print("✓ Factory function tests passed")

    test_helpers = TestHelperMethods()
    test_helpers.test_process_size_param(rng_keys)
    print("✓ Helper method tests passed")

    print("\nAll manual tests completed successfully!")
    print("Run 'pytest' for the complete test suite with forward pass tests.")
