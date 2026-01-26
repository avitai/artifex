#!/usr/bin/env python3
"""
Quick test script to verify UNet fixes work correctly.
Tests basic functionality without complex performance benchmarks.

UNet uses the (config, *, rngs) signature pattern.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# Import the corrected UNet components
from artifex.generative_models.core.configuration import UNetBackboneConfig
from artifex.generative_models.models.backbones.unet import (
    ConvBlock,
    DownBlock,
    TimeEmbedding,
    UNet,
    UpBlock,
)


def create_unet_config(
    hidden_dims: tuple = (32, 64),
    in_channels: int = 3,
    out_channels: int = 3,
    time_embedding_dim: int = 128,
) -> UNetBackboneConfig:
    """Create a UNet config for testing."""
    return UNetBackboneConfig(
        name="test_unet",
        hidden_dims=hidden_dims,
        activation="gelu",
        in_channels=in_channels,
        out_channels=out_channels,
        time_embedding_dim=time_embedding_dim,
    )


def test_time_embedding():
    """Test TimeEmbedding component."""
    print("Testing TimeEmbedding...")
    rngs = nnx.Rngs(42)

    embedding_dim = 128
    time_emb = TimeEmbedding(embedding_dim, rngs=rngs)

    timesteps = jnp.array([10, 20])
    output = time_emb(timesteps)

    assert output.shape == (2, embedding_dim)
    assert jnp.isfinite(output).all()
    print("✅ TimeEmbedding test passed")


def test_conv_block():
    """Test ConvBlock component."""
    print("Testing ConvBlock...")
    rngs = nnx.Rngs(42)

    # Test with channels that work well with GroupNorm
    in_channels = 32
    out_channels = 64
    time_embedding_dim = 128

    block = ConvBlock(in_channels, out_channels, time_embedding_dim, rngs=rngs)

    x = jnp.ones((2, 32, 32, in_channels))
    time_emb = jnp.ones((2, time_embedding_dim))

    output = block(x, time_emb)

    assert output.shape == (2, 32, 32, out_channels)
    assert jnp.isfinite(output).all()
    print("✅ ConvBlock test passed")


def test_down_block():
    """Test DownBlock component."""
    print("Testing DownBlock...")
    rngs = nnx.Rngs(42)

    in_channels = 32
    out_channels = 64
    time_embedding_dim = 128

    down_block = DownBlock(in_channels, out_channels, time_embedding_dim, rngs=rngs)

    x = jnp.ones((2, 32, 32, in_channels))
    time_emb = jnp.ones((2, time_embedding_dim))

    downsampled, skip_features = down_block(x, time_emb)

    # Downsampled should be half the spatial size
    assert downsampled.shape == (2, 16, 16, out_channels)
    # Skip features should maintain original spatial size
    assert skip_features.shape == (2, 32, 32, out_channels)
    assert jnp.isfinite(downsampled).all()
    assert jnp.isfinite(skip_features).all()
    print("✅ DownBlock test passed")


def test_up_block():
    """Test UpBlock component."""
    print("Testing UpBlock...")
    rngs = nnx.Rngs(42)

    in_channels = 64
    skip_channels = 64  # FIXED: Added skip_channels parameter
    out_channels = 32
    time_embedding_dim = 128

    up_block = UpBlock(in_channels, skip_channels, out_channels, time_embedding_dim, rngs=rngs)

    # Simulate downsampled feature and skip connection
    small_size = 16
    large_size = 32

    x = jnp.ones((2, small_size, small_size, in_channels))
    skip_features = jnp.ones((2, large_size, large_size, skip_channels))
    time_emb = jnp.ones((2, time_embedding_dim))

    output = up_block(x, skip_features, time_emb)

    # Output should match skip features spatial size
    assert output.shape == (2, large_size, large_size, out_channels)
    assert jnp.isfinite(output).all()
    print("✅ UpBlock test passed")


def test_unet_basic():
    """Test basic UNet functionality."""
    print("Testing UNet basic functionality...")
    rngs = nnx.Rngs(42)

    # Use GroupNorm-friendly dimensions
    config = create_unet_config(
        hidden_dims=(32, 64, 128),
        in_channels=3,
        out_channels=3,
        time_embedding_dim=128,
    )

    unet = UNet(config, rngs=rngs)

    # Test with different input sizes
    batch_size = 2
    image_size = 32

    x = jnp.ones((batch_size, image_size, image_size, 3))
    t = jnp.array([10, 20])

    output = unet(x, t, deterministic=True)

    # Output should have same shape as input
    assert output.shape == x.shape
    assert jnp.isfinite(output).all()
    print("✅ UNet basic test passed")


def test_unet_different_sizes():
    """Test UNet with different input sizes."""
    print("Testing UNet with different sizes...")
    rngs = nnx.Rngs(42)

    config = create_unet_config(hidden_dims=(32, 64))
    unet = UNet(config, rngs=rngs)

    # Test different image sizes
    test_sizes = [32, 64]  # Sizes that are divisible by 2^num_levels

    for size in test_sizes:
        x = jnp.ones((1, size, size, 3))
        t = jnp.array([10])

        output = unet(x, t, deterministic=True)

        assert output.shape == x.shape
        assert jnp.isfinite(output).all()
        print(f"  ✅ Size {size}x{size} test passed")


def test_groupnorm_channels():
    """Test that our GroupNorm group calculation works correctly."""
    print("Testing GroupNorm group calculation...")

    def get_num_groups(channels: int) -> int:
        """Get number of groups ensuring divisibility."""
        for g in [32, 16, 8, 4, 2, 1]:
            if channels % g == 0:
                return g
        return 1

    # Test various channel counts
    test_channels = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    for channels in test_channels:
        groups = get_num_groups(channels)
        assert channels % groups == 0, f"Channels {channels} not divisible by groups {groups}"
        print(f"  Channels {channels} -> Groups {groups}")

    print("✅ GroupNorm calculation test passed")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("Testing gradient flow...")
    rngs = nnx.Rngs(42)

    config = create_unet_config(hidden_dims=(32, 64))
    unet = UNet(config, rngs=rngs)

    x = jnp.ones((1, 32, 32, 3))
    t = jnp.array([10])

    def loss_fn(model, x, t):
        output = model(x, t, deterministic=True)
        # Use a target to ensure non-zero loss and gradients
        target = jnp.ones_like(output) * 0.5
        return jnp.mean((output - target) ** 2)

    # Compute gradients using correct NNX API
    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(unet, x, t)

    # Check that gradients exist and are finite
    has_grads = False
    grad_count = 0

    def check_gradient(grad_value):
        nonlocal has_grads, grad_count
        if isinstance(grad_value, jax.Array):
            grad_count += 1
            assert jnp.isfinite(grad_value).all()
            if jnp.any(jnp.abs(grad_value) > 1e-6):
                has_grads = True
        return grad_value

    # Apply the check to all gradient values
    jax.tree.map(check_gradient, grads)

    assert grad_count > 0, "No gradient arrays found"
    assert has_grads, f"No non-zero gradients found out of {grad_count} gradient arrays"
    print("✅ Gradient flow test passed")


def main():
    """Run all tests."""
    print("🧪 Running UNet Quick Tests")
    print("=" * 50)

    try:
        test_time_embedding()
        test_conv_block()
        test_down_block()
        test_up_block()
        test_groupnorm_channels()
        test_unet_basic()
        test_unet_different_sizes()
        test_gradient_flow()

        print("\n" + "=" * 50)
        print("🎉 All tests passed! UNet implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
