"""Core functionality tests for UNet implementation (no performance tests)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import UNetBackboneConfig
from artifex.generative_models.models.backbones.unet import (
    ConvBlock,
    DownBlock,
    TimeEmbedding,
    UNet,
    UpBlock,
)


@pytest.fixture
def rngs():
    """Fixture for nnx random number generator."""
    return nnx.Rngs(42)


@pytest.fixture
def batch_size():
    """Standard batch size for testing."""
    return 2


@pytest.fixture
def image_size():
    """Standard image size for testing."""
    return 32


@pytest.fixture
def channels():
    """Standard number of channels for testing."""
    return 3


@pytest.fixture
def sample_image(batch_size, image_size, channels):
    """Sample image tensor for testing."""
    return jnp.ones((batch_size, image_size, image_size, channels))


@pytest.fixture
def sample_timesteps(batch_size):
    """Sample timestep indices for testing."""
    return jnp.array([10, 20])  # Different timesteps for each batch item


@pytest.fixture
def default_unet_config():
    """Default UNet configuration for testing."""
    return UNetBackboneConfig(
        name="test_unet",
        hidden_dims=(32, 64, 128, 256),
        activation="gelu",
        in_channels=3,
        out_channels=3,
        time_embedding_dim=128,
    )


class TestTimeEmbedding:
    """Test cases for TimeEmbedding module."""

    def test_initialization(self, rngs):
        """Test TimeEmbedding initialization."""
        embedding_dim = 128
        time_emb = TimeEmbedding(embedding_dim, rngs=rngs)

        assert time_emb.embedding_dim == embedding_dim
        assert hasattr(time_emb, "dense1")
        assert hasattr(time_emb, "dense2")

    def test_forward_pass_shape(self, rngs, batch_size):
        """Test TimeEmbedding forward pass produces correct shapes."""
        embedding_dim = 256
        time_emb = TimeEmbedding(embedding_dim, rngs=rngs)

        timesteps = jnp.array([5, 15])
        output = time_emb(timesteps)

        assert output.shape == (batch_size, embedding_dim)
        assert jnp.isfinite(output).all()

    def test_different_timesteps(self, rngs):
        """Test that different timesteps produce different embeddings."""
        embedding_dim = 128
        time_emb = TimeEmbedding(embedding_dim, rngs=rngs)

        t1 = jnp.array([5])
        t2 = jnp.array([10])

        emb1 = time_emb(t1)
        emb2 = time_emb(t2)

        # Different timesteps should produce different embeddings
        assert not jnp.allclose(emb1, emb2, atol=1e-5)

    def test_odd_embedding_dimension(self, rngs):
        """Test TimeEmbedding with odd embedding dimension."""
        embedding_dim = 127  # Odd dimension
        time_emb = TimeEmbedding(embedding_dim, rngs=rngs)

        timesteps = jnp.array([1, 2])
        output = time_emb(timesteps)

        assert output.shape == (2, embedding_dim)
        assert jnp.isfinite(output).all()


class TestConvBlock:
    """Test cases for ConvBlock module."""

    def test_initialization_same_channels(self, rngs):
        """Test ConvBlock initialization with same input/output channels."""
        in_channels = 32
        out_channels = 32
        block = ConvBlock(in_channels, out_channels, rngs=rngs)

        assert block.in_channels == in_channels
        assert block.out_channels == out_channels
        assert not block.use_skip_connection  # No skip needed for same channels

    def test_initialization_different_channels(self, rngs):
        """Test ConvBlock initialization with different input/output channels."""
        in_channels = 32
        out_channels = 64
        block = ConvBlock(in_channels, out_channels, rngs=rngs)

        assert block.in_channels == in_channels
        assert block.out_channels == out_channels
        assert block.use_skip_connection  # Skip connection needed

    def test_initialization_with_time_embedding(self, rngs):
        """Test ConvBlock initialization with time embedding."""
        in_channels = 32
        out_channels = 64
        time_embedding_dim = 128

        block = ConvBlock(in_channels, out_channels, time_embedding_dim, rngs=rngs)

        assert block.time_emb is not None
        assert hasattr(block, "time_emb")

    def test_forward_pass_without_time(self, rngs, batch_size, image_size):
        """Test ConvBlock forward pass without time embedding."""
        in_channels = 32
        out_channels = 64
        block = ConvBlock(in_channels, out_channels, rngs=rngs)

        x = jnp.ones((batch_size, image_size, image_size, in_channels))
        output = block(x)

        assert output.shape == (batch_size, image_size, image_size, out_channels)
        assert jnp.isfinite(output).all()

    def test_forward_pass_with_time(self, rngs, batch_size, image_size):
        """Test ConvBlock forward pass with time embedding."""
        in_channels = 32
        out_channels = 64
        time_embedding_dim = 128

        block = ConvBlock(in_channels, out_channels, time_embedding_dim, rngs=rngs)

        x = jnp.ones((batch_size, image_size, image_size, in_channels))
        time_emb = jnp.ones((batch_size, time_embedding_dim))

        output = block(x, time_emb)

        assert output.shape == (batch_size, image_size, image_size, out_channels)
        assert jnp.isfinite(output).all()

    def test_time_embedding_effect(self, rngs, batch_size, image_size):
        """Test that time embedding actually affects the output."""
        in_channels = 32
        out_channels = 64
        time_embedding_dim = 128

        block = ConvBlock(in_channels, out_channels, time_embedding_dim, rngs=rngs)

        x = jnp.ones((batch_size, image_size, image_size, in_channels))
        time_emb1 = jnp.ones((batch_size, time_embedding_dim))
        time_emb2 = jnp.ones((batch_size, time_embedding_dim)) * 2

        output1 = block(x, time_emb1)
        output2 = block(x, time_emb2)

        # Different time embeddings should produce different outputs
        assert not jnp.allclose(output1, output2, atol=1e-5)


class TestDownBlock:
    """Test cases for DownBlock module."""

    def test_initialization(self, rngs):
        """Test DownBlock initialization."""
        in_channels = 32
        out_channels = 64
        down_block = DownBlock(in_channels, out_channels, rngs=rngs)

        assert hasattr(down_block, "block1")
        assert hasattr(down_block, "block2")
        assert hasattr(down_block, "downsample")

    def test_forward_pass_shape(self, rngs, batch_size, image_size):
        """Test DownBlock forward pass produces correct shapes."""
        in_channels = 32
        out_channels = 64
        down_block = DownBlock(in_channels, out_channels, rngs=rngs)

        x = jnp.ones((batch_size, image_size, image_size, in_channels))
        downsampled, skip_features = down_block(x)

        # Downsampled should be half the spatial size
        expected_size = image_size // 2
        assert downsampled.shape == (batch_size, expected_size, expected_size, out_channels)

        # Skip features should maintain original spatial size
        assert skip_features.shape == (batch_size, image_size, image_size, out_channels)

        assert jnp.isfinite(downsampled).all()
        assert jnp.isfinite(skip_features).all()

    def test_with_time_embedding(self, rngs, batch_size, image_size):
        """Test DownBlock with time embedding."""
        in_channels = 32
        out_channels = 64
        time_embedding_dim = 128

        down_block = DownBlock(in_channels, out_channels, time_embedding_dim, rngs=rngs)

        x = jnp.ones((batch_size, image_size, image_size, in_channels))
        time_emb = jnp.ones((batch_size, time_embedding_dim))

        downsampled, skip_features = down_block(x, time_emb)

        expected_size = image_size // 2
        assert downsampled.shape == (batch_size, expected_size, expected_size, out_channels)
        assert skip_features.shape == (batch_size, image_size, image_size, out_channels)


class TestUpBlock:
    """Test cases for UpBlock module."""

    def test_initialization(self, rngs):
        """Test UpBlock initialization."""
        in_channels = 64
        skip_channels = 64  # FIXED: Added skip_channels parameter
        out_channels = 32
        up_block = UpBlock(in_channels, skip_channels, out_channels, rngs=rngs)

        assert hasattr(up_block, "block1")
        assert hasattr(up_block, "block2")
        assert hasattr(up_block, "upsample")

    def test_forward_pass_shape(self, rngs, batch_size):
        """Test UpBlock forward pass produces correct shapes."""
        in_channels = 64
        skip_channels = 64  # FIXED: Added skip_channels parameter
        out_channels = 32
        up_block = UpBlock(in_channels, skip_channels, out_channels, rngs=rngs)

        # Simulate downsampled feature and skip connection
        small_size = 16
        large_size = 32

        x = jnp.ones((batch_size, small_size, small_size, in_channels))
        skip_features = jnp.ones((batch_size, large_size, large_size, skip_channels))

        output = up_block(x, skip_features)

        # Output should match skip features spatial size
        assert output.shape == (batch_size, large_size, large_size, out_channels)
        assert jnp.isfinite(output).all()

    def test_with_time_embedding(self, rngs, batch_size):
        """Test UpBlock with time embedding."""
        in_channels = 64
        skip_channels = 64  # FIXED: Added skip_channels parameter
        out_channels = 32
        time_embedding_dim = 128

        up_block = UpBlock(in_channels, skip_channels, out_channels, time_embedding_dim, rngs=rngs)

        small_size = 16
        large_size = 32

        x = jnp.ones((batch_size, small_size, small_size, in_channels))
        skip_features = jnp.ones((batch_size, large_size, large_size, skip_channels))
        time_emb = jnp.ones((batch_size, time_embedding_dim))

        output = up_block(x, skip_features, time_emb)

        assert output.shape == (batch_size, large_size, large_size, out_channels)
        assert jnp.isfinite(output).all()


class TestUNet:
    """Test cases for the complete UNet model."""

    def test_initialization_default(self, rngs, default_unet_config):
        """Test UNet initialization with default config."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        assert unet.hidden_dims == [32, 64, 128, 256]
        assert unet.in_channels == 3
        assert unet.out_channels == 3
        assert hasattr(unet, "time_embedding")
        assert hasattr(unet, "in_conv")
        assert len(unet.down_blocks) == 3  # len(hidden_dims) - 1
        assert len(unet.up_blocks) == 3

    def test_initialization_custom(self, rngs):
        """Test UNet initialization with custom config."""
        config = UNetBackboneConfig(
            name="custom_unet",
            hidden_dims=(32, 64, 128),
            activation="silu",
            in_channels=1,
            out_channels=1,
            time_embedding_dim=64,
        )

        unet = UNet(config=config, rngs=rngs)

        assert unet.hidden_dims == [32, 64, 128]
        assert unet.in_channels == 1
        assert unet.out_channels == 1
        assert len(unet.down_blocks) == 2  # len(hidden_dims) - 1
        assert len(unet.up_blocks) == 2

    def test_forward_pass_shape_preservation(
        self, rngs, default_unet_config, sample_image, sample_timesteps
    ):
        """Test that UNet preserves input shape in output."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        output = unet(sample_image, sample_timesteps, deterministic=True)

        # Output should have same shape as input
        assert output.shape == sample_image.shape
        assert jnp.isfinite(output).all()

    def test_different_image_sizes(self, rngs, default_unet_config, batch_size, sample_timesteps):
        """Test UNet with different input image sizes."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        # Test different image sizes (must be divisible by 2^num_levels)
        test_sizes = [32, 64, 128]  # Divisible by 8 (2^3 levels)

        for size in test_sizes:
            x = jnp.ones((batch_size, size, size, 3))
            output = unet(x, sample_timesteps, deterministic=True)

            assert output.shape == x.shape
            assert jnp.isfinite(output).all()

    def test_different_channels(self, rngs, batch_size, image_size, sample_timesteps):
        """Test UNet with different numbers of input channels."""
        test_channels = [1, 3, 4]

        for channels in test_channels:
            config = UNetBackboneConfig(
                name="test_unet",
                hidden_dims=(32, 64, 128, 256),
                activation="gelu",
                in_channels=channels,
                out_channels=channels,
            )
            unet = UNet(config=config, rngs=rngs)
            x = jnp.ones((batch_size, image_size, image_size, channels))

            output = unet(x, sample_timesteps, deterministic=True)

            assert output.shape == x.shape
            assert jnp.isfinite(output).all()

    def test_different_batch_sizes(self, rngs, default_unet_config, image_size, channels):
        """Test UNet with different batch sizes."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        test_batch_sizes = [1, 2, 4]

        for batch_size in test_batch_sizes:
            x = jnp.ones((batch_size, image_size, image_size, channels))
            t = jnp.ones((batch_size,), dtype=jnp.int32) * 10

            output = unet(x, t, deterministic=True)

            assert output.shape == x.shape
            assert jnp.isfinite(output).all()

    def test_different_timesteps(self, rngs, default_unet_config, sample_image):
        """Test UNet with different timestep values."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        # Test different timestep scenarios
        test_timesteps = [
            jnp.array([0, 1]),  # Small timesteps
            jnp.array([100, 200]),  # Medium timesteps
            jnp.array([999, 1000]),  # Large timesteps
            jnp.array([5, 5]),  # Same timesteps
        ]

        for timesteps in test_timesteps:
            output = unet(sample_image, timesteps, deterministic=True)
            assert output.shape == sample_image.shape
            assert jnp.isfinite(output).all()

    def test_deterministic_behavior(
        self, rngs, default_unet_config, sample_image, sample_timesteps
    ):
        """Test deterministic vs stochastic behavior."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        # Deterministic should give same output
        output1 = unet(sample_image, sample_timesteps, deterministic=True)
        output2 = unet(sample_image, sample_timesteps, deterministic=True)

        assert jnp.allclose(output1, output2)

    def test_gradient_flow(self, rngs, default_unet_config, sample_image, sample_timesteps):
        """Test that gradients can flow through the model."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        def loss_fn(model, x, t):
            output = model(x, t, deterministic=True)
            # Use a target to ensure non-zero loss and gradients
            target = jnp.ones_like(output) * 0.5
            return jnp.mean((output - target) ** 2)

        # Compute gradients using the correct NNX API
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(unet, sample_image, sample_timesteps)

        # Check that gradients exist and are finite
        has_nonzero_grad = False
        grad_count = 0

        def check_gradient(grad_value):
            nonlocal has_nonzero_grad, grad_count
            if isinstance(grad_value, jax.Array):
                grad_count += 1
                # Check that gradients are finite
                assert jnp.isfinite(grad_value).all(), "Non-finite gradient found"
                # Check for non-zero gradients
                if jnp.any(jnp.abs(grad_value) > 1e-6):
                    has_nonzero_grad = True
            return grad_value

        # Apply the check to all gradient values
        jax.tree.map(check_gradient, grads)

        # Ensure we actually found some gradients
        assert grad_count > 0, "No gradient arrays found"
        assert has_nonzero_grad, f"No non-zero gradients found out of {grad_count} gradient arrays"

    def test_time_embedding_influence(self, rngs, default_unet_config, sample_image):
        """Test that different timesteps produce different outputs."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        t1 = jnp.array([10, 20])
        t2 = jnp.array([100, 200])

        output1 = unet(sample_image, t1, deterministic=True)
        output2 = unet(sample_image, t2, deterministic=True)

        # Different timesteps should produce different outputs
        assert not jnp.allclose(output1, output2, atol=1e-4)

    def test_jit_compilation(self, rngs, default_unet_config, sample_image, sample_timesteps):
        """Test that UNet can be JIT compiled."""
        unet = UNet(config=default_unet_config, rngs=rngs)

        @nnx.jit
        def forward(model, x, t):
            return model(x, t, deterministic=True)

        # First call (compilation)
        output1 = forward(unet, sample_image, sample_timesteps)

        # Second call (should use compiled version)
        output2 = forward(unet, sample_image, sample_timesteps)

        assert jnp.allclose(output1, output2)
        assert output1.shape == sample_image.shape


class TestGroupNormCompatibility:
    """Test GroupNorm compatibility with various channel counts."""

    def test_group_calculation(self):
        """Test the group calculation function."""

        def get_num_groups(channels: int) -> int:
            """Get number of groups ensuring divisibility."""
            for g in [32, 16, 8, 4, 2, 1]:
                if channels % g == 0:
                    return g
            return 1

        test_channels = [1, 2, 4, 8, 16, 32, 64, 128, 256]

        for channels in test_channels:
            groups = get_num_groups(channels)
            assert channels % groups == 0
            assert groups <= channels
            assert groups >= 1

    def test_conv_block_various_channels(self, rngs):
        """Test ConvBlock with various channel configurations."""
        # Test configurations that should work well with GroupNorm
        channel_configs = [
            (32, 32),  # Same channels
            (32, 64),  # Power of 2 channels
            (64, 128),  # Larger channels
            (8, 16),  # Smaller channels
        ]

        for in_channels, out_channels in channel_configs:
            block = ConvBlock(in_channels, out_channels, rngs=rngs)

            x = jnp.ones((1, 16, 16, in_channels))
            output = block(x)

            assert output.shape == (1, 16, 16, out_channels)
            assert jnp.isfinite(output).all()

    def test_unet_various_hidden_dims(self, rngs):
        """Test UNet with various hidden dimension configurations."""
        # Test configurations that work well with GroupNorm
        hidden_dim_configs = [
            (32, 64),
            (32, 64, 128),
            (64, 128, 256),
        ]

        for hidden_dims in hidden_dim_configs:
            config = UNetBackboneConfig(
                name="test_unet",
                hidden_dims=hidden_dims,
                activation="gelu",
                in_channels=3,
                out_channels=3,
            )
            unet = UNet(config=config, rngs=rngs)

            x = jnp.ones((1, 32, 32, 3))
            t = jnp.array([10])

            output = unet(x, t, deterministic=True)

            assert output.shape == x.shape
            assert jnp.isfinite(output).all()
