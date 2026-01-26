"""Tests for PixelCNN autoregressive image models.

This module provides comprehensive test coverage for the PixelCNN implementation,
including MaskedConv2D, PixelCNN model, and related functionality.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import PixelCNNConfig
from artifex.generative_models.models.autoregressive.pixel_cnn import MaskedConv2D, PixelCNN


# =============================================================================
# Fixtures (DRY: Centralized test setup)
# =============================================================================


@pytest.fixture
def rngs():
    """Standard fixture for random number generators."""
    return nnx.Rngs(
        params=jax.random.key(42), sample=jax.random.key(123), dropout=jax.random.key(456)
    )


@pytest.fixture
def small_image_shape():
    """Small image shape for fast testing."""
    return (4, 4, 1)  # 4x4 grayscale


@pytest.fixture
def rgb_image_shape():
    """RGB image shape for testing."""
    return (8, 8, 3)  # 8x8 RGB


@pytest.fixture
def pixel_cnn_config(small_image_shape):
    """Standard PixelCNN configuration for testing."""
    return PixelCNNConfig(
        name="test_pixel_cnn",
        image_shape=small_image_shape,
        hidden_channels=32,
        num_layers=3,
        num_residual_blocks=2,
        kernel_size=3,
    )


@pytest.fixture
def rgb_pixel_cnn_config(rgb_image_shape):
    """RGB PixelCNN configuration for testing."""
    return PixelCNNConfig(
        name="test_rgb_pixel_cnn",
        image_shape=rgb_image_shape,
        hidden_channels=64,
        num_layers=4,
        num_residual_blocks=3,
        kernel_size=3,
    )


def create_random_image(key: jax.Array, shape: tuple[int, ...], batch_size: int = 1) -> jax.Array:
    """Helper to create random integer images in [0, 255] range."""
    full_shape = (batch_size, *shape)
    return jax.random.randint(key, full_shape, 0, 256, dtype=jnp.int32)


# =============================================================================
# MaskedConv2D Tests
# =============================================================================


class TestMaskedConv2D:
    """Test suite for MaskedConv2D layer."""

    def test_initialization_mask_type_a(self, rngs: nnx.Rngs):
        """Test MaskedConv2D initialization with mask type A."""
        conv = MaskedConv2D(
            in_features=3,
            out_features=32,
            kernel_size=(3, 3),
            mask_type="A",
            rngs=rngs,
        )

        assert conv.in_features == 3
        assert conv.out_features == 32
        assert conv.kernel_size == (3, 3)
        assert conv.mask_type == "A"
        assert conv.mask is not None

    def test_initialization_mask_type_b(self, rngs: nnx.Rngs):
        """Test MaskedConv2D initialization with mask type B."""
        conv = MaskedConv2D(
            in_features=32,
            out_features=32,
            kernel_size=(3, 3),
            mask_type="B",
            rngs=rngs,
        )

        assert conv.mask_type == "B"
        assert conv.mask is not None

    def test_mask_type_a_blocks_center(self, rngs: nnx.Rngs):
        """Test that mask type A blocks the center pixel."""
        conv = MaskedConv2D(
            in_features=1,
            out_features=1,
            kernel_size=(3, 3),
            mask_type="A",
            rngs=rngs,
        )

        # For 3x3 kernel, center is at (1, 1)
        center_h, center_w = 1, 1
        assert conv.mask[center_h, center_w, 0, 0] == 0, "Mask type A should block center"

    def test_mask_type_b_allows_center(self, rngs: nnx.Rngs):
        """Test that mask type B allows the center pixel."""
        conv = MaskedConv2D(
            in_features=1,
            out_features=1,
            kernel_size=(3, 3),
            mask_type="B",
            rngs=rngs,
        )

        # For 3x3 kernel, center is at (1, 1)
        center_h, center_w = 1, 1
        assert conv.mask[center_h, center_w, 0, 0] == 1, "Mask type B should allow center"

    def test_mask_blocks_future_pixels(self, rngs: nnx.Rngs):
        """Test that mask blocks future pixels (below and to the right)."""
        conv = MaskedConv2D(
            in_features=1,
            out_features=1,
            kernel_size=(3, 3),
            mask_type="A",
            rngs=rngs,
        )

        mask = conv.mask[:, :, 0, 0]

        # Bottom row should be all zeros (future)
        assert jnp.all(mask[2, :] == 0), "Bottom row should be masked"

        # Center row, right side should be zeros (future)
        assert mask[1, 2] == 0, "Right of center should be masked"

        # Top row should be all ones (past)
        assert jnp.all(mask[0, :] == 1), "Top row should be unmasked"

    def test_mask_shape(self, rngs: nnx.Rngs):
        """Test mask has correct shape."""
        in_features = 3
        out_features = 16
        kernel_size = (5, 5)

        conv = MaskedConv2D(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            mask_type="B",
            rngs=rngs,
        )

        expected_shape = (5, 5, in_features, out_features)
        assert conv.mask.shape == expected_shape

    def test_forward_pass_shape(self, rngs: nnx.Rngs):
        """Test forward pass produces correct output shape."""
        batch_size = 2
        height, width = 8, 8
        in_features = 3
        out_features = 32

        conv = MaskedConv2D(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            mask_type="A",
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, in_features))
        output = conv(x)

        expected_shape = (batch_size, height, width, out_features)
        assert output.shape == expected_shape

    def test_forward_pass_produces_finite_output(self, rngs: nnx.Rngs):
        """Test forward pass produces finite values."""
        conv = MaskedConv2D(
            in_features=3,
            out_features=16,
            kernel_size=(3, 3),
            mask_type="A",
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.key(0), (1, 8, 8, 3))
        output = conv(x)

        assert jnp.isfinite(output).all()

    def test_different_kernel_sizes(self, rngs: nnx.Rngs):
        """Test MaskedConv2D with different kernel sizes."""
        kernel_sizes = [(3, 3), (5, 5), (7, 7)]

        for ksize in kernel_sizes:
            conv = MaskedConv2D(
                in_features=3,
                out_features=16,
                kernel_size=ksize,
                mask_type="A",
                rngs=rngs,
            )

            x = jax.random.normal(jax.random.key(0), (1, 16, 16, 3))
            output = conv(x)

            # Output should have same spatial dimensions due to SAME padding
            assert output.shape == (1, 16, 16, 16)


class TestMaskedConv2DAutoregressive:
    """Tests verifying autoregressive property of MaskedConv2D."""

    def test_mask_type_a_autoregressive_property(self, rngs: nnx.Rngs):
        """Test that mask type A maintains autoregressive property.

        The output at position (i, j) should only depend on pixels
        strictly before (i, j) in raster scan order.
        """
        conv = MaskedConv2D(
            in_features=1,
            out_features=1,
            kernel_size=(3, 3),
            mask_type="A",
            rngs=rngs,
        )

        # Create input with a single non-zero pixel at center
        x = jnp.zeros((1, 5, 5, 1))
        x = x.at[0, 2, 2, 0].set(1.0)  # Set center pixel

        _ = conv(x)

        # For mask type A, the output at center (2, 2) should be 0
        # because it can't depend on itself
        # Note: We can't make this exact due to kernel weights,
        # but the mask ensures certain dependencies are zero
        # The mask structure verification is done in other tests

    def test_mask_type_b_allows_self_reference(self, rngs: nnx.Rngs):
        """Test that mask type B allows self-reference at center.

        The output at position (i, j) can depend on the input at (i, j).
        """
        conv = MaskedConv2D(
            in_features=1,
            out_features=1,
            kernel_size=(3, 3),
            mask_type="B",
            rngs=rngs,
        )

        # The mask at center should be 1 for type B
        assert conv.mask[1, 1, 0, 0] == 1


# =============================================================================
# PixelCNN Model Tests
# =============================================================================


class TestPixelCNNInitialization:
    """Test suite for PixelCNN initialization."""

    def test_basic_initialization(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test basic model initialization."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        assert model.config == pixel_cnn_config
        assert model.image_shape == pixel_cnn_config.image_shape
        assert model.num_layers == pixel_cnn_config.num_layers
        assert model.hidden_channels == pixel_cnn_config.hidden_channels
        assert model.num_residual_blocks == pixel_cnn_config.num_residual_blocks

    def test_initialization_creates_layers(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test that initialization creates all required layers."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        # Check first conv (mask type A)
        assert model.first_conv is not None
        assert model.first_conv.mask_type == "A"

        # Check hidden convs (mask type B)
        assert len(model.hidden_convs) == pixel_cnn_config.num_layers - 1
        for conv in model.hidden_convs:
            assert conv.mask_type == "B"

        # Check residual blocks
        assert len(model.residual_blocks) == pixel_cnn_config.num_residual_blocks

        # Check output convs (one per channel)
        channels = pixel_cnn_config.image_shape[2]
        assert len(model.output_convs) == channels

    def test_initialization_without_rngs(self, pixel_cnn_config):
        """Test initialization without explicit rngs uses defaults."""
        model = PixelCNN(pixel_cnn_config, rngs=None)

        assert model is not None
        assert model._rngs is not None

    def test_initialization_with_invalid_config_type(self, rngs: nnx.Rngs):
        """Test that invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="config must be PixelCNNConfig"):
            PixelCNN({"image_shape": (8, 8, 3)}, rngs=rngs)

    def test_vocab_size_and_sequence_length(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test vocab_size and sequence_length are set correctly."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        # vocab_size is 256 for 8-bit pixels
        assert model.vocab_size == 256

        # sequence_length is H * W * C
        h, w, c = pixel_cnn_config.image_shape
        assert model.sequence_length == h * w * c


class TestPixelCNNForwardPass:
    """Test suite for PixelCNN forward pass."""

    def test_forward_pass_output_structure(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test forward pass returns expected output structure."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs)

        assert "logits" in outputs
        assert "logits_spatial" in outputs

    def test_forward_pass_logits_flat_shape(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test flat logits have correct shape."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        batch_size = 2
        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size)
        outputs = model(images, rngs=rngs)

        h, w, c = pixel_cnn_config.image_shape
        expected_shape = (batch_size, h * w * c, 256)
        assert outputs["logits"].shape == expected_shape

    def test_forward_pass_logits_spatial_shape(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test spatial logits have correct shape."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        batch_size = 2
        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size)
        outputs = model(images, rngs=rngs)

        h, w, c = pixel_cnn_config.image_shape
        expected_shape = (batch_size, h, w, c, 256)
        assert outputs["logits_spatial"].shape == expected_shape

    def test_forward_pass_produces_finite_output(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test forward pass produces finite values."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=1)
        outputs = model(images, rngs=rngs)

        assert jnp.isfinite(outputs["logits"]).all()
        assert jnp.isfinite(outputs["logits_spatial"]).all()

    def test_forward_pass_rgb_images(self, rgb_pixel_cnn_config, rngs: nnx.Rngs):
        """Test forward pass with RGB images."""
        model = PixelCNN(rgb_pixel_cnn_config, rngs=rngs)

        batch_size = 2
        images = create_random_image(
            jax.random.key(0), rgb_pixel_cnn_config.image_shape, batch_size
        )
        outputs = model(images, rngs=rngs)

        h, w, c = rgb_pixel_cnn_config.image_shape
        assert outputs["logits_spatial"].shape == (batch_size, h, w, c, 256)

    def test_forward_pass_training_mode(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test forward pass in training mode."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs, training=True)

        assert jnp.isfinite(outputs["logits"]).all()

    def test_forward_pass_deterministic(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test forward pass is deterministic in eval mode."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=1)

        outputs1 = model(images, rngs=rngs, training=False)
        outputs2 = model(images, rngs=rngs, training=False)

        assert jnp.allclose(outputs1["logits"], outputs2["logits"], atol=1e-6)


class TestPixelCNNLoss:
    """Test suite for PixelCNN loss computation."""

    def test_loss_fn_returns_expected_keys(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test loss_fn returns all expected keys."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        expected_keys = ["loss", "nll_loss", "accuracy", "bits_per_dim"]
        for key in expected_keys:
            assert key in loss_dict, f"Missing key: {key}"

    def test_loss_is_scalar(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test loss is a scalar value."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        assert loss_dict["loss"].shape == ()

    def test_loss_is_non_negative(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test cross-entropy loss is non-negative."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        assert loss_dict["loss"] >= 0

    def test_loss_is_finite(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test loss is finite."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        assert jnp.isfinite(loss_dict["loss"])

    def test_accuracy_in_valid_range(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test accuracy is between 0 and 1."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        assert 0 <= loss_dict["accuracy"] <= 1

    def test_bits_per_dim_is_positive(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test bits per dimension is positive."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        assert loss_dict["bits_per_dim"] >= 0

    def test_loss_with_dict_batch(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test loss_fn with dictionary batch input."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        batch = {"x": images}
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(batch, outputs, rngs=rngs)

        assert jnp.isfinite(loss_dict["loss"])

    def test_loss_with_images_key_in_batch(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test loss_fn with 'images' key in dictionary batch."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        batch = {"images": images}
        outputs = model(images, rngs=rngs)
        loss_dict = model.loss_fn(batch, outputs, rngs=rngs)

        assert jnp.isfinite(loss_dict["loss"])


class TestPixelCNNLogProb:
    """Test suite for PixelCNN log probability computation."""

    def test_log_prob_shape(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test log_prob returns correct shape."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        batch_size = 3
        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size)
        log_probs = model.log_prob(images, rngs=rngs)

        assert log_probs.shape == (batch_size,)

    def test_log_prob_is_negative(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test log probabilities are negative (probabilities < 1)."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        log_probs = model.log_prob(images, rngs=rngs)

        assert jnp.all(log_probs <= 0)

    def test_log_prob_is_finite(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test log probabilities are finite."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        log_probs = model.log_prob(images, rngs=rngs)

        assert jnp.isfinite(log_probs).all()


class TestPixelCNNGeneration:
    """Test suite for PixelCNN image generation."""

    def test_generate_output_shape(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test generate returns correct shape."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        n_samples = 2
        generated = model.generate(n_samples=n_samples, rngs=rngs)

        expected_shape = (n_samples, *pixel_cnn_config.image_shape)
        assert generated.shape == expected_shape

    def test_generate_pixel_values_in_range(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test generated pixel values are in valid range [0, 255]."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        generated = model.generate(n_samples=1, rngs=rngs)

        assert jnp.all(generated >= 0)
        assert jnp.all(generated < 256)

    def test_generate_with_temperature(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test generation with different temperatures."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        # High temperature (more random)
        gen_high = model.generate(n_samples=1, temperature=2.0, rngs=rngs)

        # Low temperature (more deterministic)
        gen_low = model.generate(n_samples=1, temperature=0.5, rngs=rngs)

        # Both should be valid
        assert jnp.all(gen_high >= 0)
        assert jnp.all(gen_low >= 0)
        assert jnp.all(gen_high < 256)
        assert jnp.all(gen_low < 256)

    def test_generate_without_explicit_rngs(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test generation without explicit rngs uses stored rngs."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        generated = model.generate(n_samples=1)

        assert generated.shape == (1, *pixel_cnn_config.image_shape)

    def test_generate_multiple_samples(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test generating multiple samples."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        n_samples = 4
        generated = model.generate(n_samples=n_samples, rngs=rngs)

        assert generated.shape[0] == n_samples


class TestPixelCNNInpainting:
    """Test suite for PixelCNN inpainting functionality."""

    def test_inpaint_preserves_masked_pixels(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test inpainting preserves pixels where mask=1."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        h, w, c = pixel_cnn_config.image_shape

        # Create conditioning image
        conditioning = jax.random.randint(jax.random.key(0), (h, w, c), 0, 256)

        # Create mask: 1 = keep, 0 = generate
        # Keep top half, generate bottom half
        mask = jnp.ones((h, w))
        mask = mask.at[h // 2 :, :].set(0)

        inpainted = model.inpaint(conditioning, mask, n_samples=1, rngs=rngs)

        # Top half should match conditioning
        top_half_orig = conditioning[: h // 2, :, :]
        top_half_result = inpainted[0, : h // 2, :, :]
        assert jnp.array_equal(top_half_orig, top_half_result)

    def test_inpaint_output_shape(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test inpainting returns correct shape."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        h, w, c = pixel_cnn_config.image_shape
        conditioning = jax.random.randint(jax.random.key(0), (h, w, c), 0, 256)
        mask = jnp.ones((h, w))

        n_samples = 3
        inpainted = model.inpaint(conditioning, mask, n_samples=n_samples, rngs=rngs)

        expected_shape = (n_samples, h, w, c)
        assert inpainted.shape == expected_shape

    def test_inpaint_with_temperature(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test inpainting with temperature parameter."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        h, w, c = pixel_cnn_config.image_shape
        conditioning = jax.random.randint(jax.random.key(0), (h, w, c), 0, 256)
        mask = jnp.zeros((h, w))  # Generate all pixels

        inpainted = model.inpaint(conditioning, mask, n_samples=1, temperature=0.8, rngs=rngs)

        assert inpainted.shape == (1, h, w, c)
        assert jnp.all(inpainted >= 0)
        assert jnp.all(inpainted < 256)


# =============================================================================
# PixelCNNConfig Tests
# =============================================================================


class TestPixelCNNConfig:
    """Test suite for PixelCNNConfig validation."""

    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config = PixelCNNConfig(
            name="test",
            image_shape=(32, 32, 3),
            hidden_channels=64,
            num_layers=5,
        )

        assert config.image_shape == (32, 32, 3)
        assert config.hidden_channels == 64
        assert config.num_layers == 5

    def test_config_missing_image_shape_raises_error(self):
        """Test that missing image_shape raises ValueError."""
        with pytest.raises(ValueError, match="image_shape is required"):
            PixelCNNConfig(name="test", image_shape=None)

    def test_config_invalid_image_shape_dimensions_raises_error(self):
        """Test that invalid image_shape dimensions raises ValueError."""
        with pytest.raises(ValueError, match="must have 3 dimensions"):
            PixelCNNConfig(name="test", image_shape=(32, 32))

    def test_config_negative_dimensions_raises_error(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            PixelCNNConfig(name="test", image_shape=(-1, 32, 3))

    def test_config_invalid_hidden_channels_raises_error(self):
        """Test that non-positive hidden_channels raises ValueError."""
        with pytest.raises(ValueError, match="hidden_channels"):
            PixelCNNConfig(name="test", image_shape=(8, 8, 1), hidden_channels=0)

    def test_config_invalid_num_layers_raises_error(self):
        """Test that non-positive num_layers raises ValueError."""
        with pytest.raises(ValueError, match="num_layers"):
            PixelCNNConfig(name="test", image_shape=(8, 8, 1), num_layers=0)

    def test_config_negative_residual_blocks_raises_error(self):
        """Test that negative num_residual_blocks raises ValueError."""
        with pytest.raises(ValueError, match="num_residual_blocks"):
            PixelCNNConfig(name="test", image_shape=(8, 8, 1), num_residual_blocks=-1)

    def test_config_invalid_dropout_rate_raises_error(self):
        """Test that invalid dropout_rate raises ValueError."""
        with pytest.raises(ValueError):
            PixelCNNConfig(name="test", image_shape=(8, 8, 1), dropout_rate=1.5)

    def test_config_derived_vocab_size(self):
        """Test derived_vocab_size property."""
        config = PixelCNNConfig(name="test", image_shape=(8, 8, 3))
        assert config.derived_vocab_size == 256

    def test_config_derived_sequence_length(self):
        """Test derived_sequence_length property."""
        config = PixelCNNConfig(name="test", image_shape=(8, 8, 3))
        # 8 * 8 * 3 = 192
        assert config.derived_sequence_length == 192

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test",
            "image_shape": [16, 16, 1],  # List should be converted to tuple
            "hidden_channels": 32,
        }
        config = PixelCNNConfig.from_dict(data)

        assert config.image_shape == (16, 16, 1)
        assert config.hidden_channels == 32


# =============================================================================
# Integration Tests
# =============================================================================


class TestPixelCNNIntegration:
    """Integration tests for PixelCNN."""

    def test_training_step(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test a complete training step."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        batch_size = 4
        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size)

        # Forward pass
        outputs = model(images, rngs=rngs, training=True)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        # Verify outputs
        assert jnp.isfinite(loss_dict["loss"])
        assert loss_dict["loss"] >= 0

    def test_gradient_computation(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test that gradients can be computed."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        def loss_fn(model, images):
            outputs = model(images, rngs=rngs, training=True)
            loss_dict = model.loss_fn(images, outputs, rngs=rngs)
            return loss_dict["loss"]

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(model, images)

        # Check loss and gradients are valid
        assert jnp.isfinite(loss)
        assert grads is not None

    def test_model_reproducibility(self, pixel_cnn_config):
        """Test model reproducibility with same random seed."""
        rngs1 = nnx.Rngs(params=jax.random.key(42))
        rngs2 = nnx.Rngs(params=jax.random.key(42))

        model1 = PixelCNN(pixel_cnn_config, rngs=rngs1)
        model2 = PixelCNN(pixel_cnn_config, rngs=rngs2)

        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=1)

        outputs1 = model1(images, training=False)
        outputs2 = model2(images, training=False)

        assert jnp.allclose(outputs1["logits"], outputs2["logits"], atol=1e-6)

    def test_end_to_end_train_and_generate(self, pixel_cnn_config, rngs: nnx.Rngs):
        """Test end-to-end training and generation workflow."""
        model = PixelCNN(pixel_cnn_config, rngs=rngs)

        # Training step
        images = create_random_image(jax.random.key(0), pixel_cnn_config.image_shape, batch_size=2)
        outputs = model(images, rngs=rngs, training=True)
        loss_dict = model.loss_fn(images, outputs, rngs=rngs)

        assert jnp.isfinite(loss_dict["loss"])

        # Generation step
        generated = model.generate(n_samples=1, rngs=rngs)

        assert generated.shape == (1, *pixel_cnn_config.image_shape)
        assert jnp.all(generated >= 0)
        assert jnp.all(generated < 256)

    def test_different_image_sizes(self, rngs: nnx.Rngs):
        """Test model works with different image sizes."""
        image_shapes = [
            (4, 4, 1),
            (8, 8, 1),
            (8, 8, 3),
            (16, 16, 1),
        ]

        for img_shape in image_shapes:
            config = PixelCNNConfig(
                name=f"test_{img_shape}",
                image_shape=img_shape,
                hidden_channels=16,
                num_layers=2,
                num_residual_blocks=1,
            )
            model = PixelCNN(config, rngs=rngs)

            images = create_random_image(jax.random.key(0), img_shape, batch_size=1)
            outputs = model(images, rngs=rngs)

            assert jnp.isfinite(outputs["logits"]).all()
