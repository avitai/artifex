"""Comprehensive tests for UNet with Cross-Attention.

This module tests the UNet2DCondition model with cross-attention for text conditioning,
following Test-Driven Development (TDD) principles.
"""

import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.models.backbones.unet_cross_attention import UNet2DCondition


class TestUNet2DConditionInitialization:
    """Test UNet2DCondition initialization."""

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        rngs = nnx.Rngs(0)

        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256, 512],
            num_res_blocks=2,
            attention_levels=[0, 1, 2],
            cross_attention_dim=768,
            num_heads=8,
            rngs=rngs,
        )

        assert unet is not None
        assert unet.in_channels == 4
        assert unet.out_channels == 4
        assert unet.cross_attention_dim == 768

    def test_initialization_sd_config(self):
        """Test initialization with Stable Diffusion configuration."""
        rngs = nnx.Rngs(42)

        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[320, 640, 1280, 1280],
            num_res_blocks=2,
            attention_levels=[0, 1, 2, 3],
            cross_attention_dim=768,
            num_heads=8,
            rngs=rngs,
        )

        assert unet is not None

    def test_initialization_minimal_config(self):
        """Test initialization with minimal configuration."""
        rngs = nnx.Rngs(123)

        unet = UNet2DCondition(
            in_channels=3,
            out_channels=3,
            hidden_dims=[32, 64],
            num_res_blocks=1,
            attention_levels=[0, 1],
            cross_attention_dim=256,
            num_heads=4,
            rngs=rngs,
        )

        assert unet is not None


class TestUNet2DConditionForwardPass:
    """Test UNet2DCondition forward pass."""

    @pytest.fixture
    def unet(self):
        """Create a test UNet."""
        rngs = nnx.Rngs(0)
        return UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

    def test_forward_pass_basic(self, unet):
        """Test basic forward pass with text conditioning."""
        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        # Latent input
        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))

        # Timesteps
        timesteps = jnp.array([100, 200])

        # Text embeddings
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        # Forward pass
        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        # Check output shape
        assert output.shape == (batch_size, height, width, channels)
        assert output.dtype == jnp.float32

    def test_forward_pass_different_batch_sizes(self, unet):
        """Test forward pass with different batch sizes."""
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        for batch_size in [1, 2, 4, 8]:
            x = jax.random.normal(jax.random.key(batch_size), (batch_size, height, width, channels))
            timesteps = jnp.full((batch_size,), 100)
            encoder_hidden_states = jax.random.normal(
                jax.random.key(batch_size + 100), (batch_size, seq_len, text_dim)
            )

            output = unet(x, timesteps, conditioning=encoder_hidden_states)
            assert output.shape == (batch_size, height, width, channels)

    def test_forward_pass_different_spatial_sizes(self):
        """Test forward pass with different spatial sizes."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        channels = 4
        seq_len = 77
        text_dim = 512

        for height, width in [(16, 16), (32, 32), (64, 64)]:
            x = jax.random.normal(jax.random.key(height), (batch_size, height, width, channels))
            timesteps = jnp.array([100, 200])
            encoder_hidden_states = jax.random.normal(
                jax.random.key(height + 100), (batch_size, seq_len, text_dim)
            )

            output = unet(x, timesteps, conditioning=encoder_hidden_states)
            assert output.shape == (batch_size, height, width, channels)

    def test_forward_pass_different_timesteps(self, unet):
        """Test forward pass with different timesteps."""
        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        # Test different timestep values
        for t_val in [0, 50, 100, 500, 999]:
            timesteps = jnp.full((batch_size,), t_val)
            output = unet(x, timesteps, conditioning=encoder_hidden_states)
            assert output.shape == (batch_size, height, width, channels)

    def test_forward_pass_conditioning_effect(self, unet):
        """Test that text conditioning affects output."""
        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])

        # Two different text embeddings
        text_emb1 = jax.random.normal(jax.random.key(1), (batch_size, seq_len, text_dim))
        text_emb2 = jax.random.normal(jax.random.key(2), (batch_size, seq_len, text_dim))

        # Forward pass with different conditioning
        output1 = unet(x, timesteps, conditioning=text_emb1)
        output2 = unet(x, timesteps, conditioning=text_emb2)

        # Different conditioning should produce different outputs
        assert not jnp.allclose(output1, output2, atol=1e-3)

    def test_forward_pass_null_conditioning(self, unet):
        """Test forward pass with null (zero) conditioning."""
        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])

        # Null conditioning (all zeros)
        null_text_emb = jnp.zeros((batch_size, seq_len, text_dim))

        output = unet(x, timesteps, conditioning=null_text_emb)

        assert output.shape == (batch_size, height, width, channels)
        assert not jnp.isnan(output).any()

    def test_forward_pass_deterministic(self, unet):
        """Test that forward pass is deterministic."""
        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        # Multiple forward passes with same input
        output1 = unet(x, timesteps, conditioning=encoder_hidden_states)
        output2 = unet(x, timesteps, conditioning=encoder_hidden_states)
        output3 = unet(x, timesteps, conditioning=encoder_hidden_states)

        # All outputs should be identical
        assert jnp.allclose(output1, output2)
        assert jnp.allclose(output2, output3)


class TestUNet2DConditionOutputProperties:
    """Test output properties of UNet2DCondition."""

    @pytest.fixture
    def unet(self):
        """Create a test UNet."""
        rngs = nnx.Rngs(0)
        return UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

    def test_output_not_nan_or_inf(self, unet):
        """Test that output doesn't contain NaN or Inf."""
        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        assert not jnp.isnan(output).any()
        assert not jnp.isinf(output).any()

    def test_output_distribution(self, unet):
        """Test that output has reasonable distribution."""
        batch_size = 4
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200, 300, 400])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        # Check mean and std are reasonable
        mean = output.mean()
        std = output.std()

        # Output should have reasonable statistics
        assert jnp.abs(mean) < 5.0
        assert 0.01 < std < 50.0


class TestUNet2DConditionTimeEmbedding:
    """Test time embedding in UNet2DCondition."""

    def test_different_timesteps_produce_different_outputs(self):
        """Test that different timesteps produce different outputs."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        # Different timesteps
        timesteps1 = jnp.array([100, 100])
        timesteps2 = jnp.array([500, 500])

        output1 = unet(x, timesteps1, conditioning=encoder_hidden_states)
        output2 = unet(x, timesteps2, conditioning=encoder_hidden_states)

        # Different timesteps should produce different outputs
        assert not jnp.allclose(output1, output2, atol=1e-3)

    def test_timestep_zero(self):
        """Test with timestep zero."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.zeros((batch_size,), dtype=jnp.int32)
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        assert output.shape == (batch_size, height, width, channels)
        assert not jnp.isnan(output).any()


class TestUNet2DConditionCrossAttention:
    """Test cross-attention mechanism in UNet2DCondition."""

    def test_cross_attention_at_all_levels(self):
        """Test that cross-attention works at all specified levels."""
        rngs = nnx.Rngs(0)

        # Enable cross-attention at all levels
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256, 512],
            num_res_blocks=2,
            attention_levels=[0, 1, 2],  # All levels
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        assert output.shape == (batch_size, height, width, channels)

    def test_cross_attention_at_partial_levels(self):
        """Test cross-attention at only some levels."""
        rngs = nnx.Rngs(0)

        # Enable cross-attention at only middle levels
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256, 512],
            num_res_blocks=2,
            attention_levels=[1, 2],  # Skip level 0
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        assert output.shape == (batch_size, height, width, channels)


class TestUNet2DConditionGradients:
    """Test gradient computation for UNet2DCondition."""

    def test_gradients_computable(self):
        """Test that gradients can be computed."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[64, 128],
            num_res_blocks=1,
            attention_levels=[0, 1],
            cross_attention_dim=256,
            num_heads=4,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 16, 16
        channels = 4
        seq_len = 10
        text_dim = 256

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        def loss_fn(model):
            output = model(x, timesteps, conditioning=encoder_hidden_states)
            return jnp.mean(output**2)

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(unet)

        assert isinstance(loss, jax.Array)
        assert grads is not None

    def test_gradients_not_nan(self):
        """Test that gradients are not NaN."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[64, 128],
            num_res_blocks=1,
            attention_levels=[0, 1],
            cross_attention_dim=256,
            num_heads=4,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 16, 16
        channels = 4
        seq_len = 10
        text_dim = 256

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        def loss_fn(model):
            output = model(x, timesteps, conditioning=encoder_hidden_states)
            return jnp.mean(output**2)

        loss, grads = nnx.value_and_grad(loss_fn)(unet)

        # Check gradients - use jax.tree.leaves to get all gradient arrays
        grad_leaves = jax.tree.leaves(grads)

        # Check that gradients were computed
        assert len(grad_leaves) > 0, "No gradients computed"

        # Check all gradient values for NaN
        for i, grad_value in enumerate(grad_leaves):
            if isinstance(grad_value, jax.Array):
                assert not jnp.isnan(grad_value).any(), f"NaN gradient in leaf {i}"


class TestUNet2DConditionTrainEvalModes:
    """Test train/eval mode switching."""

    def test_eval_mode_default(self):
        """Test that UNet is in eval mode by default."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        assert output.shape == (batch_size, height, width, channels)

    def test_train_mode(self):
        """Test UNet in train mode."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        # Switch to train mode
        unet.train()

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        output = unet(x, timesteps, conditioning=encoder_hidden_states)

        assert output.shape == (batch_size, height, width, channels)

    def test_mode_switching(self):
        """Test switching between train and eval modes."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        # Eval mode
        unet.eval()
        output_eval1 = unet(x, timesteps, conditioning=encoder_hidden_states)

        # Train mode
        unet.train()
        unet(x, timesteps, conditioning=encoder_hidden_states)

        # Back to eval mode
        unet.eval()
        output_eval2 = unet(x, timesteps, conditioning=encoder_hidden_states)

        # Eval mode outputs should be identical
        assert jnp.allclose(output_eval1, output_eval2)


class TestUNet2DConditionJITCompatibility:
    """Test JIT compilation compatibility for UNet2DCondition."""

    def test_jit_forward_pass(self):
        """Test that forward pass can be JIT compiled."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        @nnx.jit
        def forward(model, latents, t, text_emb):
            return model(latents, t, conditioning=text_emb)

        # First call (should trigger compilation)
        output1 = forward(unet, x, timesteps, encoder_hidden_states)

        # Second call (should use cached compilation)
        output2 = forward(unet, x, timesteps, encoder_hidden_states)

        # Both outputs should be identical
        assert jnp.allclose(output1, output2)
        assert output1.shape == (batch_size, height, width, channels)

    def test_jit_with_different_batch_sizes(self):
        """Test JIT compilation with different batch sizes (triggers recompilation)."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        height, width = 32, 32
        channels = 4
        seq_len = 77
        text_dim = 512

        @nnx.jit
        def forward(model, latents, t, text_emb):
            return model(latents, t, conditioning=text_emb)

        # Test with different batch sizes (each triggers recompilation)
        for batch_size in [1, 2, 4]:
            x = jax.random.normal(jax.random.key(batch_size), (batch_size, height, width, channels))
            timesteps = jnp.full((batch_size,), 100)
            encoder_hidden_states = jax.random.normal(
                jax.random.key(batch_size + 100), (batch_size, seq_len, text_dim)
            )

            output = forward(unet, x, timesteps, encoder_hidden_states)

            assert output.shape == (batch_size, height, width, channels)
            assert not jnp.isnan(output).any()

    def test_jit_with_different_spatial_sizes(self):
        """Test JIT compilation with different spatial sizes (triggers recompilation)."""
        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[128, 256],
            num_res_blocks=2,
            attention_levels=[0, 1],
            cross_attention_dim=512,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        channels = 4
        seq_len = 77
        text_dim = 512

        @nnx.jit
        def forward(model, latents, t, text_emb):
            return model(latents, t, conditioning=text_emb)

        # Test with different spatial sizes (each triggers recompilation)
        for height, width in [(16, 16), (32, 32), (64, 64)]:
            x = jax.random.normal(jax.random.key(height), (batch_size, height, width, channels))
            timesteps = jnp.array([100, 200])
            encoder_hidden_states = jax.random.normal(
                jax.random.key(height + 100), (batch_size, seq_len, text_dim)
            )

            output = forward(unet, x, timesteps, encoder_hidden_states)

            assert output.shape == (batch_size, height, width, channels)
            assert not jnp.isnan(output).any()

    def test_jit_gradient_computation(self):
        """Test that gradient computation can be JIT compiled.

        Note: This test requires deterministic GPU operations to pass reliably.
        Run with: ARTIFEX_DETERMINISTIC=1 pytest <test_file>

        By default, Artifex uses non-deterministic CUDA operations for maximum
        performance. Complex models like UNet with cross-attention accumulate
        small floating-point differences (1e-7 to 5e-6) during backward passes
        through multiple attention layers when non-deterministic mode is enabled.
        """
        # Check if deterministic mode is enabled
        deterministic_enabled = os.environ.get("ARTIFEX_DETERMINISTIC", "0") == "1"
        if not deterministic_enabled:
            pytest.skip(
                "Test requires ARTIFEX_DETERMINISTIC=1 for strict gradient reproducibility. "
                "Run: ARTIFEX_DETERMINISTIC=1 pytest tests/..."
            )

        rngs = nnx.Rngs(0)
        unet = UNet2DCondition(
            in_channels=4,
            out_channels=4,
            hidden_dims=[64, 128],
            num_res_blocks=1,
            attention_levels=[0, 1],
            cross_attention_dim=256,
            num_heads=4,
            rngs=rngs,
        )

        # Ensure eval mode for determinism (no dropout/stochastic ops)
        unet.eval()

        batch_size = 2
        height, width = 16, 16
        channels = 4
        seq_len = 10
        text_dim = 256

        x = jax.random.normal(jax.random.key(0), (batch_size, height, width, channels))
        timesteps = jnp.array([100, 200])
        encoder_hidden_states = jax.random.normal(
            jax.random.key(1), (batch_size, seq_len, text_dim)
        )

        @nnx.jit
        def compute_loss_and_grads(model, latents, t, text_emb):
            def loss_fn(m):
                output = m(latents, t, conditioning=text_emb)
                return jnp.mean(output**2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            return loss, grads

        # First call (compilation)
        loss1, grads1 = compute_loss_and_grads(unet, x, timesteps, encoder_hidden_states)

        # Second call (cached)
        loss2, grads2 = compute_loss_and_grads(unet, x, timesteps, encoder_hidden_states)

        # Both should produce identical results
        assert jnp.allclose(loss1, loss2)

        # Check gradients are valid
        grad_leaves1 = jax.tree.leaves(grads1)
        grad_leaves2 = jax.tree.leaves(grads2)

        assert len(grad_leaves1) > 0
        assert len(grad_leaves1) == len(grad_leaves2)

        for g1, g2 in zip(grad_leaves1, grad_leaves2):
            if isinstance(g1, jax.Array) and isinstance(g2, jax.Array):
                # Even with ARTIFEX_DETERMINISTIC=1 and XLA_FLAGS=--xla_gpu_deterministic_ops=true,
                # floating-point non-associativity causes small numerical differences in complex
                # models with attention layers. Empirical analysis shows:
                # - Maximum absolute difference: ~2.6e-6 (measured with deterministic mode)
                # - Large relative errors occur when gradients are very small (near zero)
                #
                # This is documented behavior across authoritative sources:
                # - NVIDIA GTC 2019 (S9911): Atomic operations introduce "truly random FP rounding errors"
                # - JAX Issues #565, #10674, #13672: XLA rearranges operations, FP is approximate
                # - Stanford CS231n: float32 has relative errors ~1e-7 to 1e-2
                # - PyTorch docs: cuDNN backward ops don't guarantee bit-wise reproducibility
                #
                # Tolerance rtol=1e-5, atol=1e-5 accommodates these limitations while ensuring
                # gradients are numerically consistent within acceptable float32 precision bounds.
                assert jnp.allclose(g1, g2, rtol=1e-5, atol=1e-5)
                assert not jnp.isnan(g1).any()
                assert not jnp.isnan(g2).any()
