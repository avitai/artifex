"""Tests for Diffusion Transformer (DiT) implementation."""

import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.configuration import (
    DiTConfig,
    NoiseScheduleConfig,
)
from artifex.generative_models.models.backbones.dit import (
    DiffusionTransformer,
    DiTBlock,
    FinalLayer,
    get_2d_sincos_pos_embed,
    LabelEmbedder,
    modulate,
    TimestepEmbedder,
)
from artifex.generative_models.models.diffusion.dit import DiTModel


@pytest.fixture
def rngs():
    """Standard fixture for random number generators."""
    return nnx.Rngs(42)


class TestDiTComponents:
    """Test individual DiT components."""

    def test_positional_embeddings(self):
        """Test 2D sincos positional embeddings."""
        # Test square grid
        pos_embed = get_2d_sincos_pos_embed(256, 16)
        assert pos_embed.shape == (16 * 16, 256)

        # Test non-square grid
        pos_embed = get_2d_sincos_pos_embed(256, (8, 16))
        assert pos_embed.shape == (8 * 16, 256)

        # Test with class token
        pos_embed = get_2d_sincos_pos_embed(256, 16, add_cls_token=True)
        assert pos_embed.shape == (1 + 16 * 16, 256)

        # Check values are in reasonable range
        assert jnp.abs(pos_embed).max() <= 1.0

    def test_modulate_function(self):
        """Test adaptive layer norm modulation."""
        x = jnp.ones((2, 10, 256))
        shift = jnp.ones((2, 10, 256)) * 0.1
        scale = jnp.ones((2, 10, 256)) * 0.2

        result = modulate(x, shift, scale)
        expected = x * (1 + scale) + shift
        assert jnp.allclose(result, expected)

    def test_timestep_embedder(self, rngs):
        """Test timestep embedding generation."""
        embedder = TimestepEmbedder(hidden_size=256, rngs=rngs)

        # Test with different timesteps
        t = jnp.array([0, 100, 500, 999])
        emb = embedder(t)

        assert emb.shape == (4, 256)
        # Check embeddings are different for different timesteps
        assert not jnp.allclose(emb[0], emb[1])

    def test_label_embedder(self, rngs):
        """Test label embedding with dropout."""
        num_classes = 10
        embedder = LabelEmbedder(
            num_classes=num_classes, hidden_size=256, dropout_rate=0.1, rngs=rngs
        )

        labels = jnp.array([0, 5, 9])

        # Test without dropout
        emb = embedder(labels, deterministic=True)
        assert emb.shape == (3, 256)

        # Test with forced dropout
        force_drop = jnp.array([True, False, True])
        emb_dropped = embedder(labels, force_drop_ids=force_drop)

        # Check that dropped labels use unconditional embedding
        uncond_emb = embedder(jnp.array([num_classes]))
        assert jnp.allclose(emb_dropped[0], uncond_emb[0])
        assert not jnp.allclose(emb_dropped[1], uncond_emb[0])

    def test_dit_block(self, rngs):
        """Test DiT transformer block."""
        block = DiTBlock(hidden_size=256, num_heads=8, mlp_ratio=4.0, dropout_rate=0.1, rngs=rngs)

        batch_size = 2
        seq_len = 16
        x = jnp.ones((batch_size, seq_len, 256))
        c = jnp.ones((batch_size, 256))

        # Test forward pass
        output = block(x, c, deterministic=True)
        assert output.shape == x.shape

        # Test that output changes with different conditioning
        c2 = jnp.zeros((batch_size, 256))
        output2 = block(x, c2, deterministic=True)
        assert not jnp.allclose(output, output2)

    def test_final_layer(self, rngs):
        """Test final output layer."""
        final = FinalLayer(hidden_size=256, patch_size=2, out_channels=3, rngs=rngs)

        batch_size = 2
        num_patches = 16
        x = jnp.ones((batch_size, num_patches, 256))
        c = jnp.ones((batch_size, 256))

        output = final(x, c)
        assert output.shape == (batch_size, num_patches, 2 * 2 * 3)


class TestDiffusionTransformer:
    """Test complete Diffusion Transformer backbone."""

    def test_initialization(self, rngs):
        """Test DiT initialization with different configurations."""
        # Basic initialization
        dit = DiffusionTransformer(
            img_size=32,
            patch_size=4,
            in_channels=3,
            hidden_size=256,
            depth=6,
            num_heads=8,
            rngs=rngs,
        )

        assert dit.num_patches == (32 // 4) ** 2
        assert len(dit.blocks) == 6

    def test_forward_pass_unconditional(self, rngs):
        """Test forward pass without class conditioning."""
        dit = DiffusionTransformer(
            img_size=32,
            patch_size=4,
            in_channels=3,
            hidden_size=256,
            depth=4,
            num_heads=8,
            rngs=rngs,
        )

        batch_size = 2
        x = jnp.ones((batch_size, 32, 32, 3))
        t = jnp.array([100, 500])

        output = dit(x, t, deterministic=True)
        assert output.shape == (batch_size, 32, 32, 3)

    def test_forward_pass_conditional(self, rngs):
        """Test forward pass with class conditioning."""
        dit = DiffusionTransformer(
            img_size=32,
            patch_size=4,
            in_channels=3,
            hidden_size=256,
            depth=4,
            num_heads=8,
            num_classes=10,
            rngs=rngs,
        )

        batch_size = 2
        x = jnp.ones((batch_size, 32, 32, 3))
        t = jnp.array([100, 500])
        y = jnp.array([2, 7])

        output = dit(x, t, y, deterministic=True)
        assert output.shape == (batch_size, 32, 32, 3)

    def test_learn_sigma(self, rngs):
        """Test DiT with learned variance."""
        dit = DiffusionTransformer(
            img_size=16,
            patch_size=2,
            in_channels=3,
            hidden_size=128,
            depth=2,
            num_heads=4,
            learn_sigma=True,
            rngs=rngs,
        )

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([100, 500])

        output = dit(x, t, deterministic=True)
        # With learn_sigma, output should have 2x channels
        assert output.shape == (batch_size, 16, 16, 6)

    def test_unpatchify(self, rngs):
        """Test unpatchify operation."""
        dit = DiffusionTransformer(img_size=32, patch_size=4, rngs=rngs)

        batch_size = 2
        num_patches = (32 // 4) ** 2
        # Create patches [batch, num_patches, patch_size^2 * channels]
        patches = jnp.ones((batch_size, num_patches, 4 * 4 * 3))

        images = dit.unpatchify(patches)
        assert images.shape == (batch_size, 32, 32, 3)


class TestDiTModel:
    """Test complete DiT diffusion model."""

    def test_dit_model_initialization(self, rngs):
        """Test DiTModel initialization."""
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule",
            num_timesteps=1000,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        config = DiTConfig(
            name="test_dit",
            noise_schedule=noise_schedule,
            input_shape=(3, 32, 32),  # (C, H, W) format
            patch_size=4,
            hidden_size=256,
            depth=4,
            num_heads=8,
        )

        model = DiTModel(config, rngs=rngs)
        assert model.backbone is not None
        assert hasattr(model, "num_classes")
        assert hasattr(model, "cfg_scale")

    def test_dit_model_forward(self, rngs):
        """Test DiTModel forward pass."""
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_forward",
            num_timesteps=1000,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        config = DiTConfig(
            name="test_dit_forward",
            noise_schedule=noise_schedule,
            input_shape=(3, 16, 16),  # (C, H, W) format
            patch_size=2,
            hidden_size=128,
            depth=2,
            num_heads=4,
        )

        model = DiTModel(config, rngs=rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([100, 500])

        output = model(x, t, deterministic=True)
        assert output.shape == (batch_size, 16, 16, 3)

    def test_dit_model_with_cfg(self, rngs):
        """Test DiTModel with classifier-free guidance."""
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_cfg",
            num_timesteps=1000,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        config = DiTConfig(
            name="test_dit_cfg",
            noise_schedule=noise_schedule,
            input_shape=(3, 16, 16),  # (C, H, W) format
            patch_size=2,
            hidden_size=128,
            depth=2,
            num_heads=4,
            num_classes=10,
            cfg_scale=2.0,
        )

        model = DiTModel(config, rngs=rngs)

        batch_size = 2
        x = jnp.ones((batch_size, 16, 16, 3))
        t = jnp.array([100, 500])
        y = jnp.array([2, 7])

        output = model(x, t, y, deterministic=True, cfg_scale=3.0)
        assert output.shape == (batch_size, 16, 16, 3)

    def test_dit_model_generate(self, rngs):
        """Test DiTModel generation."""
        noise_schedule = NoiseScheduleConfig(
            name="test_schedule_generate",
            num_timesteps=10,
            schedule_type="linear",
            beta_start=0.0001,
            beta_end=0.02,
        )
        config = DiTConfig(
            name="test_dit_generate",
            noise_schedule=noise_schedule,
            input_shape=(3, 8, 8),  # (C, H, W) format
            patch_size=2,
            hidden_size=64,
            depth=1,
            num_heads=4,
        )

        model = DiTModel(config, rngs=rngs)

        samples = model.generate(n_samples=2, rngs=rngs, num_steps=10, img_size=8)

        assert samples.shape == (2, 8, 8, 3)
