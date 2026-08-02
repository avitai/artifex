"""Integration tests placing flash attention inside real model structures.

Element-wise parity with :class:`flax.nnx.MultiHeadAttention` is asserted in
``test_flash_attention.py``; this file covers what parity alone does not, namely
that the module composes correctly inside transformer blocks and encoder-decoder
stacks, and that it survives ``grad``, ``jit`` and ``vmap``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.nnx.nn.attention import MultiHeadAttention as FlaxMultiHeadAttention

from artifex.generative_models.core.layers.flash_attention import FlashMultiHeadAttention


DIM = 128
NUM_HEADS = 8


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Seeded RNG collection."""
    return nnx.Rngs(42)


def copy_attention_weights(source: nnx.Module, target: nnx.Module) -> None:
    """Copy every projection weight from one attention module to another."""
    for name in ("query", "key", "value", "out"):
        source_layer = getattr(source, name)
        target_layer = getattr(target, name)
        target_layer.kernel[...] = source_layer.kernel[...]
        if getattr(source_layer, "bias", None) is not None:
            target_layer.bias[...] = source_layer.bias[...]


class TransformerBlock(nnx.Module):
    """A pre-norm transformer block parameterised by its attention class."""

    def __init__(self, attention_cls, dim: int, num_heads: int, rngs: nnx.Rngs) -> None:
        """Build the block."""
        super().__init__()
        self.attention = attention_cls(
            num_heads=num_heads, in_features=dim, dropout_rate=0.0, decode=False, rngs=rngs
        )
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(dim, dim * 4, rngs=rngs),
            nnx.gelu,
            nnx.Linear(dim * 4, dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:
        """Apply attention and the feed-forward network with residuals."""
        x = x + self.attention(self.norm1(x), deterministic=deterministic)
        return x + self.ffn(self.norm2(x))


class EncoderDecoder(nnx.Module):
    """Encoder self-attention, decoder self-attention and cross-attention."""

    def __init__(self, attention_cls, dim: int, num_heads: int, rngs: nnx.Rngs) -> None:
        """Build the three attention stacks."""
        super().__init__()
        build = lambda: attention_cls(  # noqa: E731
            num_heads=num_heads, in_features=dim, decode=False, rngs=rngs
        )
        self.encoder_attn = build()
        self.decoder_attn = build()
        self.cross_attn = build()

    def __call__(
        self, encoder_input: jax.Array, decoder_input: jax.Array, *, deterministic: bool = True
    ) -> jax.Array:
        """Encode, decode, then attend from the decoder to the encoder."""
        encoded = self.encoder_attn(encoder_input, deterministic=deterministic)
        decoded = self.decoder_attn(decoder_input, deterministic=deterministic)
        return self.cross_attn(decoded, encoded, encoded, deterministic=deterministic)


class TestDropInReplacement:
    """Swapping the attention class must not change a model's output."""

    def test_transformer_block_replacement(self, rngs: nnx.Rngs) -> None:
        """A transformer block must be unchanged by the swap.

        The portable path delegates to the same nnx kernel, so the outputs agree
        to floating-point noise rather than to the loose 1e-3 the previous
        implementation needed.
        """
        flax_block = TransformerBlock(FlaxMultiHeadAttention, DIM, NUM_HEADS, rngs)
        flash_block = TransformerBlock(FlashMultiHeadAttention, DIM, NUM_HEADS, rngs)

        copy_attention_weights(flax_block.attention, flash_block.attention)
        for name in ("norm1", "norm2"):
            getattr(flash_block, name).scale[...] = getattr(flax_block, name).scale[...]
            getattr(flash_block, name).bias[...] = getattr(flax_block, name).bias[...]
        for flash_layer, flax_layer in zip(flash_block.ffn.layers, flax_block.ffn.layers):
            if isinstance(flash_layer, nnx.Linear) and isinstance(flax_layer, nnx.Linear):
                flash_layer.kernel[...] = flax_layer.kernel[...]
                if flash_layer.bias is not None and flax_layer.bias is not None:
                    flash_layer.bias[...] = flax_layer.bias[...]

        x = jax.random.normal(rngs(), (2, 64, DIM))
        np.testing.assert_allclose(flax_block(x), flash_block(x), rtol=1e-6, atol=1e-6)

    def test_encoder_decoder_replacement(self, rngs: nnx.Rngs) -> None:
        """An encoder-decoder stack must be unchanged by the swap."""
        flax_model = EncoderDecoder(FlaxMultiHeadAttention, DIM, NUM_HEADS, rngs)
        flash_model = EncoderDecoder(FlashMultiHeadAttention, DIM, NUM_HEADS, rngs)
        for name in ("encoder_attn", "decoder_attn", "cross_attn"):
            copy_attention_weights(getattr(flax_model, name), getattr(flash_model, name))

        encoder_input = jax.random.normal(rngs(), (2, 32, DIM))
        decoder_input = jax.random.normal(rngs(), (2, 16, DIM))
        np.testing.assert_allclose(
            flax_model(encoder_input, decoder_input),
            flash_model(encoder_input, decoder_input),
            rtol=1e-6,
            atol=1e-6,
        )


class TestTransformations:
    """The module must survive the JAX transformations models rely on."""

    def make_module(self, rngs: nnx.Rngs, **kwargs) -> FlashMultiHeadAttention:
        """Build a module for transformation tests."""
        return FlashMultiHeadAttention(
            num_heads=NUM_HEADS, in_features=DIM, decode=False, rngs=rngs, **kwargs
        )

    def test_gradients_match_reference(self, rngs: nnx.Rngs) -> None:
        """Gradients must match those of the reference module."""
        flax_module = FlaxMultiHeadAttention(
            num_heads=NUM_HEADS, in_features=DIM, decode=False, rngs=rngs
        )
        flash_module = self.make_module(rngs)
        copy_attention_weights(flax_module, flash_module)
        x = jax.random.normal(rngs(), (2, 16, DIM))

        def loss(module: nnx.Module) -> jax.Array:
            return jnp.sum(module(x) ** 2)

        flax_grads = nnx.grad(loss)(flax_module)
        flash_grads = nnx.grad(loss)(flash_module)
        np.testing.assert_allclose(
            flax_grads["query"]["kernel"].value,
            flash_grads["query"]["kernel"].value,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_gradients_are_finite(self, rngs: nnx.Rngs) -> None:
        """Gradients must not contain NaN or infinity."""
        module = self.make_module(rngs)
        x = jax.random.normal(rngs(), (2, 16, DIM))
        grads = nnx.grad(lambda m: jnp.sum(m(x) ** 2))(module)
        assert jnp.all(jnp.isfinite(grads["query"]["kernel"].value))

    def test_jit_is_stable(self, rngs: nnx.Rngs) -> None:
        """Repeated jitted calls must agree, and match the eager result."""
        module = self.make_module(rngs)
        x = jax.random.normal(rngs(), (2, 64, DIM))
        forward = nnx.jit(lambda m, inputs: m(inputs))
        assert jnp.allclose(forward(module, x), forward(module, x))
        assert jnp.allclose(forward(module, x), module(x), atol=1e-6)

    def test_vmap_preserves_shape(self, rngs: nnx.Rngs) -> None:
        """The module must vectorise over an extra leading axis."""
        module = self.make_module(rngs)
        x = jax.random.normal(rngs(), (8, 64, DIM))
        vmapped = jax.vmap(lambda sample: module(sample[None, ...])[0])
        assert vmapped(x).shape == (8, 64, DIM)

    def test_training_step_produces_scalar_loss(self, rngs: nnx.Rngs) -> None:
        """A realistic training step must run with dropout live."""

        class SimpleModel(nnx.Module):
            def __init__(self, rngs: nnx.Rngs) -> None:
                super().__init__()
                self.attention = FlashMultiHeadAttention(
                    num_heads=NUM_HEADS,
                    in_features=DIM,
                    dropout_rate=0.1,
                    decode=False,
                    rngs=rngs,
                )
                self.output_proj = nnx.Linear(DIM, 10, rngs=rngs)

            def __call__(self, x: jax.Array, *, training: bool = True) -> jax.Array:
                x = self.attention(x, deterministic=not training)
                return self.output_proj(jnp.mean(x, axis=1))

        model = SimpleModel(rngs)
        x = jax.random.normal(rngs(), (4, 32, DIM))
        y = jax.random.normal(rngs(), (4, 10))
        loss, grads = nnx.value_and_grad(lambda m: jnp.mean((m(x) - y) ** 2))(model)

        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(grads["output_proj"]["kernel"].value))
