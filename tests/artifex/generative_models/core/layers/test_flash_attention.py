"""Tests for the flash attention layer.

These tests assert that ``FlashMultiHeadAttention`` is *numerically correct*, not
merely that it runs. The previous suite exercised every public argument without
ever comparing against a reference, which is why a dead Triton kernel, an
unimplemented ``broadcast_dropout``, an ignored ``sow_weights``, dropout applied
to the wrong tensor, and a decode cache that could not be traced all passed.

The reference is ``nnx.MultiHeadAttention``: the module advertises itself as a
drop-in replacement for it, so parity with it is the contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.layers.attention_backend import (
    AttentionBackend,
    is_cudnn_eligible,
    select_attention_backend,
)
from artifex.generative_models.core.layers.flash_attention import (
    flash_dot_product_attention,
    FlashMultiHeadAttention,
)


NUM_HEADS = 4
IN_FEATURES = 32
HEAD_DIM = IN_FEATURES // NUM_HEADS
BATCH = 2
SEQ_LEN = 8


def make_inputs(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    features: int = IN_FEATURES,
    dtype: jnp.dtype = jnp.float32,
    seed: int = 0,
) -> jax.Array:
    """Build a deterministic input batch."""
    return jax.random.normal(jax.random.key(seed), (batch, seq_len, features), dtype=dtype)


def make_pair(**kwargs) -> tuple[FlashMultiHeadAttention, nnx.MultiHeadAttention]:
    """Build a flash module and an nnx module with identical parameters.

    Both are seeded identically and create their sublayers in the same order, so
    their parameters are equal element-wise.
    """
    common = {
        "num_heads": NUM_HEADS,
        "in_features": IN_FEATURES,
        "decode": False,
        **kwargs,
    }
    flash = FlashMultiHeadAttention(**common, rngs=nnx.Rngs(0))
    reference = nnx.MultiHeadAttention(**common, rngs=nnx.Rngs(0))
    return flash, reference


class TestParityWithNNX:
    """The module must match ``nnx.MultiHeadAttention`` element-wise."""

    def test_parameters_are_identical(self) -> None:
        """Identical seeds must produce identical parameters."""
        flash, reference = make_pair()
        assert jnp.array_equal(flash.query.kernel[...], reference.query.kernel[...])
        assert jnp.array_equal(flash.out.kernel[...], reference.out.kernel[...])

    def test_self_attention_output_matches(self) -> None:
        """Self-attention output must match the reference."""
        flash, reference = make_pair()
        x = make_inputs()
        assert jnp.allclose(flash(x), reference(x), atol=1e-6)

    def test_cross_attention_output_matches(self) -> None:
        """Cross-attention output must match the reference."""
        flash, reference = make_pair()
        q = make_inputs(seed=1)
        kv = make_inputs(seq_len=SEQ_LEN * 2, seed=2)
        assert jnp.allclose(flash(q, kv), reference(q, kv), atol=1e-6)

    def test_masked_output_matches(self) -> None:
        """A boolean mask must be honoured exactly as the reference honours it."""
        flash, reference = make_pair()
        x = make_inputs()
        mask = nnx.make_causal_mask(jnp.ones((BATCH, SEQ_LEN)))
        assert jnp.allclose(flash(x, mask=mask), reference(x, mask=mask), atol=1e-6)

    def test_normalize_qk_matches(self) -> None:
        """QK normalisation must match the reference."""
        flash, reference = make_pair(normalize_qk=True)
        x = make_inputs()
        assert jnp.allclose(flash(x), reference(x), atol=1e-6)

    def test_wider_qkv_features_match(self) -> None:
        """A projection wider than the input must match the reference."""
        flash, reference = make_pair(qkv_features=IN_FEATURES * 2)
        x = make_inputs()
        assert jnp.allclose(flash(x), reference(x), atol=1e-6)


class TestDropout:
    """Dropout must be applied to the attention weights, as nnx does."""

    def test_dropout_matches_reference(self) -> None:
        """Live dropout must match nnx given the same dropout key.

        The previous implementation applied dropout to the attention *output*
        rather than the weights, so this comparison failed by construction.
        """
        flash, reference = make_pair(dropout_rate=0.5)
        x = make_inputs()
        flash_out = flash(x, deterministic=False, rngs=nnx.Rngs(7))
        reference_out = reference(x, deterministic=False, rngs=nnx.Rngs(7))
        assert jnp.allclose(flash_out, reference_out, atol=1e-6)

    def test_dropout_changes_output(self) -> None:
        """Live dropout must actually perturb the output."""
        flash, _ = make_pair(dropout_rate=0.5)
        x = make_inputs()
        wet = flash(x, deterministic=False, rngs=nnx.Rngs(7))
        dry = flash(x, deterministic=True)
        assert not jnp.allclose(wet, dry, atol=1e-6)

    def test_broadcast_dropout_is_honoured(self) -> None:
        """``broadcast_dropout`` must change behaviour.

        It was stored and never read, so True and False gave identical output.
        """
        x = make_inputs()
        broadcast, _ = make_pair(dropout_rate=0.5, broadcast_dropout=True)
        per_element, _ = make_pair(dropout_rate=0.5, broadcast_dropout=False)
        broadcast_out = broadcast(x, deterministic=False, rngs=nnx.Rngs(7))
        per_element_out = per_element(x, deterministic=False, rngs=nnx.Rngs(7))
        assert not jnp.allclose(broadcast_out, per_element_out, atol=1e-6)

    def test_deterministic_ignores_dropout(self) -> None:
        """Deterministic mode must be reproducible."""
        flash, _ = make_pair(dropout_rate=0.5)
        x = make_inputs()
        assert jnp.array_equal(flash(x, deterministic=True), flash(x, deterministic=True))


class TestSowWeights:
    """``sow_weights`` must actually sow, and sow the pre-dropout weights."""

    def test_sow_weights_yields_one_intermediate(self) -> None:
        """The previous implementation yielded zero intermediates."""
        flash, reference = make_pair()
        x = make_inputs()
        flash(x, sow_weights=True)
        reference(x, sow_weights=True)
        flash_sown = nnx.pop(flash, nnx.Intermediate)
        reference_sown = nnx.pop(reference, nnx.Intermediate)
        assert len(jax.tree.leaves(flash_sown)) == len(jax.tree.leaves(reference_sown)) == 1

    def test_sown_weights_match_reference(self) -> None:
        """Sown weights must equal the reference's sown weights."""
        flash, reference = make_pair()
        x = make_inputs()
        flash(x, sow_weights=True)
        reference(x, sow_weights=True)
        flash_weights = jax.tree.leaves(nnx.pop(flash, nnx.Intermediate))[0]
        reference_weights = jax.tree.leaves(nnx.pop(reference, nnx.Intermediate))[0]
        assert jnp.allclose(flash_weights, reference_weights, atol=1e-6)

    def test_sown_weights_are_a_probability_distribution(self) -> None:
        """Sown weights are post-softmax and pre-dropout, so rows sum to one."""
        flash, _ = make_pair()
        flash(make_inputs(), sow_weights=True)
        weights = jax.tree.leaves(nnx.pop(flash, nnx.Intermediate))[0]
        assert jnp.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)

    def test_no_sowing_by_default(self) -> None:
        """Weights must not be sown unless asked for."""
        flash, _ = make_pair()
        flash(make_inputs())
        assert len(jax.tree.leaves(nnx.pop(flash, nnx.Intermediate))) == 0


class TestDecode:
    """Autoregressive decoding must be correct and traceable."""

    def test_decode_matches_full_causal_attention(self) -> None:
        """Token-by-token decoding must equal one causal forward pass."""
        flash, _ = make_pair()
        x = make_inputs()
        mask = nnx.make_causal_mask(jnp.ones((BATCH, SEQ_LEN)))
        expected = flash(x, mask=mask)

        flash.init_cache(x.shape)
        flash.decode = True
        decoded = jnp.stack(
            [flash(x[:, step : step + 1, :])[:, 0, :] for step in range(SEQ_LEN)], axis=1
        )
        assert jnp.allclose(decoded, expected, atol=1e-5)

    def test_decode_is_jittable(self) -> None:
        """Decoding must survive ``jax.jit``.

        ``_update_cache`` previously returned ``cached_key[:, : cur_index + 1]``,
        a traced-length slice, which raised ``IndexError`` under jit.
        """
        flash, _ = make_pair()
        x = make_inputs()
        flash.init_cache(x.shape)
        flash.decode = True

        @nnx.jit
        def decode_step(module: FlashMultiHeadAttention, token: jax.Array) -> jax.Array:
            return module(token)

        out = decode_step(flash, x[:, 0:1, :])
        assert out.shape == (BATCH, 1, IN_FEATURES)

    def test_decode_without_cache_raises(self) -> None:
        """Decoding before ``init_cache`` must fail loudly."""
        flash, _ = make_pair()
        flash.decode = True
        with pytest.raises((ValueError, AttributeError)):
            flash(make_inputs(seq_len=1))

    def test_causal_decode_matches_full_causal_attention(self) -> None:
        """A causal module must decode correctly.

        During decode a single query attends to the whole cache, so applying
        ``is_causal`` on top of the cache mask would build a ``tril`` of shape
        ``(1, max_length)`` and confine the token to position zero. Causality
        must come from the cache mask alone.
        """
        causal = FlashMultiHeadAttention(
            num_heads=NUM_HEADS,
            in_features=IN_FEATURES,
            decode=False,
            causal=True,
            rngs=nnx.Rngs(0),
        )
        x = make_inputs()
        expected = causal(x)

        causal.init_cache(x.shape)
        causal.decode = True
        decoded = jnp.stack(
            [causal(x[:, step : step + 1, :])[:, 0, :] for step in range(SEQ_LEN)], axis=1
        )
        assert jnp.allclose(decoded, expected, atol=1e-5)

    def test_decode_matches_reference(self) -> None:
        """Decoded output must match the reference module's decoded output."""
        flash, reference = make_pair()
        x = make_inputs()
        for module in (flash, reference):
            module.init_cache(x.shape)
            module.decode = True
        flash_out = flash(x[:, 0:1, :])
        reference_out = reference(x[:, 0:1, :])
        assert jnp.allclose(flash_out, reference_out, atol=1e-6)


class TestPadding:
    """Padding is expressed with sequence lengths, which cuDNN accelerates."""

    def test_sequence_lengths_match_explicit_mask(self) -> None:
        """``query_seq_lengths`` must equal the equivalent boolean mask."""
        q = jax.random.normal(jax.random.key(1), (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM))
        k = jax.random.normal(jax.random.key(2), (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM))
        v = jax.random.normal(jax.random.key(3), (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM))
        lengths = jnp.array([SEQ_LEN - 3, SEQ_LEN], dtype=jnp.int32)

        actual = flash_dot_product_attention(
            q, k, v, query_seq_lengths=lengths, key_value_seq_lengths=lengths
        )
        valid = jnp.arange(SEQ_LEN)[None, :] < lengths[:, None]
        mask = valid[:, None, :, None] & valid[:, None, None, :]
        expected = flash_dot_product_attention(q, k, v, mask=mask)

        rows = valid[:, :, None, None]
        assert jnp.allclose(jnp.where(rows, actual, 0.0), jnp.where(rows, expected, 0.0), atol=1e-5)


class TestScale:
    """A custom logit scale must survive the portable path.

    The reference kernel has no ``scale`` argument and always divides by
    ``sqrt(depth)``, so the helper pre-multiplies the query to compensate. That
    compensation is easy to get wrong and needs its own check.
    """

    @staticmethod
    def make_qkv() -> tuple[jax.Array, jax.Array, jax.Array]:
        """Build query, key and value arrays."""
        shape = (BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM)
        return (
            jax.random.normal(jax.random.key(1), shape),
            jax.random.normal(jax.random.key(2), shape),
            jax.random.normal(jax.random.key(3), shape),
        )

    def test_default_scale_is_inverse_sqrt_depth(self) -> None:
        """Passing the default explicitly must change nothing."""
        q, k, v = self.make_qkv()
        implicit = flash_dot_product_attention(q, k, v)
        explicit = flash_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        assert jnp.allclose(implicit, explicit, atol=1e-6)

    def test_custom_scale_matches_manual_attention(self) -> None:
        """A non-default scale must match a hand-computed reference."""
        q, k, v = self.make_qkv()
        scale = 0.25
        actual = flash_dot_product_attention(q, k, v, scale=scale)
        scores = jnp.einsum("bqnh,bknh->bnqk", q, k) * scale
        expected = jnp.einsum("bnqk,bknh->bqnh", jax.nn.softmax(scores, axis=-1), v)
        assert jnp.allclose(actual, expected, atol=1e-5)

    def test_custom_scale_differs_from_default(self) -> None:
        """The scale must actually take effect."""
        q, k, v = self.make_qkv()
        assert not jnp.allclose(
            flash_dot_product_attention(q, k, v),
            flash_dot_product_attention(q, k, v, scale=0.25),
            atol=1e-6,
        )


class TestBackendSelection:
    """Backend choice must be explicit, honest, and testable off-GPU."""

    def test_float32_is_not_cudnn_eligible(self) -> None:
        """cuDNN rejects float32; the default parameter dtype is float32."""
        assert not is_cudnn_eligible(
            dtype=jnp.float32, head_dim=64, device_kind="gpu", compute_capability=(9, 0)
        )

    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
    def test_half_precision_is_eligible_on_gpu(self, dtype: jnp.dtype) -> None:
        """bf16 and fp16 are the supported fused dtypes."""
        assert is_cudnn_eligible(
            dtype=dtype, head_dim=64, device_kind="gpu", compute_capability=(8, 0)
        )

    def test_head_dim_must_be_a_multiple_of_eight(self) -> None:
        """cuDNN requires ``head_dim % 8 == 0``."""
        assert not is_cudnn_eligible(
            dtype=jnp.bfloat16, head_dim=60, device_kind="gpu", compute_capability=(9, 0)
        )

    def test_head_dim_limit_depends_on_architecture(self) -> None:
        """The limit is 128 below sm90 and 256 from sm90 up."""
        pre_hopper = {"dtype": jnp.bfloat16, "head_dim": 256, "device_kind": "gpu"}
        assert not is_cudnn_eligible(**pre_hopper, compute_capability=(8, 0))
        assert is_cudnn_eligible(**pre_hopper, compute_capability=(9, 0))

    def test_cpu_is_never_eligible(self) -> None:
        """There is no fused kernel off GPU."""
        assert not is_cudnn_eligible(
            dtype=jnp.bfloat16, head_dim=64, device_kind="cpu", compute_capability=None
        )

    def test_live_dropout_never_selects_cudnn(self) -> None:
        """The fused kernel's dropout seed is a compile-time constant.

        ``fused_attention_stablehlo.py`` marks ``seed`` static and bakes it into
        the custom-call backend config, so a jitted training step would reuse one
        dropout mask forever. Live dropout must therefore never be fused.
        """
        decision = select_attention_backend(
            dtype=jnp.bfloat16,
            head_dim=64,
            device_kind="gpu",
            compute_capability=(9, 0),
            deterministic=False,
            dropout_rate=0.1,
            sow_weights=False,
        )
        assert decision is AttentionBackend.XLA

    def test_sow_weights_never_selects_cudnn(self) -> None:
        """Fused kernels never materialise the weights, so they cannot be sown."""
        decision = select_attention_backend(
            dtype=jnp.bfloat16,
            head_dim=64,
            device_kind="gpu",
            compute_capability=(9, 0),
            deterministic=True,
            dropout_rate=0.0,
            sow_weights=True,
        )
        assert decision is AttentionBackend.XLA

    def test_eligible_inference_selects_cudnn(self) -> None:
        """With nothing blocking it, the fused kernel must be chosen."""
        decision = select_attention_backend(
            dtype=jnp.bfloat16,
            head_dim=64,
            device_kind="gpu",
            compute_capability=(9, 0),
            deterministic=True,
            dropout_rate=0.0,
            sow_weights=False,
        )
        assert decision is AttentionBackend.CUDNN

    def test_dropout_rate_zero_is_fusable_even_when_training(self) -> None:
        """A zero rate means no dropout at all, so fusion stays available."""
        decision = select_attention_backend(
            dtype=jnp.bfloat16,
            head_dim=64,
            device_kind="gpu",
            compute_capability=(9, 0),
            deterministic=False,
            dropout_rate=0.0,
            sow_weights=False,
        )
        assert decision is AttentionBackend.CUDNN


class TestSurface:
    """The published surface must describe only what the code does."""

    def test_no_triton_symbols_remain(self) -> None:
        """The orphaned Triton kernel and its scaffolding are gone."""
        import artifex.generative_models.core.layers.flash_attention as module

        for removed in (
            "flash_attention_forward_kernel",
            "TRITON_AVAILABLE",
            "FlashAttentionConfig",
            "AttentionMask",
            "PADDING_SEGMENT_ID",
        ):
            assert not hasattr(module, removed), f"{removed} should have been removed"

    def test_segment_ids_are_not_silently_accepted(self) -> None:
        """Segment IDs never masked across segments, so they must not be accepted.

        Passing ``[0, 0, 1, 1]`` produced output bit-identical to ``[0, 0, 0, 0]``:
        the packing was ignored. Accepting the argument again would restore a
        silent wrong answer, so it must raise instead.
        """
        x = make_inputs()
        flash, _ = make_pair()
        with pytest.raises(TypeError):
            flash(x, query_segment_ids=jnp.zeros((BATCH, SEQ_LEN), dtype=jnp.int32))  # type: ignore[call-arg]

    def test_flash_attention_is_a_drop_in_attention_fn(self) -> None:
        """The helper must satisfy the nnx ``attention_fn`` contract."""
        flash = nnx.MultiHeadAttention(
            num_heads=NUM_HEADS,
            in_features=IN_FEATURES,
            decode=False,
            attention_fn=flash_dot_product_attention,
            rngs=nnx.Rngs(0),
        )
        reference = nnx.MultiHeadAttention(
            num_heads=NUM_HEADS, in_features=IN_FEATURES, decode=False, rngs=nnx.Rngs(0)
        )
        x = make_inputs()
        assert jnp.allclose(flash(x), reference(x), atol=1e-6)


class TestCausal:
    """The ``causal`` convenience must agree with an explicit causal mask."""

    @staticmethod
    def make_causal() -> FlashMultiHeadAttention:
        """Build a causal module. ``causal`` is an artifex addition to nnx."""
        return FlashMultiHeadAttention(
            num_heads=NUM_HEADS,
            in_features=IN_FEATURES,
            decode=False,
            causal=True,
            rngs=nnx.Rngs(0),
        )

    def test_causal_matches_explicit_mask(self) -> None:
        """``causal=True`` must equal passing ``nnx.make_causal_mask``."""
        causal = self.make_causal()
        explicit, _ = make_pair()
        x = make_inputs()
        mask = nnx.make_causal_mask(jnp.ones((BATCH, SEQ_LEN)))
        assert jnp.allclose(causal(x), explicit(x, mask=mask), atol=1e-6)

    def test_causal_forbids_future_tokens(self) -> None:
        """Changing a later token must not change an earlier output."""
        causal = self.make_causal()
        x = make_inputs()
        perturbed = x.at[:, -1, :].add(10.0)
        assert jnp.allclose(causal(x)[:, 0, :], causal(perturbed)[:, 0, :], atol=1e-6)


@pytest.mark.gpu
class TestFusedBackendOnGpu:
    """Paths the CPU-only jaxlib cannot reach."""

    def test_bfloat16_fused_output_matches_reference(self) -> None:
        """The fused kernel must agree with the reference within bf16 tolerance.

        The reference runs nnx's XLA kernel over identical weights, so this
        compares fused against unfused without switching devices.
        """
        flash, reference = make_pair(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        assert flash.resolve_backend() is AttentionBackend.CUDNN, "expected the fused path"
        x = make_inputs(dtype=jnp.bfloat16)
        assert jnp.allclose(
            flash(x).astype(jnp.float32), reference(x).astype(jnp.float32), atol=2e-2
        )

    def test_cpu_scope_falls_back_to_xla(self) -> None:
        """A CPU-scoped block must not dispatch a CUDA-only primitive.

        ``jax.default_device`` redirects computation to the CPU while a GPU is
        still present. Selecting cuDNN there raises ``NotImplementedError`` for
        the missing CPU lowering, so the selector has to honour the scope.
        """
        flash, _ = make_pair(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        with jax.default_device(jax.devices("cpu")[0]):
            assert flash.resolve_backend() is AttentionBackend.XLA
            assert flash(make_inputs(dtype=jnp.bfloat16)).shape == (BATCH, SEQ_LEN, IN_FEATURES)

    def test_selected_backend_is_cudnn_for_bfloat16(self) -> None:
        """An eligible bf16 module must actually reach the fused path."""
        flash, _ = make_pair(dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
        assert flash.resolve_backend() is AttentionBackend.CUDNN

    def test_float32_module_falls_back_to_xla(self) -> None:
        """The default float32 module cannot be fused, and must say so."""
        flash, _ = make_pair()
        assert flash.resolve_backend() is AttentionBackend.XLA
