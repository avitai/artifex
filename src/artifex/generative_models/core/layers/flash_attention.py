"""Multi-head attention that dispatches to NVIDIA's fused cuDNN kernel.

:class:`FlashMultiHeadAttention` is a drop-in replacement for
:class:`flax.nnx.MultiHeadAttention`. It subclasses it rather than reimplementing
it, so projections, the decode cache, sown weights and dropout are the reference
implementations. The only thing this module adds is the choice of attention
kernel, plus a ``causal`` convenience flag.

Two capabilities always take the portable XLA path, for reasons documented in
:mod:`artifex.generative_models.core.layers.attention_backend`: sown attention
weights, which a fused kernel never materialises, and live dropout, whose seed
the fused kernel bakes into the compiled executable.

Based on:
- Flash Attention paper: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes
from flax.nnx.nn.attention import combine_masks, dot_product_attention as _nnx_attention
from flax.typing import Dtype, PrecisionLike, PromoteDtypeFn

from artifex.generative_models.core.layers.attention_backend import (
    AttentionBackend,
    local_device_profile,
    select_attention_backend,
)


Array = jax.Array

#: Axis holding the sequence length in ``[batch..., length, heads, depth]``.
SEQUENCE_AXIS = -3


def _sequence_length_mask(
    lengths: Array,
    max_length: int,
) -> Array:
    """Build a ``[batch, length]`` validity mask from per-example lengths."""
    return jnp.arange(max_length)[None, :] < lengths[:, None]


def _padding_mask(
    query_seq_lengths: Array | None,
    key_value_seq_lengths: Array | None,
    query_length: int,
    key_length: int,
) -> Array | None:
    """Express sequence lengths as a broadcastable attention mask.

    The fused kernel consumes the lengths directly; the portable path needs them
    as a mask, because :func:`flax.nnx.nn.attention.dot_product_attention` has no
    sequence-length argument.

    Args:
        query_seq_lengths: Unpadded query lengths, shape ``[batch]``.
        key_value_seq_lengths: Unpadded key/value lengths, shape ``[batch]``.
        query_length: Padded query length.
        key_length: Padded key/value length.

    Returns:
        A ``[batch, 1, query_length, key_length]`` boolean mask, or ``None`` when
        no lengths were supplied.
    """
    if query_seq_lengths is None and key_value_seq_lengths is None:
        return None

    query_valid = (
        _sequence_length_mask(query_seq_lengths, query_length)
        if query_seq_lengths is not None
        else jnp.ones((1, query_length), dtype=bool)
    )
    key_valid = (
        _sequence_length_mask(key_value_seq_lengths, key_length)
        if key_value_seq_lengths is not None
        else jnp.ones((1, key_length), dtype=bool)
    )
    return query_valid[:, None, :, None] & key_valid[:, None, None, :]


def _resolve_causal(is_causal: bool, query_length: int, key_length: int) -> bool:
    """Decide whether ``is_causal`` may be applied to these shapes.

    A causal mask is only meaningful when queries and keys index the same
    positions. During cached decoding a single query attends to the whole cache,
    and ``nnx`` already supplies an ``arange(max_length) <= cache_index`` mask
    that enforces causality; applying ``is_causal`` on top would build a
    ``tril`` of shape ``(1, max_length)`` and wrongly restrict the token to
    position zero.

    Args:
        is_causal: Whether causal masking was requested.
        query_length: Number of query positions.
        key_length: Number of key positions.

    Returns:
        Whether to apply causal masking in the attention kernel.

    Raises:
        ValueError: If causal masking is requested for mismatched lengths that
            are not the cached-decoding case.
    """
    if not is_causal or query_length == key_length:
        return is_causal
    if query_length == 1:
        # Cached decoding: causality already comes from the cache mask.
        return False
    raise ValueError(
        f"Causal masking needs equal query and key lengths, got {query_length} "
        f"and {key_length}. Pass an explicit `mask` for cross-attention."
    )


def flash_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    *,
    broadcast_dropout: bool = True,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: nnx.Module | None = None,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    is_causal: bool = False,
    scale: float | None = None,
    query_seq_lengths: Array | None = None,
    key_value_seq_lengths: Array | None = None,
) -> Array:
    """Compute dot-product attention, fusing the kernel when that is possible.

    The signature satisfies the ``attention_fn`` contract of
    :class:`flax.nnx.MultiHeadAttention`, so this may be passed to any nnx
    attention module.

    Args:
        query: Queries, shape ``[batch..., length, heads, depth]``.
        key: Keys, shape ``[batch..., length, heads, depth]``.
        value: Values, shape ``[batch..., length, heads, depth]``.
        bias: Additive bias broadcastable to ``[batch..., heads, q_len, kv_len]``.
        mask: Boolean mask broadcastable to ``[batch..., heads, q_len, kv_len]``,
            where ``True`` means the position takes part in attention.
        broadcast_dropout: Share the dropout mask across batch and head axes.
        dropout_rng: Key for attention dropout.
        dropout_rate: Attention dropout rate.
        deterministic: Disable dropout when ``True``.
        dtype: Computation dtype; inferred from the inputs when ``None``.
        precision: Matmul precision for the portable path.
        module: Module that sows the attention weights, or ``None``.
        promote_dtype: Function promoting the query, key and value dtypes.
        is_causal: Apply causal masking.
        scale: Logit scale; defaults to ``1 / sqrt(depth)``.
        query_seq_lengths: Unpadded query lengths, shape ``[batch]``.
        key_value_seq_lengths: Unpadded key/value lengths, shape ``[batch]``.

    Returns:
        Attention output with the same shape as ``query``.
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    compute_dtype = query.dtype
    query_length, key_length = query.shape[SEQUENCE_AXIS], key.shape[SEQUENCE_AXIS]
    is_causal = _resolve_causal(is_causal, query_length, key_length)

    device_kind, compute_capability = local_device_profile()
    backend = select_attention_backend(
        dtype=compute_dtype,
        head_dim=query.shape[-1],
        device_kind=device_kind,
        compute_capability=compute_capability,
        deterministic=deterministic,
        dropout_rate=dropout_rate,
        sow_weights=module is not None,
    )

    if backend is AttentionBackend.CUDNN:
        return jax.nn.dot_product_attention(
            query,
            key,
            value,
            bias=bias,
            mask=mask,
            scale=scale,
            is_causal=is_causal,
            query_seq_lengths=query_seq_lengths,
            key_value_seq_lengths=key_value_seq_lengths,
            implementation="cudnn",
        )

    padding = _padding_mask(query_seq_lengths, key_value_seq_lengths, query_length, key_length)
    if padding is not None:
        mask = padding if mask is None else combine_masks(mask, padding)

    if scale is not None:
        query = query * jnp.asarray(scale * (query.shape[-1] ** 0.5), dtype=compute_dtype)

    return _nnx_attention(
        query,
        key,
        value,
        bias,
        mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=compute_dtype,
        precision=precision,
        module=module,
        is_causal=is_causal,
    )


class FlashMultiHeadAttention(nnx.MultiHeadAttention):
    """Multi-head attention that uses the fused cuDNN kernel where it applies.

    Behaviour is identical to :class:`flax.nnx.MultiHeadAttention`; only the
    attention kernel differs, and only when the inputs are eligible. Note that
    the inherited ``param_dtype`` default of ``float32`` is *not* eligible, since
    cuDNN accepts only ``bfloat16`` and ``float16``. Call :meth:`resolve_backend`
    to see which kernel a given configuration will reach.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        in_kv_features: int | None = None,
        *,
        causal: bool = False,
        attention_fn: Callable[..., Array] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the module.

        Args:
            num_heads: Number of attention heads.
            in_features: Size of the query input feature dimension.
            qkv_features: Size of the projected query/key/value dimension.
            out_features: Size of the output feature dimension.
            in_kv_features: Size of the key/value input feature dimension.
            causal: Apply causal masking without building an explicit mask. The
                fused kernel skips the masked region rather than computing and
                discarding it.
            attention_fn: Override for the attention kernel.
            **kwargs: Forwarded to :class:`flax.nnx.MultiHeadAttention`.
        """
        super().__init__(
            num_heads,
            in_features,
            qkv_features,
            out_features,
            in_kv_features,
            attention_fn=(
                attention_fn
                if attention_fn is not None
                else functools.partial(flash_dot_product_attention, is_causal=causal)
            ),
            **kwargs,
        )
        self.causal = causal

    def resolve_backend(
        self,
        *,
        deterministic: bool = True,
        sow_weights: bool = False,
    ) -> AttentionBackend:
        """Report which kernel this configuration reaches on the current device.

        Args:
            deterministic: Whether dropout would be disabled for the call.
            sow_weights: Whether the call would ask for the attention weights.

        Returns:
            The backend that :func:`flash_attention` would dispatch to.
        """
        device_kind, compute_capability = local_device_profile()
        return select_attention_backend(
            dtype=self.dtype if self.dtype is not None else self.param_dtype,
            head_dim=self.head_dim,
            device_kind=device_kind,
            compute_capability=compute_capability,
            deterministic=deterministic,
            dropout_rate=self.dropout_rate,
            sow_weights=sow_weights,
        )
