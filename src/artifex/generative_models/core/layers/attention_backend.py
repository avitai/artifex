"""Backend selection for scaled dot-product attention.

JAX ships two attention implementations reachable from the public API: the
portable XLA lowering and NVIDIA's fused cuDNN kernel. Choosing between them has
to happen here because
:func:`jax.nn.dot_product_attention` does *not* select one itself --- passing
``implementation=None`` always takes the XLA path (see the standing ``TODO`` in
``jax/_src/nn/functions.py``). Eligibility is therefore decided explicitly, and
the reasons are kept close to the JAX source that enforces them.

The constraints below mirror ``check_is_flash_attention`` and the dtype guard in
``jax/_src/cudnn/fused_attention_stablehlo.py``.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike


logger = logging.getLogger(__name__)

#: cuDNN rejects float32 outright; only these half-precision dtypes are fused.
CUDNN_SUPPORTED_DTYPES: tuple[DTypeLike, ...] = (jnp.bfloat16, jnp.float16)

#: The fused kernel requires the head dimension to be a multiple of eight.
HEAD_DIM_MULTIPLE = 8

#: Maximum head dimension below and from Hopper (sm90) respectively.
HEAD_DIM_MAX_PRE_HOPPER = 128
HEAD_DIM_MAX_HOPPER = 256

#: Compute capability at which the larger head-dimension limit becomes available.
HOPPER_COMPUTE_CAPABILITY = (9, 0)


class AttentionBackend(StrEnum):
    """An attention implementation that this library will actually dispatch to."""

    CUDNN = "cudnn"
    XLA = "xla"


def is_cudnn_eligible(
    *,
    dtype: DTypeLike,
    head_dim: int,
    device_kind: str,
    compute_capability: tuple[int, int] | None,
) -> bool:
    """Report whether the fused cuDNN kernel supports this shape and dtype.

    This is a pure predicate so that the rules stay testable without a GPU.

    Args:
        dtype: Computation dtype of the query, key and value arrays.
        head_dim: Size of each attention head.
        device_kind: Device platform, such as ``"gpu"`` or ``"cpu"``.
        compute_capability: CUDA compute capability, or ``None`` when unknown.

    Returns:
        ``True`` when every cuDNN constraint is satisfied.
    """
    if device_kind != "gpu" or compute_capability is None:
        return False

    if jnp.dtype(dtype) not in {jnp.dtype(supported) for supported in CUDNN_SUPPORTED_DTYPES}:
        return False

    if head_dim % HEAD_DIM_MULTIPLE != 0:
        return False

    is_hopper_or_later = compute_capability >= HOPPER_COMPUTE_CAPABILITY
    head_dim_max = HEAD_DIM_MAX_HOPPER if is_hopper_or_later else HEAD_DIM_MAX_PRE_HOPPER
    return head_dim <= head_dim_max


def local_device_profile() -> tuple[str, tuple[int, int] | None]:
    """Describe the device attention will run on as ``(kind, compute_capability)``.

    Honours ``jax.default_device``, so a CPU-scoped block on a GPU host selects
    the portable kernel instead of dispatching a CUDA primitive that has no CPU
    lowering. The device is read globally rather than from the query array
    because under ``jit`` the array is a tracer with no device to inspect.

    Returns:
        The platform string and the CUDA compute capability, which is ``None``
        for non-CUDA devices or when the runtime does not report one.
    """
    # jax.config exposes this at runtime but ships no stub entry for it.
    default_device = getattr(jax.config, "jax_default_device", None)
    device = default_device or jax.devices()[0]
    kind = device.platform
    raw = getattr(device, "compute_capability", None)
    if raw is None:
        return kind, None

    text = str(raw)
    # JAX reports either "9.0" or the packed form "90" depending on the version.
    parts = text.split(".") if "." in text else [text[:-1], text[-1]]
    try:
        major, minor = (int(part) for part in parts)
    except ValueError:
        logger.debug("Unrecognised compute capability %r; treating as unknown.", raw)
        return kind, None
    return kind, (major, minor)


def select_attention_backend(
    *,
    dtype: DTypeLike,
    head_dim: int,
    device_kind: str,
    compute_capability: tuple[int, int] | None,
    deterministic: bool,
    dropout_rate: float,
    sow_weights: bool,
) -> AttentionBackend:
    """Choose the attention backend for one call.

    Two capabilities force the portable path even on eligible hardware:

    * **Sown weights.** A fused kernel never materialises the attention matrix,
      so there is nothing to sow.
    * **Live dropout.** ``fused_attention_stablehlo`` marks the dropout ``seed``
      as a static argument and serialises it into the custom-call backend
      config, so it is baked into the compiled executable. A jitted training
      step would reuse a single dropout mask for every step, which is silently
      wrong rather than merely slow.

    Args:
        dtype: Computation dtype of the query, key and value arrays.
        head_dim: Size of each attention head.
        device_kind: Device platform, such as ``"gpu"`` or ``"cpu"``.
        compute_capability: CUDA compute capability, or ``None`` when unknown.
        deterministic: Whether dropout is disabled for this call.
        dropout_rate: Configured attention dropout rate.
        sow_weights: Whether the caller asked for the attention weights.

    Returns:
        The backend to dispatch to.
    """
    if sow_weights:
        return AttentionBackend.XLA

    has_live_dropout = not deterministic and dropout_rate > 0.0
    if has_live_dropout:
        return AttentionBackend.XLA

    if is_cudnn_eligible(
        dtype=dtype,
        head_dim=head_dim,
        device_kind=device_kind,
        compute_capability=compute_capability,
    ):
        return AttentionBackend.CUDNN
    return AttentionBackend.XLA
