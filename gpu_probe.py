"""Report which fused attention paths the active GPU admits.

Run inside a CUDA-enabled environment. Prints the JAX backend, the device and its
compute capability, and then probes each attention implementation across the
dtypes and head dimensions that decide backend eligibility, so the results come
from the hardware rather than from reading the constraint table.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp


HEAD_DIMS = (64, 128, 256)
DTYPES = (("bf16", jnp.bfloat16), ("fp16", jnp.float16), ("fp32", jnp.float32))


def _describe_devices() -> None:
    """Print the backend and per-device compute capability."""
    print(f"jax {jax.__version__} | backend: {jax.default_backend()}")
    for device in jax.devices():
        capability = getattr(device, "compute_capability", "unknown")
        print(f"  device: {device.device_kind} | compute capability: {capability}")


def _probe(implementation: Literal["xla", "cudnn"], dtype: jnp.dtype, head_dim: int) -> str:
    """Return 'ok' or the reason the implementation rejected this configuration."""
    batch, seq_len, num_heads = 2, 256, 4
    shape = (batch, seq_len, num_heads, head_dim)
    query = jnp.ones(shape, dtype=dtype)
    try:
        out = jax.nn.dot_product_attention(query, query, query, implementation=implementation)
        out.block_until_ready()
    except Exception as error:  # noqa: BLE001 - probing which errors the backend raises
        return f"{type(error).__name__}: {str(error).splitlines()[0][:88]}"
    return "ok"


def main() -> None:
    """Probe every implementation/dtype/head-dim combination."""
    _describe_devices()

    implementations: tuple[Literal["xla", "cudnn"], ...] = ("xla", "cudnn")
    for implementation in implementations:
        print(f"\nimplementation={implementation!r}")
        for label, dtype in DTYPES:
            for head_dim in HEAD_DIMS:
                status = _probe(implementation, dtype, head_dim)
                print(f"  dtype={label:5s} head_dim={head_dim:>3d} -> {status}")

    try:
        from jax.experimental.pallas.ops.gpu import attention as pallas_attention
    except ImportError as error:
        print(f"\npallas gpu attention unavailable: {error}")
        return

    print("\npallas mha")
    for label, dtype in DTYPES:
        for head_dim in HEAD_DIMS:
            query = jnp.ones((2, 256, 4, head_dim), dtype=dtype)
            try:
                out = jnp.asarray(pallas_attention.mha(query, query, query, segment_ids=None))
                out.block_until_ready()
                status = "ok"
            except Exception as error:  # noqa: BLE001 - probing kernel support
                status = f"{type(error).__name__}: {str(error).splitlines()[0][:88]}"
            print(f"  dtype={label:5s} head_dim={head_dim:>3d} -> {status}")


if __name__ == "__main__":
    main()
