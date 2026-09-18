"""Benchmark the attention backends JAX can dispatch to, on the active GPU.

Timing follows the JAX conventions rather than a naive wall-clock loop:

* Inputs are placed on the device up front, so transfers are not timed.
* Every candidate is compiled ahead of time and warmed up before measurement,
  so neither tracing nor autotuning lands inside the timed region.
* `jax.block_until_ready` is applied to the whole output pytree, because JAX
  dispatches asynchronously and an unblocked call measures dispatch, not work.
* Where CUPTI is available, `jax.experimental.mosaic.gpu.profiler.measure`
  reports true on-device kernel time, excluding host dispatch overhead. That is
  the same utility JAX's own Mosaic kernel benchmarks use. Otherwise the
  fallback times many iterations with `time.perf_counter` and reports the
  median, which is robust to scheduler noise in a way the mean is not.

Results are reported as achieved TFLOP/s and as a speedup over the XLA
baseline, mirroring how JAX's matmul benchmarks present kernel comparisons.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp


# Batch, sequence length, heads, head dim. Head dims span the cuDNN cutoff at
# 128 so the eligibility boundary shows up in the numbers.
SHAPES = (
    (2, 1024, 8, 64),
    (2, 1024, 8, 128),
    (4, 4096, 8, 64),
    (1, 8192, 16, 64),
)
DTYPES: tuple[tuple[str, Any], ...] = (("bf16", jnp.bfloat16), ("fp16", jnp.float16))

WARMUP_ITERATIONS = 3
TIMED_ITERATIONS = 20


def attention_flops(batch: int, seq_len: int, heads: int, head_dim: int) -> float:
    """Return the FLOPs of one non-causal attention forward pass.

    Two matmuls of equal cost: scores = q @ k^T and out = weights @ v, each
    2 * batch * heads * seq_len * seq_len * head_dim.
    """
    return 4.0 * batch * heads * seq_len * seq_len * head_dim


def _time_with_cupti(fn: Callable[[], Any]) -> float | None:
    """Return median on-device milliseconds via CUPTI, or None if unavailable."""
    try:
        from jax.experimental.mosaic.gpu import profiler
    except ImportError:
        return None

    try:
        measured = profiler.measure(fn, iterations=TIMED_ITERATIONS)
        _, timings = measured()
    except Exception:
        # CUPTI is unavailable on this platform, or another subscriber holds it.
        return None

    if timings is None:
        return None
    if isinstance(timings, float):
        return timings
    return statistics.median(timings)


def _time_with_wall_clock(fn: Callable[[], Any]) -> float:
    """Return median milliseconds measured on the host, after warmup."""
    for _ in range(WARMUP_ITERATIONS):
        jax.block_until_ready(fn())

    samples: list[float] = []
    for _ in range(TIMED_ITERATIONS):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def benchmark(fn: Callable[[], Any]) -> tuple[float, str]:
    """Return (median milliseconds, timer used) for a compiled callable.

    The timer is returned rather than hidden: CUPTI reports on-device kernel
    time while the fallback reports host wall clock including dispatch, so the
    two are not interchangeable and a silent downgrade would misreport results.
    """
    jax.block_until_ready(fn())  # Compile and warm up before any measurement.
    device_ms = _time_with_cupti(fn)
    if device_ms is not None:
        return device_ms, "cupti"
    return _time_with_wall_clock(fn), "wall"


def _make_candidates(
    query: jax.Array,
) -> dict[str, Callable[[], Any]]:
    """Build one compiled zero-argument callable per available backend."""
    candidates: dict[str, Callable[[], Any]] = {}

    for implementation in ("xla", "cudnn"):

        def call(impl: Literal["xla", "cudnn"] = implementation) -> jax.Array:  # type: ignore[assignment]
            return jax.nn.dot_product_attention(query, query, query, implementation=impl)

        candidates[implementation] = jax.jit(call)

    try:
        from jax.experimental.pallas.ops.gpu import attention as pallas_attention
    except ImportError:
        return candidates

    def pallas_call() -> Any:
        return pallas_attention.mha(query, query, query, segment_ids=None)

    candidates["pallas"] = jax.jit(pallas_call)
    return candidates


def main() -> None:
    """Benchmark every backend across the configured shapes and dtypes."""
    print(f"jax {jax.__version__} | backend: {jax.default_backend()}")
    for device in jax.devices():
        capability = getattr(device, "compute_capability", "unknown")
        print(f"  device: {device.device_kind} | compute capability: {capability}")

    header = (
        f"{'dtype':>5} {'shape (B,T,H,D)':>22} {'backend':>8} "
        f"{'ms':>9} {'TFLOP/s':>9} {'vs xla':>8} {'timer':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for label, dtype in DTYPES:
        for batch, seq_len, heads, head_dim in SHAPES:
            shape = (batch, seq_len, heads, head_dim)
            query = jax.device_put(jnp.ones(shape, dtype=dtype))
            flops = attention_flops(batch, seq_len, heads, head_dim)

            baseline_ms: float | None = None
            for name, fn in _make_candidates(query).items():
                try:
                    elapsed, timer = benchmark(fn)
                except Exception as error:  # noqa: BLE001 - backend rejects this config
                    reason = str(error).splitlines()[0][:52]
                    print(f"{label:>5} {str(shape):>22} {name:>8} {'-':>9} {'-':>9}   {reason}")
                    continue

                if name == "xla":
                    baseline_ms = elapsed
                tflops = flops / (elapsed / 1e3) / 1e12
                speedup = f"{baseline_ms / elapsed:.2f}x" if baseline_ms else "-"
                print(
                    f"{label:>5} {str(shape):>22} {name:>8} "
                    f"{elapsed:>9.3f} {tflops:>9.1f} {speedup:>8} {timer:>6}"
                )


if __name__ == "__main__":
    main()
