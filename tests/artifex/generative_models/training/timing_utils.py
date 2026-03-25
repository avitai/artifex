"""Timing helpers for CI-stable training callback tests."""

from __future__ import annotations


def best_average_us_per_call(dispatch, *, iterations: int, repeats: int = 7) -> float:
    """Return the best observed per-call latency in microseconds."""
    import gc
    import time

    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(200):
            dispatch()

        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                dispatch()
            elapsed_ns = time.perf_counter_ns() - start
            samples.append(elapsed_ns / iterations / 1_000.0)
        return min(samples)
    finally:
        if gc_enabled:
            gc.enable()
