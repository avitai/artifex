"""Calibrax-first benchmark registry helpers.

Artifex keeps only its registration helper here. Benchmarks register in calibrax's
singleton, `calibrax.core.BenchmarkRegistry`; look them up with calibrax's
`get_benchmark` and `list_benchmarks`.
"""

from __future__ import annotations

from typing import Any, cast

import calibrax.core

from artifex.benchmarks.core import Benchmark, BenchmarkConfig


__all__ = ["register_benchmark"]


def register_benchmark(
    name: str,
    benchmark: Benchmark | None = None,
    config: BenchmarkConfig | None = None,
) -> Any:
    """Register a benchmark with the Calibrax singleton registry."""
    registry = calibrax.core.BenchmarkRegistry()

    def decorator(cls_or_fn: Any) -> Any:
        if isinstance(cls_or_fn, type) and issubclass(cls_or_fn, Benchmark):
            benchmark_config = config or BenchmarkConfig(
                name=name,
                description=f"Benchmark {name}",
                metric_names=[],
            )
            benchmark_instance = cls_or_fn(benchmark_config)
            registry.register(name, benchmark_instance)
            return cls_or_fn

        benchmark_instance = cast(Benchmark, cls_or_fn)
        registry.register(name, benchmark_instance)
        return cls_or_fn

    if benchmark is not None:
        registry.register(name, benchmark)
        return None

    return decorator
