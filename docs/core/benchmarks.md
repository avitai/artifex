# Benchmarks

The benchmark foundation moved out of `artifex.generative_models.core` and
now lives in `artifex.benchmarks.core`.

## Retained Owners

- `Benchmark`, `BenchmarkConfig`, `BenchmarkResult`, and `BenchmarkSuite`
  live in `artifex.benchmarks.core.foundation`.
- `BenchmarkBase` and `BenchmarkWithValidation` live in
  `artifex.benchmarks.core.nnx` and keep the Artifex-specific NNX glue.
- `BenchmarkRunner` and `PerformanceTracker` live in
  `artifex.benchmarks.core.runner`.

## Core Boundary

`core.evaluation` is metrics-only. `core.protocols` no longer owns a
benchmark base layer.

## Use This Surface

Import from `artifex.benchmarks.core` in benchmark and suite code.
