# Benchmarks

The benchmark foundation moved out of `artifex.generative_models.core` and
now lives in `artifex.benchmarks.core`.

## Retained Owners

- `Benchmark`, `BenchmarkConfig` and `BenchmarkSuite` live in
  `artifex.benchmarks.core.foundation`. `BenchmarkResult` is calibrax's
  (`calibrax.core.BenchmarkResult`), re-exported unchanged; `Benchmark.result(...)`
  and `benchmark_result(...)` build one from a metric dictionary, and
  `metric_values(result)` reads the plain `name -> value` view back.
- `BenchmarkBase` and `BenchmarkWithValidation` live in
  `artifex.benchmarks.core.nnx` and keep the Artifex-specific NNX glue.
- `BenchmarkRunner` and `PerformanceTracker` live in
  `artifex.benchmarks.core.runner`.

## Core Boundary

`core` owns no evaluation package. Metric classes live in
`artifex.benchmarks.metrics`, and `core.protocols` keeps only the `MetricBase`
protocol, not a benchmark base layer.

## Use This Surface

Import from `artifex.benchmarks.core` in benchmark and suite code.
