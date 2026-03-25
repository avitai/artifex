# Runner

`BenchmarkRunner` and `PerformanceTracker` live in
`artifex.benchmarks.core.runner`.

## Current Contract

- `BenchmarkRunner` executes the retained NNX benchmark workflow and keeps
  result history for comparisons.
- `PerformanceTracker` records explicit metric snapshots against the
  `EvaluationConfig.metric_params["target_metrics"]` contract.
- Unsupported workflow behavior should fail in the benchmark or metric
  implementation rather than silently fabricating results.

## Import Guidance

Use `from artifex.benchmarks.core import BenchmarkRunner, PerformanceTracker`
in new code.
