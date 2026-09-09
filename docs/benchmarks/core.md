# Core

**Module:** `benchmarks.protocols.core`

**Source:** `benchmarks/protocols/core.py`

## Overview

Core benchmark protocols and infrastructure.

## Classes

### BenchmarkConfig

```python
class BenchmarkConfig
```

### BenchmarkResult

```python
class BenchmarkResult  # calibrax.core.BenchmarkResult, re-exported
```

Results are calibrax's frozen `BenchmarkResult`: `name`, `tags` (with
`model_name`), `metrics` (`name -> Metric`), `metadata`, `config` and
`timestamp`, with `save`/`load` in calibrax's JSON layout. Build one with
`Benchmark.result(model_name, metrics, metadata=...)` (named after the config and
carrying it) or `benchmark_result(name, model_name, metrics, ...)`; read the
values back with `metric_values(result)`.

## Module Statistics

- **Classes:** 2
- **Functions:** 0
- **Imports:** 3
