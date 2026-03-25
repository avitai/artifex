# Performance

**Module:** `artifex.generative_models.core.performance`

**Source:** `src/artifex/generative_models/core/performance.py`

## Overview

This module provides estimation-oriented performance helpers for Artifex runtime
analysis. The retained contract is intentionally narrow: platform and device
count come from the live JAX runtime, while memory capacity, compute capability,
peak FLOPs, and memory bandwidth may be heuristic estimates rather than measured
hardware facts.

## Hardware Detection Contract

`HardwareDetector.detect_hardware()` returns a `HardwareSpecs` record that labels
where each hardware field came from. Use the source markers before treating a
value as runtime fact:

- `memory_source`
- `compute_capability_source`
- `peak_flops_source`
- `memory_bandwidth_source`

Source values are one of `"detected"`, `"estimated"`, or `"unavailable"`.
Current GPU, TPU, and CPU helpers mostly populate heuristic estimates for the
capacity and throughput fields.

## Roofline Contract

`PerformanceEstimator.analyze_roofline(...)` performs arithmetic-intensity and
roofline calculations, but it no longer fabricates peak hardware numbers.
`analyze_roofline(...)` requires explicit `peak_flops_per_second` and
`memory_bandwidth_gb_per_second` values on the `HardwareSpecs` input.

If those values are unavailable, supply them explicitly from a trusted source or
stop at the higher-level estimation helpers instead of presenting a fake
roofline result as measured runtime truth.

## Benchmarking Notes

The profiling helpers benchmark JAX functions by timing compiled execution and
blocking on device completion. Interpret their outputs as local runtime
measurements, and interpret any populated hardware-capacity fields through the
source labels above.

## Public Surface

Classes:

- `HardwareDetector`
- `HardwareSpecs`
- `PerformanceEstimator`
- `RooflineMetrics`

Representative methods:

- `detect_hardware()`
- `analyze_roofline()`
- `benchmark_operation()`
- `profile_jax_function()`
- `estimate_transformer_layer_performance()`
