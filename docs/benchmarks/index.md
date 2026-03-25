# Benchmarks

Artifex retains benchmark and evaluation modules, but the public benchmark surface is intentionally narrow.
There is no supported public benchmark CLI runner, and several retained suites now require explicit
`demo_mode=True` or synthetic/demo dataset settings before mock or placeholder paths are allowed to run.

## Current Contract

- Use Python APIs for benchmark work; do not expect `artifex benchmark` to execute a supported runtime flow.
- Treat retained mock or synthetic suites as demo-only teaching surfaces.
- Expect supported-mode evaluation to fail fast when benchmark-grade assets or third-party dependencies are missing.

## Retained Demo Walkthroughs

- [Geometric Benchmark Demo](../examples/geometric/geometric-benchmark-demo.md)
- [Protein-Ligand Benchmark Demo](../examples/protein/protein-ligand-benchmark-demo.md)
- [Multi-β VAE Benchmark Demo](../examples/vae/multi-beta-vae-benchmark-demo.md)

## CLI Status

The CLI benchmark page is a retirement notice, not an execution guide: [Benchmark CLI](../cli/benchmark.md).

## Explicit Demo Opt-In

```python
from artifex.benchmarks.suites.multi_beta_vae_suite import MultiBetaVAEBenchmarkSuite

suite = MultiBetaVAEBenchmarkSuite(
    dataset_config={"num_samples": 100, "image_size": 64, "include_attributes": True},
    benchmark_config={"num_samples": 50, "batch_size": 10},
    demo_mode=True,
    rngs=rngs,
)
```

The same rule applies to retained synthetic dataset paths such as `data_source="synthetic"` and to mock metric
backends such as `mock_inception=True` or `mock_implementation=True`: those switches are explicit demo-mode opt-ins,
not the primary supported benchmark runtime.
