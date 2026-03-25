# Geometric Benchmark Demo

**Status:** `Demo-only benchmark walkthrough`
**Device:** `CPU-compatible`

This walkthrough keeps the historical geometric benchmark example, but the public claim is now narrow: the shipped
pair uses retained synthetic ShapeNet-style data and demo-only benchmark wiring to illustrate point-cloud evaluation.
It is not the canonical benchmark-grade ShapeNet runtime.

## Files

- Python script: [geometric_benchmark_demo.py](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_benchmark_demo.py)
- Jupyter notebook: [geometric_benchmark_demo.ipynb](https://github.com/avitai/artifex/blob/main/examples/generative_models/geometric/geometric_benchmark_demo.ipynb)

## Run It

```bash
python examples/generative_models/geometric/geometric_benchmark_demo.py
jupyter lab examples/generative_models/geometric/geometric_benchmark_demo.ipynb
```

## What This Demo Actually Uses

- `ShapeNetDataset` with `data_source="synthetic"` and explicit demo-mode metadata
- `PointCloudGenerationBenchmark` through typed benchmark configuration objects
- `PointCloudMetrics` and `PointCloudModel` for retained point-cloud evaluation
- training and plotting helpers around a synthetic airplane-style subset

## Why It Is Demo-Only

- benchmark-grade ShapeNet assets are not auto-downloaded by the supported runtime
- the shipped walkthrough intentionally opts into retained synthetic data for reproducibility
- the public CLI does not ship a supported benchmark runner for this workflow

## Use This When

Use this pair when you want to inspect the retained geometric benchmark API shape, run a synthetic point-cloud demo,
or adapt the code to your own benchmark-grade assets from Python.
