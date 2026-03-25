# Multi-β VAE Benchmark Demo

**Status:** `Demo-only benchmark walkthrough`
**Device:** `CPU-compatible`

This walkthrough keeps the retained multi-β VAE benchmark example, but the shipped contract is explicit: it runs a
mock-model demo through the retained suite with demo-mode image metrics. It is useful for understanding the API shape,
not for claiming a benchmark-grade VAE evaluation stack out of the box.

## Files

- Python script: [multi_beta_vae_benchmark_demo.py](https://github.com/avitai/artifex/blob/main/examples/generative_models/vae/multi_beta_vae_benchmark_demo.py)
- Jupyter notebook: [multi_beta_vae_benchmark_demo.ipynb](https://github.com/avitai/artifex/blob/main/examples/generative_models/vae/multi_beta_vae_benchmark_demo.ipynb)

## Run It

```bash
python examples/generative_models/vae/multi_beta_vae_benchmark_demo.py
jupyter lab examples/generative_models/vae/multi_beta_vae_benchmark_demo.ipynb
```

## What This Demo Actually Uses

- `MultiBetaVAEBenchmarkSuite(..., demo_mode=True)`
- retained disentanglement metrics plus demo-mode `FIDMetric` and `LPIPSMetric`
- a checked-in mock `MultiBetaVAE` implementation that exposes the expected benchmark interface
- small CelebA-style batches for a quick local walkthrough

## Why It Is Demo-Only

- the shipped suite relies on retained mock perceptual/image-quality backends in demo mode
- the example model is a pedagogical mock, not a benchmark-ready trained VAE
- the public benchmark surface is Python-first and intentionally narrower than the historical docs implied

## Use This When

Use this pair when you want to understand the retained suite inputs and outputs, compare mock β-VAE quality tiers, or
swap in your own model implementation while keeping the demo-only expectations explicit.
