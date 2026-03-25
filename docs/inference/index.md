# Inference Reference

**Status:** `Supported runtime inference surface`

The shared inference package is intentionally narrow today.

`artifex.inference` does not exist.
`artifex.generative_models.inference` exports no public helpers from `__all__`.
The only retained shared inference owner is
`artifex.generative_models.inference.optimization.production`.

## Current Inference Pages

- [Inference Reference](index.md): truthful catalog for the current shared
  surface.
- [Production](production.md): experimental production inference helpers around
  `ProductionOptimizer`, `OptimizationTarget`, `ProductionPipeline`, and
  `ProductionMonitor`.

## Family-Owned Generation Entry Points

Generation remains family-owned; the family-owned generation entrypoints
remain the supported path, and the shared `docs/inference` catalog does not
replace model-specific generation APIs.

- VAE: `VAE.sample(...)` and `VAE.reconstruct(...)`
- GAN: `GAN.generate(...)`
- DDPM: `DDPMModel.generate(...)`
- Normalizing flows: `NormalizingFlow.sample(...)` and
  `NormalizingFlow.log_prob(...)`

See [Inference Overview](../user-guide/inference/overview.md) and
[Sampling Guide](../user-guide/inference/sampling.md) for the retained loading
and generation workflow.

## Coming Soon

The remaining pages in this section stay published only as `Coming soon`
placeholders for not-yet-shipped shared inference modules. They remain relevant
future surface areas, but they are not shipped yet as part of the current
runtime.
