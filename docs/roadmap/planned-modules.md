# Planned Modules Roadmap

This page is a checked status surface over the current package tree. It is not a
freehand completeness matrix, and it intentionally avoids summary totals that
drift away from the live runtime inventory.

## Status Legend

| Status | Meaning |
| --- | --- |
| **Shipped** | Importable and part of the retained runtime surface |
| **Experimental** | Importable, but intentionally narrow, unstable, or placeholder-heavy |
| **Roadmap-only** | Not importable today, or intentionally not presented as current support |

## Current Runtime Status

### Generative Models

| Family | Status | Runtime evidence | Notes |
| --- | --- | --- | --- |
| VAE | Shipped | `artifex.generative_models.models.vae` | Retained VAE owners and typed configs |
| GAN | Shipped | `artifex.generative_models.models.gan` | Retained GAN owners such as `DCGAN` and `PatchGAN` |
| Diffusion | Shipped | `artifex.generative_models.models.diffusion` | Retained diffusion owners such as `DDPM` and `DiT` |
| Flow | Shipped | `artifex.generative_models.models.flow` | Retained flow owners include `RealNVP`, `Glow`, `MAF`, `IAF`, `NeuralSplineFlow`, and `ConditionalRealNVP` |
| Autoregressive | Shipped | `artifex.generative_models.models.autoregressive` | Retained autoregressive owners |
| Energy-Based | Shipped | `artifex.generative_models.models.energy` | Retained EBM owners |
| Geometric | Shipped | `artifex.generative_models.models.geometric` | Retained point-cloud, graph, mesh, voxel, and protein owners |
| Audio models | Experimental | `artifex.generative_models.models.audio` | Importable, but narrower and less settled than the core image/model families |
| Backbones | Shipped | `artifex.generative_models.models.backbones` | Shared retained backbones used by multiple families |

### Core Framework

| Surface | Status | Runtime evidence | Notes |
| --- | --- | --- | --- |
| Base protocols and classes | Shipped | `artifex.generative_models.models.base`, `artifex.generative_models.core.base` | Current public protocol split includes `GenerativeModelProtocol` and `TrainableGenerativeModelProtocol` |
| Losses, distributions, and sampling | Shipped | `artifex.generative_models.core.losses`, `core.distributions`, `core.sampling` | Retained runtime owners |
| Layers and configuration | Shipped | `artifex.generative_models.core.layers`, `core.configuration` | Typed-config runtime surface |
| Evaluation package | Experimental | `artifex.generative_models.core.evaluation` | Current top-level exports are only `benchmarks` and `metrics` |
| Device management | Shipped | `artifex.generative_models.core.device_manager` | Retained runtime owner |

### Training

| Surface | Status | Runtime evidence | Notes |
| --- | --- | --- | --- |
| Trainers and training loops | Shipped | `artifex.generative_models.training` | Retained trainer and loop surface |
| Callbacks | Shipped | `artifex.generative_models.training.callbacks` | Current callback surface |
| Distributed data parallel helper | Shipped | `artifex.generative_models.training.distributed.data_parallel` | Narrow retained distributed helper |

### Extensions

| Surface | Status | Runtime evidence | Notes |
| --- | --- | --- | --- |
| Protein extensions | Shipped | `artifex.generative_models.extensions.protein` | Retained protein-extension bundle |

### Modalities

| Family | Status | Runtime evidence | Notes |
| --- | --- | --- | --- |
| Image | Shipped | `artifex.generative_models.modalities` | `ImageModality` |
| Text | Shipped | `artifex.generative_models.modalities` | `TextModality` |
| Protein | Shipped | `artifex.generative_models.modalities` | `ProteinModality` |
| Timeseries | Shipped | `artifex.generative_models.modalities` | `TimeseriesModality` |
| Audio | Experimental | `artifex.generative_models.modalities` | `AudioModality` remains narrower than the core retained image/text surface |
| Molecular | Experimental | `artifex.generative_models.modalities` | `MolecularModality` |
| Tabular | Experimental | `artifex.generative_models.modalities` | `TabularModality` |
| Multi-modal helper | Experimental | `artifex.generative_models.modalities.multi_modal` | Importable helper package, but not registry-backed |

### Inference Pipeline

| Surface | Status | Runtime evidence | Notes |
| --- | --- | --- | --- |
| Inference package namespace | Experimental | `artifex.generative_models.inference` | Live package namespace; the retained runtime surface is narrow |
| Production optimization pocket | Experimental | `artifex.generative_models.inference.optimization.production` | `ProductionOptimizer` plus `OptimizationTarget`, `OptimizationResult`, and `MonitoringMetrics`; only `jit_compilation` is a real applied optimization today |

### Benchmarks

| Surface | Status | Runtime evidence | Notes |
| --- | --- | --- | --- |
| Benchmark package | Experimental | `artifex.benchmarks` | Registry plus retained suites and model adapters |
| Retained suites | Experimental | `GeometricBenchmarkSuite`, `ProteinLigandBenchmarkSuite`, `MultiBetaVAEBenchmarkSuite` | Useful checked-in coverage, but not a complete stable benchmark matrix |

## Roadmap-Only Surfaces

These entries are still coming soon and should not be read as part of the
current importable runtime.

| Surface | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Video modality | Roadmap-only | `artifex.generative_models.modalities.video` is not importable | Keep video in roadmap pages until a real module exists |
| Neural ODE flow owners | Roadmap-only | No `Neural ODE` owner is exported from `artifex.generative_models.models.flow` | Do not publish as current support |
| CNF flow owners | Roadmap-only | No `CNF` owner is exported from `artifex.generative_models.models.flow` | Do not publish as current support |
| Broader inference stacks | Roadmap-only | Adaptation, batching, conversion, serving, and per-family generator layers are not a checked-in supported inference package | Keep these in roadmap language until real modules land |
| Complete benchmark matrix | Roadmap-only | Current benchmark layer is partial and experimental | Additional datasets, metrics, suites, and reporting remain roadmap work |
| StyleGAN, standalone UNet, conditioning package pages | Roadmap-only | The corresponding standalone runtime modules are not currently shipped | Keep them as roadmap references only |

## Fine-Tuning

There is no standalone `artifex.fine_tuning` runtime package today. Current RL
trainers live under `artifex.generative_models.training`, while LoRA, prefix
tuning, prompt tuning, distillation, few-shot transfer, and RLHF remain
roadmap-only topics.

## Data Pipeline

A broad internal data package is still roadmap-only. Current example and
tutorial flows use direct arrays plus `datarax` helpers where appropriate.
Dataset, loader, preprocessing, tokenizer, and streaming families beyond that
retained surface remain planned work.

## CLI

The supported CLI surface is the retained config command entrypoint under
`artifex.core.cli.config_commands`. Broader train/generate/serve/evaluate CLI
families remain roadmap-only.

## Utilities

The retained utility surface is small. `artifex.utils.file_utils`,
`artifex.generative_models.utils.jax.device`, the logging helpers, code-analysis
helpers, and protein visualization compatibility layers are live. Most other
utility families remain coming soon and should stay on roadmap-only pages until
real importable modules exist.

## Configuration

The retained configuration runtime is the frozen-dataclass-based surface under
`artifex.generative_models.core.configuration` plus the checked-in config loader
and template management helpers. Older freehand utility families such as config
merge or conversion helpers remain roadmap-only unless they are reintroduced as
real importable owners.

## Maintenance Notes

- Keep current-runtime rows tied to importable modules or exported owners.
- Keep missing families in the roadmap-only section until a real runtime module exists.
- Do not reintroduce summary-count tables unless they are mechanically derived from the checked-in inventory.
