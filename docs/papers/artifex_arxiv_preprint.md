# Artifex: A Generative Modeling Research Library Under Active Development

**Authors:** Mahdi Shafiei
**Affiliation:** Avitai Bio
**Email:** <mahdi@avitai.bio>
**Date:** January 26, 2026
**Status:** Research preprint describing an actively changing codebase
**arXiv ID:** [To be assigned]

---

## Abstract

Artifex is a JAX/Flax NNX generative modeling library under active development.
The current checked-in runtime exposes typed model protocols, a centralized
model factory, retained model-family packages for VAE, GAN, diffusion, flow,
autoregressive, energy-based, and geometric work, and retained modality owners
for image, text, audio, protein, molecular, tabular, and time-series data.
The repository also keeps narrower experimental surfaces for benchmarking,
experimental production optimization, and multi-modal helpers. This preprint is
therefore a status document for a research library in motion, not a completeness
claim about one finished end-to-end platform. Video modalities, Neural ODE/CNF
flows, broader serving and adaptation stacks, and a larger evaluation and
benchmark matrix remain roadmap work rather than live importable API.

## Shipped Runtime Surface

The public runtime surface described here is limited to importable owners that
exist in the checked-in package today.

### Shared model protocols

The base model protocol does not own one universal training objective. The live
runtime splits generic generation from the narrower single-objective training
surface:

```python
from typing import Any, Protocol, runtime_checkable

import jax
from flax import nnx


@runtime_checkable
class GenerativeModelProtocol(Protocol):
    def __call__(self, x: Any, *, rngs: nnx.Rngs | None = None, **kwargs) -> dict[str, Any]:
        ...

    def generate(self, n_samples: int = 1, *, rngs: nnx.Rngs | None = None, **kwargs) -> jax.Array:
        ...


@runtime_checkable
class TrainableGenerativeModelProtocol(GenerativeModelProtocol, Protocol):
    def loss_fn(
        self,
        batch: Any,
        model_outputs: dict[str, Any],
        *,
        rngs: nnx.Rngs | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        ...
```

`GenerativeModelProtocol` covers forward and generation behavior shared across
families. `TrainableGenerativeModelProtocol` is the narrower runtime contract for
families that still own a single `loss_fn(...)` surface.

### Modality adapters

The shared modality adapter contract is a typed creation and adaptation surface.
It is not a generic helper-trio contract layered above the real adapter owner:

```python
from typing import Any, Protocol, runtime_checkable

from flax import nnx

from artifex.generative_models.core.base import GenerativeModel


@runtime_checkable
class ModelAdapter(Protocol):
    def adapt(self, model: Any, config: Any) -> Any:
        ...

    def create(self, config: Any, *, rngs: nnx.Rngs, **kwargs: Any) -> GenerativeModel:
        ...
```

Preprocessing and postprocessing remain modality-local concerns on the concrete
modality or processor owners, not part of one shared adapter API.

### Current package inventory

The retained runtime currently includes:

- model-family packages for VAE, GAN, diffusion, flow, autoregressive,
  energy-based, and geometric owners
- a live flow package built around retained owners such as `RealNVP`, `Glow`,
  `MAF`, `IAF`, `NeuralSplineFlow`, and `ConditionalRealNVP`
- retained modality exports including `ImageModality`, `TextModality`,
  `ProteinModality`, `AudioModality`, `MolecularModality`,
  `TabularModality`, and `TimeseriesModality`
- a centralized factory layer for typed-config model creation

## Experimental But Importable Surfaces

Several checked-in surfaces are importable today, but they should be read as
narrow experimental runtime pockets rather than complete platform guarantees.

### Evaluation and benchmarks

`artifex.generative_models.core.evaluation` currently exports only the
`benchmarks` and `metrics` packages:

```python
import artifex.generative_models.core.evaluation as evaluation

assert evaluation.__all__ == ["benchmarks", "metrics"]
```

The retained benchmark layer lives in `artifex.benchmarks` and currently exposes
registry helpers plus a small set of retained suites such as
`BenchmarkRegistry`, `GeometricBenchmarkSuite`,
`ProteinLigandBenchmarkSuite`, and `MultiBetaVAEBenchmarkSuite`. This is an
experimental benchmark substrate, not a claim that the repository ships one
complete cross-modality benchmark platform.

### Experimental inference optimization

The only checked-in inference optimization pocket is
`artifex.generative_models.inference.optimization.production`:

```python
from artifex.generative_models.inference.optimization.production import (
    MonitoringMetrics,
    OptimizationResult,
    OptimizationTarget,
    ProductionOptimizer,
)
```

That module keeps one real optimization technique today, `jit_compilation`, plus
latency and throughput measurement helpers around a compiled pipeline. It should
be read as experimental optimization infrastructure rather than a broad serving
or deployment stack.

### Experimental modality helpers

`artifex.generative_models.modalities.multi_modal` remains importable as an
experimental helper package for aligned datasets, fusion helpers, and evaluation
utilities. It is not registry-backed and should not be confused with the shared
current modality registry surface.

## Roadmap-Only And Future Work

The following topics remain future-work or roadmap-only surfaces rather than
current runtime imports:

- video modality owners such as `artifex.generative_models.modalities.video`
- Neural ODE and CNF flow owners
- broader inference layers for adaptation, batching, serving, conversion, and
  per-family generator modules beyond the retained optimization pocket
- a larger benchmark matrix with stable datasets, metrics, suites, and
  reporting layers across every model family and modality
- deployment claims that would require a validated, stable serving contract

The companion [planned modules roadmap](../roadmap/planned-modules.md) is the
truthful status page for those future surfaces.

## Research Positioning

Artifex is aimed at researchers who want a typed, modular JAX codebase for
exploration across multiple generative families. The checked-in code already has
useful retained runtime owners, but the library is still tightening its public
contract surface. This preprint should therefore be read as a research
positioning document for the importable runtime that exists today, with explicit
separation between retained runtime, experimental subsystems, and roadmap work.

## Availability

Source code: <https://github.com/avitai/artifex>

## Citation

```bibtex
@software{artifex_2025,
  title = {Artifex: Generative Modeling Research Library},
  author = {Shafiei, Mahdi and contributors},
  year = {2025},
  url = {https://github.com/avitai/artifex},
  version = {0.1.0}
}
```
