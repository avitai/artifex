# Core

The retained `artifex.generative_models.core` package is a narrower shared-runtime layer.

## Overview

```python
from artifex.generative_models import core

configuration = core.configuration
evaluation = core.evaluation
layers = core.layers
losses = core.losses
protocols = core.protocols
sampling = core.sampling
```

`core` stays lazy at import time and should be read through its surviving
child-module boundaries.

## Sampling

Sampling helpers live under `core.sampling`.

```python
from artifex.generative_models.core.sampling import BlackJAXNUTS, mcmc_sampling, sde_sampling
```

## Evaluation Metrics

Evaluation metrics live under `core.evaluation.metrics`.

```python
from artifex.generative_models.core.evaluation.metrics import (
    EvaluationPipeline,
    FrechetInceptionDistance,
    InceptionScore,
    PrecisionRecall,
)
```

## Layers

Shared layers live under `core.layers`.

```python
from artifex.generative_models.core.layers import (
    FlashMultiHeadAttention,
    ResNetBlock,
    TransformerEncoder,
)
```

Two backbone families ship without an artifex consumer yet: the
Kolmogorov-Arnold Network layers and the Clifford-algebra layers. They are
documented on their own pages.

[:octicons-arrow-right-24: KAN Layers](kan.md) | [:octicons-arrow-right-24: Clifford Layers](clifford.md)

## Protocols

Core protocol types live under `core.protocols`.

```python
from artifex.generative_models.core.protocols import (
    BatchableDatasetProtocol,
    MetricBase,
    NoiseScheduleProtocol,
)

protocol_types = (BatchableDatasetProtocol, MetricBase, NoiseScheduleProtocol)
```

The remaining model-facing evaluation protocol surface includes
`BenchmarkModelProtocol` and `DatasetProtocol`.

Benchmark runtime types now live under `artifex.benchmarks.core`, not under
`core.protocols` or `core.evaluation`. Device meshes and sharding strategies are
substrax's (`substrax.mesh`), not artifex's.

[:octicons-arrow-right-24: Evaluation Protocols](evaluation.md) | [:octicons-arrow-right-24: Benchmark Runtime](benchmarks.md)

## Configuration, Losses, And Distributions

The remaining core child modules stay package-owned and should be read through their real module boundaries:

- `core.configuration` for the surviving typed configuration surface
- `core.losses` for functional loss primitives and family-local objective helpers
- `core.distributions` for shared distribution implementations

See the owner pages in this section for the detailed module-level contracts.

## Surface Map

| Surface | Owner |
| --- | --- |
| `core.configuration` | shared configuration package and typed config helpers |
| `core.distributions` | distribution implementations and transforms |
| `core.evaluation.metrics` | evaluation metrics and explicit dependency pipelines |
| `core.layers` | shared layers and architectural building blocks |
| `core.losses` | loss primitives and objective helpers |
| `core.protocols` | evaluation, metric, and training protocols |
| `core.sampling` | BlackJAX, MCMC, and SDE sampling helpers |
| `artifex.benchmarks.core` | benchmark configs, results, NNX benchmark bases, and runners |
| `core` top-level helpers | checkpointing, rematerialization, and device helpers |

## Related Documentation

- [API Reference](../api/core/base.md)
- [Sampling API](../api/sampling.md)
- [Evaluation Protocols](evaluation.md)
- [Benchmark Runtime](benchmarks.md)
