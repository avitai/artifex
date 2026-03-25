# Metrics

Artifex ships a narrow evaluation surface with explicit dependencies. It is
not a full built-in evaluation catalog with pretrained backends.

## Supported Runtime Metrics

- `image:fid` via `FrechetInceptionDistance` with a caller-supplied
  `feature_extractor`
- `image:is` via `InceptionScore` with a caller-supplied `classifier`
- `text:perplexity` via `Perplexity` with a caller-supplied `model` or
  explicit `log_probs` at compute time

Unsupported metric specs raise during pipeline construction. The retained
pipeline does not silently skip `bleu`, `rouge`, or other unimplemented
names.

## Explicit Dependency Example

```python
import jax.numpy as jnp
from flax import nnx

from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.evaluation.metrics import EvaluationPipeline


def feature_extractor(images):
    means = jnp.mean(images, axis=(1, 2, 3))
    stds = jnp.std(images, axis=(1, 2, 3))
    return jnp.stack([means, stds], axis=1)


def classifier(images):
    logits = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    return jnp.tile(logits, (1, 10))


def language_model(inputs):
    return jnp.full(inputs.shape, -0.5)


config = EvaluationConfig(
    name="eval",
    metrics=["image:fid", "image:is", "text:perplexity"],
    metric_params={
        "fid": {"feature_extractor": feature_extractor},
        "is": {"classifier": classifier},
        "perplexity": {"model": language_model},
    },
)

pipeline = EvaluationPipeline(config, rngs=nnx.Rngs(0))
```

## Pipeline Rules

- `EvaluationPipeline` requires `modality:metric` specs.
- Supported metric specs are `image:fid`, `image:is`, and
  `text:perplexity`.
- Unsupported metric specs raise instead of being skipped.

## Registry Ownership

Registry-backed lookup, collections, and suites live in
`calibrax.metrics.MetricRegistry` and the surrounding CalibraX metric
composition helpers. Artifex does not ship a parallel registry wrapper in the
core evaluation package.
