# Metrics

Artifex ships one metric class layer, `artifex.benchmarks.metrics`, and computes
every score through calibrax's functions. There is no second metric layer under
`core`: the benchmark metrics are the runtime metrics.

## The Layer

Each metric is a `MetricBase` subclass built from an `EvaluationConfig` whose
`metric_params` carry the metric's settings. Backbones are caller-supplied:
artifex does not ship an Inception network, a perceptual network or a language
model, and it does not fall back to a placeholder outside an explicit demo mode.

| Metric | Class | Caller-supplied dependency | calibrax function |
| --- | --- | --- | --- |
| FID | `FIDMetric` | `feature_extractor` | `generative.frechet_feature_distance` |
| Inception score | `ISMetric` | `classifier` (logits) | `generative.inception_score_per_split` |
| Precision and recall | `PrecisionRecallMetric` | `feature_extractor` (optional) | `generative.manifold_precision`, `manifold_recall` and the density-weighted pair |
| Perplexity | `PerplexityMetric` | `model` (token log-probabilities) | `text.perplexity` with its mask |
| SSIM | `SSIMMetric` | none | `image.ssim` |

The `create_*_metric` factories build the configuration for you:

```python
from flax import nnx

from artifex.benchmarks.metrics.image import create_fid_metric, create_is_metric
from artifex.benchmarks.metrics.precision_recall import create_precision_recall_metric
from artifex.benchmarks.metrics.text import create_perplexity_metric

rngs = nnx.Rngs(0)
fid = create_fid_metric(rngs, feature_extractor=feature_extractor)
inception = create_is_metric(rngs, classifier=classifier, splits=10)
precision_recall = create_precision_recall_metric(rngs, feature_extractor=feature_extractor, k=3)
perplexity = create_perplexity_metric(rngs=rngs, model=language_model)

fid.compute(real_images, generated_images)          # {"fid_score": ...}
inception.compute(real_images, generated_images)    # {"inception_score": ..., "inception_score_std": ...}
precision_recall.compute(real_images, generated_images)  # {"precision", "recall", "f1_score"}
perplexity.compute(None, token_ids, mask=attention_mask)  # {"perplexity": ...}
```

A missing dependency raises at construction (`ValueError` naming the parameter);
a non-callable one raises `TypeError`. `ISMetric` raises when `splits` exceeds
the number of samples instead of shrinking the split count.

## Tier 1 Registration

Importing `artifex.benchmarks.metrics` registers the backbone-based scores in
`calibrax.metrics.MetricRegistry` under the `frozen_backbone` tier, next to
calibrax's own Tier 0 functions:

| Registered name | Signature |
| --- | --- |
| `fid_from_backbone` | `(real, generated, *, feature_extractor)` |
| `inception_score_from_backbone` | `(generated, *, classifier, splits=10)` |
| `precision_from_backbone` | `(real, generated, *, feature_extractor=None, k=3, density_weighted=False)` |
| `recall_from_backbone` | `(real, generated, *, feature_extractor=None, k=3, density_weighted=False)` |
| `perplexity_from_model` | `(inputs, *, model, mask=None)` |

```python
from calibrax.metrics import MetricRegistry, MetricTier

import artifex.benchmarks.metrics  # registers the Tier 1 entries

registry = MetricRegistry()
score = registry.get_function("fid_from_backbone")
fid_value = score(real_images, generated_images, feature_extractor=feature_extractor)
tier_one = [entry.name for entry in registry.list_by_tier(MetricTier.FROZEN_BACKBONE)]
```

The Tier 0 functions themselves (`frechet_feature_distance`, `inception_score`,
`manifold_precision`, `perplexity`, ...) stay calibrax's; artifex adds the
feature-extraction step and the config-driven classes, nothing else.

## Demo Mode

`mock_inception=True`, `mock_implementation=True` and `use_mock=True` select
retained demo backends for the example suites. They are explicit opt-ins and
never a default; see the [benchmarks overview](../benchmarks/index.md).
