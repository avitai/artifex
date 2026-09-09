# Precision Recall

**Module:** `benchmarks.metrics.precision_recall`

**Source:** `benchmarks/metrics/precision_recall.py`

## Overview

Precision and recall of a generative model over k-NN manifolds, the metrics of
"Improved Precision and Recall Metric for Assessing Generative Models"
(Kynkäänniemi et al., 2019). Precision is the fraction of generated samples
inside the k-nearest-neighbour manifold of the real features, recall the
fraction of real samples inside the generated manifold. The manifold estimates
and their density-weighted variants are calibrax's (`manifold_precision`,
`manifold_recall`, `density_weighted_precision`, `density_weighted_recall`);
this module adds feature extraction, the config-driven metric class and the
benchmark that samples a model.

Samples of any shape are flattened to one row per sample after the optional
`feature_extractor` runs. `k` must be at least 1 and below both sample counts;
empty or too-small sets raise `ValueError`, as do mismatched feature dimensions.

## Functions

### precision_from_backbone

```python
def precision_from_backbone(real, generated, *, feature_extractor=None, k=3, density_weighted=False) -> float
```

### recall_from_backbone

```python
def recall_from_backbone(real, generated, *, feature_extractor=None, k=3, density_weighted=False) -> float
```

Both register in `calibrax.metrics.MetricRegistry` as `frozen_backbone` metrics.

### f1_score

```python
def f1_score(precision: float, recall: float) -> float
```

### create_precision_recall_metric

```python
def create_precision_recall_metric(rngs, *, feature_extractor=None, k=3, density_weighted=False, batch_size=32, config_name="precision_recall_metric") -> PrecisionRecallMetric
```

## Classes

### PrecisionRecallMetric

```python
class PrecisionRecallMetric(MetricBase)
```

Built from an `EvaluationConfig`; reads `feature_extractor`, `k` and
`density_weighted` from `metric_params["precision_recall"]`. `compute(real, generated)`
returns `precision`, `recall` and `f1_score`.

### PrecisionRecallBenchmark

```python
class PrecisionRecallBenchmark(Benchmark)
```

`PrecisionRecallBenchmark(k=3, num_samples=1000, random_seed=None, *, density_weighted=False)`
draws `num_samples` from the model, takes the dataset as an array or subsamples an
indexable one, and returns a `BenchmarkResult` named `precision_recall` with the
three metrics. A model's name never changes how it is scored.
