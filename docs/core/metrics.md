# Metrics

Artifex provides a comprehensive metrics system for evaluating generative models. The metrics framework is built on Flax NNX and follows JAX-compatible patterns for efficient computation.

## Overview

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } **Image Metrics**

    ---

    FID, Inception Score, and feature-based metrics for image quality

    [:octicons-arrow-right-24: Image Metrics](#image-metrics)

- :material-text:{ .lg .middle } **Text Metrics**

    ---

    Perplexity and sequence-based metrics for language models

    [:octicons-arrow-right-24: Text Metrics](#text-metrics)

- :material-chart-line:{ .lg .middle } **Distribution Metrics**

    ---

    Statistical comparison between real and generated distributions

    [:octicons-arrow-right-24: Distribution Metrics](#distribution-metrics)

- :material-cog:{ .lg .middle } **Metric Pipeline**

    ---

    Compose and orchestrate metrics across modalities

    [:octicons-arrow-right-24: Evaluation Pipeline](#evaluation-pipeline)

</div>

## Quick Start

```python
from artifex.generative_models.core.evaluation.metrics import (
    FrechetInceptionDistance,
    InceptionScore,
    Perplexity,
    MetricsRegistry,
    EvaluationPipeline,
)
from flax import nnx

# Create a metric instance
key = jax.random.key(42)
fid_metric = FrechetInceptionDistance(
    batch_size=32,
    rngs=nnx.Rngs(params=key),
)

# Compute FID between real and generated images
results = fid_metric.compute(real_images, generated_images)
print(f"FID: {results['fid']:.2f}")
```

---

## Base Classes

All metrics inherit from base classes that provide consistent interfaces and JAX compatibility.

### MetricModule

The foundation class for all metrics, providing the basic compute interface.

```python
from artifex.generative_models.core.evaluation.metrics.base import MetricModule
from flax import nnx

class MyMetric(MetricModule):
    """Custom metric implementation."""

    def __init__(
        self,
        name: str = "my_metric",
        batch_size: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(name=name, batch_size=batch_size, rngs=rngs)

    def compute(self, predictions, targets) -> dict[str, float]:
        """Compute the metric value.

        Returns:
            Dictionary with metric name and value
        """
        value = ...  # Your computation
        return {self.name: float(value)}
```

### FeatureBasedMetric

For metrics requiring feature extraction (e.g., FID, Inception Score).

```python
from artifex.generative_models.core.evaluation.metrics.base import FeatureBasedMetric

class MyFeatureMetric(FeatureBasedMetric):
    def __init__(
        self,
        feature_extractor=None,
        batch_size: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(
            name="my_feature_metric",
            feature_extractor=feature_extractor,
            batch_size=batch_size,
            rngs=rngs,
        )

    def compute(self, real_data, generated_data) -> dict[str, float]:
        # Extract features using batch processing
        real_features = self.extract_features(real_data)
        gen_features = self.extract_features(generated_data)

        # Compare features
        score = ...
        return {self.name: float(score)}
```

### DistributionMetric

For metrics comparing statistical distributions.

```python
from artifex.generative_models.core.evaluation.metrics.base import DistributionMetric

class MyDistributionMetric(DistributionMetric):
    def compute(self, real_data, generated_data) -> dict[str, float]:
        # Compute statistics for each distribution
        real_stats = self.compute_statistics(real_data)
        gen_stats = self.compute_statistics(generated_data)

        # Returns: {"mean": ..., "covariance": ..., "std": ...}
        distance = self._compare_distributions(real_stats, gen_stats)
        return {self.name: float(distance)}
```

### SequenceMetric

For metrics operating on sequences (text, time series).

```python
from artifex.generative_models.core.evaluation.metrics.base import SequenceMetric

class MySequenceMetric(SequenceMetric):
    def compute(self, sequences, masks=None) -> dict[str, float]:
        # Process sequences with optional masking
        processed = self.process_sequences(sequences, masks)

        score = ...
        return {self.name: float(score)}
```

---

## Image Metrics

### FID (Fréchet Inception Distance)

Measures the distance between feature distributions of real and generated images.

```python
from artifex.generative_models.core.evaluation.metrics import FrechetInceptionDistance

fid = FrechetInceptionDistance(
    feature_extractor=None,  # Uses default Inception-v3
    batch_size=32,
    rngs=nnx.Rngs(params=key),
)

# Compute FID
results = fid.compute(real_images, generated_images)
print(f"FID: {results['fid']:.2f}")  # Lower is better
```

**Interpretation:**

| FID Range | Quality |
|-----------|---------|
| < 10 | Excellent |
| 10-50 | Good |
| 50-100 | Moderate |
| > 100 | Poor |

**How FID Works:**

1. Extract features from both real and generated images using Inception-v3
2. Fit multivariate Gaussians to both feature sets
3. Compute Fréchet distance between the two Gaussians

```python
# The Fréchet distance formula:
# FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^½)
```

### Inception Score (IS)

Measures both quality and diversity of generated images.

```python
from artifex.generative_models.core.evaluation.metrics import InceptionScore

is_metric = InceptionScore(
    classifier=None,  # Uses default Inception classifier
    batch_size=32,
    splits=10,  # Number of splits for computing variance
    rngs=nnx.Rngs(params=key),
)

# Compute IS
results = is_metric.compute(generated_images)
print(f"IS: {results['is']:.2f}")  # Higher is better
```

**Interpretation:**

- Higher IS indicates better quality and diversity
- Real CIFAR-10: ~11.24
- Real ImageNet: ~100+

### Precision and Recall

Measures coverage (recall) and quality (precision) separately.

```python
from artifex.generative_models.core.evaluation.metrics import PrecisionRecall

pr_metric = PrecisionRecall(
    k=3,  # k-nearest neighbors
    rngs=nnx.Rngs(params=key),
)

results = pr_metric.compute(real_images, generated_images)
print(f"Precision: {results['precision']:.3f}")  # Quality
print(f"Recall: {results['recall']:.3f}")        # Coverage
```

---

## Text Metrics

### Perplexity

Measures how well a model predicts text sequences.

```python
from artifex.generative_models.core.evaluation.metrics import Perplexity

ppl_metric = Perplexity(
    model=language_model,
    batch_size=32,
    rngs=nnx.Rngs(params=key),
)

results = ppl_metric.compute(text_sequences)
print(f"Perplexity: {results['perplexity']:.2f}")  # Lower is better
```

**Interpretation:**

- Lower perplexity indicates better prediction
- Typical values: 10-100 depending on dataset complexity

---

## Distribution Metrics

### Statistical Comparison

Compare distributions using mean and covariance statistics.

```python
from artifex.generative_models.core.evaluation.metrics.base import DistributionMetric

# Compute statistics
stats = DistributionMetric.compute_statistics(features)
# Returns:
# {
#     "mean": jax.Array,        # Feature means
#     "covariance": jax.Array,  # Covariance matrix
#     "std": jax.Array,         # Standard deviations
# }
```

---

## Metrics Registry

The registry provides centralized management of metrics.

### Registering Metrics

```python
from artifex.generative_models.core.evaluation.metrics import MetricsRegistry

registry = MetricsRegistry()

# Register a custom metric
def my_custom_metric(predictions, targets):
    error = jnp.mean((predictions - targets) ** 2)
    return {"my_metric": float(error)}

registry.register_metric_computer("my_metric", my_custom_metric)
```

### Using the Registry

```python
# List available metrics
available = registry.list_available_metrics()
print(f"Available metrics: {available}")

# Check if metric exists
if registry.has_metric("mse"):
    results = registry.compute_metrics("mse", predictions, targets)
    print(f"MSE: {results['mse']:.4f}")
```

### Built-in Metrics

The registry comes with standard metrics pre-registered:

| Metric | Description |
|--------|-------------|
| `accuracy` | Classification accuracy |
| `mse` | Mean Squared Error |
| `mae` | Mean Absolute Error |

---

## Evaluation Pipeline

The evaluation pipeline orchestrates metrics across multiple modalities.

### Basic Usage

```python
from artifex.generative_models.core.evaluation.metrics import EvaluationPipeline
from artifex.generative_models.core.configuration import EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    metrics=["image:fid", "image:is", "text:perplexity"],
    metric_params={
        "fid": {"batch_size": 64},
        "is": {"splits": 10},
    },
)

# Create pipeline
pipeline = EvaluationPipeline(config, rngs=nnx.Rngs(params=key))

# Run evaluation
data = {
    "image": {"real": real_images, "generated": gen_images},
    "text": {"real": real_text, "generated": gen_text},
}

results = pipeline.evaluate(data)
# Returns: {"image": {"fid": ..., "is": ...}, "text": {"perplexity": ...}}
```

### Metric Composer

Compose and aggregate metrics with custom rules.

```python
from artifex.generative_models.core.evaluation.metrics import MetricComposer

config = EvaluationConfig(
    metrics=["image:fid", "image:is"],
    metric_params={
        "composition_rules": {
            "quality_score": {
                "weights": {"fid": -0.5, "is": 0.5},  # FID negative (lower is better)
                "normalization": "min_max",
            },
        },
    },
)

composer = MetricComposer(config, rngs=nnx.Rngs(params=key))

# Compose metrics into single score
composed = composer.compose({"fid": 25.0, "is": 8.5})
print(f"Quality Score: {composed['quality_score']:.3f}")
```

### Cross-Modality Aggregation

```python
# Aggregate results across modalities
config = EvaluationConfig(
    metrics=["image:fid", "text:perplexity"],
    metric_params={
        "composer_settings": {
            "aggregation_strategy": "weighted_average",
            "modality_weights": {"image": 0.6, "text": 0.4},
        },
    },
)

composer = MetricComposer(config, rngs=nnx.Rngs(params=key))

modality_results = {
    "image": {"fid": 25.0},
    "text": {"perplexity": 45.0},
}

aggregated = composer.aggregate_modalities(modality_results)
print(f"Cross-modality Score: {aggregated['cross_modality_score']:.3f}")
```

### Modality Metrics Manager

Select appropriate metrics based on modality and quality requirements.

```python
from artifex.generative_models.core.evaluation.metrics import ModalityMetrics

config = EvaluationConfig(
    metrics=["image:fid", "image:is", "text:bleu"],
    metric_params={
        "quality_levels": {
            "fast": ["fid"],
            "standard": ["fid", "is"],
            "comprehensive": ["fid", "is", "lpips", "precision_recall"],
        },
    },
)

manager = ModalityMetrics(config, rngs=nnx.Rngs(params=key))

# Get supported modalities
modalities = manager.get_supported_modalities()
print(f"Supported: {modalities}")

# Select metrics for quality level
metrics = manager.select_metrics("image", quality_level="standard")
print(f"Selected metrics: {metrics}")  # ["fid", "is"]
```

---

## Creating Custom Metrics

### Step-by-Step Guide

1. **Choose the appropriate base class:**
   - `MetricModule`: General metrics
   - `FeatureBasedMetric`: Metrics requiring feature extraction
   - `DistributionMetric`: Distribution comparison metrics
   - `SequenceMetric`: Sequence-based metrics

2. **Implement the `compute` method:**

```python
from artifex.generative_models.core.evaluation.metrics.base import MetricModule
import jax.numpy as jnp

class SSIMMetric(MetricModule):
    """Structural Similarity Index Metric."""

    def __init__(
        self,
        window_size: int = 11,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(name="ssim", batch_size=32, rngs=rngs)
        self.window_size = window_size

    def compute(
        self,
        image1: jax.Array,
        image2: jax.Array,
    ) -> dict[str, float]:
        """Compute SSIM between two images.

        Args:
            image1: First image batch
            image2: Second image batch

        Returns:
            Dictionary with SSIM score
        """
        # SSIM computation logic
        ssim_value = self._compute_ssim(image1, image2)
        return {self.name: float(ssim_value)}

    def _compute_ssim(self, img1, img2):
        # Implementation details...
        pass
```

3. **Register with the registry (optional):**

```python
registry = MetricsRegistry()
registry.register_metric_computer(
    "ssim",
    lambda img1, img2: SSIMMetric().compute(img1, img2)
)
```

---

## Best Practices

!!! success "DO"
    - Use appropriate metrics for each modality
    - Report multiple metrics for comprehensive evaluation
    - Include confidence intervals with multiple runs
    - Use consistent sample sizes for fair comparison

!!! danger "DON'T"
    - Don't cherry-pick metrics that favor your model
    - Don't use incompatible feature extractors for comparison
    - Don't ignore statistical significance
    - Don't compare metrics computed on different datasets

---

## Summary

The Artifex metrics system provides:

- **Base Classes**: `MetricModule`, `FeatureBasedMetric`, `DistributionMetric`, `SequenceMetric`
- **Image Metrics**: FID, Inception Score, Precision/Recall
- **Text Metrics**: Perplexity and sequence metrics
- **Registry**: Centralized metric management and discovery
- **Pipeline**: Multi-modal evaluation orchestration
- **Composition**: Metric aggregation and cross-modality scoring

Use the evaluation pipeline for comprehensive model assessment across modalities.
