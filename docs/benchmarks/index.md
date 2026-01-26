# Benchmarks & Evaluation

Artifex provides a comprehensive benchmarking framework for evaluating generative models across different modalities. This includes standardized metrics, evaluation protocols, and benchmark suites for rigorous model comparison.

## Overview

<div class="grid cards" markdown>

- :material-image:{ .lg .middle } **Image Metrics**

    ---

    FID, Inception Score, LPIPS, Precision/Recall for image generation quality

    [:octicons-arrow-right-24: Image Metrics](#image-metrics)

- :material-text:{ .lg .middle } **Text Metrics**

    ---

    Perplexity, BLEU, ROUGE, and diversity scores for text generation

    [:octicons-arrow-right-24: Text Metrics](#text-metrics)

- :material-waveform:{ .lg .middle } **Audio Metrics**

    ---

    MCD, spectral metrics, and perceptual quality for audio generation

    [:octicons-arrow-right-24: Audio Metrics](#audio-metrics)

- :material-chart-scatter-plot:{ .lg .middle } **Disentanglement**

    ---

    MIG, SAP, DCI, and FactorVAE scores for latent space evaluation

    [:octicons-arrow-right-24: Disentanglement Metrics](#disentanglement-metrics)

</div>

## Quick Start

Run benchmarks on your models with just a few lines of code:

```python
from artifex.benchmarks import BenchmarkRunner
from artifex.benchmarks.metrics.image import FIDMetric, InceptionScoreMetric
from artifex.benchmarks.suites.image_suite import ImageBenchmarkSuite

# Create benchmark suite
suite = ImageBenchmarkSuite(
    metrics=[FIDMetric(), InceptionScoreMetric()],
    dataset_name="cifar10",
)

# Run benchmarks
results = suite.evaluate(model, num_samples=10000)

# Print results
print(f"FID: {results['fid']:.2f}")
print(f"IS: {results['inception_score']:.2f} +/- {results['inception_score_std']:.2f}")
```

---

## Image Metrics

Metrics for evaluating image generation quality, diversity, and realism.

### FID (Frechet Inception Distance)

Measures the distance between feature distributions of real and generated images.

```python
from artifex.benchmarks.metrics.image import FIDMetric

fid_metric = FIDMetric()
fid_score = fid_metric.compute(real_images, generated_images)
print(f"FID: {fid_score:.2f}")  # Lower is better
```

**Interpretation:**

- FID < 10: Excellent quality
- FID 10-50: Good quality
- FID 50-100: Moderate quality
- FID > 100: Poor quality

### Inception Score (IS)

Measures both quality and diversity of generated images.

```python
from artifex.benchmarks.metrics.image import InceptionScoreMetric

is_metric = InceptionScoreMetric()
is_score, is_std = is_metric.compute(generated_images)
print(f"IS: {is_score:.2f} +/- {is_std:.2f}")  # Higher is better
```

### LPIPS (Learned Perceptual Image Patch Similarity)

Measures perceptual similarity using deep features.

```python
from artifex.benchmarks.metrics.image import LPIPSMetric

lpips_metric = LPIPSMetric()
lpips_score = lpips_metric.compute(image1, image2)
print(f"LPIPS: {lpips_score:.4f}")  # Lower means more similar
```

### Precision and Recall

Measures coverage and quality separately.

```python
from artifex.benchmarks.metrics.precision_recall import PrecisionRecallMetric

pr_metric = PrecisionRecallMetric()
precision, recall = pr_metric.compute(real_images, generated_images)
print(f"Precision: {precision:.3f}")  # Quality (higher is better)
print(f"Recall: {recall:.3f}")        # Coverage (higher is better)
```

### Documentation

- [Image Metrics](image.md) - Detailed image metric implementations
- [Unified Image Metrics](image_unified.md) - Combined image evaluation
- [Inception Metrics](inception_metrics.md) - FID and IS details
- [Precision/Recall](precision_recall.md) - P/R implementation

---

## Text Metrics

Metrics for evaluating text generation quality and coherence.

### Perplexity

Measures how well a model predicts text.

```python
from artifex.benchmarks.metrics.text import PerplexityMetric

ppl_metric = PerplexityMetric()
perplexity = ppl_metric.compute(model, test_text)
print(f"Perplexity: {perplexity:.2f}")  # Lower is better
```

### BLEU Score

Measures n-gram overlap with reference text.

```python
from artifex.benchmarks.metrics.text import BLEUMetric

bleu_metric = BLEUMetric()
bleu_score = bleu_metric.compute(generated_text, reference_text)
print(f"BLEU: {bleu_score:.3f}")  # Higher is better (0-1)
```

### Diversity Metrics

Measures vocabulary diversity and uniqueness.

```python
from artifex.benchmarks.metrics.diversity import DiversityMetric

diversity_metric = DiversityMetric()
scores = diversity_metric.compute(generated_texts)
print(f"Distinct-1: {scores['distinct_1']:.3f}")
print(f"Distinct-2: {scores['distinct_2']:.3f}")
```

### Documentation

- [Text Metrics](text.md) - Text evaluation metrics
- [Perplexity](perplexity.md) - Perplexity implementation
- [Diversity](diversity.md) - Diversity metrics

---

## Audio Metrics

Metrics for evaluating audio generation quality.

### MCD (Mel-Cepstral Distortion)

Measures spectral difference between audio signals.

```python
from artifex.benchmarks.metrics.audio import MCDMetric

mcd_metric = MCDMetric()
mcd_score = mcd_metric.compute(generated_audio, reference_audio)
print(f"MCD: {mcd_score:.2f} dB")  # Lower is better
```

### Spectral Metrics

Measures frequency-domain characteristics.

```python
from artifex.benchmarks.metrics.audio import SpectralMetrics

spectral_metrics = SpectralMetrics()
results = spectral_metrics.compute(generated_audio, reference_audio)
print(f"Spectral Convergence: {results['spectral_convergence']:.4f}")
print(f"Log STFT Magnitude: {results['log_stft_magnitude']:.4f}")
```

### Documentation

- [Audio Metrics](audio.md) - Audio evaluation metrics

---

## Disentanglement Metrics

Metrics for evaluating VAE latent space quality.

### MIG (Mutual Information Gap)

Measures how well individual latents capture single factors.

```python
from artifex.benchmarks.metrics.disentanglement import MIGMetric

mig_metric = MIGMetric()
mig_score = mig_metric.compute(model, dataset)
print(f"MIG: {mig_score:.3f}")  # Higher is better (0-1)
```

### SAP (Separated Attribute Predictability)

Measures predictability of factors from latents.

```python
from artifex.benchmarks.metrics.disentanglement import SAPMetric

sap_metric = SAPMetric()
sap_score = sap_metric.compute(model, dataset)
print(f"SAP: {sap_score:.3f}")  # Higher is better (0-1)
```

### DCI (Disentanglement, Completeness, Informativeness)

Comprehensive disentanglement evaluation.

```python
from artifex.benchmarks.metrics.disentanglement import DCIMetric

dci_metric = DCIMetric()
results = dci_metric.compute(model, dataset)
print(f"Disentanglement: {results['disentanglement']:.3f}")
print(f"Completeness: {results['completeness']:.3f}")
print(f"Informativeness: {results['informativeness']:.3f}")
```

### Documentation

- [Disentanglement Metrics](disentanglement.md) - Latent space evaluation

---

## Benchmark Suites

Pre-configured evaluation protocols for different use cases.

### Image Suite

```python
from artifex.benchmarks.suites.image_suite import ImageBenchmarkSuite

suite = ImageBenchmarkSuite(
    dataset_name="cifar10",
    metrics=["fid", "is", "lpips", "precision_recall"],
)

results = suite.run(model, num_samples=50000)
```

### Multi-Beta VAE Suite

```python
from artifex.benchmarks.suites.multi_beta_vae_suite import MultiBetaVAESuite

suite = MultiBetaVAESuite(
    beta_values=[0.1, 0.5, 1.0, 2.0, 4.0],
    metrics=["reconstruction", "kl", "mig", "sap"],
)

results = suite.run(model, dataset)
```

### Geometric Suite

```python
from artifex.benchmarks.suites.geometric_suite import GeometricBenchmarkSuite

suite = GeometricBenchmarkSuite(
    metrics=["chamfer", "earth_mover", "coverage"],
)

results = suite.run(model, point_cloud_dataset)
```

### Documentation

- [Image Suite](image_suite.md) - Image benchmark suite
- [Text Suite](text_suite.md) - Text benchmark suite
- [Audio Suite](audio_suite.md) - Audio benchmark suite
- [Geometric Suite](geometric_suite.md) - 3D geometry benchmarks
- [Multi-Beta VAE Suite](multi_beta_vae_suite.md) - VAE disentanglement
- [Standard Suite](standard.md) - Standard evaluation protocols

---

## Performance Benchmarks

Measure computational performance of models.

### Latency

```python
from artifex.benchmarks.performance.latency import LatencyBenchmark

latency_bench = LatencyBenchmark()
results = latency_bench.measure(model, batch_size=32, num_iterations=100)
print(f"Mean latency: {results['mean_ms']:.2f} ms")
print(f"P99 latency: {results['p99_ms']:.2f} ms")
```

### Throughput

```python
from artifex.benchmarks.performance.throughput import ThroughputBenchmark

throughput_bench = ThroughputBenchmark()
results = throughput_bench.measure(model, batch_sizes=[1, 8, 32, 64, 128])
print(f"Peak throughput: {results['peak_samples_per_sec']:.1f} samples/sec")
```

### Memory

```python
from artifex.benchmarks.performance.memory import MemoryBenchmark

memory_bench = MemoryBenchmark()
results = memory_bench.measure(model, batch_size=32)
print(f"Peak memory: {results['peak_mb']:.1f} MB")
```

### Documentation

- [Latency](latency.md) - Latency measurements
- [Throughput](throughput.md) - Throughput benchmarks
- [Memory](memory.md) - Memory profiling
- [Scaling](scaling.md) - Scaling analysis
- [Optimization](optimization.md) - Performance optimization

---

## Visualization

Tools for visualizing benchmark results.

```python
from artifex.benchmarks.visualization.plots import plot_metrics
from artifex.benchmarks.visualization.comparison import compare_models

# Plot metrics over training
plot_metrics(training_history, metrics=["fid", "is"])

# Compare multiple models
compare_models(
    models={"VAE": vae_results, "GAN": gan_results, "Diffusion": diffusion_results},
    metrics=["fid", "is", "lpips"],
)
```

### Documentation

- [Plots](plots.md) - Plotting utilities
- [Comparison](comparison.md) - Model comparison
- [Dashboard](dashboard.md) - Interactive dashboard
- [Image Grid](image_grid.md) - Sample visualization

---

## Datasets

Benchmark datasets available in Artifex.

| Dataset | Modality | Size | Use Case |
|---------|----------|------|----------|
| CIFAR-10 | Image | 60K | General image benchmarks |
| CelebA | Image | 200K | Face generation |
| FFHQ | Image | 70K | High-quality faces |
| QM9 | Molecular | 134K | Molecular generation |
| CrossDocked | Protein | 22M | Protein-ligand docking |

### Documentation

- [CelebA](celeba.md) - CelebA dataset
- [FFHQ](ffhq.md) - FFHQ dataset
- [QM9](qm9.md) - QM9 molecular dataset
- [CrossDocked](crossdocked.md) - Protein-ligand dataset
- [Synthetic Datasets](synthetic_datasets.md) - Synthetic data generation

---

## Best Practices

!!! success "DO"
    - Use standardized evaluation protocols for fair comparison
    - Report confidence intervals with multiple runs
    - Use appropriate metrics for your modality
    - Compare against established baselines

!!! danger "DON'T"
    - Don't cherry-pick metrics that favor your model
    - Don't use different sample sizes for comparisons
    - Don't ignore statistical significance
    - Don't compare models trained on different datasets

---

## Summary

The Artifex benchmarking framework provides:

- **Image Metrics**: FID, IS, LPIPS, Precision/Recall
- **Text Metrics**: Perplexity, BLEU, ROUGE, diversity
- **Audio Metrics**: MCD, spectral analysis
- **Disentanglement**: MIG, SAP, DCI, FactorVAE
- **Performance**: Latency, throughput, memory profiling
- **Visualization**: Plots, comparisons, dashboards

Use benchmark suites for standardized evaluation and fair model comparison.
