"""Evaluation pipeline and metric composition for artifex.generative_models.core.evaluation."""

from typing import Any

import flax.nnx as nnx

from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.protocols.metrics import MetricBase


class EvaluationPipeline(nnx.Module):
    """Complete evaluation pipeline for multi-modal benchmarks.

    Orchestrates evaluation across multiple modalities and metrics,
    providing comprehensive assessment capabilities.

    Attributes:
        config: Pipeline configuration
        metrics: Dictionary of modality-specific metrics
        rngs: NNX Rngs for stochastic operations
    """

    def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
        """Initialize evaluation pipeline.

        Args:
            config: Evaluation configuration
            rngs: NNX Rngs for stochastic operations

        Raises:
            TypeError: If config is not EvaluationConfig
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")

        self.config = config
        # Extract modalities from metrics
        modalities = set()
        for metric in config.metrics:
            if ":" in metric:
                modality, _ = metric.split(":", 1)
                modalities.add(modality)

        self.rngs = rngs
        self.metrics = {}

        # Initialize modality-specific metrics
        for modality in modalities:
            self.metrics[modality] = self._create_modality_metrics(modality)

    def _create_modality_metrics(self, modality: str) -> list[MetricBase]:
        """Create metrics for a specific modality."""
        metrics = []

        # Extract metrics for this modality from typed config
        metric_names = [
            metric.split(":", 1)[1] if ":" in metric else metric
            for metric in self.config.metrics
            if metric.startswith(f"{modality}:") or ":" not in metric
        ]

        for metric_name in metric_names:
            # Get metric params from typed config
            metric_config = {
                "name": metric_name,
                "modality": modality,
                **self.config.metric_params.get(metric_name, {}),
            }

            # Create actual metrics with their specific constructors
            if metric_name == "fid":
                from artifex.generative_models.core.evaluation.metrics.image import (
                    FrechetInceptionDistance,
                )

                # Extract FID-specific params
                batch_size = metric_config.get("batch_size", 32)
                feature_extractor = metric_config.get("feature_extractor", None)
                metrics.append(
                    FrechetInceptionDistance(
                        batch_size=batch_size,
                        feature_extractor=feature_extractor,
                        name=metric_name,
                        rngs=self.rngs,
                    )
                )
            elif metric_name == "is":
                from artifex.generative_models.core.evaluation.metrics.image import InceptionScore

                # Extract IS-specific params
                batch_size = metric_config.get("batch_size", 32)
                splits = metric_config.get("splits", 10)
                classifier = metric_config.get("classifier", None)
                metrics.append(
                    InceptionScore(
                        classifier=classifier,
                        batch_size=batch_size,
                        splits=splits,
                        name=metric_name,
                        rngs=self.rngs,
                    )
                )
            elif metric_name == "perplexity":
                from artifex.generative_models.core.evaluation.metrics.text import Perplexity

                # Extract perplexity-specific params
                model = metric_config.get("model", None)
                batch_size = metric_config.get("batch_size", 32)
                metrics.append(
                    Perplexity(model=model, batch_size=batch_size, name=metric_name, rngs=self.rngs)
                )
            # For unsupported metrics, create a mock for now
            elif metric_name in ["bleu", "rouge"]:
                # These metrics aren't implemented yet, so we'll skip them
                pass

        return metrics

    def evaluate(self, data: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Evaluate all modalities and metrics.

        Args:
            data: Dictionary with modality -> {real, generated} data

        Returns:
            Dictionary of results by modality and metric
        """
        results = {}

        for modality, modality_data in data.items():
            if modality in self.metrics:
                modality_results = {}

                for metric in self.metrics[modality]:
                    metric_results = metric.compute(
                        modality_data["real"], modality_data["generated"]
                    )
                    modality_results.update(metric_results)

                results[modality] = modality_results

        return results


class MetricComposer(nnx.Module):
    """Compose and aggregate metrics across modalities.

    Provides sophisticated metric composition capabilities including
    weighted combinations and cross-modality aggregation.

    Attributes:
        config: Composer configuration
        rngs: NNX Rngs for stochastic operations
    """

    def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
        """Initialize metric composer.

        Args:
            config: Composer configuration
            rngs: NNX Rngs for stochastic operations

        Raises:
            TypeError: If config is not EvaluationConfig
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")
        self.config = config
        self.rngs = rngs

    def compose(self, metrics: dict[str, float]) -> dict[str, float]:
        """Compose metrics using configured rules.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Dictionary with composed metrics
        """
        composed = {}

        # Get composition rules from metric_params
        composition_rules = self.config.metric_params.get("composition_rules", {})
        for rule_name, rule_config in composition_rules.items():
            weights = rule_config.get("weights", {})
            normalization = rule_config.get("normalization", "none")

            # Compute weighted combination
            score = 0.0
            for metric_name, weight in weights.items():
                if metric_name in metrics:
                    value = metrics[metric_name]

                    # Apply normalization if specified
                    if normalization == "min_max":
                        # Simple mock normalization for testing
                        value = (value - 0) / (100 - 0)  # Assume range [0, 100]

                    score += weight * value

            composed[rule_name] = score

        return composed

    def aggregate_modalities(
        self, modality_results: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Aggregate results across modalities.

        Args:
            modality_results: Results by modality

        Returns:
            Aggregated cross-modality metrics
        """
        # Get aggregation settings from metric_params
        # Check if there's a composer_settings dict first
        composer_settings = self.config.metric_params.get("composer_settings", {})
        strategy = composer_settings.get("aggregation_strategy", "weighted_average")
        weights = composer_settings.get("modality_weights", {})

        if strategy == "weighted_average":
            # Compute weighted average across modalities
            total_score = 0.0
            total_weight = 0.0

            for modality, results in modality_results.items():
                weight = weights.get(modality, 1.0)
                # Use first metric as representative score for simplicity
                modality_score = next(iter(results.values()), 0.0)

                total_score += weight * modality_score
                total_weight += weight

            if total_weight > 0:
                cross_modality_score = total_score / total_weight
            else:
                cross_modality_score = 0.0

            return {"cross_modality_score": cross_modality_score}

        return {}


class ModalityMetrics(nnx.Module):
    """Manage modality-specific metrics and selection.

    Provides centralized management of metrics by modality with
    automatic selection capabilities based on quality requirements.

    Attributes:
        config: Modality metrics configuration
        rngs: NNX Rngs for stochastic operations
    """

    def __init__(self, config: EvaluationConfig, *, rngs: nnx.Rngs):
        """Initialize modality metrics manager.

        Args:
            config: Modality metrics configuration
            rngs: NNX Rngs for stochastic operations

        Raises:
            TypeError: If config is not EvaluationConfig
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")
        self.config = config
        self.rngs = rngs

        # Extract modalities from metrics
        modalities = set()
        for metric in config.metrics:
            if ":" in metric:
                modality, _ = metric.split(":", 1)
                modalities.add(modality)
        self.supported_modalities = modalities

    def get_supported_modalities(self) -> list[str]:
        """Get list of supported modalities."""
        return list(self.supported_modalities)

    def select_metrics(self, modality: str, quality_level: str = "standard") -> list[str]:
        """Select appropriate metrics for modality and quality level.

        Args:
            modality: Target modality
            quality_level: Quality requirement level

        Returns:
            List of recommended metric names
        """
        if modality not in self.supported_modalities:
            return []

        # Get quality levels from metric_params
        quality_levels = self.config.metric_params.get("quality_levels", {})

        if quality_level in quality_levels:
            return quality_levels[quality_level]

        # Default metric selection by modality
        default_metrics: dict[str, list[str]] = {
            "image": ["fid", "is"],
            "text": ["bleu", "rouge"],
            "audio": ["spectral", "mcd"],
        }

        return default_metrics.get(modality, [])
