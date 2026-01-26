"""
Metrics logging utilities for the Artifex library.

This module provides utility functions for logging metrics from various
evaluation metrics and integrating them with loggers.
"""

import jax
import numpy as np

from artifex.generative_models.core.evaluation.metrics.base import MetricModule as Metric
from artifex.generative_models.utils.logging.logger import Logger


class MetricsLogger:
    """
    Class for logging metrics from various evaluation metrics.

    This class integrates evaluation metrics with loggers to streamline
    the process of computing and logging metrics during training and evaluation.
    """

    def __init__(
        self,
        logger: Logger,
        metrics: dict[str, Metric] | None = None,
        prefix: str = "",
        compute_frequency: int = 100,
    ):
        """
        Initialize the metrics logger.

        Args:
            logger: Logger instance to use for logging metrics.
            metrics: Dictionary of metric name to Metric instance.
            prefix: Prefix to add to metric names when logging.
            compute_frequency: How often to compute metrics (in steps).
        """
        self.logger = logger
        self.metrics = metrics or {}
        self.prefix = prefix
        self.compute_frequency = compute_frequency
        self.last_computed_step = -compute_frequency  # Ensure first step computes

        # Log the initialization of metrics
        metric_names = list(self.metrics.keys())
        if metric_names:
            self.logger.info(
                f"Initialized metrics logger with {len(metric_names)} metrics: {metric_names}"
            )
        else:
            self.logger.info("Initialized metrics logger with no metrics")

    def add_metric(self, name: str, metric: Metric) -> None:
        """
        Add a metric to the logger.

        Args:
            name: Name of the metric.
            metric: Metric instance.
        """
        self.metrics[name] = metric
        self.logger.info(f"Added metric: {name}")

    def remove_metric(self, name: str) -> bool:
        """
        Remove a metric from the logger.

        Args:
            name: Name of the metric to remove.

        Returns:
            True if the metric was removed, False if it wasn't found.
        """
        if name in self.metrics:
            del self.metrics[name]
            self.logger.info(f"Removed metric: {name}")
            return True
        else:
            self.logger.warning(f"Attempted to remove non-existent metric: {name}")
            return False

    def compute_metrics(
        self,
        real_data: jax.Array,
        generated_data: jax.Array,
        step: int | None = None,
        log_results: bool = True,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """
        Compute all metrics and optionally log them.

        Args:
            real_data: Real data samples.
            generated_data: Generated data samples.
            step: Current step number.
            log_results: Whether to log the results using the logger.
            **kwargs: Additional keyword arguments to pass to metric computation.

        Returns:
            Dictionary mapping metric names to their results.
        """
        # Skip if no metrics are registered
        if not self.metrics:
            self.logger.warning("No metrics registered for computation")
            return {}

        # Skip if not at compute frequency
        if step is not None and (step - self.last_computed_step) < self.compute_frequency:
            return {}

        results = {}
        all_scalar_metrics = {}

        # Compute each metric
        for name, metric in self.metrics.items():
            try:
                # Handle shape differences gracefully
                if real_data.shape != generated_data.shape:
                    self.logger.warning(
                        f"Shape mismatch for metric {name}: "
                        f"real_data {real_data.shape}, generated_data {generated_data.shape}"
                    )

                # Compute the metric
                metric_result = metric(real_data, generated_data, **kwargs)
                results[name] = metric_result

                # Extract scalar values for logging
                scalar_metrics = {}
                for key, value in metric_result.items():
                    # Add prefix if specified
                    metric_name = f"{self.prefix}{name}/{key}" if self.prefix else f"{name}/{key}"

                    # Only log scalar values
                    if isinstance(value, (int, float)):
                        scalar_metrics[metric_name] = value
                        all_scalar_metrics[metric_name] = value

                # Log individual metric results
                if log_results and scalar_metrics:
                    self.logger.log_scalars(scalar_metrics, step=step)

            except Exception as e:
                self.logger.error(f"Error computing metric {name}: {e}")

        # Log all metrics together if requested
        if log_results and all_scalar_metrics:
            self.logger.info(
                f"Computed {len(results)} metrics at step {step}: {', '.join(results.keys())}"
            )

        # Update last computed step
        if step is not None:
            self.last_computed_step = step

        return results

    def should_compute(self, step: int) -> bool:
        """
        Check if metrics should be computed at the given step.

        Args:
            step: Current step number.

        Returns:
            True if metrics should be computed, False otherwise.
        """
        return (step - self.last_computed_step) >= self.compute_frequency

    def log_training_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str = "train/",
    ) -> None:
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metric names to values.
            step: Current step number.
            prefix: Prefix to add to metric names.
        """
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}{name}": value for name, value in metrics.items()}

        # Log metrics
        self.logger.log_scalars(prefixed_metrics, step=step)

    def log_validation_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str = "val/",
    ) -> None:
        """
        Log validation metrics.

        Args:
            metrics: Dictionary of metric names to values.
            step: Current step number.
            prefix: Prefix to add to metric names.
        """
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}{name}": value for name, value in metrics.items()}

        # Log metrics
        self.logger.log_scalars(prefixed_metrics, step=step)

    def log_test_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str = "test/",
    ) -> None:
        """
        Log test metrics.

        Args:
            metrics: Dictionary of metric names to values.
            step: Current step number.
            prefix: Prefix to add to metric names.
        """
        # Add prefix to metric names
        prefixed_metrics = {f"{prefix}{name}": value for name, value in metrics.items()}

        # Log metrics
        self.logger.log_scalars(prefixed_metrics, step=step)

    def log_generated_samples(
        self,
        samples: jax.Array,
        name: str = "generated_samples",
        step: int | None = None,
        max_samples: int = 16,
    ) -> None:
        """
        Log generated samples as images.

        Args:
            samples: Generated samples to log.
            name: Name for the logged images.
            step: Current step number.
            max_samples: Maximum number of samples to log.
        """
        # Ensure we don't exceed the maximum number of samples
        num_samples = min(samples.shape[0], max_samples)
        samples_to_log = samples[:num_samples]

        # Log the samples
        self.logger.log_image(name, samples_to_log, step=step)

    def log_comparison(
        self,
        real_samples: jax.Array,
        generated_samples: jax.Array,
        name: str = "real_vs_generated",
        step: int | None = None,
        max_samples: int = 8,
    ) -> None:
        """
        Log a comparison of real and generated samples.

        Args:
            real_samples: Real data samples.
            generated_samples: Generated data samples.
            name: Name for the comparison.
            step: Current step number.
            max_samples: Maximum number of samples to log (per category).
        """
        # Handle potential shape differences
        if real_samples.shape[1:] != generated_samples.shape[1:]:
            self.logger.warning(
                f"Shape mismatch for sample comparison: "
                f"real_samples {real_samples.shape}, generated_samples {generated_samples.shape}"
            )
            # Log them separately to avoid errors
            self.log_generated_samples(
                real_samples, name="real_samples", step=step, max_samples=max_samples
            )
            self.log_generated_samples(
                generated_samples, name="generated_samples", step=step, max_samples=max_samples
            )
            return

        # Ensure we don't exceed the maximum number of samples
        num_real = min(real_samples.shape[0], max_samples)
        num_gen = min(generated_samples.shape[0], max_samples)
        num_samples = min(num_real, num_gen)

        real_to_log = real_samples[:num_samples]
        gen_to_log = generated_samples[:num_samples]

        try:
            # Stack vertically for comparison (real on top, generated on bottom)
            comparison = []
            for i in range(num_samples):
                comparison.append(real_to_log[i])
                comparison.append(gen_to_log[i])

            # Log the comparison
            self.logger.log_image(name, comparison, step=step)

        except Exception as e:
            self.logger.warning(f"Failed to create comparison image: {e}")
            # Fall back to logging separately
            self.log_generated_samples(real_to_log, name="real_samples", step=step)
            self.log_generated_samples(gen_to_log, name="generated_samples", step=step)


# Convenience functions


def get_default_metrics() -> dict[str, Metric]:
    """
    Get default metrics for generative model evaluation.

    Returns:
        Dictionary of default metrics.
    """
    from artifex.generative_models.core.metrics.fid import FrechetInceptionDistance
    from artifex.generative_models.core.metrics.inception_score import InceptionScore
    from artifex.generative_models.core.metrics.precision_recall import PrecisionRecall

    metrics = {
        "inception_score": InceptionScore(),
        "fid": FrechetInceptionDistance(),
        "precision_recall": PrecisionRecall(),
    }

    return metrics


def log_distribution_metrics(
    logger: Logger,
    real_samples: jax.Array,
    generated_samples: jax.Array,
    step: int | None = None,
) -> dict[str, float]:
    """
    Log basic distribution metrics between real and generated samples.

    Args:
        logger: Logger instance to use for logging.
        real_samples: Real data samples.
        generated_samples: Generated data samples.
        step: Current step number.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}

    # Compute basic statistics
    real_mean = np.mean(real_samples)
    real_std = np.std(real_samples)
    real_min = np.min(real_samples)
    real_max = np.max(real_samples)

    gen_mean = np.mean(generated_samples)
    gen_std = np.std(generated_samples)
    gen_min = np.min(generated_samples)
    gen_max = np.max(generated_samples)

    # Compute differences
    mean_diff = abs(real_mean - gen_mean)
    std_diff = abs(real_std - gen_std)

    # Log real statistics
    real_stats = {
        "real/mean": real_mean,
        "real/std": real_std,
        "real/min": real_min,
        "real/max": real_max,
    }
    logger.log_scalars(real_stats, step=step)
    metrics.update(real_stats)

    # Log generated statistics
    gen_stats = {
        "gen/mean": gen_mean,
        "gen/std": gen_std,
        "gen/min": gen_min,
        "gen/max": gen_max,
    }
    logger.log_scalars(gen_stats, step=step)
    metrics.update(gen_stats)

    # Log differences
    diff_stats = {
        "diff/mean": mean_diff,
        "diff/std": std_diff,
    }
    logger.log_scalars(diff_stats, step=step)
    metrics.update(diff_stats)

    return metrics
