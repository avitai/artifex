"""Benchmark runner implementations for artifex.generative_models.core.evaluation."""

import time
from typing import Any

import jax.numpy as jnp

from artifex.generative_models.core.configuration import EvaluationConfig
from artifex.generative_models.core.protocols.benchmarks import BenchmarkBase


class PerformanceTracker:
    """Tracks performance metrics during benchmark execution.

    This class provides functionality to:
    - Track metrics over time
    - Check target achievement
    - Store performance history
    - Generate performance summaries
    """

    def __init__(self, config: EvaluationConfig):
        """Initialize performance tracker.

        Args:
            config: Evaluation configuration containing target metrics

        Raises:
            TypeError: If config is not EvaluationConfig
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")
        self.config = config
        self.metrics_history: list[dict[str, Any]] = []

    def track_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Track metrics at a specific step.

        Args:
            metrics: dictionary of metric name to value
            step: Current step/epoch number
        """
        entry = {"step": step, "timestamp": time.time(), "metrics": metrics.copy()}
        self.metrics_history.append(entry)

    def get_current_performance(self) -> dict[str, Any] | None:
        """Get the most recent performance metrics.

        Returns:
            Most recent metrics entry or None if no metrics tracked
        """
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]

    def check_target_achievement(self) -> bool:
        """Check if current performance meets targets.

        Returns:
            True if targets are achieved, False otherwise
        """
        current = self.get_current_performance()
        if current is None:
            return False

        # Get target metrics from metric_params
        targets = self.config.metric_params.get("target_metrics", {})
        current_metrics = current["metrics"]

        for metric_name, target_value in targets.items():
            if metric_name not in current_metrics:
                return False

            current_value = current_metrics[metric_name]

            # For latency metrics, lower is better
            if "latency" in metric_name.lower() or "time" in metric_name.lower():
                if current_value > target_value:
                    return False
            else:
                # For most metrics (accuracy, etc.), higher is better
                if current_value < target_value:
                    return False

        return True

    def get_performance_summary(self) -> dict[str, Any]:
        """Generate summary of performance over time.

        Returns:
            Summary statistics for tracked metrics
        """
        if not self.metrics_history:
            return {"num_steps": 0, "metrics": {}}

        # Collect all metric values
        all_metrics: dict[str, list[float]] = {}
        for entry in self.metrics_history:
            for metric_name, value in entry["metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Compute summary statistics
        summary: dict[str, Any] = {"num_steps": len(self.metrics_history), "metrics": {}}

        for metric_name, values in all_metrics.items():
            summary["metrics"][metric_name] = {
                "latest": values[-1],
                "mean": float(jnp.mean(jnp.array(values))),
                "std": float(jnp.std(jnp.array(values))),
                "min": float(jnp.min(jnp.array(values))),
                "max": float(jnp.max(jnp.array(values))),
            }

        return summary


class BenchmarkRunner:
    """Orchestrates benchmark execution and results management.

    This class provides high-level functionality to:
    - Execute complete benchmark workflows
    - Track results over multiple runs
    - Compare performance across runs
    - Generate comprehensive reports
    """

    def __init__(self, benchmark: BenchmarkBase) -> None:
        """Initialize benchmark runner.

        Args:
            benchmark: Benchmark instance to run
        """
        self.benchmark = benchmark
        self.results_history: list[dict[str, Any]] = []

    def run_full_benchmark(self) -> dict[str, Any]:
        """Execute complete benchmark workflow.

        This runs both training and evaluation phases,
        collects all metrics, and validates target achievement.

        Returns:
            Comprehensive results dictionary containing:
            - training_results: Training phase metrics
            - evaluation_results: Evaluation phase metrics
            - performance_summary: Overall performance summary
            - targets_achieved: Whether targets were met
            - timestamp: When benchmark was run
        """
        timestamp = time.time()

        # Execute training phase
        training_results = self.benchmark.run_training()

        # Execute evaluation phase
        evaluation_results = self.benchmark.run_evaluation()

        # Get performance targets
        targets = self.benchmark.get_performance_targets()

        # Check if targets are achieved
        all_metrics = {**training_results, **evaluation_results}
        targets_achieved = self.benchmark.validate_targets_achieved(all_metrics)

        # Create comprehensive results
        results = {
            "timestamp": timestamp,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "performance_summary": {"targets": targets, "all_metrics": all_metrics},
            "targets_achieved": targets_achieved,
        }

        # Store in history
        self.results_history.append(results)

        return results

    def compare_performance(self) -> dict[str, Any]:
        """Compare performance across multiple benchmark runs.

        Returns:
            Performance comparison summary
        """
        if not self.results_history:
            return {"num_runs": 0, "message": "No benchmark runs available"}

        comparison: dict[str, Any] = {"num_runs": len(self.results_history), "runs_summary": []}

        # Collect metrics from all runs
        all_runs_metrics: list[dict[str, Any]] = []
        for run in self.results_history:
            run_summary = {
                "timestamp": run["timestamp"],
                "targets_achieved": run["targets_achieved"],
                "training_loss": run["training_results"].get("training_loss"),
                "evaluation_metrics": run["evaluation_results"],
            }
            comparison["runs_summary"].append(run_summary)

            # Collect all metrics for statistical analysis
            all_metrics = {**run["training_results"], **run["evaluation_results"]}
            all_runs_metrics.append(all_metrics)

        # Compute cross-run statistics if multiple runs
        if len(all_runs_metrics) > 1:
            metrics_summary: dict[str, Any] = {}

            # Get all metric names
            all_metric_names: set[str] = set()
            for metrics in all_runs_metrics:
                all_metric_names.update(metrics.keys())

            # Compute statistics for each metric
            for metric_name in all_metric_names:
                values: list[float] = []
                for metrics in all_runs_metrics:
                    if metric_name in metrics:
                        values.append(metrics[metric_name])

                if values:
                    values_array = jnp.array(values)
                    metrics_summary[metric_name] = {
                        "mean": float(jnp.mean(values_array)),
                        "std": float(jnp.std(values_array)),
                        "min": float(jnp.min(values_array)),
                        "max": float(jnp.max(values_array)),
                    }

            comparison["metrics_summary"] = metrics_summary

        return comparison

    def get_latest_results(self) -> dict[str, Any] | None:
        """Get results from most recent benchmark run.

        Returns:
            Latest results or None if no runs completed
        """
        if not self.results_history:
            return None
        return self.results_history[-1]

    def get_run_count(self) -> int:
        """Get number of completed benchmark runs.

        Returns:
            Number of runs in history
        """
        return len(self.results_history)

    def clear_history(self) -> None:
        """Clear benchmark run history."""
        self.results_history.clear()
