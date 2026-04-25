"""Retained benchmark execution helpers for Artifex."""

from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp

from artifex.benchmarks.core.nnx import BenchmarkBase
from artifex.generative_models.core.configuration import EvaluationConfig


class PerformanceTracker:
    """Track performance metrics during benchmark execution."""

    def __init__(self, config: EvaluationConfig):
        """Initialize performance tracking for a benchmark run."""
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")
        self.config = config
        self.metrics_history: list[dict[str, Any]] = []

    def track_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Track metrics at a specific step."""
        entry = {"step": step, "timestamp": time.time(), "metrics": metrics.copy()}
        self.metrics_history.append(entry)

    def get_current_performance(self) -> dict[str, Any] | None:
        """Get the most recent performance metrics."""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]

    def check_target_achievement(self) -> bool:
        """Check if the current performance meets targets."""
        current = self.get_current_performance()
        if current is None:
            return False
        targets = self.config.metric_params.get("target_metrics", {})
        current_metrics = current["metrics"]
        for metric_name, target_value in targets.items():
            if metric_name not in current_metrics:
                return False
            current_value = current_metrics[metric_name]
            if "latency" in metric_name.lower() or "time" in metric_name.lower():
                if current_value > target_value:
                    return False
            elif current_value < target_value:
                return False
        return True

    def get_performance_summary(self) -> dict[str, Any]:
        """Generate summary statistics for tracked metrics."""
        if not self.metrics_history:
            return {"num_steps": 0, "metrics": {}}

        all_metrics: dict[str, list[float]] = {}
        for entry in self.metrics_history:
            for metric_name, value in entry["metrics"].items():
                all_metrics.setdefault(metric_name, []).append(value)

        summary: dict[str, Any] = {
            "num_steps": len(self.metrics_history),
            "metrics": {},
        }
        for metric_name, values in all_metrics.items():
            values_array = jnp.array(values)
            summary["metrics"][metric_name] = {
                "latest": values[-1],
                "mean": float(jnp.mean(values_array)),
                "std": float(jnp.std(values_array)),
                "min": float(jnp.min(values_array)),
                "max": float(jnp.max(values_array)),
            }
        return summary


class BenchmarkRunner:
    """Orchestrate benchmark execution and results management."""

    def __init__(self, benchmark: BenchmarkBase) -> None:
        """Initialize the runner for a benchmark."""
        self.benchmark = benchmark
        self.results_history: list[dict[str, Any]] = []

    def run_full_benchmark(self) -> dict[str, Any]:
        """Execute the full benchmark workflow."""
        timestamp = time.time()
        training_results = self.benchmark.run_training()
        evaluation_results = self.benchmark.run_evaluation()
        targets = self.benchmark.get_performance_targets()
        all_metrics = {**training_results, **evaluation_results}
        targets_achieved = self.benchmark.validate_targets_achieved(all_metrics)
        results = {
            "timestamp": timestamp,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "performance_summary": {"targets": targets, "all_metrics": all_metrics},
            "targets_achieved": targets_achieved,
        }
        self.results_history.append(results)
        return results

    def compare_performance(self) -> dict[str, Any]:
        """Compare performance across multiple benchmark runs."""
        if not self.results_history:
            return {"num_runs": 0, "message": "No benchmark runs available"}

        comparison: dict[str, Any] = {
            "num_runs": len(self.results_history),
            "runs_summary": [],
        }
        all_runs_metrics: list[dict[str, Any]] = []
        for run in self.results_history:
            comparison["runs_summary"].append(
                {
                    "timestamp": run["timestamp"],
                    "targets_achieved": run["targets_achieved"],
                    "training_loss": run["training_results"].get("training_loss"),
                    "evaluation_metrics": run["evaluation_results"],
                }
            )
            all_runs_metrics.append({**run["training_results"], **run["evaluation_results"]})

        if len(all_runs_metrics) > 1:
            metrics_summary: dict[str, Any] = {}
            all_metric_names: set[str] = set()
            for metrics in all_runs_metrics:
                all_metric_names.update(metrics.keys())
            for metric_name in all_metric_names:
                values = [
                    metrics[metric_name] for metrics in all_runs_metrics if metric_name in metrics
                ]
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
        """Get results from the most recent benchmark run."""
        if not self.results_history:
            return None
        return self.results_history[-1]

    def get_run_count(self) -> int:
        """Get the number of completed benchmark runs."""
        return len(self.results_history)

    def clear_history(self) -> None:
        """Clear benchmark run history."""
        self.results_history.clear()


__all__ = [
    "BenchmarkRunner",
    "PerformanceTracker",
]
