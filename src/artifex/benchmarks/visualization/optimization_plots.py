"""Visualization tools for optimization benchmark results.

The plots read their benchmark's metadata through the record layer in
:mod:`artifex.benchmarks.performance.optimization`, so they validate it exactly as the
benchmark wrote it.
"""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from calibrax.core import MetadataValue, read_metadata, read_metadata_entry
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from artifex.benchmarks import BenchmarkResult
from artifex.benchmarks.performance.optimization import (
    training_curve_from_metadata,
    TrainingCurvePoint,
)


_SAVE_DPI = 300


def _metric_curve(
    result: BenchmarkResult, metric_name: str
) -> tuple[list[TrainingCurvePoint], list[float]]:
    """The result's training curve and one metric's value at each of its points."""
    curve = training_curve_from_metadata(result.metadata)
    available = curve[0].metrics
    if metric_name not in available:
        raise ValueError(
            f"Metric '{metric_name}' not found in training data. "
            f"Available metrics: {', '.join(sorted(available))}"
        )
    return curve, [point.metrics[metric_name] for point in curve]


def _timestamps(curve: list[TrainingCurvePoint]) -> list[float]:
    """The time of each curve point, for a time-based plot."""
    times = [point.timestamp for point in curve if point.timestamp is not None]
    if len(times) != len(curve):
        raise ValueError("Cannot create time-based plot: not all curve points have timestamp data.")
    return times


def _optimizer_bars(
    metadata: Mapping[str, MetadataValue], metric_name: str
) -> tuple[list[str], list[float]]:
    """Each optimizer's name and its value of ``metric_name``."""
    for key, what in (
        ("individual_results", "individual optimizer results"),
        ("optimizer_configs", "optimizer configurations"),
    ):
        if key not in metadata:
            raise ValueError(f"Benchmark result does not contain {what}.")
    results = read_metadata(
        list[dict[str, float]], metadata["individual_results"], "individual_results"
    )
    configs = read_metadata(
        list[dict[str, MetadataValue]], metadata["optimizer_configs"], "optimizer_configs"
    )
    if not results:
        raise ValueError("No optimizer results found.")
    if len(results) != len(configs):
        raise ValueError(
            f"Mismatch between number of optimizer results ({len(results)}) "
            f"and configurations ({len(configs)})."
        )
    if metric_name not in results[0]:
        raise ValueError(
            f"Metric '{metric_name}' not found in optimizer results. "
            f"Available metrics: {sorted(results[0])}"
        )
    names = [
        read_metadata_entry(
            str, config, "name", default=f"optimizer_{i}", name=f"optimizer_configs[{i}].name"
        )
        for i, config in enumerate(configs)
    ]
    return names, [row[metric_name] for row in results]


def _finish(fig: Figure, save_path: str | None) -> Figure:
    """Save ``fig`` at ``save_path`` when one is given, and return it."""
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=_SAVE_DPI)
    return fig


def _mark_convergence(ax: Axes, x: float, y: float, metric_name: str, *, time_based: bool) -> None:
    """Mark and annotate the first point at or below the target."""
    ax.plot(x, y, "ro", markersize=10, label="Convergence Point")
    where = f"Time: {x:.2f}s" if time_based else f"Iter: {int(x)}"
    ax.annotate(
        f"{where}\n{metric_name}: {y:.4f}",
        xy=(x, y),
        xytext=(10, -20),
        textcoords="offset points",
        ha="left",
        bbox={"boxstyle": "round,pad=0.5", "fc": "yellow", "alpha": 0.5},
    )


def plot_training_curve(
    result: BenchmarkResult,
    metric_name: str = "loss",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    xlabel: str = "Iterations",
    ylabel: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Plot a training curve from benchmark results.

    Args:
        result: Benchmark result from an optimization benchmark.
        metric_name: Name of the metric to plot.
        title: Title for the plot.
        figsize: Figure size (width, height).
        xlabel: X-axis label.
        ylabel: Y-axis label. If None, uses metric_name.
        save_path: Path to save the figure to.

    Returns:
        Matplotlib figure object.
    """
    curve, values = _metric_curve(result, metric_name)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([point.iteration for point in curve], values, marker="o", linewidth=2)
    ax.set_title(title if title is not None else f"{metric_name.capitalize()} vs. Iterations")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else metric_name.capitalize())
    ax.grid(True, linestyle="--", alpha=0.7)
    return _finish(fig, save_path)


def plot_optimizer_comparison(
    result: BenchmarkResult,
    metric_name: str = "final_loss",
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> Figure:
    """Plot a comparison of optimizer performance.

    Args:
        result: Benchmark result from OptimizerComparisonBenchmark.
        metric_name: Name of the metric to compare.
        title: Title for the plot.
        figsize: Figure size (width, height).
        save_path: Path to save the figure to.

    Returns:
        Matplotlib figure object.

    Raises:
        ValueError: If the result is not from OptimizerComparisonBenchmark.
    """
    if result.name != "optimizer_comparison":
        raise ValueError(
            "Benchmark result is not from OptimizerComparisonBenchmark. "
            f"Found benchmark name: {result.name}"
        )
    names, values = _optimizer_bars(result.metadata, metric_name)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    for bar in ax.bar(x, values, width=0.6):
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_title(
        title if title is not None else f"Optimizer Comparison - {metric_name.capitalize()}"
    )
    ax.set_ylabel(metric_name.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return _finish(fig, save_path)


def plot_convergence_speed(
    result: BenchmarkResult,
    metric_name: str = "loss",
    target_value: float | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    highlight_convergence: bool = True,
    time_based: bool = False,
    save_path: str | None = None,
) -> Figure:
    """Plot convergence speed from benchmark results.

    Args:
        result: Benchmark result from TrainingConvergenceBenchmark.
        metric_name: Name of the metric to plot.
        target_value: Target value for convergence. If None, uses the ``target_loss`` the
            benchmark recorded, if any.
        title: Title for the plot.
        figsize: Figure size (width, height).
        highlight_convergence: Whether to highlight the convergence point.
        time_based: Whether to plot against time instead of iterations.
        save_path: Path to save the figure to.

    Returns:
        Matplotlib figure object.
    """
    curve, values = _metric_curve(result, metric_name)
    if target_value is None and "target_loss" in result.metadata:
        target_value = read_metadata(float, result.metadata["target_loss"], "target_loss")
    x_values = _timestamps(curve) if time_based else [float(point.iteration) for point in curve]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_values, values, marker="o", linewidth=2)
    if target_value is not None:
        ax.axhline(
            y=target_value, color="r", linestyle="--", alpha=0.7, label=f"Target ({target_value})"
        )
        converged = next((i for i, value in enumerate(values) if value <= target_value), None)
        if highlight_convergence and converged is not None:
            _mark_convergence(
                ax, x_values[converged], values[converged], metric_name, time_based=time_based
            )
        ax.legend(loc="best")

    if title is None:
        title = f"{metric_name.capitalize()} Convergence"
        if time_based:
            title += " (Time-based)"
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)" if time_based else "Iterations")
    ax.set_ylabel(metric_name.capitalize())
    ax.grid(True, linestyle="--", alpha=0.7)
    return _finish(fig, save_path)
