"""Visualization tools for optimization benchmark results."""

import matplotlib.pyplot as plt
import numpy as np

from artifex.benchmarks.base import BenchmarkResult
from artifex.utils.file_utils import ensure_valid_output_path


def plot_training_curve(
    result: BenchmarkResult,
    metric_name: str = "loss",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    xlabel: str = "Iterations",
    ylabel: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
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

    Raises:
        TypeError: If the result is not a BenchmarkResult instance.
        ValueError: If the result does not contain training curve data or
            if the specified metric is not found in training data.
        IOError: If saving the figure fails.
    """
    # Validate input
    if not isinstance(result, BenchmarkResult):
        raise TypeError(f"Expected BenchmarkResult, got {type(result).__name__}")

    # Check for training curve data
    if not result.metadata:
        raise ValueError("Benchmark result metadata is empty or None.")

    if "training_curve" not in result.metadata:
        raise ValueError(
            "Benchmark result does not contain training curve data. "
            "Use a result from OptimizationBenchmark."
        )

    # Extract training curve data
    curve_data = result.metadata["training_curve"]
    if not curve_data:
        raise ValueError("Training curve data is empty.")

    if not isinstance(curve_data, list) or len(curve_data) == 0:
        raise ValueError("Training curve data is not a valid list or is empty.")

    # Validate metric name
    if not curve_data[0].get("metrics"):
        raise ValueError("Training curve points do not contain metrics data.")

    if not isinstance(curve_data[0]["metrics"], dict):
        raise TypeError(
            f"Metrics must be a dictionary, got {type(curve_data[0]['metrics']).__name__}"
        )

    available_metrics = curve_data[0]["metrics"].keys()
    if metric_name not in available_metrics:
        avail_metrics_str = ", ".join(sorted(available_metrics))
        raise ValueError(
            f"Metric '{metric_name}' not found in training data. "
            f"Available metrics: {avail_metrics_str}"
        )

    # Extract metrics
    iterations = [point["iteration"] for point in curve_data]
    metric_values = [point["metrics"][metric_name] for point in curve_data]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curve
    ax.plot(iterations, metric_values, marker="o", linewidth=2)

    # Set title and labels
    if title is None:
        title = f"{metric_name.capitalize()} vs. Iterations"
    ax.set_title(title)

    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = metric_name.capitalize()
    ax.set_ylabel(ylabel)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save if requested
    if save_path:
        if not isinstance(save_path, str):
            raise TypeError(f"save_path must be a string, got {type(save_path).__name__}")

        try:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        except Exception as e:
            raise IOError(
                f"Error saving figure to {save_path}: {e}. "
                f"Make sure the directory exists and you have write permissions."
            ) from e

    return fig


def plot_optimizer_comparison(
    result: BenchmarkResult,
    metric_name: str = "final_loss",
    title: str | None = None,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
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
        TypeError: If the result is not a BenchmarkResult instance.
        ValueError: If the result does not contain optimizer comparison data or
            if the specified metric is not found in optimizer data.
        IOError: If saving the figure fails.
    """
    # Validate input
    if not isinstance(result, BenchmarkResult):
        raise TypeError(f"Expected BenchmarkResult, got {type(result).__name__}")

    # Check for optimizer comparison data
    if result.benchmark_name != "optimizer_comparison":
        raise ValueError(
            "Benchmark result is not from OptimizerComparisonBenchmark. "
            f"Found benchmark_name: {result.benchmark_name}"
        )

    if "individual_results" not in result.metadata:
        raise ValueError("Benchmark result does not contain individual optimizer results.")

    if "optimizer_configs" not in result.metadata:
        raise ValueError("Benchmark result does not contain optimizer configurations.")

    # Get optimizer names and results
    optimizer_results = result.metadata["individual_results"]
    optimizer_configs = result.metadata["optimizer_configs"]

    if not optimizer_results or len(optimizer_results) == 0:
        raise ValueError("No optimizer results found.")

    if len(optimizer_results) != len(optimizer_configs):
        raise ValueError(
            f"Mismatch between number of optimizer results ({len(optimizer_results)}) "
            f"and configurations ({len(optimizer_configs)})."
        )

    # Validate metric name
    if metric_name not in optimizer_results[0]:
        avail_metrics = sorted(optimizer_results[0].keys())
        raise ValueError(
            f"Metric '{metric_name}' not found in optimizer results. "
            f"Available metrics: {avail_metrics}"
        )

    # Extract optimizer names and metric values
    optimizer_names = []
    metric_values = []

    for i, (config, result_data) in enumerate(zip(optimizer_configs, optimizer_results)):
        # Get optimizer name from config or use index
        optimizer_name = config.get("name", f"optimizer_{i}")
        optimizer_names.append(optimizer_name)

        # Get metric value
        metric_values.append(result_data[metric_name])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    bar_width = 0.6
    x = np.arange(len(optimizer_names))
    bars = ax.bar(x, metric_values, width=bar_width)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Set title and labels
    if title is None:
        title = f"Optimizer Comparison - {metric_name.capitalize()}"
    ax.set_title(title)

    ax.set_ylabel(metric_name.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(optimizer_names, rotation=45, ha="right")

    # Add grid for y-axis only
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        try:
            # Ensure save_path is within benchmark_results directory
            valid_save_path = ensure_valid_output_path(save_path, "benchmark_results")
            plt.savefig(valid_save_path, bbox_inches="tight", dpi=300)
        except Exception as e:
            raise IOError(f"Error saving figure to {save_path}: {e}") from e

    return fig


def plot_convergence_speed(
    result: BenchmarkResult,
    metric_name: str = "loss",
    target_value: float | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    highlight_convergence: bool = True,
    time_based: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot convergence speed from benchmark results.

    Args:
        result: Benchmark result from TrainingConvergenceBenchmark.
        metric_name: Name of the metric to plot.
        target_value: Target value for convergence. If None, uses target_loss
            from benchmark config if available.
        title: Title for the plot.
        figsize: Figure size (width, height).
        highlight_convergence: Whether to highlight the convergence point.
        time_based: Whether to plot against time instead of iterations.
        save_path: Path to save the figure to.

    Returns:
        Matplotlib figure object.

    Raises:
        TypeError: If the result is not a BenchmarkResult instance.
        ValueError: If the result does not contain training curve data or
            if the specified metric is not found in training data.
        IOError: If saving the figure fails.
    """
    # Validate input
    if not isinstance(result, BenchmarkResult):
        raise TypeError(f"Expected BenchmarkResult, got {type(result).__name__}")

    # Check for training curve data
    if "training_curve" not in result.metadata:
        raise ValueError(
            "Benchmark result does not contain training curve data. "
            "Use a result from OptimizationBenchmark or "
            "TrainingConvergenceBenchmark."
        )

    # Extract training curve data
    curve_data = result.metadata["training_curve"]
    if not curve_data:
        raise ValueError("Training curve data is empty.")

    # Validate metric name
    if not curve_data[0].get("metrics"):
        raise ValueError("Training curve points do not contain metrics data.")

    available_metrics = curve_data[0]["metrics"].keys()
    if metric_name not in available_metrics:
        avail_metrics_str = ", ".join(sorted(available_metrics))
        raise ValueError(
            f"Metric '{metric_name}' not found in training data. "
            f"Available metrics: {avail_metrics_str}"
        )

    # Determine target value
    if target_value is None and "target_loss" in result.metadata:
        target_value = result.metadata["target_loss"]

    # Validate timestamps are present if time_based is True
    if time_based:
        if not all("timestamp" in point for point in curve_data):
            raise ValueError(
                "Cannot create time-based plot: not all curve points have timestamp data."
            )

    # Extract metrics
    iterations = [point["iteration"] for point in curve_data]
    timestamps = [point.get("timestamp", i) for i, point in enumerate(curve_data)]
    metric_values = [point["metrics"][metric_name] for point in curve_data]

    # Determine convergence point if target is specified
    convergence_idx = None
    if target_value is not None and highlight_convergence:
        # Find first point where metric <= target for minimization metrics
        # or metric >= target for maximization metrics (default: minimization)
        for i, value in enumerate(metric_values):
            if value <= target_value:
                convergence_idx = i
                break

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curve
    x_values = timestamps if time_based else iterations
    x_label = "Time (seconds)" if time_based else "Iterations"

    ax.plot(x_values, metric_values, marker="o", linewidth=2)

    # Add target line if specified
    if target_value is not None:
        ax.axhline(
            y=target_value,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Target ({target_value})",
        )

    # Highlight convergence point if found
    if convergence_idx is not None and highlight_convergence:
        x_conv = x_values[convergence_idx]
        y_conv = metric_values[convergence_idx]

        # Add marker
        ax.plot(
            x_conv,
            y_conv,
            "ro",
            markersize=10,
            label="Convergence Point",
        )

        # Annotate with iteration/time and value
        if time_based:
            text = f"Time: {x_conv:.2f}s\n{metric_name}: {y_conv:.4f}"
        else:
            text = f"Iter: {int(x_conv)}\n{metric_name}: {y_conv:.4f}"

        ax.annotate(
            text,
            xy=(x_conv, y_conv),
            xytext=(10, -20),
            textcoords="offset points",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        )

    # Set title and labels
    if title is None:
        title_metric = metric_name.capitalize()
        title = f"{title_metric} Convergence"
        if time_based:
            title += " (Time-based)"
    ax.set_title(title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_name.capitalize())

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add legend if needed
    if (target_value is not None) or (convergence_idx is not None and highlight_convergence):
        ax.legend(loc="best")

    # Save if requested
    if save_path:
        try:
            # Ensure save_path is within benchmark_results directory
            valid_save_path = ensure_valid_output_path(save_path, "benchmark_results")
            plt.savefig(valid_save_path, bbox_inches="tight", dpi=300)
        except Exception as e:
            raise IOError(f"Error saving figure to {save_path}: {e}") from e

    return fig
