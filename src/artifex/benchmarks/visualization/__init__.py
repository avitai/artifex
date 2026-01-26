"""Visualization tools for benchmark results.

This package provides tools for visualizing benchmark results,
including plots for training curves, optimizer comparisons, and more.
"""

from artifex.benchmarks.visualization.optimization_plots import (
    plot_convergence_speed,
    plot_optimizer_comparison,
    plot_training_curve,
)


__all__ = [
    "plot_training_curve",
    "plot_optimizer_comparison",
    "plot_convergence_speed",
]
