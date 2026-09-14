"""Benchmark figures are saved at the path the caller passes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from calibrax.core.result import Metric
from matplotlib.figure import Figure

from artifex.benchmarks import BenchmarkResult
from artifex.benchmarks.suites.protein_benchmarks import ProteinBenchmarkSuite
from artifex.benchmarks.visualization.optimization_plots import (
    plot_convergence_speed,
    plot_optimizer_comparison,
    plot_training_curve,
)


_CURVE = [{"iteration": step, "metrics": {"loss": 1.0 / (step + 1)}} for step in range(4)]


def _entries(directory: Path) -> list[str]:
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*"))


def _training_curve(save_path: str) -> Figure:
    result = BenchmarkResult(name="optimization", metadata={"training_curve": _CURVE})
    return plot_training_curve(result, save_path=save_path)


def _optimizer_comparison(save_path: str) -> Figure:
    result = BenchmarkResult(
        name="optimizer_comparison",
        metadata={
            "individual_results": [{"final_loss": 0.2}, {"final_loss": 0.1}],
            "optimizer_configs": [{"name": "sgd"}, {"name": "adam"}],
        },
    )
    return plot_optimizer_comparison(result, save_path=save_path)


def _convergence_speed(save_path: str) -> Figure:
    result = BenchmarkResult(name="convergence", metadata={"training_curve": _CURVE})
    return plot_convergence_speed(result, target_value=0.3, save_path=save_path)


@pytest.mark.parametrize("plot", [_training_curve, _optimizer_comparison, _convergence_speed])
def test_optimization_plots_save_a_relative_path_where_it_is_given(
    plot: Callable[[str], Figure], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    plt.close(plot("figure.png"))

    assert _entries(tmp_path) == ["figure.png"]


def test_protein_suite_figure_is_saved_where_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    suite = ProteinBenchmarkSuite(num_samples=4, random_seed=42, demo_mode=True)
    metrics = {name: Metric(value=0.5) for name in ("precision", "recall", "f1_score")}
    suite.results = {
        "model": {"precision_recall": BenchmarkResult(name="precision_recall", metrics=metrics)}
    }

    plt.close(suite.visualize_results(output_path="suite.png"))

    assert _entries(tmp_path) == ["suite.png"]
