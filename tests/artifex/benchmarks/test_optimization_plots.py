"""Optimization plots read their benchmark metadata through one typed layer.

A training curve is written into ``BenchmarkResult.metadata`` by ``TrainingCurvePoint`` and read
back by it, so the plots and the benchmark agree on one record format and one validation.
"""

from __future__ import annotations

from typing import Any

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - the backend is chosen first
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from artifex.benchmarks import BenchmarkResult  # noqa: E402
from artifex.benchmarks.performance.optimization import TrainingCurvePoint  # noqa: E402
from artifex.benchmarks.visualization.optimization_plots import (  # noqa: E402
    plot_convergence_speed,
    plot_optimizer_comparison,
    plot_training_curve,
)


_POINTS = [
    TrainingCurvePoint(iteration=0, timestamp=0.0, metrics={"loss": 1.0, "accuracy": 0.1}),
    TrainingCurvePoint(iteration=10, timestamp=0.5, metrics={"loss": 0.4, "accuracy": 0.6}),
    TrainingCurvePoint(iteration=20, timestamp=1.0, metrics={"loss": 0.1, "accuracy": 0.9}),
]


def _curve_result(**metadata: Any) -> BenchmarkResult:
    record = {"training_curve": [point.to_record() for point in _POINTS]}
    return BenchmarkResult(name="optimization", metadata={**record, **metadata})


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


class TestTrainingCurveRecords:
    def test_a_point_round_trips_through_its_record(self) -> None:
        for point in _POINTS:
            assert TrainingCurvePoint.from_record(point.to_record()) == point

    def test_array_scalars_read_as_the_numbers_they_hold(self) -> None:
        record = {
            "iteration": 3,
            "timestamp": np.float32(0.5),
            "metrics": {"loss": np.float32(0.25)},
        }

        point = TrainingCurvePoint.from_record(record)

        assert point == TrainingCurvePoint(iteration=3, timestamp=0.5, metrics={"loss": 0.25})
        assert type(point.metrics["loss"]) is float

    def test_a_point_without_metrics_is_refused(self) -> None:
        with pytest.raises(ValueError, match="do not contain metrics data"):
            TrainingCurvePoint.from_record({"iteration": 0, "timestamp": 0.0})

    def test_metrics_that_are_not_a_mapping_are_refused(self) -> None:
        with pytest.raises(TypeError, match="Metrics must be a dictionary"):
            TrainingCurvePoint.from_record({"iteration": 0, "metrics": [1.0]})

    def test_a_point_that_is_not_a_mapping_is_refused(self) -> None:
        with pytest.raises(TypeError, match="Training curve point must be a dictionary"):
            TrainingCurvePoint.from_record(3)


class TestPlotTrainingCurve:
    def test_the_line_holds_the_iterations_and_the_metric(self) -> None:
        fig = plot_training_curve(_curve_result(), metric_name="accuracy")

        xy = fig.axes[0].lines[0].get_xydata()
        np.testing.assert_array_equal(xy[:, 0], [0, 10, 20])
        np.testing.assert_array_equal(xy[:, 1], [0.1, 0.6, 0.9])

    def test_a_result_without_a_curve_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not contain training curve data"):
            plot_training_curve(BenchmarkResult(name="optimization", metadata={"other": 1}))

    def test_an_empty_curve_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Training curve data is empty"):
            plot_training_curve(
                BenchmarkResult(name="optimization", metadata={"training_curve": []})
            )

    def test_a_curve_that_is_not_a_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a valid list"):
            plot_training_curve(
                BenchmarkResult(name="optimization", metadata={"training_curve": "points"})
            )

    def test_an_unknown_metric_names_the_available_ones(self) -> None:
        with pytest.raises(ValueError, match="Available metrics: accuracy, loss"):
            plot_training_curve(_curve_result(), metric_name="f1")


class TestPlotConvergenceSpeed:
    def test_the_target_comes_from_the_metadata_and_marks_the_first_point_below_it(self) -> None:
        fig = plot_convergence_speed(_curve_result(target_loss=0.5))

        ax = fig.axes[0]
        assert ax.lines[1].get_ydata()[0] == 0.5  # the target line
        np.testing.assert_array_equal(ax.lines[2].get_xydata(), [[10, 0.4]])  # the first <= 0.5

    def test_a_time_based_plot_uses_the_timestamps(self) -> None:
        fig = plot_convergence_speed(_curve_result(), time_based=True)

        np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(), [0.0, 0.5, 1.0])

    def test_a_time_based_plot_refuses_points_without_timestamps(self) -> None:
        curve = [{"iteration": 0, "metrics": {"loss": 1.0}}]
        result = BenchmarkResult(name="optimization", metadata={"training_curve": curve})

        with pytest.raises(ValueError, match="not all curve points have timestamp data"):
            plot_convergence_speed(result, time_based=True)


class TestPlotOptimizerComparison:
    @staticmethod
    def _result(**overrides: Any) -> BenchmarkResult:
        metadata = {
            "optimizer_configs": [{"name": "adam"}, {}],
            "individual_results": [{"final_loss": 0.2}, {"final_loss": 0.3}],
            **overrides,
        }
        return BenchmarkResult(name="optimizer_comparison", metadata=metadata)

    def test_one_bar_per_optimizer_named_by_its_config_or_its_index(self) -> None:
        fig = plot_optimizer_comparison(self._result())

        ax = fig.axes[0]
        assert [bar.get_height() for bar in ax.patches] == [0.2, 0.3]
        assert [label.get_text() for label in ax.get_xticklabels()] == ["adam", "optimizer_1"]

    def test_results_and_configs_of_different_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Mismatch between number of optimizer results"):
            plot_optimizer_comparison(self._result(optimizer_configs=[{"name": "adam"}]))

    def test_an_unknown_metric_names_the_available_ones(self) -> None:
        with pytest.raises(ValueError, match="not found in optimizer results"):
            plot_optimizer_comparison(self._result(), metric_name="accuracy")
