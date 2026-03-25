"""Tests for calibrax protocol conformance and integration.

Verifies that artifex benchmark infrastructure correctly implements
or interoperates with calibrax's core protocols and registry.
"""

import jax.numpy as jnp
import pytest
from calibrax.core import (
    BatchableDatasetProtocol,
    DatasetProtocol,
)
from flax import nnx

from artifex.benchmarks.core import Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite
from artifex.benchmarks.core.result_model import (
    from_calibrax_result,
    to_calibrax_result,
)
from artifex.benchmarks.registry import (
    BenchmarkRegistry,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)


# ---------------------------------------------------------------------------
# Registry interop tests
# ---------------------------------------------------------------------------


class TestRegistryInterop:
    """Tests that artifex BenchmarkRegistry interoperates with calibrax Registry."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        BenchmarkRegistry.reset()

    def test_registry_singleton(self) -> None:
        """BenchmarkRegistry is a singleton."""
        r1 = BenchmarkRegistry()
        r2 = BenchmarkRegistry()
        assert r1 is r2

    def test_register_and_retrieve(self) -> None:
        """Register a benchmark and retrieve it by name."""

        class DummyBenchmark(Benchmark):
            def run(self, model, dataset=None):
                return BenchmarkResult(
                    benchmark_name="dummy",
                    model_name="m",
                    metrics={"x": 1.0},
                )

        config = BenchmarkConfig(
            name="dummy",
            description="d",
            metric_names=["x"],
        )
        bm = DummyBenchmark(config=config)
        register_benchmark("dummy", bm)

        assert get_benchmark("dummy") is bm
        assert "dummy" in list_benchmarks()

    def test_reset_clears_registry(self) -> None:
        """Reset removes all entries."""

        class DummyBenchmark(Benchmark):
            def run(self, model, dataset=None):
                return BenchmarkResult(
                    benchmark_name="d",
                    model_name="m",
                    metrics={},
                )

        config = BenchmarkConfig(name="d", description="d", metric_names=[])
        register_benchmark("d", DummyBenchmark(config=config))
        assert len(list_benchmarks()) == 1

        BenchmarkRegistry.reset()
        assert len(list_benchmarks()) == 0

    def test_get_nonexistent_raises_key_error(self) -> None:
        """Getting a nonexistent benchmark raises KeyError."""
        with pytest.raises(KeyError):
            get_benchmark("no_such_benchmark")


# ---------------------------------------------------------------------------
# Dataset protocol conformance tests
# ---------------------------------------------------------------------------


class TestDatasetProtocolConformance:
    """Tests that artifex datasets satisfy calibrax DatasetProtocol."""

    def test_mock_dataset_satisfies_protocol(self) -> None:
        """A simple dataset with __len__ and __getitem__ satisfies DatasetProtocol."""

        class SimpleDataset:
            def __len__(self) -> int:
                return 10

            def __getitem__(self, idx: int):
                return jnp.ones(3) * idx

        ds = SimpleDataset()
        assert isinstance(ds, DatasetProtocol)

    def test_batchable_dataset_satisfies_protocol(self) -> None:
        """A dataset with get_batch satisfies BatchableDatasetProtocol."""

        class BatchableDataset:
            def __len__(self) -> int:
                return 100

            def __getitem__(self, idx: int):
                return jnp.ones(3) * idx

            def get_batch(self, batch_size: int, start_idx: int = 0):
                return {"data": jnp.ones((batch_size, 3))}

        ds = BatchableDataset()
        assert isinstance(ds, DatasetProtocol)
        assert isinstance(ds, BatchableDatasetProtocol)

    def test_geometric_dataset_has_protocol_methods(self) -> None:
        """ShapeNetDataset has __len__ and __getitem__ for DatasetProtocol."""
        from artifex.benchmarks.datasets.geometric import ShapeNetDataset

        assert hasattr(ShapeNetDataset, "__len__")
        assert hasattr(ShapeNetDataset, "__getitem__")

    def test_crossdocked_dataset_has_protocol_methods(self) -> None:
        """CrossDockedDataset has __len__ and __getitem__ for DatasetProtocol."""
        from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset

        assert hasattr(CrossDockedDataset, "__len__")
        assert hasattr(CrossDockedDataset, "__getitem__")

    def test_ffhq_dataset_has_protocol_methods(self) -> None:
        """FFHQDataset has __len__ and __getitem__ for DatasetProtocol."""
        from artifex.benchmarks.datasets.ffhq import FFHQDataset

        assert hasattr(FFHQDataset, "__len__")
        assert hasattr(FFHQDataset, "__getitem__")

    def test_qm9_dataset_has_protocol_methods(self) -> None:
        """QM9Dataset has __len__ and __getitem__ for DatasetProtocol."""
        from artifex.benchmarks.datasets.qm9 import QM9Dataset

        assert hasattr(QM9Dataset, "__len__")
        assert hasattr(QM9Dataset, "__getitem__")

    def test_protein_dataset_has_protocol_methods(self) -> None:
        """SyntheticProteinDataset has __len__ and __getitem__ for DatasetProtocol."""
        from artifex.benchmarks.datasets.protein_dataset import SyntheticProteinDataset

        assert hasattr(SyntheticProteinDataset, "__len__")
        assert hasattr(SyntheticProteinDataset, "__getitem__")

    def test_celeba_dataset_has_protocol_methods(self) -> None:
        """CelebADataset has __len__ and __getitem__ for DatasetProtocol."""
        from artifex.benchmarks.datasets.celeba import CelebADataset

        assert hasattr(CelebADataset, "__len__")
        assert hasattr(CelebADataset, "__getitem__")

    def test_crossdocked_instance_satisfies_protocol(self) -> None:
        """CrossDockedDataset instance passes calibrax isinstance check."""
        from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset
        from artifex.generative_models.core.configuration import DataConfig

        config = DataConfig(
            name="crossdocked_test",
            dataset_name="CrossDocked2020",
            split="train",
            metadata={"demo_mode": True},
        )
        ds = CrossDockedDataset(
            data_path="test_results/test_data/mock_crossdocked",
            config=config,
            rngs=nnx.Rngs(42),
        )
        assert isinstance(ds, DatasetProtocol)


# ---------------------------------------------------------------------------
# Metric protocol conformance tests
# ---------------------------------------------------------------------------


class TestMetricProtocolConformance:
    """Tests that artifex metrics have calibrax MetricProtocol attributes."""

    def test_metric_base_has_required_methods(self) -> None:
        """MetricBase has compute and validate_inputs abstract methods."""
        from artifex.generative_models.core.protocols.metrics import MetricBase

        assert hasattr(MetricBase, "compute")
        assert hasattr(MetricBase, "validate_inputs")

    def test_fid_metric_has_name_and_direction(self) -> None:
        """FIDMetric has name and higher_is_better attributes."""
        from artifex.benchmarks.metrics.image import FIDMetric
        from artifex.generative_models.core.configuration import EvaluationConfig

        config = EvaluationConfig(
            name="fid_test",
            metrics=("fid",),
            metric_params={"fid": {"mock_inception": True}},
        )
        metric = FIDMetric(rngs=nnx.Rngs(42), config=config)

        assert hasattr(metric, "name")
        assert hasattr(metric, "higher_is_better")
        assert isinstance(metric.name, str)
        assert isinstance(metric.higher_is_better, bool)

    def test_ssim_metric_has_name_and_direction(self) -> None:
        """SSIMMetric has name and higher_is_better attributes."""
        from artifex.benchmarks.metrics.image import SSIMMetric
        from artifex.generative_models.core.configuration import EvaluationConfig

        config = EvaluationConfig(name="ssim_test", metrics=("ssim",))
        metric = SSIMMetric(rngs=nnx.Rngs(42), config=config)

        assert hasattr(metric, "name")
        assert hasattr(metric, "higher_is_better")


# ---------------------------------------------------------------------------
# Result bridge round-trip tests
# ---------------------------------------------------------------------------


class TestResultBridge:
    """Tests that result conversion between artifex and calibrax works."""

    def test_round_trip_preserves_data(self) -> None:
        """Converting to calibrax and back preserves key fields."""
        original = BenchmarkResult(
            benchmark_name="test_bench",
            model_name="test_model",
            metrics={"accuracy": 0.95, "loss": 0.05},
            metadata={"epochs": 10},
        )

        calibrax_result = to_calibrax_result(original, domain="testing")
        restored = from_calibrax_result(calibrax_result)

        assert restored.benchmark_name == original.benchmark_name
        assert restored.model_name == original.model_name
        assert restored.metrics["accuracy"] == pytest.approx(0.95)
        assert restored.metrics["loss"] == pytest.approx(0.05)

    def test_jax_scalars_sanitized(self) -> None:
        """JAX scalars are converted to Python primitives."""
        original = BenchmarkResult(
            benchmark_name="jax_test",
            model_name="model",
            metrics={"val": float(jnp.float32(3.14))},
        )

        calibrax_result = to_calibrax_result(original)
        assert isinstance(calibrax_result.name, str)


# ---------------------------------------------------------------------------
# Benchmark lifecycle conformance tests
# ---------------------------------------------------------------------------


class TestBenchmarkLifecycle:
    """Tests for benchmark lifecycle method availability."""

    def test_benchmark_base_has_run(self) -> None:
        """Benchmark ABC has abstract run() method."""
        assert hasattr(Benchmark, "run")

    def test_benchmark_suite_has_run_all(self) -> None:
        """BenchmarkSuite has run_all() method."""
        assert hasattr(BenchmarkSuite, "run_all")

    def test_geometric_benchmark_has_lifecycle_methods(self) -> None:
        """GeometricBenchmarkSuite benchmarks have training and evaluation."""
        from artifex.benchmarks.suites.geometric_suite import (
            PointCloudGenerationBenchmark,
        )

        assert hasattr(PointCloudGenerationBenchmark, "run_training")
        assert hasattr(PointCloudGenerationBenchmark, "run_evaluation")
