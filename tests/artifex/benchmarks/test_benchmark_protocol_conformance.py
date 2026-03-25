"""Tests for benchmark protocol conformance with calibrax.core.BenchmarkProtocol."""

import flax.nnx as nnx
import pytest
from calibrax.core.protocols import BenchmarkProtocol

from artifex.benchmarks.core import Benchmark, BenchmarkBase, BenchmarkConfig
from artifex.generative_models.core.configuration import EvaluationConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs():
    """Provide RNGs for tests."""
    return nnx.Rngs(42)


# ---------------------------------------------------------------------------
# Base class conformance
# ---------------------------------------------------------------------------


class TestBenchmarkBaseConformance:
    """Test that Benchmark ABC conforms to BenchmarkProtocol."""

    def test_benchmark_has_setup(self):
        """Benchmark ABC exposes a setup() method."""
        assert hasattr(Benchmark, "setup")

    def test_benchmark_has_run_training(self):
        """Benchmark ABC exposes a run_training() method."""
        assert hasattr(Benchmark, "run_training")

    def test_benchmark_has_run_evaluation(self):
        """Benchmark ABC exposes a run_evaluation() method."""
        assert hasattr(Benchmark, "run_evaluation")

    def test_benchmark_has_teardown(self):
        """Benchmark ABC exposes a teardown() method."""
        assert hasattr(Benchmark, "teardown")

    def test_benchmark_has_get_performance_targets(self):
        """Benchmark ABC exposes a get_performance_targets() method."""
        assert hasattr(Benchmark, "get_performance_targets")


class TestBenchmarkBaseNNXConformance:
    """Test that BenchmarkBase (nnx.Module) conforms to BenchmarkProtocol."""

    def test_benchmarkbase_has_setup(self):
        """BenchmarkBase exposes a setup() method."""
        assert hasattr(BenchmarkBase, "setup")

    def test_benchmarkbase_has_teardown(self):
        """BenchmarkBase exposes a teardown() method."""
        assert hasattr(BenchmarkBase, "teardown")

    def test_benchmarkbase_has_run_training(self):
        """BenchmarkBase exposes run_training()."""
        assert hasattr(BenchmarkBase, "run_training")

    def test_benchmarkbase_has_run_evaluation(self):
        """BenchmarkBase exposes run_evaluation()."""
        assert hasattr(BenchmarkBase, "run_evaluation")

    def test_benchmarkbase_has_get_performance_targets(self):
        """BenchmarkBase exposes get_performance_targets()."""
        assert hasattr(BenchmarkBase, "get_performance_targets")


# ---------------------------------------------------------------------------
# Concrete benchmark isinstance checks
# ---------------------------------------------------------------------------


class _MinimalBenchmark(Benchmark):
    """Minimal concrete Benchmark for testing protocol conformance."""

    def run(self, model, dataset=None):
        """No-op run."""
        return None


class _MinimalBenchmarkBase(BenchmarkBase):
    """Minimal concrete BenchmarkBase for testing."""

    def _setup_benchmark_components(self) -> None:
        """No-op setup."""

    def run_training(self) -> dict[str, float]:
        """No-op training."""
        return {"loss": 0.0}

    def run_evaluation(self) -> dict[str, float]:
        """No-op evaluation."""
        return {"accuracy": 1.0}

    def get_performance_targets(self) -> dict[str, float]:
        """Return targets."""
        return {"accuracy": 0.9}


class TestInstanceofBenchmarkProtocol:
    """Test concrete benchmarks pass isinstance(b, BenchmarkProtocol)."""

    def test_minimal_benchmark_is_benchmark_protocol(self):
        """A minimal Benchmark subclass satisfies BenchmarkProtocol."""
        config = BenchmarkConfig(
            name="test",
            description="test",
            metric_names=["accuracy"],
        )
        b = _MinimalBenchmark(config)
        assert isinstance(b, BenchmarkProtocol)

    def test_minimal_benchmarkbase_is_benchmark_protocol(self, rngs):
        """A minimal BenchmarkBase subclass satisfies BenchmarkProtocol."""
        config = EvaluationConfig(
            name="test",
            metrics=["accuracy"],
            metric_params={},
            eval_batch_size=4,
        )
        b = _MinimalBenchmarkBase(config, rngs=rngs)
        assert isinstance(b, BenchmarkProtocol)


# ---------------------------------------------------------------------------
# Lifecycle method behaviour
# ---------------------------------------------------------------------------


class TestLifecycleDefaults:
    """Test that default lifecycle methods on Benchmark work correctly."""

    def test_setup_is_noop(self):
        """Default setup() does not raise."""
        config = BenchmarkConfig(
            name="test",
            description="test",
            metric_names=[],
        )
        b = _MinimalBenchmark(config)
        b.setup()  # should not raise

    def test_teardown_is_noop(self):
        """Default teardown() does not raise."""
        config = BenchmarkConfig(
            name="test",
            description="test",
            metric_names=[],
        )
        b = _MinimalBenchmark(config)
        b.teardown()  # should not raise

    def test_run_training_returns_dict(self):
        """Default run_training() returns an empty dict."""
        config = BenchmarkConfig(
            name="test",
            description="test",
            metric_names=[],
        )
        b = _MinimalBenchmark(config)
        result = b.run_training()
        assert isinstance(result, dict)

    def test_run_evaluation_returns_dict(self):
        """Default run_evaluation() returns an empty dict."""
        config = BenchmarkConfig(
            name="test",
            description="test",
            metric_names=[],
        )
        b = _MinimalBenchmark(config)
        result = b.run_evaluation()
        assert isinstance(result, dict)

    def test_get_performance_targets_returns_dict(self):
        """Default get_performance_targets() returns an empty dict."""
        config = BenchmarkConfig(
            name="test",
            description="test",
            metric_names=[],
        )
        b = _MinimalBenchmark(config)
        targets = b.get_performance_targets()
        assert isinstance(targets, dict)


class TestBenchmarkBaseLifecycle:
    """Test lifecycle on BenchmarkBase (nnx.Module)."""

    def test_setup_is_noop(self, rngs):
        """BenchmarkBase.setup() does not raise."""
        config = EvaluationConfig(
            name="test",
            metrics=["m"],
            metric_params={},
            eval_batch_size=4,
        )
        b = _MinimalBenchmarkBase(config, rngs=rngs)
        b.setup()

    def test_teardown_is_noop(self, rngs):
        """BenchmarkBase.teardown() does not raise."""
        config = EvaluationConfig(
            name="test",
            metrics=["m"],
            metric_params={},
            eval_batch_size=4,
        )
        b = _MinimalBenchmarkBase(config, rngs=rngs)
        b.teardown()

    def test_run_training_returns_dict(self, rngs):
        """BenchmarkBase.run_training() returns dict."""
        config = EvaluationConfig(
            name="test",
            metrics=["m"],
            metric_params={},
            eval_batch_size=4,
        )
        b = _MinimalBenchmarkBase(config, rngs=rngs)
        result = b.run_training()
        assert isinstance(result, dict)
        assert "loss" in result

    def test_get_performance_targets_returns_dict(self, rngs):
        """BenchmarkBase.get_performance_targets() returns dict."""
        config = EvaluationConfig(
            name="test",
            metrics=["m"],
            metric_params={},
            eval_batch_size=4,
        )
        b = _MinimalBenchmarkBase(config, rngs=rngs)
        targets = b.get_performance_targets()
        assert isinstance(targets, dict)
        assert "accuracy" in targets


# ---------------------------------------------------------------------------
# Concrete suite isinstance checks (import-level verification)
# ---------------------------------------------------------------------------


class TestConcreteSuiteConformance:
    """Verify all concrete benchmark classes have lifecycle methods."""

    def test_point_cloud_benchmark_has_lifecycle(self):
        """PointCloudGenerationBenchmark has all lifecycle methods."""
        from artifex.benchmarks.suites.geometric_suite import (
            PointCloudGenerationBenchmark,
        )

        for method in (
            "setup",
            "run_training",
            "run_evaluation",
            "teardown",
            "get_performance_targets",
        ):
            assert hasattr(PointCloudGenerationBenchmark, method), f"Missing {method}"

    def test_protein_structure_benchmark_has_lifecycle(self):
        """ProteinStructureBenchmark has all lifecycle methods."""
        from artifex.benchmarks.suites.protein_benchmarks import (
            ProteinStructureBenchmark,
        )

        for method in (
            "setup",
            "run_training",
            "run_evaluation",
            "teardown",
            "get_performance_targets",
        ):
            assert hasattr(ProteinStructureBenchmark, method), f"Missing {method}"

    def test_protein_ligand_benchmark_has_lifecycle(self):
        """ProteinLigandCoDesignBenchmark has all lifecycle methods."""
        from artifex.benchmarks.suites.protein_ligand_suite import (
            ProteinLigandCoDesignBenchmark,
        )

        for method in (
            "setup",
            "run_training",
            "run_evaluation",
            "teardown",
            "get_performance_targets",
        ):
            assert hasattr(ProteinLigandCoDesignBenchmark, method), f"Missing {method}"

    def test_multi_beta_vae_benchmark_has_lifecycle(self):
        """MultiBetaVAEBenchmark has all lifecycle methods."""
        from artifex.benchmarks.suites.multi_beta_vae_suite import (
            MultiBetaVAEBenchmark,
        )

        for method in (
            "setup",
            "run_training",
            "run_evaluation",
            "teardown",
            "get_performance_targets",
        ):
            assert hasattr(MultiBetaVAEBenchmark, method), f"Missing {method}"

    def test_stylegan3_benchmark_has_lifecycle(self):
        """StyleGAN3Benchmark has all lifecycle methods."""
        from artifex.benchmarks.suites.stylegan3_suite import StyleGAN3Benchmark

        for method in (
            "setup",
            "run_training",
            "run_evaluation",
            "teardown",
            "get_performance_targets",
        ):
            assert hasattr(StyleGAN3Benchmark, method), f"Missing {method}"

    def test_se3_molecular_flows_benchmark_has_lifecycle(self):
        """SE3MolecularFlowsBenchmark has all lifecycle methods."""
        from artifex.benchmarks.suites.se3_molecular_flows_suite import (
            SE3MolecularFlowsBenchmark,
        )

        for method in (
            "setup",
            "run_training",
            "run_evaluation",
            "teardown",
            "get_performance_targets",
        ):
            assert hasattr(SE3MolecularFlowsBenchmark, method), f"Missing {method}"
