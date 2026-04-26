"""Tests for the CalibraX-backed benchmark and suite registries."""

import pytest
from calibrax.core import (
    BenchmarkRegistry as CalibraxBenchmarkRegistry,
    Registry as CalibraxRegistry,
)

from artifex.benchmarks.core import Benchmark, BenchmarkConfig, BenchmarkResult
from artifex.benchmarks.registry import (
    BenchmarkRegistry,
    get_benchmark,
    list_benchmarks,
    register_benchmark,
)
from artifex.benchmarks.suites.registry import (
    get_suite,
    list_suites,
    register_suite,
)


class _DummyBenchmark(Benchmark):
    """Minimal benchmark for registry tests."""

    def run(self, model, dataset=None):
        return BenchmarkResult(
            benchmark_name=self.config.name,
            model_name="test",
            metrics={"x": 1.0},
        )


def _make_benchmark(name: str = "test") -> _DummyBenchmark:
    config = BenchmarkConfig(name=name, description=name, metric_names=["x"])
    return _DummyBenchmark(config=config)


class TestBenchmarkRegistryCalibrax:
    """Verify the benchmark registry surface reuses the CalibraX singleton."""

    def setup_method(self) -> None:
        BenchmarkRegistry.reset()

    def test_registry_aliases_calibrax_singleton(self) -> None:
        assert BenchmarkRegistry is CalibraxBenchmarkRegistry

    def test_singleton(self) -> None:
        r1 = BenchmarkRegistry()
        r2 = BenchmarkRegistry()
        assert r1 is r2

    def test_register_and_get(self) -> None:
        bm = _make_benchmark("reg_get")
        registry = BenchmarkRegistry()
        registry.register("reg_get", bm)
        assert registry.get("reg_get") is bm

    def test_list(self) -> None:
        bm1 = _make_benchmark("a")
        bm2 = _make_benchmark("b")
        registry = BenchmarkRegistry()
        registry.register("a", bm1)
        registry.register("b", bm2)
        names = registry.list_names()
        assert "a" in names
        assert "b" in names

    def test_get_nonexistent_raises(self) -> None:
        registry = BenchmarkRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_reset_clears(self) -> None:
        registry = BenchmarkRegistry()
        registry.register("x", _make_benchmark("x"))
        assert len(registry.list_names()) == 1
        BenchmarkRegistry.reset()
        assert len(registry.list_names()) == 0


class TestConvenienceFunctions:
    """Verify Artifex convenience helpers sit on top of the CalibraX singleton."""

    def setup_method(self) -> None:
        BenchmarkRegistry.reset()

    def test_register_and_get_benchmark(self) -> None:
        bm = _make_benchmark("conv")
        register_benchmark("conv", bm)
        assert get_benchmark("conv") is bm

    def test_list_benchmarks(self) -> None:
        register_benchmark("p", _make_benchmark("p"))
        register_benchmark("q", _make_benchmark("q"))
        names = list_benchmarks()
        assert "p" in names
        assert "q" in names

    def test_decorator_registration(self) -> None:
        @register_benchmark("decorated")
        class DecoratedBenchmark(Benchmark):
            """Decorated benchmark."""

            def __init__(self, config=None):
                if config is None:
                    config = BenchmarkConfig(
                        name="decorated",
                        description="decorated",
                        metric_names=[],
                    )
                super().__init__(config=config)

            def run(self, model, dataset=None):
                return BenchmarkResult(
                    benchmark_name="decorated",
                    model_name="m",
                    metrics={},
                )

        assert "decorated" in list_benchmarks()
        assert isinstance(get_benchmark("decorated"), DecoratedBenchmark)

    def test_get_nonexistent_raises(self) -> None:
        with pytest.raises(KeyError):
            get_benchmark("no_such")


class TestSuiteRegistryCalibrax:
    """Verify suite registry uses a direct CalibraX Registry instance."""

    def setup_method(self) -> None:
        from artifex.benchmarks.suites.registry import _suite_registry

        _suite_registry.clear()

    def teardown_method(self) -> None:
        from artifex.benchmarks.suites import standard
        from artifex.benchmarks.suites.registry import _suite_registry

        _suite_registry.clear()
        register_suite("quality", standard.get_quality_suite)
        register_suite("performance", standard.get_performance_suite)
        register_suite("standard", standard.get_standard_suite)

    def test_backed_by_calibrax_registry(self) -> None:
        from artifex.benchmarks.suites.registry import _suite_registry

        assert isinstance(_suite_registry, CalibraxRegistry)

    def test_register_and_get_suite(self) -> None:
        def my_suite() -> list[Benchmark]:
            return [_make_benchmark("s1")]

        register_suite("my_suite", my_suite)
        result = get_suite("my_suite")
        assert len(result) == 1
        assert result[0].config.name == "s1"

    def test_list_suites(self) -> None:
        register_suite("a", lambda: [])
        register_suite("b", lambda: [])
        names = list_suites()
        assert "a" in names
        assert "b" in names

    def test_get_nonexistent_suite_raises(self) -> None:
        with pytest.raises(KeyError):
            get_suite("unknown_suite")
