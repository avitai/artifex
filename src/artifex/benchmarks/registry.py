"""Registry for artifex.benchmarks."""

from __future__ import annotations

from typing import Any, Callable, cast, TypeVar

from artifex.benchmarks.base import Benchmark, BenchmarkConfig


T = TypeVar("T", bound=Benchmark)
BenchmarkFactory = Callable[[], Benchmark]


class BenchmarkRegistry:
    """Registry for benchmark implementations.

    This class implements the Singleton pattern to ensure that there's only one
    registry instance.
    """

    _instance: "BenchmarkRegistry" | None = None
    _initialized = False

    def __new__(cls) -> "BenchmarkRegistry":
        """Create a new registry instance or return the existing one."""
        if cls._instance is None:
            cls._instance = super(BenchmarkRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if not BenchmarkRegistry._initialized:
            self.benchmarks: dict[str, Benchmark] = {}
            BenchmarkRegistry._initialized = True

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (for testing purposes)."""
        if cls._instance is not None:
            cls._instance.benchmarks = {}

    def register(self, name: str, benchmark: Benchmark) -> None:
        """Register a benchmark.

        Args:
            name: Name to register the benchmark under.
            benchmark: Benchmark to register.
        """
        self.benchmarks[name] = benchmark

    def get(self, name: str) -> Benchmark:
        """Get a benchmark by name.

        Args:
            name: Name of the benchmark to get.

        Returns:
            The benchmark.

        Raises:
            KeyError: If the benchmark isn't registered.
        """
        if name not in self.benchmarks:
            raise KeyError(f"Benchmark '{name}' not found in registry")
        return self.benchmarks[name]

    def list(self) -> list[str]:
        """List all registered benchmarks.

        Returns:
            List of benchmark names.
        """
        return list(self.benchmarks.keys())


def register_benchmark(
    name: str, benchmark: Benchmark | None = None, config: BenchmarkConfig | None = None
) -> Any:
    """Register a benchmark with the given name.

    This function can be used as a decorator or as a regular function.

    Args:
        name: Name to register the benchmark under.
        benchmark: Benchmark to register (if used as a regular function).
        config: Config to use when creating a benchmark from a class (when used as a decorator).

    Returns:
        Decorator function or None.
    """
    registry = BenchmarkRegistry()

    def decorator(cls_or_fn: Any) -> Any:
        """Decorator function."""
        if isinstance(cls_or_fn, type) and issubclass(cls_or_fn, Benchmark):
            # Used as a class decorator
            benchmark_config = config or BenchmarkConfig(
                name=name, description=f"Benchmark {name}", metric_names=[]
            )
            benchmark_instance = cls_or_fn(benchmark_config)
            registry.register(name, benchmark_instance)
            return cls_or_fn
        else:
            # Used as a function - cls_or_fn is already a Benchmark instance
            benchmark_instance = cast(Benchmark, cls_or_fn)
            registry.register(name, benchmark_instance)
            return cls_or_fn

    if benchmark is not None:
        # Used as a regular function
        registry.register(name, benchmark)
        return None

    # Used as a decorator
    return decorator


def get_benchmark(name: str) -> Benchmark:
    """Get a benchmark by name.

    Args:
        name: Name of the benchmark to get.

    Returns:
        The benchmark.

    Raises:
        KeyError: If the benchmark isn't registered.
    """
    registry = BenchmarkRegistry()
    return registry.get(name)


def list_benchmarks() -> list[str]:
    """List all registered benchmarks.

    Returns:
        List of benchmark names.
    """
    registry = BenchmarkRegistry()
    return registry.list()
