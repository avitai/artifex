"""Registry for benchmark suites.

Uses calibrax.core.Registry as the backing store for suite registration
and discovery by name.
"""

from collections.abc import Callable

from calibrax.core import Registry

from artifex.benchmarks import Benchmark


# Module-level calibrax Registry for suite factories.
_suite_registry: Registry[Callable[[], list[Benchmark]]] = Registry()


def register_suite(name: str, suite_fn: Callable[[], list[Benchmark]]) -> None:
    """Register a benchmark suite.

    Args:
        name: Name to register the suite under.
        suite_fn: Function that returns a list of benchmarks.
    """
    _suite_registry.register(name, suite_fn)


def get_suite(name: str) -> list[Benchmark]:
    """Get a benchmark suite by name.

    Args:
        name: Name of the suite to get.

    Returns:
        List of benchmarks in the suite.

    Raises:
        KeyError: If the suite isn't registered.
    """
    return _suite_registry.get(name)()


def list_suites() -> list[str]:
    """List all registered benchmark suites.

    Returns:
        List of suite names.
    """
    return _suite_registry.list_names()
