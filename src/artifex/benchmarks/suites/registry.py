"""Registry for benchmark suites.

This module provides a registry for benchmark suites, allowing suites to be
registered and discovered by name.
"""

from typing import Callable

from artifex.benchmarks import Benchmark


# Registry for benchmark suites
_suite_registry: dict[str, Callable[[], list[Benchmark]]] = {}


def register_suite(name: str, suite_fn: Callable[[], list[Benchmark]]) -> None:
    """Register a benchmark suite.

    Args:
        name: Name to register the suite under.
        suite_fn: Function that returns a list of benchmarks.
    """
    _suite_registry[name] = suite_fn


def get_suite(name: str) -> list[Benchmark]:
    """Get a benchmark suite by name.

    Args:
        name: Name of the suite to get.

    Returns:
        List of benchmarks in the suite.

    Raises:
        KeyError: If the suite isn't registered.
    """
    if name not in _suite_registry:
        raise KeyError(f"Benchmark suite '{name}' not found in registry")
    return _suite_registry[name]()


def list_suites() -> list[str]:
    """List all registered benchmark suites.

    Returns:
        List of suite names.
    """
    return list(_suite_registry.keys())
