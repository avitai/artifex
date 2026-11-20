"""Repository contracts for the Calibrax-first benchmark foundation."""

from __future__ import annotations

from pathlib import Path

from tests.utils.fresh_interpreter import run_repo_json


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_benchmark_registry_surface_reuses_calibrax_singleton() -> None:
    """Benchmarks register in the Calibrax singleton, which artifex neither wraps nor aliases."""
    payload = run_repo_json(
        "from substrax.testing import report; "
        "import artifex.benchmarks, artifex.benchmarks.registry; "
        "from calibrax.core import BenchmarkRegistry; "
        "from artifex.benchmarks.registry import register_benchmark; "
        "marker = object(); "
        "register_benchmark('contract_probe', marker); "
        "report({"
        "'package_alias': hasattr(artifex.benchmarks, 'BenchmarkRegistry'), "
        "'module_alias': hasattr(artifex.benchmarks.registry, 'BenchmarkRegistry'), "
        "'registered_in_calibrax': BenchmarkRegistry().get('contract_probe') is marker"
        "})"
    )

    assert payload["package_alias"] is False
    assert payload["module_alias"] is False
    assert payload["registered_in_calibrax"] is True

    registry_source = (REPO_ROOT / "src/artifex/benchmarks/registry.py").read_text()
    assert "class BenchmarkRegistry" not in registry_source


def test_dataset_registry_surface_uses_calibrax_singleton_without_items_escape_hatch() -> None:
    """Dataset registry should subclass Calibrax singleton storage without exposing .datasets."""
    payload = run_repo_json(
        "from substrax.testing import report; "
        "from calibrax.core import SingletonRegistry; "
        "from artifex.benchmarks.datasets.base import DatasetRegistry; "
        "registry = DatasetRegistry(); "
        "report({"
        "'is_subclass': issubclass(DatasetRegistry, SingletonRegistry), "
        "'module': DatasetRegistry.__mro__[1].__module__, "
        "'has_datasets_attr': hasattr(registry, 'datasets')"
        "})"
    )

    assert payload["is_subclass"] is True
    assert payload["module"] == "calibrax.core.registry"
    assert payload["has_datasets_attr"] is False


def test_dataset_loader_registry_reuses_the_shared_dataset_registry() -> None:
    """Dataset loader helpers should reuse the shared DatasetRegistry instance."""
    payload = run_repo_json(
        "from substrax.testing import report; "
        "from artifex.benchmarks.datasets.base import DatasetRegistry; "
        "from artifex.benchmarks.datasets.dataset_loaders import _dataset_registry; "
        "report({"
        "'same_type': type(_dataset_registry) is DatasetRegistry, "
        "'module': type(_dataset_registry).__module__"
        "})"
    )

    assert payload["same_type"] is True
    assert payload["module"] == "artifex.benchmarks.datasets.base"


def test_benchmark_protocol_exports_narrow_to_calibrax_protocols() -> None:
    """The benchmark protocol package should export Calibrax protocol owners only."""
    payload = run_repo_json(
        "from substrax.testing import report; "
        "import artifex.benchmarks.protocols as protocols; "
        "report({"
        "'exports': list(getattr(protocols, '__all__')), "
        "'benchmark_protocol_module': protocols.BenchmarkProtocol.__module__, "
        "'dataset_protocol_module': protocols.DatasetProtocol.__module__, "
        "'metric_protocol_module': protocols.MetricProtocol.__module__"
        "})"
    )

    assert payload["exports"] == [
        "BenchmarkProtocol",
        "DatasetProtocol",
        "BatchableDatasetProtocol",
        "MetricProtocol",
        "StatefulMetricProtocol",
        "MetricLearningProtocol",
    ]
    assert payload["benchmark_protocol_module"] == "calibrax.core.protocols"
    assert payload["dataset_protocol_module"] == "calibrax.core.protocols"
    assert payload["metric_protocol_module"] == "calibrax.core.protocols"


def test_legacy_protein_adapter_module_reexports_the_retained_adapter_only() -> None:
    """The legacy protein adapter module should be a compatibility re-export only."""
    payload = run_repo_json(
        "from substrax.testing import report; "
        "from artifex.benchmarks import protein_model_adapters as legacy; "
        "from artifex.benchmarks.model_adapters import protein_adapters as retained; "
        "report({"
        "'same_object': legacy.ProteinPointCloudAdapter is retained.ProteinPointCloudAdapter, "
        "'legacy_exports': list(getattr(legacy, '__all__', []))"
        "})"
    )

    assert payload["same_object"] is True
    assert payload["legacy_exports"] == ["ProteinPointCloudAdapter"]

    legacy_source = (REPO_ROOT / "src/artifex/benchmarks/protein_model_adapters.py").read_text()
    assert "class ProteinPointCloudAdapter" not in legacy_source
    assert "register_adapter(" not in legacy_source
