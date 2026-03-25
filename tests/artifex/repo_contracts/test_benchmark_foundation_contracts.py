"""Repository contracts for the CalibraX-first benchmark foundation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_benchmark_registry_surface_reuses_calibrax_singleton() -> None:
    """The benchmark registry should be the CalibraX singleton, not a local wrapper."""
    payload = _run_python(
        "import json; "
        "from calibrax.core import BenchmarkRegistry as CalibraxBenchmarkRegistry; "
        "from artifex.benchmarks.registry import BenchmarkRegistry; "
        "print(json.dumps({"
        "'same_object': BenchmarkRegistry is CalibraxBenchmarkRegistry, "
        "'module': BenchmarkRegistry.__module__"
        "}))"
    )

    assert payload["same_object"] is True
    assert payload["module"] == "calibrax.core.registry"

    registry_source = (REPO_ROOT / "src/artifex/benchmarks/registry.py").read_text()
    assert "class BenchmarkRegistry" not in registry_source


def test_dataset_registry_surface_uses_calibrax_singleton_without_items_escape_hatch() -> None:
    """Dataset registry should subclass CalibraX singleton storage without exposing .datasets."""
    payload = _run_python(
        "import json; "
        "from calibrax.core import SingletonRegistry; "
        "from artifex.benchmarks.datasets.base import DatasetRegistry; "
        "registry = DatasetRegistry(); "
        "print(json.dumps({"
        "'is_subclass': issubclass(DatasetRegistry, SingletonRegistry), "
        "'module': DatasetRegistry.__mro__[1].__module__, "
        "'has_datasets_attr': hasattr(registry, 'datasets')"
        "}))"
    )

    assert payload["is_subclass"] is True
    assert payload["module"] == "calibrax.core.registry"
    assert payload["has_datasets_attr"] is False


def test_dataset_loader_registry_reuses_the_shared_dataset_registry() -> None:
    """Dataset loader helpers should reuse the shared DatasetRegistry instance."""
    payload = _run_python(
        "import json; "
        "from artifex.benchmarks.datasets.base import DatasetRegistry; "
        "from artifex.benchmarks.datasets.dataset_loaders import _dataset_registry; "
        "print(json.dumps({"
        "'same_type': type(_dataset_registry) is DatasetRegistry, "
        "'module': type(_dataset_registry).__module__"
        "}))"
    )

    assert payload["same_type"] is True
    assert payload["module"] == "artifex.benchmarks.datasets.base"


def test_benchmark_protocol_exports_narrow_to_calibrax_protocols() -> None:
    """The benchmark protocol package should export CalibraX protocol owners only."""
    payload = _run_python(
        "import json; "
        "import artifex.benchmarks.protocols as protocols; "
        "print(json.dumps({"
        "'exports': list(getattr(protocols, '__all__')), "
        "'benchmark_protocol_module': protocols.BenchmarkProtocol.__module__, "
        "'dataset_protocol_module': protocols.DatasetProtocol.__module__, "
        "'metric_protocol_module': protocols.MetricProtocol.__module__"
        "}))"
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
    payload = _run_python(
        "import json; "
        "from artifex.benchmarks import protein_model_adapters as legacy; "
        "from artifex.benchmarks.model_adapters import protein_adapters as retained; "
        "print(json.dumps({"
        "'same_object': legacy.ProteinPointCloudAdapter is retained.ProteinPointCloudAdapter, "
        "'legacy_exports': list(getattr(legacy, '__all__', []))"
        "}))"
    )

    assert payload["same_object"] is True
    assert payload["legacy_exports"] == ["ProteinPointCloudAdapter"]

    legacy_source = (REPO_ROOT / "src/artifex/benchmarks/protein_model_adapters.py").read_text()
    assert "class ProteinPointCloudAdapter" not in legacy_source
    assert "register_adapter(" not in legacy_source
