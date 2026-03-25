"""Repository contracts for paper and roadmap truthfulness."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER = REPO_ROOT / "docs/papers/artifex_arxiv_preprint.md"
ROADMAP = REPO_ROOT / "docs/roadmap/planned-modules.md"


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _section(contents: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)",
        contents,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section: {heading}"
    return match.group("body")


def test_preprint_separates_shipped_experimental_and_roadmap_surfaces() -> None:
    """The paper should distinguish live runtime imports from experimental and roadmap claims."""
    payload = _run_python(
        "import json; "
        "from artifex.generative_models.models.base import ("
        "GenerativeModelProtocol, TrainableGenerativeModelProtocol); "
        "from artifex.generative_models.modalities.base import ModelAdapter; "
        "from artifex.generative_models.inference.optimization.production import ProductionOptimizer; "
        "from artifex.benchmarks import BenchmarkRegistry; "
        "import artifex.generative_models.core.evaluation as evaluation; "
        "print(json.dumps({"
        "'model_protocol': GenerativeModelProtocol.__name__, "
        "'trainable_protocol': TrainableGenerativeModelProtocol.__name__, "
        "'adapter_protocol': ModelAdapter.__name__, "
        "'optimizer': ProductionOptimizer.__name__, "
        "'benchmark_registry': BenchmarkRegistry.__name__, "
        "'evaluation_exports': sorted(evaluation.__all__)"
        "}))"
    )

    contents = PAPER.read_text(encoding="utf-8")

    for heading in (
        "Shipped Runtime Surface",
        "Experimental But Importable Surfaces",
        "Roadmap-Only And Future Work",
    ):
        assert f"## {heading}" in contents

    model_protocol = payload["model_protocol"]
    trainable_protocol = payload["trainable_protocol"]
    adapter_protocol = payload["adapter_protocol"]
    optimizer = payload["optimizer"]
    benchmark_registry = payload["benchmark_registry"]
    evaluation_exports = payload["evaluation_exports"]

    assert isinstance(model_protocol, str)
    assert isinstance(trainable_protocol, str)
    assert isinstance(adapter_protocol, str)
    assert isinstance(optimizer, str)
    assert isinstance(benchmark_registry, str)
    assert isinstance(evaluation_exports, list)
    assert all(isinstance(name, str) for name in evaluation_exports)

    required_tokens = [
        model_protocol,
        trainable_protocol,
        adapter_protocol,
        optimizer,
        benchmark_registry,
        "create(",
        "jit_compilation",
        *evaluation_exports,
    ]

    for token in required_tokens:
        assert token in contents


def test_preprint_omits_dead_pipeline_names_and_false_completeness_claims() -> None:
    """The paper should not publish non-exported or placeholder-heavy owners as current API."""
    contents = PAPER.read_text(encoding="utf-8")

    banned_tokens = [
        "EvaluationPipeline",
        "complete benchmarking system",
        "adapt_model(...)",
        "preprocess(...)",
        "postprocess(...)",
        "production-ready",
    ]

    for banned_token in banned_tokens:
        assert banned_token not in contents


def test_roadmap_current_runtime_status_tracks_live_inventory() -> None:
    """The roadmap current-runtime section should match importable flow, modality, inference, and benchmark surfaces."""
    payload = _run_python(
        "import json; "
        "import artifex.benchmarks as benchmarks; "
        "import artifex.generative_models.inference as inference; "
        "import artifex.generative_models.models.flow as flow; "
        "import artifex.generative_models.modalities as modalities; "
        "from artifex.generative_models.inference.optimization.production import ProductionOptimizer; "
        "print(json.dumps({"
        "'flow_exports': [name for name in ['RealNVP', 'Glow', 'MAF', 'IAF', 'NeuralSplineFlow', 'ConditionalRealNVP'] if hasattr(flow, name)], "
        "'timeseries_name': modalities.TimeseriesModality.__name__, "
        "'inference_package': inference.__name__, "
        "'production_module': ProductionOptimizer.__module__, "
        "'benchmarks_package': benchmarks.__name__"
        "}))"
    )

    contents = ROADMAP.read_text(encoding="utf-8")
    current = _section(contents, "Current Runtime Status")

    assert "coming soon" in contents.lower()
    assert "| Category | Supported | Experimental | Planned |" not in contents
    assert "All inference modules are planned." not in contents

    flow_exports = payload["flow_exports"]
    timeseries_name = payload["timeseries_name"]
    inference_package = payload["inference_package"]
    production_module = payload["production_module"]
    benchmarks_package = payload["benchmarks_package"]

    assert isinstance(flow_exports, list)
    assert all(isinstance(name, str) for name in flow_exports)
    assert isinstance(timeseries_name, str)
    assert isinstance(inference_package, str)
    assert isinstance(production_module, str)
    assert isinstance(benchmarks_package, str)

    for export_name in flow_exports:
        assert export_name in current

    for required_token in (
        timeseries_name,
        inference_package,
        production_module,
        benchmarks_package,
        "artifex.generative_models.modalities.multi_modal",
    ):
        assert required_token in current


def test_roadmap_moves_missing_video_and_unshipped_flow_families_to_roadmap_only() -> None:
    """Missing families should stay in the roadmap-only section, not the current runtime tables."""
    contents = ROADMAP.read_text(encoding="utf-8")
    current = _section(contents, "Current Runtime Status")
    roadmap_only = _section(contents, "Roadmap-Only Surfaces")

    for token in (
        "Neural ODE",
        "CNF",
        "artifex.generative_models.modalities.video",
    ):
        assert token not in current
        assert token in roadmap_only
