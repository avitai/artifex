"""Repository contracts for the evaluation surface: one metric layer, over calibrax."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DOC = REPO_ROOT / "docs/core/benchmarks.md"
RUNNER_DOC = REPO_ROOT / "docs/core/runner.md"
INDEX_DOC = REPO_ROOT / "docs/core/index.md"
METRICS_DOC = REPO_ROOT / "docs/core/metrics.md"
REGISTRY_DOC = REPO_ROOT / "docs/core/registry.md"
GAN_GUIDE = REPO_ROOT / "docs/user-guide/models/gan-guide.md"
TRAINING_OVERVIEW = REPO_ROOT / "docs/user-guide/training/overview.md"
TIER_ONE_NAMES = (
    "fid_from_backbone",
    "inception_score_from_backbone",
    "precision_from_backbone",
    "recall_from_backbone",
    "perplexity_from_model",
)


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_benchmark_foundation_lives_only_under_artifex_benchmarks_core() -> None:
    """Benchmark ownership should live under artifex.benchmarks.core only."""
    payload = _run_python(
        textwrap.dedent(
            """
            import importlib
            import json

            import artifex.generative_models.core as core
            import artifex.generative_models.core.protocols as protocols
            from artifex.benchmarks.core import (
                Benchmark,
                BenchmarkBase,
                BenchmarkConfig,
                BenchmarkResult,
                BenchmarkRunner,
                BenchmarkSuite,
                PerformanceTracker,
            )

            errors = {}
            for module_name in (
                'artifex.generative_models.core.evaluation',
                'artifex.generative_models.core.evaluation.metrics',
                'artifex.generative_models.core.protocols.benchmarks',
            ):
                try:
                    importlib.import_module(module_name)
                except Exception as exc:
                    errors[module_name] = type(exc).__name__

            print(json.dumps({
                'benchmark_module': Benchmark.__module__,
                'benchmark_base_module': BenchmarkBase.__module__,
                'config_module': BenchmarkConfig.__module__,
                'result_module': BenchmarkResult.__module__,
                'runner_module': BenchmarkRunner.__module__,
                'suite_module': BenchmarkSuite.__module__,
                'tracker_module': PerformanceTracker.__module__,
                'core_exports': list(getattr(core, '__all__')),
                'protocol_exports': list(getattr(protocols, '__all__')),
                'errors': errors,
            }))
            """
        )
    )

    assert payload["benchmark_module"] == "artifex.benchmarks.core.foundation"
    assert payload["benchmark_base_module"] == "artifex.benchmarks.core.nnx"
    assert payload["config_module"] == "artifex.benchmarks.core.foundation"
    assert payload["result_module"] == "calibrax.core.result"
    assert payload["runner_module"] == "artifex.benchmarks.core.runner"
    assert payload["suite_module"] == "artifex.benchmarks.core.foundation"
    assert payload["tracker_module"] == "artifex.benchmarks.core.runner"
    assert "evaluation" not in payload["core_exports"]
    assert "BenchmarkBase" not in payload["protocol_exports"]
    assert "BenchmarkWithValidation" not in payload["protocol_exports"]
    assert payload["errors"] == {
        "artifex.generative_models.core.evaluation": "ModuleNotFoundError",
        "artifex.generative_models.core.evaluation.metrics": "ModuleNotFoundError",
        "artifex.generative_models.core.protocols.benchmarks": "ModuleNotFoundError",
    }

    assert not (REPO_ROOT / "src/artifex/generative_models/core/evaluation").exists()
    assert not (REPO_ROOT / "src/artifex/generative_models/core/protocols/benchmarks.py").exists()


def test_core_docs_route_benchmark_readers_to_artifex_benchmarks_core() -> None:
    """Core docs should point benchmark readers to the benchmark package, not legacy core paths."""
    benchmark_doc = BENCHMARK_DOC.read_text(encoding="utf-8")
    runner_doc = RUNNER_DOC.read_text(encoding="utf-8")
    index_doc = INDEX_DOC.read_text(encoding="utf-8")

    for contents in (benchmark_doc, runner_doc, index_doc):
        assert "artifex.benchmarks.core" in contents

    banned_tokens = [
        "core.protocols.benchmarks",
        "core.evaluation",
    ]
    for banned in banned_tokens:
        assert banned not in benchmark_doc
        assert banned not in runner_doc
        assert banned not in index_doc

    assert "artifex.benchmarks.metrics" in index_doc
    assert "Benchmark runtime" in index_doc


def test_benchmark_metrics_are_the_one_class_layer_and_register_tier_one() -> None:
    """The metric classes live in artifex.benchmarks.metrics and register in calibrax's registry."""
    payload = _run_python(
        textwrap.dedent(
            """
            import json

            from calibrax.metrics import MetricRegistry, MetricTier

            import artifex.benchmarks.metrics as benchmark_metrics
            import artifex.benchmarks.metrics.core as benchmark_metrics_core
            from artifex.benchmarks.metrics.image import FIDMetric, ISMetric
            from artifex.benchmarks.metrics.precision_recall import PrecisionRecallMetric
            from artifex.benchmarks.metrics.text import PerplexityMetric
            from artifex.generative_models.core.protocols.metrics import MetricBase

            registry = MetricRegistry()
            print(json.dumps({
                'tier_one': sorted(
                    entry.name for entry in registry.list_by_tier(MetricTier.FROZEN_BACKBONE)
                ),
                'class_modules': {
                    cls.__name__: cls.__module__
                    for cls in (FIDMetric, ISMetric, PrecisionRecallMetric, PerplexityMetric)
                },
                'all_metric_base': all(
                    issubclass(cls, MetricBase)
                    for cls in (FIDMetric, ISMetric, PrecisionRecallMetric, PerplexityMetric)
                ),
                'benchmarks_has_pipeline': hasattr(benchmark_metrics, 'EvaluationPipeline'),
                'benchmarks_core_has_pipeline': hasattr(benchmark_metrics_core, 'EvaluationPipeline'),
            }))
            """
        )
    )

    assert payload["tier_one"] == sorted(TIER_ONE_NAMES)
    assert payload["class_modules"] == {
        "FIDMetric": "artifex.benchmarks.metrics.image",
        "ISMetric": "artifex.benchmarks.metrics.image",
        "PrecisionRecallMetric": "artifex.benchmarks.metrics.precision_recall",
        "PerplexityMetric": "artifex.benchmarks.metrics.text",
    }
    assert payload["all_metric_base"] is True
    assert payload["benchmarks_has_pipeline"] is False
    assert payload["benchmarks_core_has_pipeline"] is False


def test_metric_docs_describe_the_benchmark_layer_over_calibrax() -> None:
    """The metrics and registry pages document the one layer and its calibrax registration."""
    docs = [
        METRICS_DOC.read_text(encoding="utf-8"),
        REGISTRY_DOC.read_text(encoding="utf-8"),
    ]
    combined = "\n".join(docs)

    required_tokens = [
        "artifex.benchmarks.metrics",
        "calibrax.metrics.MetricRegistry",
        "caller-supplied",
        "feature_extractor",
        "classifier",
        "model",
        "frozen_backbone",
        *TIER_ONE_NAMES,
    ]
    for token in required_tokens:
        assert token in combined

    banned_patterns = [
        r"\bEvaluationPipeline\b",
        r"\bcore\.evaluation\b",
        r"\bimage:fid\b",
        r"\bMetricsRegistry\b",
        r"\bMetricComposer\b",
        r"\bModalityMetrics\b",
        r"complete metrics system",
        r"Uses default Inception",
    ]
    for pattern in banned_patterns:
        assert re.search(pattern, combined) is None, pattern


def test_guides_show_the_benchmark_metric_classes() -> None:
    """The GAN and training guides construct metrics from the benchmark layer."""
    gan_guide = GAN_GUIDE.read_text(encoding="utf-8")
    training_overview = TRAINING_OVERVIEW.read_text(encoding="utf-8")

    assert "create_is_metric" in gan_guide
    assert "create_fid_metric" in gan_guide
    assert "inception_score_std" in gan_guide
    assert "create_fid_metric" in training_overview
    assert "create_precision_recall_metric" in training_overview
    for contents in (gan_guide, training_overview):
        assert re.search(r"\bcore\.evaluation\b", contents) is None
        assert "FrechetInceptionDistance(" not in contents
        assert "InceptionScore(" not in contents
        assert "PrecisionRecall(" not in contents
