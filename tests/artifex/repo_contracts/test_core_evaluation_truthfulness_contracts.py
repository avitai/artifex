"""Repository contracts for the narrowed core evaluation surface."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DOC = REPO_ROOT / "docs/core/benchmarks.md"
RUNNER_DOC = REPO_ROOT / "docs/core/runner.md"
INDEX_DOC = REPO_ROOT / "docs/core/index.md"
METRICS_DOC = REPO_ROOT / "docs/core/metrics.md"
PIPELINE_DOC = REPO_ROOT / "docs/core/pipeline.md"
REGISTRY_DOC = REPO_ROOT / "docs/core/registry.md"


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

            import artifex.generative_models.core.evaluation as evaluation
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
                'artifex.generative_models.core.evaluation.benchmarks',
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
                'evaluation_exports': list(getattr(evaluation, '__all__')),
                'protocol_exports': list(getattr(protocols, '__all__')),
                'errors': errors,
            }))
            """
        )
    )

    assert payload["benchmark_module"] == "artifex.benchmarks.core.foundation"
    assert payload["benchmark_base_module"] == "artifex.benchmarks.core.nnx"
    assert payload["config_module"] == "artifex.benchmarks.core.foundation"
    assert payload["result_module"] == "artifex.benchmarks.core.foundation"
    assert payload["runner_module"] == "artifex.benchmarks.core.runner"
    assert payload["suite_module"] == "artifex.benchmarks.core.foundation"
    assert payload["tracker_module"] == "artifex.benchmarks.core.runner"
    assert payload["evaluation_exports"] == ["metrics"]
    assert "BenchmarkBase" not in payload["protocol_exports"]
    assert "BenchmarkWithValidation" not in payload["protocol_exports"]
    assert (
        payload["errors"]["artifex.generative_models.core.evaluation.benchmarks"]
        == "ModuleNotFoundError"
    )
    assert (
        payload["errors"]["artifex.generative_models.core.protocols.benchmarks"]
        == "ModuleNotFoundError"
    )

    assert not (REPO_ROOT / "src/artifex/generative_models/core/evaluation/benchmarks").exists()
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
        "core.evaluation.benchmarks",
    ]
    for banned in banned_tokens:
        assert banned not in benchmark_doc
        assert banned not in runner_doc
        assert banned not in index_doc

    assert "metrics-only" in benchmark_doc
    assert "Benchmark runtime" in index_doc


def test_evaluation_pipeline_and_docs_keep_registry_ownership_in_calibrax() -> None:
    """Core evaluation should stay narrow and delegate registry ownership to CalibraX."""
    payload = _run_python(
        textwrap.dedent(
            """
            import json

            from flax import nnx

            import artifex.benchmarks.metrics as benchmark_metrics
            import artifex.benchmarks.metrics.core as benchmark_metrics_core
            import artifex.generative_models.core.evaluation.metrics as core_metrics
            from artifex.generative_models.core.configuration import EvaluationConfig
            from artifex.generative_models.core.evaluation.metrics.pipeline import EvaluationPipeline

            errors = {}
            cases = {
                'missing_fid': EvaluationConfig(name='missing_fid', metrics=['image:fid']),
                'missing_is': EvaluationConfig(name='missing_is', metrics=['image:is']),
                'missing_ppl': EvaluationConfig(name='missing_ppl', metrics=['text:perplexity']),
                'unsupported': EvaluationConfig(name='unsupported', metrics=['text:bleu']),
            }
            for name, config in cases.items():
                try:
                    EvaluationPipeline(config, rngs=nnx.Rngs(0))
                except Exception as exc:
                    errors[name] = str(exc)

            print(json.dumps({
                'errors': errors,
                'core_exports': list(getattr(core_metrics, '__all__')),
                'core_has_metrics_registry': hasattr(core_metrics, 'MetricsRegistry'),
                'core_has_metric_module': hasattr(core_metrics, 'MetricModule'),
                'core_has_feature_based_metric': hasattr(core_metrics, 'FeatureBasedMetric'),
                'core_has_distribution_metric': hasattr(core_metrics, 'DistributionMetric'),
                'core_has_sequence_metric': hasattr(core_metrics, 'SequenceMetric'),
                'core_has_metric_composer': hasattr(core_metrics, 'MetricComposer'),
                'core_has_modality_metrics': hasattr(core_metrics, 'ModalityMetrics'),
                'benchmarks_has_pipeline': hasattr(benchmark_metrics, 'EvaluationPipeline'),
                'benchmarks_has_metric_composer': hasattr(benchmark_metrics, 'MetricComposer'),
                'benchmarks_has_modality_metrics': hasattr(benchmark_metrics, 'ModalityMetrics'),
                'benchmarks_core_has_pipeline': hasattr(benchmark_metrics_core, 'EvaluationPipeline'),
                'benchmarks_core_has_metric_composer': hasattr(benchmark_metrics_core, 'MetricComposer'),
                'benchmarks_core_has_modality_metrics': hasattr(benchmark_metrics_core, 'ModalityMetrics'),
            }))
            """
        )
    )

    assert "feature_extractor" in payload["errors"]["missing_fid"]
    assert "classifier" in payload["errors"]["missing_is"]
    assert "model" in payload["errors"]["missing_ppl"]
    assert "Unsupported evaluation metric spec" in payload["errors"]["unsupported"]

    assert payload["core_exports"] == [
        "EvaluationPipeline",
        "FrechetInceptionDistance",
        "InceptionScore",
        "Perplexity",
        "PrecisionRecall",
        "DensityPrecisionRecall",
    ]
    assert "MetricsRegistry" not in payload["core_exports"]
    assert "MetricModule" not in payload["core_exports"]
    assert "FeatureBasedMetric" not in payload["core_exports"]
    assert "DistributionMetric" not in payload["core_exports"]
    assert "SequenceMetric" not in payload["core_exports"]
    assert "MetricComposer" not in payload["core_exports"]
    assert "ModalityMetrics" not in payload["core_exports"]
    assert payload["core_has_metrics_registry"] is False
    assert payload["core_has_metric_module"] is False
    assert payload["core_has_feature_based_metric"] is False
    assert payload["core_has_distribution_metric"] is False
    assert payload["core_has_sequence_metric"] is False
    assert payload["core_has_metric_composer"] is False
    assert payload["core_has_modality_metrics"] is False
    assert payload["benchmarks_has_pipeline"] is False
    assert payload["benchmarks_has_metric_composer"] is False
    assert payload["benchmarks_has_modality_metrics"] is False
    assert payload["benchmarks_core_has_pipeline"] is False
    assert payload["benchmarks_core_has_metric_composer"] is False
    assert payload["benchmarks_core_has_modality_metrics"] is False

    docs = [
        METRICS_DOC.read_text(encoding="utf-8"),
        PIPELINE_DOC.read_text(encoding="utf-8"),
        REGISTRY_DOC.read_text(encoding="utf-8"),
    ]
    combined = "\n".join(docs)

    required_tokens = [
        "caller-supplied",
        "feature_extractor",
        "classifier",
        "model",
        "image:fid",
        "image:is",
        "text:perplexity",
        "unsupported metric specs raise",
        "calibrax.metrics.MetricRegistry",
    ]
    for token in required_tokens:
        assert token in combined

    banned_tokens = [
        "complete metrics system",
        "feature_extractor=None",
        "classifier=None",
        "Uses default Inception-v3",
        "Uses default Inception classifier",
        "MetricsRegistry",
        "MetricComposer",
        "ModalityMetrics",
        '"normalization": "min_max"',
    ]
    for token in banned_tokens:
        assert token not in combined
