from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "inference"

RETAINED_RUNTIME_PAGES: dict[str, dict[str, object]] = {
    "production.md": {
        "module": "artifex.generative_models.inference.optimization.production",
        "source": "src/artifex/generative_models/inference/optimization/production.py",
        "required": [
            "ProductionOptimizer",
            "OptimizationTarget",
            "ProductionPipeline",
            "ProductionMonitor",
            "experimental production inference helpers",
        ],
    },
}

COMING_SOON_PAGES = {
    "adapter.md": "artifex.generative_models.inference.adapter",
    "ancestral.md": "artifex.generative_models.inference.ancestral",
    "autoregressive_generator.md": "artifex.generative_models.inference.autoregressive_generator",
    "base.md": "artifex.generative_models.inference.base",
    "beam_search.md": "artifex.generative_models.inference.beam_search",
    "caching.md": "artifex.generative_models.inference.caching",
    "classifier_free.md": "artifex.generative_models.inference.classifier_free",
    "compilation.md": "artifex.generative_models.inference.compilation",
    "diffusion_generator.md": "artifex.generative_models.inference.diffusion_generator",
    "distillation.md": "artifex.generative_models.inference.distillation",
    "dynamic.md": "artifex.generative_models.inference.dynamic",
    "energy_generator.md": "artifex.generative_models.inference.energy_generator",
    "flow_generator.md": "artifex.generative_models.inference.flow_generator",
    "gan_generator.md": "artifex.generative_models.inference.gan_generator",
    "grpc.md": "artifex.generative_models.inference.grpc",
    "latency.md": "artifex.generative_models.inference.latency",
    "lora.md": "artifex.generative_models.inference.lora",
    "memory.md": "artifex.generative_models.inference.memory",
    "middleware.md": "artifex.generative_models.inference.middleware",
    "nucleus.md": "artifex.generative_models.inference.nucleus",
    "onnx.md": "artifex.generative_models.inference.onnx",
    "padding.md": "artifex.generative_models.inference.padding",
    "pipeline.md": "artifex.generative_models.inference.pipeline",
    "prefix_tuning.md": "artifex.generative_models.inference.prefix_tuning",
    "prompt_tuning.md": "artifex.generative_models.inference.prompt_tuning",
    "pruning.md": "artifex.generative_models.inference.pruning",
    "quantization.md": "artifex.generative_models.inference.quantization",
    "rest.md": "artifex.generative_models.inference.rest",
    "stateless.md": "artifex.generative_models.inference.stateless",
    "streaming.md": "artifex.generative_models.inference.streaming",
    "temperature.md": "artifex.generative_models.inference.temperature",
    "tensorrt.md": "artifex.generative_models.inference.tensorrt",
    "tfjs.md": "artifex.generative_models.inference.tfjs",
    "throughput.md": "artifex.generative_models.inference.throughput",
    "top_k.md": "artifex.generative_models.inference.top_k",
    "vae_generator.md": "artifex.generative_models.inference.vae_generator",
}


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_inference_reference_pages_are_runtime_backed_or_coming_soon() -> None:
    """Each inference page should be either live or clearly marked as coming soon."""
    actual_pages = {path.name for path in DOCS_ROOT.glob("*.md") if path.name != "index.md"}
    expected_pages = set(RETAINED_RUNTIME_PAGES) | set(COMING_SOON_PAGES)

    assert actual_pages == expected_pages

    for page_name, expected in RETAINED_RUNTIME_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")

        assert "**Status:** `Supported runtime inference surface`" in contents
        assert expected["module"] in contents
        assert expected["source"] in contents

        for required_reference in expected["required"]:
            assert required_reference in contents

    for page_name, planned_module in COMING_SOON_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")

        assert "**Status:** `Coming soon`" in contents
        assert planned_module in contents
        assert "not shipped yet" in contents
        assert "artifex.generative_models.inference.optimization.production" in contents
        assert (
            "See [Inference Reference](index.md) for the current shared inference docs." in contents
        )


def test_inference_index_and_connected_docs_only_publish_live_shared_surface() -> None:
    """Connected docs should not advertise a phantom shared inference framework."""
    index_docs = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    generative_models_docs = (REPO_ROOT / "docs/generative_models/index.md").read_text(
        encoding="utf-8"
    )
    deployment_docs = (REPO_ROOT / "docs/user-guide/integrations/deployment.md").read_text(
        encoding="utf-8"
    )
    payload = _run_python(
        "import json; "
        "from pathlib import Path; "
        "import artifex.generative_models.inference as inference; "
        "import artifex.generative_models.inference.optimization as optimization; "
        "print(json.dumps({"
        "'inference_all': sorted(getattr(inference, '__all__', [])), "
        "'optimization_all': sorted(getattr(optimization, '__all__', [])), "
        "'production_exists': Path('src/artifex/generative_models/inference/optimization/production.py').exists()"
        "}))"
    )

    combined_docs = index_docs + generative_models_docs + deployment_docs

    for banned in [
        "from artifex.inference",
        "InferencePipeline",
        "VAEGenerator",
        "GANGenerator",
        "DiffusionGenerator",
        "FlowGenerator",
        "EnergyGenerator",
        "AutoregressiveGenerator",
        "TemperatureSampler",
        "TopKSampler",
        "save_model(",
        "load_model(",
    ]:
        assert banned not in combined_docs

    for required in [
        "artifex.generative_models.inference.optimization.production",
        "ProductionOptimizer",
        "family-owned generation entrypoints",
        "`artifex.generative_models.inference` exports no public helpers",
        "setup_checkpoint_manager",
        "save_checkpoint",
        "load_checkpoint",
        "application framework",
    ]:
        assert required in combined_docs

    assert payload["inference_all"] == []
    assert payload["optimization_all"] == []
    assert payload["production_exists"] is True


def test_inference_index_nav_and_mkdocs_split_supported_from_coming_soon() -> None:
    """The inference catalog should separate the real runtime page from roadmap-only pages."""
    index_contents = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    inference_reference_block = mkdocs_contents.split("      - Inference Reference:\n", 1)[1].split(
        "      - Modalities Reference:\n", 1
    )[0]
    current_block = inference_reference_block.split("        - Current Inference Pages:\n", 1)[
        1
    ].split("        - Coming Soon:\n", 1)[0]
    coming_soon_block = inference_reference_block.split("        - Coming Soon:\n", 1)[1]

    assert "**Status:** `Supported runtime inference surface`" in index_contents
    assert "## Current Inference Pages" in index_contents
    assert "## Coming Soon" in index_contents

    assert "inference/index.md" in current_block
    for page_name in RETAINED_RUNTIME_PAGES:
        assert f"inference/{page_name}" in current_block

    for page_name in COMING_SOON_PAGES:
        assert f"inference/{page_name}" not in current_block
        assert f"inference/{page_name}" in coming_soon_block
