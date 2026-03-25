from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "utils"

RETAINED_RUNTIME_PAGES: dict[str, dict[str, object]] = {
    "dependency_analyzer.md": {
        "module": "artifex.generative_models.utils.code_analysis.dependency_analyzer",
        "source": "src/artifex/generative_models/utils/code_analysis/dependency_analyzer.py",
        "required": ["DependencyAnalyzer", "detect_circular_dependencies"],
    },
    "device.md": {
        "module": "artifex.generative_models.utils.jax.device",
        "source": "src/artifex/generative_models/utils/jax/device.py",
        "required": ["verify_device_setup", "get_recommended_batch_size"],
    },
    "file_utils.md": {
        "module": "artifex.utils.file_utils",
        "source": "src/artifex/utils/file_utils.py",
        "required": ["ensure_valid_output_path", "get_valid_output_dir"],
    },
    "logger.md": {
        "module": "artifex.generative_models.utils.logging.logger",
        "source": "src/artifex/generative_models/utils/logging/logger.py",
        "required": ["Logger", "create_logger"],
    },
    "metrics.md": {
        "module": "artifex.generative_models.utils.logging.metrics",
        "source": "src/artifex/generative_models/utils/logging/metrics.py",
        "required": ["MetricsLogger", "log_distribution_metrics"],
    },
    "mlflow.md": {
        "module": "artifex.generative_models.utils.logging.mlflow",
        "source": "src/artifex/generative_models/utils/logging/mlflow.py",
        "required": ["MLFlowLogger", "log_scalars"],
    },
    "protein.md": {
        "module": "artifex.visualization.protein_viz",
        "source": "src/artifex/visualization/protein_viz.py",
        "required": [
            "ProteinVisualizer",
            "thin compatibility alias",
            "artifex.generative_models.utils.visualization.protein",
        ],
    },
    "wandb.md": {
        "module": "artifex.generative_models.utils.logging.wandb",
        "source": "src/artifex/generative_models/utils/logging/wandb.py",
        "required": ["WandbLogger", "log_scalars"],
    },
}

COMING_SOON_PAGES = {
    "attention_vis.md": "artifex.generative_models.utils.visualization.attention_vis",
    "color.md": "artifex.generative_models.utils.image.color",
    "dtype.md": "artifex.generative_models.utils.jax.dtype",
    "env.md": "artifex.generative_models.utils.misc.env",
    "file.md": "artifex.generative_models.utils.io.file",
    "flax_utils.md": "artifex.generative_models.utils.jax.flax_utils",
    "formats.md": "artifex.generative_models.utils.io.formats",
    "image_grid.md": "artifex.generative_models.utils.visualization.image_grid",
    "latent_space.md": "artifex.generative_models.utils.visualization.latent_space",
    "math.md": "artifex.generative_models.utils.numerical.math",
    "memory.md": "artifex.generative_models.utils.profiling.memory",
    "performance.md": "artifex.generative_models.utils.profiling.performance",
    "plotting.md": "artifex.generative_models.utils.visualization.plotting",
    "postprocessing.md": "artifex.generative_models.utils.text.postprocessing",
    "prng.md": "artifex.generative_models.utils.jax.prng",
    "processing.md": "artifex.generative_models.utils.text.processing",
    "registry.md": "artifex.generative_models.utils.misc.registry",
    "serialization.md": "artifex.generative_models.utils.io.serialization",
    "shapes.md": "artifex.generative_models.utils.jax.shapes",
    "stability.md": "artifex.generative_models.utils.numerical.stability",
    "stats.md": "artifex.generative_models.utils.numerical.stats",
    "timer.md": "artifex.generative_models.utils.misc.timer",
    "transforms.md": "artifex.generative_models.utils.image.transforms",
    "types.md": "artifex.generative_models.utils.misc.types",
    "xprof.md": "artifex.generative_models.utils.profiling.xprof",
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


def test_utils_reference_pages_are_runtime_backed_or_coming_soon() -> None:
    """Each utils page should be either live or clearly marked as coming soon."""
    actual_pages = {path.name for path in DOCS_ROOT.glob("*.md") if path.name != "index.md"}
    expected_pages = set(RETAINED_RUNTIME_PAGES) | set(COMING_SOON_PAGES)

    assert actual_pages == expected_pages

    for page_name, expected in RETAINED_RUNTIME_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text()

        assert "**Status:** `Supported runtime utility`" in contents
        assert expected["module"] in contents
        assert expected["source"] in contents

        for required_reference in expected["required"]:
            assert required_reference in contents

    for page_name, planned_module in COMING_SOON_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text()

        assert "**Status:** `Coming soon`" in contents
        assert planned_module in contents
        assert "not shipped yet" in contents
        assert "See [Utilities](index.md) for the current utility docs." in contents
        assert "../roadmap/planned-modules.md#utilities" in contents


def test_utils_reference_pages_match_live_imports() -> None:
    """Retained utility reference pages should only point at importable modules."""
    payload = _run_python(
        "import importlib, json; "
        "modules = ["
        "'artifex.generative_models.utils.code_analysis.dependency_analyzer',"
        "'artifex.generative_models.utils.jax.device',"
        "'artifex.generative_models.utils.logging.logger',"
        "'artifex.generative_models.utils.logging.metrics',"
        "'artifex.generative_models.utils.logging.mlflow',"
        "'artifex.generative_models.utils.logging.wandb',"
        "'artifex.utils.file_utils',"
        "'artifex.visualization.protein_viz',"
        "'artifex.generative_models.utils.visualization.protein'"
        "]; "
        "loaded = {name: importlib.import_module(name).__name__ for name in modules}; "
        "protein_owner = importlib.import_module('artifex.visualization.protein_viz').ProteinVisualizer; "
        "protein_alias = importlib.import_module('artifex.generative_models.utils.visualization.protein').ProteinVisualizer; "
        "print(json.dumps({'loaded': loaded, 'alias_matches_owner': protein_alias is protein_owner}))"
    )

    assert payload["alias_matches_owner"] is True
    loaded = payload["loaded"]
    for expected in RETAINED_RUNTIME_PAGES.values():
        module = expected["module"]
        assert loaded[module] == module


def test_utils_index_nav_and_roadmap_frame_planned_families_as_coming_soon() -> None:
    """Relevant planned utility families should be framed as coming soon, not supported."""
    index_contents = (DOCS_ROOT / "index.md").read_text()
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text()
    roadmap_contents = (REPO_ROOT / "docs/roadmap/planned-modules.md").read_text()

    assert "Current Utility Pages" in index_contents
    assert "## Coming Soon" in index_contents
    assert "coming-soon utility families" in index_contents
    assert "coming soon" in roadmap_contents

    for page_name in RETAINED_RUNTIME_PAGES:
        assert page_name in index_contents
        assert f"utils/{page_name}" in mkdocs_contents

    for page_name in COMING_SOON_PAGES:
        assert f"utils/{page_name}" in mkdocs_contents

    assert "- Current Utility Pages:" in mkdocs_contents
    assert "- Coming Soon:" in mkdocs_contents
