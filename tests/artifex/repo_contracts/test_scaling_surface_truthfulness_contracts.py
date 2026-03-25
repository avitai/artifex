"""Repository contracts for the narrowed scaling surface."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_INDEX = REPO_ROOT / "docs/core/index.md"
SCALING_INDEX = REPO_ROOT / "docs/scaling/index.md"
SHARDING_DOC = REPO_ROOT / "docs/scaling/sharding.md"
VERIFY_PATH = REPO_ROOT / "examples/verify_examples.py"


def _run_python(code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _normalized_text(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_placeholder_core_scaling_surfaces_are_removed() -> None:
    """Core should not publish placeholder scaling modules once scaling lives elsewhere."""
    payload = _run_python(
        textwrap.dedent(
            """
            import importlib
            import json

            errors = {}
            for module_name in (
                'artifex.generative_models.core.adapters',
                'artifex.generative_models.core.parallelism',
                'artifex.generative_models.core.types',
            ):
                try:
                    importlib.import_module(module_name)
                except Exception as exc:
                    errors[module_name] = type(exc).__name__

            print(json.dumps(errors))
            """
        )
    )

    assert payload["artifex.generative_models.core.adapters"] == "ModuleNotFoundError"
    assert payload["artifex.generative_models.core.parallelism"] == "ModuleNotFoundError"
    assert payload["artifex.generative_models.core.types"] == "ModuleNotFoundError"

    for relative_path in (
        "src/artifex/generative_models/core/adapters.py",
        "src/artifex/generative_models/core/parallelism.py",
        "src/artifex/generative_models/core/types.py",
        "docs/core/adapters.md",
        "docs/core/parallelism.md",
        "docs/core/types.md",
    ):
        assert not (REPO_ROOT / relative_path).exists(), relative_path

    index_doc = _normalized_text(CORE_INDEX)
    verify_script = VERIFY_PATH.read_text(encoding="utf-8")

    assert "artifex.generative_models.scaling" in index_doc
    for banned in ("core.adapters", "core.parallelism", "core.types"):
        assert banned not in index_doc
    assert "create_transformer_adapter" not in verify_script
    assert "core.adapters" not in verify_script


def test_scaling_docs_and_runtime_drop_invalid_pipeline_partition_helper() -> None:
    """Scaling docs should stop publishing the invalid pipeline partition helper."""
    payload = _run_python(
        textwrap.dedent(
            """
            import json

            from artifex.generative_models.scaling.sharding import MultiDimensionalStrategy

            print(json.dumps({
                'has_create_partition_spec': hasattr(MultiDimensionalStrategy, 'create_partition_spec'),
            }))
            """
        )
    )

    assert payload["has_create_partition_spec"] is False

    scaling_index = _normalized_text(SCALING_INDEX)
    sharding_doc = _normalized_text(SHARDING_DOC)
    combined = "\n".join((scaling_index, sharding_doc))

    required_tokens = [
        "ShardingConfig",
        "ParallelismConfig",
        "MultiDimensionalStrategy",
        "mesh_axis_names",
    ]
    for token in required_tokens:
        assert token in combined

    banned_tokens = [
        "create_partition_spec",
        "stage_0",
        "stage_1",
        "complete sharding infrastructure",
        "complete multi-dimensional scaling infrastructure",
    ]
    for token in banned_tokens:
        assert token not in combined
