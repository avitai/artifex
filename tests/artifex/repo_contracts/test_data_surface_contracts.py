"""Repository contracts for the retained top-level data surface."""

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


def test_data_package_import_keeps_protein_lazy() -> None:
    """Importing the top-level data package should stay narrow and lazy."""
    payload = _run_python(
        "import json, sys; "
        "import artifex.data as data; "
        "print(json.dumps({"
        "'protein_loaded': 'artifex.data.protein' in sys.modules, "
        "'jax_loaded': 'jax' in sys.modules, "
        "'all': list(getattr(data, '__all__'))"
        "}))"
    )

    assert payload["protein_loaded"] is False
    assert payload["jax_loaded"] is False
    assert payload["all"] == ["protein"]


def test_data_package_resolves_only_retained_protein_surface() -> None:
    """The top-level data barrel should not regrow a phantom generic API."""
    payload = _run_python(
        "import json; "
        "import artifex.data as data; "
        "protein = data.protein; "
        "print(json.dumps({"
        "'protein_module': protein.__name__, "
        "'protein_dataset_module': protein.ProteinDataset.__module__, "
        "'protein_config_module': protein.ProteinDatasetConfig.__module__, "
        "'has_load_dataset': hasattr(data, 'load_dataset'), "
        "'has_image_dataset': hasattr(data, 'ImageDataset'), "
        "'has_data_pipeline': hasattr(data, 'DataPipeline'), "
        "'has_dynamic_batch_collator': hasattr(data, 'DynamicBatchCollator')"
        "}))"
    )

    assert payload["protein_module"] == "artifex.data.protein"
    assert payload["protein_dataset_module"] == "artifex.data.protein.dataset"
    assert payload["protein_config_module"] == "artifex.data.protein.dataset"
    assert payload["has_load_dataset"] is False
    assert payload["has_image_dataset"] is False
    assert payload["has_data_pipeline"] is False
    assert payload["has_dynamic_batch_collator"] is False


def test_data_docs_and_examples_use_current_protein_dataset_contract() -> None:
    """Docs/examples should describe the retained config-based protein data surface."""
    required_references = {
        "docs/data/index.md": [
            "top-level `artifex.data` package is intentionally narrow",
            "artifex.data.protein",
            "ProteinDatasetConfig",
            "datarax",
        ],
        "docs/data/dataset.md": [
            "artifex.data.protein.dataset",
            "ProteinDatasetConfig",
            "protein_collate_fn",
        ],
        "docs/data/protein_dataset.md": [
            "There is no `artifex.data.protein_dataset` module",
            "artifex.data.protein.dataset",
            "ProteinDatasetConfig",
        ],
        "docs/getting-started/core-concepts.md": [
            "ProteinDatasetConfig(",
            "ProteinDataset(",
        ],
        "docs/examples/protein/protein-point-cloud-example.md": [
            "from artifex.data.protein import ProteinDataset, ProteinDatasetConfig",
            "ProteinDatasetConfig(",
        ],
        "docs/examples/protein/protein-diffusion-example.md": [
            "from artifex.data.protein import ProteinDataset, ProteinDatasetConfig",
            "ProteinDatasetConfig(",
        ],
        "examples/generative_models/protein/protein_point_cloud_example.py": [
            "from artifex.data.protein import (",
            "ProteinDatasetConfig",
        ],
        "examples/generative_models/protein/protein_diffusion_example.py": [
            "from artifex.data.protein import (",
            "ProteinDatasetConfig",
        ],
    }
    banned_references = {
        "docs/data/index.md": [
            "from artifex.data import load_dataset",
            "from artifex.data import ImageDataset",
            "DynamicBatchCollator",
            "DistributedSampler",
            "WebDatasetLoader",
            "RemoteLoader",
            "artifex.data.image",
            "artifex.data.audio",
            "artifex.data.text",
            "artifex.data.video",
        ],
        "docs/data/dataset.md": [
            "**Module:** `data.protein.dataset`",
            "create_synthetic_data()",
            "def __init__()",
        ],
        "docs/data/protein_dataset.md": [
            "**Module:** `data.protein_dataset`",
            "**Source:** `data/protein_dataset.py`",
            "collate_batch()",
        ],
        "docs/getting-started/core-concepts.md": [
            "pdb_dir=",
            "with_constraints=True",
        ],
        "docs/examples/protein/protein-point-cloud-example.md": [
            "backbone_only=True",
            "artifex.data.protein.dataset",
        ],
        "examples/generative_models/protein/protein_point_cloud_example.py": [
            "backbone_only=True",
            "artifex.data.protein.dataset",
        ],
    }

    for relative_path, references in required_references.items():
        contents = (REPO_ROOT / relative_path).read_text()
        for reference in references:
            assert reference in contents
        for banned_reference in banned_references.get(relative_path, []):
            assert banned_reference not in contents
