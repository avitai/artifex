from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "modalities"

OWNER_PAGES: dict[str, dict[str, object]] = {
    "registry.md": {
        "status": "**Status:** `Supported modality registry owner`",
        "scope": "**Scope:** `Shared registry owner`",
        "module": "artifex.generative_models.modalities.registry",
        "source": "src/artifex/generative_models/modalities/registry.py",
        "top_level": [
            "MODALITY_REGISTRY",
            "register_modality",
            "get_modality",
            "list_modalities",
            "clear_modalities",
        ],
        "required": ["image", "molecular", "protein"],
        "banned": ["## Functions", "## Module Statistics"],
    },
    "base.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Timeseries helper owner`",
        "module": "artifex.generative_models.modalities.timeseries.base",
        "source": "src/artifex/generative_models/modalities/timeseries/base.py",
        "top_level": [
            "TimeseriesRepresentation",
            "DecompositionMethod",
            "TimeseriesModalityConfig",
            "TimeseriesModality",
        ],
        "class_methods": {
            "TimeseriesModality": [
                "get_extensions",
                "get_adapter",
                "preprocess",
                "postprocess",
                "validate_data",
            ],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "adapters.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Timeseries helper owner`",
        "module": "artifex.generative_models.modalities.timeseries.adapters",
        "source": "src/artifex/generative_models/modalities/timeseries/adapters.py",
        "top_level": [
            "TimeseriesAdapterConfig",
            "TimeseriesTransformerAdapter",
            "TimeseriesRNNAdapter",
            "TimeseriesDiffusionAdapter",
            "TimeseriesVAEAdapter",
            "get_timeseries_adapter",
        ],
        "class_methods": {
            "TimeseriesTransformerAdapter": ["create"],
            "TimeseriesRNNAdapter": ["create"],
            "TimeseriesDiffusionAdapter": ["create"],
            "TimeseriesVAEAdapter": ["create"],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "datasets.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Timeseries helper owner`",
        "module": "artifex.generative_models.modalities.timeseries.datasets",
        "source": "src/artifex/generative_models/modalities/timeseries/datasets.py",
        "top_level": [
            "generate_synthetic_timeseries",
            "create_synthetic_timeseries_dataset",
            "create_simple_timeseries_dataset",
        ],
        "banned": ["## Functions", "## Module Statistics"],
    },
    "evaluation.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Timeseries helper owner`",
        "module": "artifex.generative_models.modalities.timeseries.evaluation",
        "source": "src/artifex/generative_models/modalities/timeseries/evaluation.py",
        "top_level": ["TimeseriesEvaluationSuite", "compute_timeseries_metrics"],
        "class_methods": {
            "TimeseriesEvaluationSuite": ["evaluate_batch", "compute_metrics"],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "representations.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Timeseries helper owner`",
        "module": "artifex.generative_models.modalities.timeseries.representations",
        "source": "src/artifex/generative_models/modalities/timeseries/representations.py",
        "top_level": [
            "TimeseriesProcessor",
            "FourierProcessor",
            "MultiScaleProcessor",
            "TrendDecompositionProcessor",
        ],
        "class_methods": {
            "TimeseriesProcessor": ["process", "reverse"],
            "MultiScaleProcessor": ["reconstruct"],
            "TrendDecompositionProcessor": ["reconstruct"],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "modality.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Protein-specific owner`",
        "module": "artifex.generative_models.modalities.protein.modality",
        "source": "src/artifex/generative_models/modalities/protein/modality.py",
        "top_level": ["ProteinModality"],
        "class_methods": {"ProteinModality": ["get_extensions", "get_adapter"]},
        "banned": ["## Functions", "## Module Statistics"],
    },
    "config.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Protein-specific owner`",
        "module": "artifex.generative_models.modalities.protein.config",
        "source": "src/artifex/generative_models/modalities/protein/config.py",
        "top_level": ["register_protein_modality", "create_default_protein_config"],
        "banned": ["## Functions", "## Module Statistics"],
    },
    "losses.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Protein-specific owner`",
        "module": "artifex.generative_models.modalities.protein.losses",
        "source": "src/artifex/generative_models/modalities/protein/losses.py",
        "top_level": [
            "create_rmsd_loss",
            "create_backbone_loss",
            "create_dihedral_loss",
            "create_protein_structure_loss",
        ],
        "banned": ["## Functions", "## Module Statistics"],
    },
    "utils.md": {
        "status": "**Status:** `Supported family-scoped modality owner`",
        "scope": "**Scope:** `Protein-specific owner`",
        "module": "artifex.generative_models.modalities.protein.utils",
        "source": "src/artifex/generative_models/modalities/protein/utils.py",
        "top_level": ["get_protein_adapter"],
        "banned": ["## Functions", "## Module Statistics"],
    },
}


def test_modalities_reference_pages_are_truthful_family_scoped_owner_pages() -> None:
    """The modalities catalog should make family scoping explicit on every retained page."""
    actual_pages = {path.name for path in DOCS_ROOT.glob("*.md") if path.name != "index.md"}
    assert actual_pages == set(OWNER_PAGES)

    for page_name, expected in OWNER_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")
        assert expected["status"] in contents
        assert expected["scope"] in contents
        assert expected["module"] in contents
        assert expected["source"] in contents
        assert "## Top-Level Module Exports" in contents

        for symbol in expected["top_level"]:
            assert f"`{symbol}`" in contents or f"`{symbol}()`" in contents

        class_methods = expected.get("class_methods", {})
        if class_methods:
            assert "## Class APIs" in contents
            for class_name, methods in class_methods.items():
                assert f"### `{class_name}`" in contents
                for method in methods:
                    assert f"`{method}()`" in contents
        else:
            assert "## Class APIs" not in contents

        for required in expected.get("required", []):
            assert required in contents

        for banned in expected.get("banned", []):
            assert banned not in contents


def test_modalities_reference_pages_map_to_importable_symbols() -> None:
    """Every retained modalities page should map to a live importable owner."""
    for expected in OWNER_PAGES.values():
        module = importlib.import_module(str(expected["module"]))

        for symbol in expected["top_level"]:
            assert hasattr(module, symbol)

        for class_name, methods in expected.get("class_methods", {}).items():
            owner = getattr(module, class_name)
            for method in methods:
                assert hasattr(owner, method)


def test_modalities_index_and_overview_stop_presenting_family_pages_as_generic_reference() -> None:
    """The shared modality overview should route through explicit family scoping."""
    index_contents = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    overview_contents = (REPO_ROOT / "docs/generative_models/index.md").read_text(encoding="utf-8")
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    modalities_block = mkdocs_contents.split("      - Modalities Reference:\n", 1)[1].split(
        "      - Model Reference:\n", 1
    )[0]

    for page_name in sorted(OWNER_PAGES):
        assert f"modalities/{page_name}" in modalities_block
    assert "modalities/index.md" in modalities_block

    for required in [
        "## Registry-Backed Modalities",
        "## Family-Scoped Owner Pages",
        "### Timeseries Helper Owners",
        "### Protein Owner Pages",
        "The shared filenames in this catalog are not modality-generic.",
        "Image, text, audio, and multi-modal helper packages keep their own package-local docs.",
    ]:
        assert required in index_contents

    assert (
        "Use [Modalities Overview](../modalities/index.md) for the retained registry-backed "
        "surface and the owner pages below only for family-scoped helper details."
        in overview_contents
    )
    for required in [
        "[Registry Owner](../modalities/registry.md)",
        "[Timeseries Base](../modalities/base.md)",
        "[Timeseries Datasets](../modalities/datasets.md)",
        "[Protein Modality](../modalities/modality.md)",
        "[Protein Losses](../modalities/losses.md)",
    ]:
        assert required in overview_contents
