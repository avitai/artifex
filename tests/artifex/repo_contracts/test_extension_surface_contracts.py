"""Repository contracts for the canonical extension surface."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_legacy_geometric_extension_shim_is_removed() -> None:
    """Protein extensions should live only under the canonical extension package."""
    assert not (
        REPO_ROOT / "src/artifex/generative_models/models/geometric/extensions/__init__.py"
    ).exists()

    geometric_init = (
        REPO_ROOT / "src/artifex/generative_models/models/geometric/__init__.py"
    ).read_text()
    banned_references = [
        "BondAngleExtension",
        "BondLengthExtension",
        "ProteinMixinExtension",
        "create_protein_extensions",
        "DeprecationWarning",
        "extensions.protein",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in geometric_init


def test_protein_extension_docs_and_examples_use_typed_bundle_surface() -> None:
    """Contributor-facing extension docs should describe the canonical typed bundle."""
    files_to_check = {
        "docs/configs/extensions.md": [
            "ProteinExtensionsConfig",
            "get_protein_extensions_config",
        ],
        "docs/extensions/index.md": [
            "ProteinExtensionsConfig",
            "create_protein_extensions",
            "get_extensions_registry",
        ],
        "src/artifex/generative_models/extensions/README.md": [
            "ProteinExtensionsConfig",
            "create_protein_extensions",
        ],
        "examples/generative_models/protein/protein_extensions_example.py": [
            "ProteinExtensionsConfig",
            "ProteinExtensionConfig",
            "ProteinMixinConfig",
        ],
        "examples/generative_models/protein/protein_extensions_with_config.py": [
            "ProteinExtensionsConfig",
            "get_protein_extensions_config",
        ],
        "examples/verify_examples.py": [
            "ProteinExtensionsConfig",
            "ProteinExtensionConfig",
            "ProteinMixinConfig",
        ],
    }

    banned_references = [
        "build_protein_extension_mapping",
        "normalize_loaded_protein_extension_mapping",
        "plain mapping",
        '"use_backbone_constraints":',
        'protein_config["use_backbone_constraints"]',
        "models.geometric.extensions",
    ]

    for relative_path, required_references in files_to_check.items():
        contents = (REPO_ROOT / relative_path).read_text()
        for required_reference in required_references:
            assert required_reference in contents
        for banned_reference in banned_references:
            assert banned_reference not in contents


def test_extension_docs_describe_current_extension_scope() -> None:
    """The extension docs should describe the real current surface, not a protein-only fiction."""
    docs_index = (REPO_ROOT / "docs/extensions/index.md").read_text()

    required_references = [
        "Protein Extensions",
        "ProteinBackboneConstraint",
        "ProteinDihedralConstraint",
        "ProteinMixinExtension",
        "ExtensionsRegistry",
        "chemical",
        "vision",
        "audio_processing",
        "nlp",
        "top-level `artifex.generative_models.extensions` barrel stays curated",
    ]
    banned_references = [
        "only concrete first-class extension surface",
        "future extension families",
    ]

    for required_reference in required_references:
        assert required_reference in docs_index
    for banned_reference in banned_references:
        assert banned_reference not in docs_index


def test_shared_extensions_readme_and_architecture_describe_registry_backed_scope() -> None:
    """Shared extension docs should admit the shipped non-protein families explicitly."""
    readme_contents = (REPO_ROOT / "src/artifex/generative_models/extensions/README.md").read_text()
    architecture_contents = (REPO_ROOT / "docs/architecture/extensions.md").read_text()
    package_contents = (
        REPO_ROOT / "src/artifex/generative_models/extensions/__init__.py"
    ).read_text()

    required_readme_references = [
        "protein extensions are the reference implementation",
        "chemical",
        "vision",
        "audio_processing",
        "nlp",
        "curated convenience",
        "get_extensions_registry()",
    ]
    required_architecture_references = [
        "protein, chemical, vision, NLP, and audio processing",
        "registry-backed",
        "curated top-level barrel",
    ]
    required_package_references = [
        "curated convenience exports",
        "registry-backed family subpackages",
    ]

    for required_reference in required_readme_references:
        assert required_reference in readme_contents
    for required_reference in required_architecture_references:
        assert required_reference in architecture_contents
    for required_reference in required_package_references:
        assert required_reference in package_contents
