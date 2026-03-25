"""Repository contracts for the retained modality and extension runtime surfaces."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_factory_docs_only_advertise_factory_ready_modalities() -> None:
    """The factory docs should only advertise retained registry-backed modalities."""
    contents = (REPO_ROOT / "docs/factory/index.md").read_text()

    banned_references = [
        'modality="text"',
        'modality="audio"',
        "- `text`:",
        "- `audio`:",
        "- `geometric`:",
    ]
    required_references = [
        'modality="image"',
        'modality="protein"',
        'modality="molecular"',
    ]

    for banned_reference in banned_references:
        assert banned_reference not in contents
    for required_reference in required_references:
        assert required_reference in contents


def test_modality_index_registry_section_matches_retained_registry() -> None:
    """The modality index should not publish the stale pre-audit registry snapshot."""
    contents = (REPO_ROOT / "docs/modalities/index.md").read_text()

    assert (
        "['image', 'text', 'audio', 'protein', 'multimodal', 'tabular', 'timeseries', 'molecular']"
        not in contents
    )
    assert "['image', 'molecular', 'protein']" in contents


def test_image_helper_docs_stay_on_retained_metric_and_augmentation_surface() -> None:
    """Image helper docs should stay on the lightweight modality-local surface."""
    guide_contents = (REPO_ROOT / "docs/user-guide/modalities/image.md").read_text()
    package_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/image/__init__.py"
    ).read_text()

    banned_guide_references = [
        "random_vertical_flip(",
        "random_rotation(",
        "random_contrast(",
        "random_saturation(",
        "random_hue(",
        "add_gaussian_noise(",
        "add_salt_pepper_noise(",
        "add_speckle_noise(",
        "random_crop(",
        "random_zoom(",
        "FID, IS, LPIPS",
    ]
    required_guide_references = [
        "AugmentationProcessor",
        "compute_image_metrics",
        "artifex.benchmarks.metrics.image",
        "horizontal flip",
        "brightness jitter",
    ]

    for banned_reference in banned_guide_references:
        assert banned_reference not in guide_contents
    for required_reference in required_guide_references:
        assert required_reference in guide_contents

    assert "Complete image evaluation metrics (FID, IS, LPIPS)" not in package_contents
    assert "benchmark framework" not in package_contents


def test_multi_modal_helper_docs_stay_off_shared_registry_story() -> None:
    """Multi-modal docs should teach the helper package as non-registry experimental scope."""
    index_contents = (REPO_ROOT / "docs/modalities/index.md").read_text()
    guide_contents = (REPO_ROOT / "docs/user-guide/modalities/multimodal.md").read_text()
    package_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/multi_modal/__init__.py"
    ).read_text()

    banned_index_references = [
        "MultiModalModality",
        "MultiModalEvaluator",
        'get_modality("multi_modal")',
        'modality="multi_modal"',
    ]
    required_guide_references = [
        "image, text, and audio",
        "not part of the shared modality registry",
        "get_modality(...)",
    ]

    for banned_reference in banned_index_references:
        assert banned_reference not in index_contents
    for required_reference in required_guide_references:
        assert required_reference in guide_contents

    assert "experimental helper" in package_contents
    assert "not registry-backed" in package_contents


def test_tabular_helper_surface_stays_on_typed_config_and_narrow_metric_story() -> None:
    """Tabular docs/package text should not drift back toward the removed quick-start story."""
    index_contents = (REPO_ROOT / "docs/modalities/index.md").read_text()
    package_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/tabular/__init__.py"
    ).read_text()
    evaluation_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/tabular/evaluation.py"
    ).read_text()

    banned_index_references = [
        "TabularEvaluator",
        "categorical_columns=",
        "continuous_columns=",
        "TabularModality(",
    ]
    required_package_references = [
        "TabularModalityConfig",
        "numerical KS",
        "privacy metrics",
    ]

    for banned_reference in banned_index_references:
        assert banned_reference not in index_contents
    for required_reference in required_package_references:
        assert required_reference in package_contents

    assert "categorical and ordinal helpers remain private" in evaluation_contents


def test_text_helper_surface_stays_on_processing_and_real_exports() -> None:
    """Text docs should stay on the retained helper surface and real exports."""
    index_contents = (REPO_ROOT / "docs/modalities/index.md").read_text()
    guide_contents = (REPO_ROOT / "docs/user-guide/modalities/text.md").read_text()
    package_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/text/__init__.py"
    ).read_text()

    banned_index_references = [
        "TextEvaluator",
        "TextModality(vocab_size=",
        "TextRepresentation(model=",
    ]
    banned_package_references = [
        "complete text generation capabilities",
        "modality.generate(",
        "Integration with benchmark framework",
    ]
    required_guide_references = [
        "not a standalone text generator",
        "`TextEvaluationSuite`",
        "`TextProcessor`",
        "`TokenizationProcessor`",
    ]
    required_package_references = [
        "default-config",
        "Standalone text generation",
        "TextGenerationProtocol",
        "`TextRepresentation` is an enum",
    ]

    for banned_reference in banned_index_references:
        assert banned_reference not in index_contents
    for banned_reference in banned_package_references:
        assert banned_reference not in package_contents
    for required_reference in required_guide_references:
        assert required_reference in guide_contents
    for required_reference in required_package_references:
        assert required_reference in package_contents


def test_timeseries_helper_surface_stays_on_typed_names() -> None:
    """Timeseries docs/package text should stay on the retained typed helper surface."""
    index_contents = (REPO_ROOT / "docs/modalities/index.md").read_text()
    package_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/timeseries/__init__.py"
    ).read_text()
    base_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/timeseries/base.py"
    ).read_text()

    banned_index_references = [
        "TimeSeriesModality",
        "TimeSeriesEvaluator",
        "TimeSeriesModality(seq_length=",
    ]
    required_package_references = [
        "TimeseriesModalityConfig",
        "aliases are not part of the supported surface",
        "TimeseriesEvaluationSuite",
        "compute_timeseries_metrics",
    ]
    required_base_references = [
        "TimeseriesModalityConfig",
        "seq_length=",
        "TimeSeries*",
    ]

    for banned_reference in banned_index_references:
        assert banned_reference not in index_contents
    for required_reference in required_package_references:
        assert required_reference in package_contents
    for required_reference in required_base_references:
        assert required_reference in base_contents


def test_protein_modality_docs_teach_generic_factory_boundary() -> None:
    """Protein modality docs should describe the retained generic-model boundary truthfully."""
    shared_contents = (REPO_ROOT / "src/artifex/generative_models/modalities/README.md").read_text()
    package_contents = (
        REPO_ROOT / "src/artifex/generative_models/modalities/protein/__init__.py"
    ).read_text()
    example_contents = (
        REPO_ROOT / "docs/examples/protein/protein-model-with-modality.md"
    ).read_text()
    example_source = (
        REPO_ROOT / "examples/generative_models/protein/protein_model_with_modality.py"
    ).read_text()

    banned_references = [
        "automatic modality-specific enhancements",
        "Automatically applies protein-specific enhancements",
        "Applies protein-specific enhancements",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in example_contents
        assert banned_reference not in example_source

    assert "generic model family selected by the typed config" in shared_contents
    assert "does not swap in `ProteinPointCloudModel` or `ProteinGraphModel`" in shared_contents
    assert "generic model family selected by the typed config" in package_contents
    assert "typed protein extension bundle" in example_contents
    assert "typed protein extension bundle" in example_source


def test_protein_geometric_examples_use_extension_bundle_contract() -> None:
    """Protein geometric examples should use ProteinExtensionsConfig, not a private shim."""
    example_paths = [
        "examples/generative_models/protein/protein_point_cloud_example.py",
        "examples/generative_models/protein/protein_point_cloud_example.ipynb",
        "examples/generative_models/protein/protein_diffusion_example.py",
        "examples/generative_models/protein/protein_diffusion_example.ipynb",
        "docs/examples/protein/protein-point-cloud-example.md",
        "docs/examples/protein/protein-diffusion-example.md",
    ]
    banned_references = [
        "ProteinConstraintConfig",
        "constraint_config=",
        "use_constraints=",
    ]
    required_references = [
        "ProteinExtensionsConfig",
        "ProteinExtensionConfig",
        "ProteinDihedralConfig",
        "extensions=",
    ]

    for relative_path in example_paths:
        contents = (REPO_ROOT / relative_path).read_text()
        for banned_reference in banned_references:
            assert banned_reference not in contents, (
                f"{relative_path} still contains {banned_reference!r}"
            )
        for required_reference in required_references:
            assert required_reference in contents, (
                f"{relative_path} is missing {required_reference!r}"
            )


def test_molecular_constraint_docs_use_typed_extension_config() -> None:
    """The molecular example surface should teach the typed chemical constraint path."""
    contents = (REPO_ROOT / "docs/examples/protein/protein-ligand-benchmark-demo.md").read_text()

    banned_references = [
        '"use_chemical_constraints": True',
        '"use_chemical_constraints": False',
        "'use_chemical_constraints': True",
        "'use_chemical_constraints': False",
        'metadata={"use_chemical_constraints"',
        "metadata={'use_chemical_constraints'",
    ]
    required_references = [
        "ChemicalConstraintConfig",
        "extensions={",
        '"chemical": ChemicalConstraintConfig',
    ]

    for banned_reference in banned_references:
        assert banned_reference not in contents
    for required_reference in required_references:
        assert required_reference in contents
