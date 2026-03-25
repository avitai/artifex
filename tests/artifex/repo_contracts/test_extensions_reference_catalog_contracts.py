from __future__ import annotations

import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "extensions"

OWNER_PAGES: dict[str, dict[str, object]] = {
    "extensions.md": {
        "module": "artifex.generative_models.extensions.base.extensions",
        "source": "src/artifex/generative_models/extensions/base/extensions.py",
        "top_level": [
            "Extension",
            "ModelExtension",
            "ConstraintExtension",
            "AugmentationExtension",
            "SamplingExtension",
            "LossExtension",
            "EvaluationExtension",
            "CallbackExtension",
            "ModalityExtension",
            "ExtensionDict",
        ],
        "class_methods": {
            "Extension": ["is_enabled"],
            "ModelExtension": ["loss_fn"],
            "ConstraintExtension": ["validate", "project"],
            "AugmentationExtension": ["augment"],
            "SamplingExtension": ["modify_score", "filter_samples"],
            "LossExtension": ["compute_loss", "get_weight_at_step"],
            "EvaluationExtension": ["compute_metrics"],
            "CallbackExtension": ["on_train_begin", "on_epoch_end", "on_batch_end"],
            "ModalityExtension": ["preprocess", "postprocess", "get_input_spec"],
            "ExtensionDict": ["__contains__"],
        },
        "required": [
            "Imported config types remain owned by `artifex.generative_models.core.configuration`.",
        ],
        "banned": ["class ExtensionConfig", "## Functions", "## Module Statistics"],
    },
    "registry.md": {
        "module": "artifex.generative_models.extensions.registry",
        "source": "src/artifex/generative_models/extensions/registry.py",
        "top_level": ["ExtensionType", "ExtensionsRegistry", "get_extensions_registry"],
        "class_methods": {
            "ExtensionsRegistry": [
                "register_extension",
                "get_extensions_for_modality",
                "get_extensions_by_capability",
                "get_extensions_by_type",
                "create_extension",
                "list_all_extensions",
                "get_extension_info",
                "validate_extension_compatibility",
                "create_extension_pipeline",
                "search_extensions",
                "get_available_modalities",
                "get_available_capabilities",
                "get_available_extension_types",
            ],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "augmentation.md": {
        "module": "artifex.generative_models.extensions.vision.augmentation",
        "source": "src/artifex/generative_models/extensions/vision/augmentation.py",
        "top_level": ["AdvancedImageAugmentation"],
        "class_methods": {
            "AdvancedImageAugmentation": [
                "augment",
                "apply_horizontal_flip",
                "apply_vertical_flip",
                "apply_cutout",
                "create_augmentation_sequence",
            ],
        },
        "banned": ["def __call__()", "## Functions", "## Module Statistics"],
    },
    "backbone.md": {
        "module": "artifex.generative_models.extensions.protein.backbone",
        "source": "src/artifex/generative_models/extensions/protein/backbone.py",
        "top_level": ["BondLengthExtension", "BondAngleExtension"],
        "class_methods": {
            "BondLengthExtension": ["loss_fn"],
            "BondAngleExtension": ["loss_fn"],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "constraints.md": {
        "module": "artifex.generative_models.extensions.protein.constraints",
        "source": "src/artifex/generative_models/extensions/protein/constraints.py",
        "top_level": [
            "ProteinBackboneConstraint",
            "ProteinDihedralConstraint",
            "calculate_bond_lengths",
            "calculate_bond_angles",
            "calculate_dihedral_angles",
        ],
        "class_methods": {
            "ProteinBackboneConstraint": ["loss_fn", "validate"],
            "ProteinDihedralConstraint": ["loss_fn", "validate"],
        },
        "banned": [
            "def __call__()",
            "## Functions",
            "## Module Statistics",
            "_extract_protein_coordinates",
        ],
    },
    "embeddings.md": {
        "module": "artifex.generative_models.extensions.nlp.embeddings",
        "source": "src/artifex/generative_models/extensions/nlp/embeddings.py",
        "top_level": [
            "precompute_rope_freqs",
            "apply_rope",
            "create_sinusoidal_positions",
            "TextEmbeddings",
        ],
        "class_methods": {
            "TextEmbeddings": [
                "embed",
                "get_token_embeddings",
                "compute_similarity",
                "create_contextual_embeddings",
                "project_to_vocabulary",
                "extract_sentence_embedding",
                "compute_attention_weights",
                "interpolate_embeddings",
                "apply_rope_embeddings",
                "get_sinusoidal_embeddings",
                "embed_with_sinusoidal_positions",
                "embed_with_rope",
                "get_embedding_statistics",
            ],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "features.md": {
        "module": "artifex.generative_models.extensions.chemical.features",
        "source": "src/artifex/generative_models/extensions/chemical/features.py",
        "top_level": ["MolecularFeatures"],
        "class_methods": {
            "MolecularFeatures": [
                "compute_descriptors",
                "compute_drug_likeness_score",
                "extract_fingerprint",
            ],
        },
        "banned": ["## Functions", "## Module Statistics"],
    },
    "mixin.md": {
        "module": "artifex.generative_models.extensions.protein.mixin",
        "source": "src/artifex/generative_models/extensions/protein/mixin.py",
        "top_level": ["ProteinMixinExtension"],
        "class_methods": {"ProteinMixinExtension": ["__call__"]},
        "banned": ["def __call__()", "## Functions", "## Module Statistics"],
    },
    "spectral.md": {
        "module": "artifex.generative_models.extensions.audio_processing.spectral",
        "source": "src/artifex/generative_models/extensions/audio_processing/spectral.py",
        "top_level": ["SpectralAnalysis"],
        "class_methods": {
            "SpectralAnalysis": [
                "compute_stft",
                "compute_spectrogram",
                "compute_mel_spectrogram",
                "compute_log_mel_spectrogram",
                "compute_mfcc",
                "compute_spectral_centroid",
                "compute_spectral_bandwidth",
                "compute_spectral_rolloff",
                "inverse_mel_spectrogram",
                "extract_spectral_features",
            ],
        },
        "banned": [
            "hz_to_mel",
            "mel_to_hz",
            "## Functions",
            "## Module Statistics",
        ],
    },
    "temporal.md": {
        "module": "artifex.generative_models.extensions.audio_processing.temporal",
        "source": "src/artifex/generative_models/extensions/audio_processing/temporal.py",
        "top_level": ["TemporalAnalysis"],
        "class_methods": {
            "TemporalAnalysis": [
                "compute_zero_crossing_rate",
                "compute_energy",
                "compute_rms",
                "estimate_tempo",
                "compute_onset_strength",
                "detect_beats",
                "compute_rhythm_features",
                "compute_temporal_features",
                "compute_pulse_clarity",
                "segment_by_energy",
            ],
        },
        "banned": ["def __call__()", "## Functions", "## Module Statistics"],
    },
    "tokenization.md": {
        "module": "artifex.generative_models.extensions.nlp.tokenization",
        "source": "src/artifex/generative_models/extensions/nlp/tokenization.py",
        "top_level": ["AdvancedTokenization"],
        "class_methods": {
            "AdvancedTokenization": [
                "tokenize",
                "detokenize",
                "encode_batch",
                "decode_batch",
                "create_attention_mask",
                "add_special_tokens",
                "compute_token_frequencies",
                "apply_masking",
                "create_position_ids",
                "truncate_sequences",
                "get_vocabulary_info",
            ],
        },
        "banned": ["def __call__()", "## Functions", "## Module Statistics"],
    },
}


def test_extensions_reference_pages_are_truthful_owner_pages() -> None:
    """Each retained extensions page should name a real owner and truthful API sections."""
    actual_pages = {
        path.name for path in DOCS_ROOT.glob("*.md") if path.name not in {"index.md", "utils.md"}
    }
    assert actual_pages == set(OWNER_PAGES)

    for page_name, expected in OWNER_PAGES.items():
        contents = (DOCS_ROOT / page_name).read_text(encoding="utf-8")
        assert "**Status:** `Supported runtime extension owner`" in contents
        assert expected["module"] in contents
        assert expected["source"] in contents
        assert "## Top-Level Module Exports" in contents
        assert "## Class APIs" in contents

        for symbol in expected["top_level"]:
            assert f"`{symbol}`" in contents or f"`{symbol}()`" in contents

        for class_name, methods in expected["class_methods"].items():
            assert f"### `{class_name}`" in contents
            for method in methods:
                assert f"`{method}()`" in contents

        for required in expected.get("required", []):
            assert required in contents

        for banned in expected.get("banned", []):
            assert banned not in contents


def test_extensions_reference_pages_map_to_importable_symbols() -> None:
    """The retained extension owner pages should map to live importable module symbols."""
    for expected in OWNER_PAGES.values():
        module = importlib.import_module(str(expected["module"]))

        for symbol in expected["top_level"]:
            assert hasattr(module, symbol)

        for class_name, methods in expected["class_methods"].items():
            owner = getattr(module, class_name)
            for method in methods:
                assert hasattr(owner, method)


def test_extensions_nav_and_overview_present_owner_pages_as_detail() -> None:
    """The shared overview and nav should route readers through the curated entrypoint."""
    overview_contents = (REPO_ROOT / "docs/generative_models/index.md").read_text(encoding="utf-8")
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    extensions_block = mkdocs_contents.split("      - Extensions Owner Pages:\n", 1)[1].split(
        "      - Factory Reference:\n", 1
    )[0]

    for page_name in sorted(OWNER_PAGES):
        assert f"extensions/{page_name}" in extensions_block

    assert "extensions/utils.md" in extensions_block
    assert "generated extension pages" not in overview_contents
    assert (
        "Use [Extensions Overview](../extensions/index.md) for the curated scope and "
        "the owner pages below for live module details." in overview_contents
    )
    for required in [
        "[Base Extensions](../extensions/extensions.md)",
        "[Registry Owner](../extensions/registry.md)",
        "[Protein Constraints](../extensions/constraints.md)",
        "[NLP Embeddings](../extensions/embeddings.md)",
        "[Audio Analysis](../extensions/temporal.md)",
    ]:
        assert required in overview_contents
