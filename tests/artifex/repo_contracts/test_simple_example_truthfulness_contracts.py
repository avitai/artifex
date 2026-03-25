from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

STANDALONE_SIMPLE_EXAMPLES = {
    "audio": {
        "page": "docs/examples/audio/simple-audio-generation.md",
        "script": "examples/generative_models/audio/simple_audio_generation.py",
        "run_command": "python examples/generative_models/audio/simple_audio_generation.py",
        "notebook_command": "jupyter lab examples/generative_models/audio/simple_audio_generation.ipynb",
        "nav_label": "Standalone Audio Generation",
    },
    "diffusion": {
        "page": "docs/examples/diffusion/simple-diffusion.md",
        "script": "examples/generative_models/diffusion/simple_diffusion_example.py",
        "run_command": "python examples/generative_models/diffusion/simple_diffusion_example.py",
        "notebook_command": "jupyter lab examples/generative_models/diffusion/simple_diffusion_example.ipynb",
        "nav_label": "Standalone Simple Diffusion",
    },
    "text": {
        "page": "docs/examples/text/simple-text-generation.md",
        "script": "examples/generative_models/text/simple_text_generation.py",
        "run_command": "python examples/generative_models/text/simple_text_generation.py",
        "notebook_command": "jupyter lab examples/generative_models/text/simple_text_generation.ipynb",
        "nav_label": "Standalone Text Generation",
    },
    "multimodal": {
        "page": "docs/examples/multimodal/simple-image-text.md",
        "script": "examples/generative_models/multimodal/simple_image_text.py",
        "run_command": "python examples/generative_models/multimodal/simple_image_text.py",
        "notebook_command": "jupyter lab examples/generative_models/multimodal/simple_image_text.ipynb",
        "nav_label": "Standalone Image-Text",
    },
}


def test_example_design_guide_defines_standalone_pedagogy_contract() -> None:
    """Standalone concept walkthroughs must be explicitly separated from runtime-backed examples."""
    guide = (REPO_ROOT / "docs/development/example-documentation-design.md").read_text(
        encoding="utf-8"
    )

    assert "Standalone pedagogy" in guide
    assert "does not instantiate shipped Artifex runtime owners" in guide
    assert "docs/examples/index.md" in guide
    assert "Standalone " in guide


def test_simple_example_pages_and_scripts_are_explicitly_labeled_standalone() -> None:
    """Simple-tier walkthroughs must not present raw NNX pedagogy as canonical runtime usage."""
    banned_phrases = {
        "using the Artifex framework",
        "Artifex framework's modality system",
        "pip install artifex",
    }

    for paths in STANDALONE_SIMPLE_EXAMPLES.values():
        page = (REPO_ROOT / paths["page"]).read_text(encoding="utf-8")
        script = (REPO_ROOT / paths["script"]).read_text(encoding="utf-8")

        assert "**Status:** `Standalone pedagogy`" in page
        assert "This walkthrough is a standalone JAX/Flax NNX concept demo." in page
        assert "It does not instantiate shipped Artifex runtime owners." in page
        assert "Standalone JAX/Flax NNX concept walkthrough." in script
        assert "This file does not instantiate shipped Artifex runtime owners." in script
        assert "source ./activate.sh" not in page
        assert paths["run_command"] in page
        assert paths["notebook_command"] in page

        for phrase in banned_phrases:
            assert phrase not in page
            assert phrase not in script


def test_simple_diffusion_walkthrough_has_no_phantom_neighbor_references() -> None:
    """The standalone diffusion walkthrough must point only to live docs and files."""
    page = (REPO_ROOT / STANDALONE_SIMPLE_EXAMPLES["diffusion"]["page"]).read_text(encoding="utf-8")
    script = (REPO_ROOT / STANDALONE_SIMPLE_EXAMPLES["diffusion"]["script"]).read_text(
        encoding="utf-8"
    )

    for missing_neighbor in ["ddpm_cifar10.py", "advanced_diffusion.py"]:
        assert missing_neighbor not in page
        assert missing_neighbor not in script

    assert "dit_demo.py" not in script
    assert "dit-demo.md" in page
    assert "../../user-guide/models/diffusion-guide.md" in page
    assert "../../api/models/diffusion.md" in page


def test_examples_catalog_and_nav_separate_standalone_simple_walkthroughs() -> None:
    """Catalog and nav labels should distinguish standalone walkthroughs from runtime-backed tutorials."""
    index_contents = (REPO_ROOT / "docs/examples/index.md").read_text(encoding="utf-8")
    normalized_index = " ".join(index_contents.split())
    standalone_section = index_contents.split("## Standalone Concept Walkthroughs\n", 1)[1].split(
        "\n## Reference Tables",
        1,
    )[0]
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    roadmap = (REPO_ROOT / "docs/roadmap/planned-examples.md").read_text(encoding="utf-8")

    assert "runtime-backed Artifex tutorials" in normalized_index
    assert "standalone concept walkthroughs" in normalized_index
    assert "Available now as standalone concept walkthroughs:" in roadmap

    for paths in STANDALONE_SIMPLE_EXAMPLES.values():
        section_page = paths["page"].removeprefix("docs/examples/")
        nav_page = paths["page"].removeprefix("docs/")
        assert section_page in standalone_section
        assert f"- {paths['nav_label']}: {nav_page}" in mkdocs_contents
