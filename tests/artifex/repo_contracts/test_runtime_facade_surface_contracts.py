from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_factory_docs_only_advertise_concrete_gan_configs() -> None:
    """Factory docs should not teach base GANConfig as factory-ready."""
    readme = (REPO_ROOT / "src/artifex/generative_models/factory/README.md").read_text()
    docs = (REPO_ROOT / "docs/factory/index.md").read_text()

    banned_references = [
        "- `GANConfig`, `DCGANConfig`, `WGANConfig`, `LSGANConfig`",
        "| `GANConfig`, `DCGANConfig`, `WGANConfig`, `LSGANConfig` | `gan` | GAN variants |",
        "- `GANConfig` → `GAN`",
    ]
    required_references = [
        "Base `GANConfig` remains an abstract shared config parent",
        "Base `GANConfig` is not factory-ready and is rejected by `create_model(...)`.",
        "ConditionalGANConfig",
        "CycleGANConfig",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in readme
        assert banned_reference not in docs

    for required_reference in required_references:
        assert required_reference in (readme + docs)


def test_cli_docs_only_publish_config_top_level_command() -> None:
    """CLI docs should stay aligned with the config-only runtime catalog."""
    cli_index = (REPO_ROOT / "docs/cli/index.md").read_text()
    removed_command_pages = [
        REPO_ROOT / "docs/cli/train.md",
        REPO_ROOT / "docs/cli/generate.md",
        REPO_ROOT / "docs/cli/evaluate.md",
        REPO_ROOT / "docs/cli/serve.md",
        REPO_ROOT / "docs/cli/benchmark.md",
        REPO_ROOT / "docs/cli/convert.md",
    ]

    for banned_reference in (
        "artifex train",
        "artifex generate",
        "artifex evaluate",
        "artifex serve",
        "artifex benchmark",
        "artifex convert",
    ):
        assert banned_reference not in cli_index

    assert "artifex config validate" in cli_index
    assert "Configuration management commands" in cli_index

    for page in removed_command_pages:
        contents = page.read_text()
        assert "not currently shipped in the runtime CLI" in contents
        assert "cli.commands." not in contents


def test_production_docs_stay_on_experimental_jit_and_monitoring_surface() -> None:
    """Production inference docs should not republish placeholder infrastructure."""
    docs = (REPO_ROOT / "docs/inference/production.md").read_text()

    banned_references = [
        "Automatic optimization pipeline selection",
        "Model adapter classes for different architectures",
        "Complete monitoring and debugging tools",
        "def compiled_forward()",
    ]
    required_references = [
        "experimental production inference helpers",
        "`jit_compilation`",
        "`memory_usage_gb` and `cache_hit_rate` are unavailable",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in docs

    for required_reference in required_references:
        assert required_reference in docs


def test_utils_docs_only_publish_file_utils_top_level_surface() -> None:
    """Top-level utils docs should stay on the surviving file-utils contract."""
    docs = (REPO_ROOT / "docs/utils/index.md").read_text()

    banned_references = [
        "artifex.utils.jax",
        "artifex.utils.logging",
        "artifex.utils.visualization",
        "artifex.utils.io",
        "artifex.utils.profiling",
        "artifex.utils.image",
        "artifex.utils.numerical",
        "artifex.utils.text",
        "from artifex.utils import Timer",
        "from artifex.utils import Registry",
    ]
    required_references = [
        "artifex.utils.file_utils",
        "get_valid_output_dir",
        "Most other helpers now live with their owning package",
    ]

    for banned_reference in banned_references:
        assert banned_reference not in docs

    for required_reference in required_references:
        assert required_reference in docs


def test_visualization_docs_only_publish_protein_specific_surface() -> None:
    """Visualization docs should stay narrowed to the retained protein helper."""
    index = (REPO_ROOT / "docs/visualization/index.md").read_text()
    protein_doc = (REPO_ROOT / "docs/visualization/protein_viz.md").read_text()
    utils_doc = (REPO_ROOT / "docs/utils/protein.md").read_text()
    readme = (REPO_ROOT / "src/artifex/visualization/README.md").read_text()

    banned_references = [
        "artifex.utils.visualization",
        "create_image_grid",
        "plot_latent_tsne",
        "plot_training_curves",
        "plot_bond_distributions",
    ]
    required_references = [
        "artifex.visualization.protein_viz",
        "ProteinVisualizer",
        "protein visualization owner",
        "thin compatibility alias",
    ]

    combined_docs = index + protein_doc + utils_doc + readme

    for banned_reference in banned_references:
        assert banned_reference not in combined_docs

    for required_reference in required_references:
        assert required_reference in combined_docs
