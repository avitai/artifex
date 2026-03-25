from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "docs" / "examples"

RUNTIME_BACKED_TUTORIAL_PAGES = {
    "advanced/advanced-training.md": "examples/generative_models/advanced_training_example",
    "advanced/advanced-vae.md": "examples/generative_models/image/vae/advanced_vae",
    "basic/diffusion-mnist-demo.md": "examples/generative_models/image/diffusion/diffusion_mnist",
    "basic/diffusion-mnist.md": "examples/generative_models/image/diffusion/diffusion_mnist_training",
    "basic/flow-mnist.md": "examples/generative_models/image/flow/flow_mnist",
    "basic/simple-gan.md": "examples/generative_models/image/gan/simple_gan",
    "basic/vae-mnist.md": "examples/generative_models/image/vae/vae_mnist",
    "diffusion/dit-demo.md": "examples/generative_models/diffusion/dit_demo",
    "energy/simple-ebm.md": "examples/generative_models/energy/simple_ebm_example",
    "framework/framework-features-demo.md": "examples/generative_models/framework_features_demo",
    "geometric/geometric-benchmark-demo.md": "examples/generative_models/geometric/geometric_benchmark_demo",
    "geometric/geometric-losses-demo.md": "examples/generative_models/geometric/geometric_losses_demo",
    "geometric/geometric-models-demo.md": "examples/generative_models/geometric/geometric_models_demo",
    "geometric/simple-point-cloud-example.md": "examples/generative_models/geometric/simple_point_cloud_example",
    "losses/loss-examples.md": "examples/generative_models/loss_examples",
    "protein/protein-extensions-example.md": "examples/generative_models/protein/protein_extensions_example",
    "protein/protein-extensions-with-config.md": "examples/generative_models/protein/protein_extensions_with_config",
    "protein/protein-ligand-benchmark-demo.md": "examples/generative_models/protein/protein_ligand_benchmark_demo",
    "protein/protein-model-extension.md": "examples/generative_models/protein/protein_model_extension",
    "protein/protein-model-with-modality.md": "examples/generative_models/protein/protein_model_with_modality",
    "protein/protein-point-cloud-example.md": "examples/generative_models/protein/protein_point_cloud_example",
    "sampling/blackjax-example.md": "examples/generative_models/sampling/blackjax_example",
    "sampling/blackjax-integration-examples.md": "examples/generative_models/sampling/blackjax_integration_examples",
    "sampling/blackjax-sampling-examples.md": "examples/generative_models/sampling/blackjax_sampling_examples",
    "vae/multi-beta-vae-benchmark-demo.md": "examples/generative_models/vae/multi_beta_vae_benchmark_demo",
}

STANDALONE_TUTORIAL_PAGES = {
    "audio/simple-audio-generation.md": "examples/generative_models/audio/simple_audio_generation",
    "diffusion/simple-diffusion.md": "examples/generative_models/diffusion/simple_diffusion_example",
    "multimodal/simple-image-text.md": "examples/generative_models/multimodal/simple_image_text",
    "text/simple-text-generation.md": "examples/generative_models/text/simple_text_generation",
}

EXPLORATORY_TUTORIAL_PAGES = {
    "advanced/advanced-gan.md": "examples/generative_models/image/gan/advanced_gan",
    "protein/protein-diffusion-example.md": "examples/generative_models/protein/protein_diffusion_example",
}

VALIDATION_TUTORIAL_PAGES = {
    "protein/protein-diffusion-tech-validation.md": (
        "examples/generative_models/protein/protein_diffusion_tech_validation"
    ),
}

PUBLISHED_EXAMPLE_PAGES = (
    RUNTIME_BACKED_TUTORIAL_PAGES
    | STANDALONE_TUTORIAL_PAGES
    | EXPLORATORY_TUTORIAL_PAGES
    | VALIDATION_TUTORIAL_PAGES
)

ROADMAP_ONLY_PAGES = {
    "advanced/advanced-ar.md",
    "advanced/advanced-diffusion.md",
    "advanced/advanced-flow.md",
    "advanced/clip-models.md",
    "advanced/cross-modal-retrieval.md",
    "advanced/image-captioning.md",
    "advanced/multimodal.md",
    "advanced/seq2seq.md",
    "advanced/text-compression.md",
    "advanced/visual-qa.md",
    "basic/ar-text.md",
    "basic/ebm-mnist.md",
    "basic/transformer-text.md",
    "diffusion/advanced-diffusion.md",
    "framework/model-deployment.md",
    "framework/training-strategies.md",
}


def test_example_design_guide_relegates_roadmap_topics_outside_docs_examples() -> None:
    """Canonical example docs must map to runnable triplets, not roadmap placeholders."""
    guide = (REPO_ROOT / "docs/development/example-documentation-design.md").read_text(
        encoding="utf-8"
    )

    assert "Roadmap-only topics belong outside" in guide
    assert "docs/examples" in guide
    assert "docs/roadmap/planned-examples.md" in guide


def test_examples_catalog_only_ships_triplet_backed_pages() -> None:
    """Every published docs/examples page should map to a real source pair."""
    actual_pages = {
        path.relative_to(DOCS_ROOT).as_posix()
        for path in DOCS_ROOT.rglob("*.md")
        if path.name not in {"index.md", "overview.md"} and "templates" not in path.parts
    }

    assert actual_pages == set(PUBLISHED_EXAMPLE_PAGES)

    for page, example_stem in PUBLISHED_EXAMPLE_PAGES.items():
        assert (DOCS_ROOT / page).exists()
        assert (REPO_ROOT / f"{example_stem}.py").exists()
        assert (REPO_ROOT / f"{example_stem}.ipynb").exists()


def test_examples_nav_and_catalog_only_publish_retained_triplet_pages() -> None:
    """The examples nav and landing catalog should exclude roadmap-only pseudo-examples."""
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    index_contents = (DOCS_ROOT / "index.md").read_text(encoding="utf-8")
    examples_nav = mkdocs_contents.split("  - Examples:\n", 1)[1].split(
        "\n\n  - Generative Models:\n", 1
    )[0]

    for page in PUBLISHED_EXAMPLE_PAGES:
        published_path = f"examples/{page}"
        assert published_path in examples_nav
        assert page in index_contents

    for page in ROADMAP_ONLY_PAGES:
        published_path = f"examples/{page}"
        assert published_path not in examples_nav
        assert page not in index_contents

    assert "roadmap/planned-examples.md" in mkdocs_contents


def test_planned_examples_page_collects_unshipped_topics() -> None:
    """Still-relevant unshipped example themes should live on one roadmap page."""
    roadmap = (REPO_ROOT / "docs/roadmap/planned-examples.md").read_text(encoding="utf-8")

    assert "**Status:** `Coming soon`" in roadmap

    for topic in [
        "Advanced diffusion",
        "Advanced flow",
        "Autoregressive and transformer text",
        "CLIP-style multimodal systems",
        "Visual question answering",
        "Image captioning",
        "Cross-modal retrieval",
        "Training strategy walkthroughs",
        "Deployment walkthroughs",
    ]:
        assert topic in roadmap
