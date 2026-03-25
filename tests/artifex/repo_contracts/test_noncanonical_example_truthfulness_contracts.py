from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

EXPLORATORY_EXAMPLES = {
    "advanced_gan": {
        "page": REPO_ROOT / "docs/examples/advanced/advanced-gan.md",
        "script": REPO_ROOT / "examples/generative_models/image/gan/advanced_gan.py",
        "run_command": "python examples/generative_models/image/gan/advanced_gan.py",
        "notebook_command": ("jupyter lab examples/generative_models/image/gan/advanced_gan.ipynb"),
        "nav_label": "Exploratory Advanced GAN",
    },
    "protein_diffusion": {
        "page": REPO_ROOT / "docs/examples/protein/protein-diffusion-example.md",
        "script": REPO_ROOT / "examples/generative_models/protein/protein_diffusion_example.py",
        "run_command": "python examples/generative_models/protein/protein_diffusion_example.py",
        "notebook_command": (
            "jupyter lab examples/generative_models/protein/protein_diffusion_example.ipynb"
        ),
        "nav_label": "Exploratory Protein Diffusion",
    },
}

VALIDATION_EXAMPLE = {
    "page": REPO_ROOT / "docs/examples/protein/protein-diffusion-tech-validation.md",
    "script": REPO_ROOT / "examples/generative_models/protein/protein_diffusion_tech_validation.py",
    "run_command": (
        "python examples/generative_models/protein/protein_diffusion_tech_validation.py"
    ),
    "notebook_command": (
        "jupyter lab examples/generative_models/protein/protein_diffusion_tech_validation.ipynb"
    ),
    "nav_label": "Validation Protein Tech Check",
}


def normalized_text(path: Path) -> str:
    """Collapse insignificant whitespace so docs can wrap naturally."""
    return " ".join(path.read_text(encoding="utf-8").split())


def stripped_lines(contents: str) -> set[str]:
    """Return trimmed non-empty lines for exact legacy-command checks."""
    return {line.strip() for line in contents.splitlines() if line.strip()}


def test_example_design_guide_defines_exploratory_and_validation_contracts() -> None:
    """The local example design guide should distinguish non-canonical tutorial types."""
    guide = normalized_text(REPO_ROOT / "docs/development/example-documentation-design.md")

    assert "Exploratory Workflows Are Opt-In" in guide
    assert "Validation Utilities Are Not Canonical Tutorials" in guide
    assert "Exploratory workflow" in guide
    assert "Validation utility" in guide


def test_examples_catalog_and_nav_expose_noncanonical_sections_explicitly() -> None:
    """Published non-canonical examples should be labeled and grouped separately."""
    index_contents = (REPO_ROOT / "docs/examples/index.md").read_text(encoding="utf-8")
    normalized_index = " ".join(index_contents.split())
    overview_contents = normalized_text(REPO_ROOT / "docs/examples/overview.md")
    mkdocs_contents = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "runtime-backed Artifex tutorials" in normalized_index
    assert "standalone concept walkthroughs" in normalized_index
    assert "exploratory workflows" in normalized_index
    assert "validation utilities" in normalized_index
    assert "## Exploratory Workflows" in index_contents
    assert "## Validation Utilities" in index_contents
    assert (
        "runtime-backed tutorials, standalone pedagogy, exploratory workflows, "
        "and validation utilities"
    ) in overview_contents

    for paths in EXPLORATORY_EXAMPLES.values():
        nav_page = paths["page"].relative_to(REPO_ROOT / "docs").as_posix()
        assert f"- {paths['nav_label']}: {nav_page}" in mkdocs_contents

    nav_page = VALIDATION_EXAMPLE["page"].relative_to(REPO_ROOT / "docs").as_posix()
    assert f"- {VALIDATION_EXAMPLE['nav_label']}: {nav_page}" in mkdocs_contents


def test_advanced_gan_pair_is_labeled_as_exploratory_lower_level_work() -> None:
    """The advanced GAN tier should stop claiming canonical top-level owner usage."""
    page = EXPLORATORY_EXAMPLES["advanced_gan"]["page"].read_text(encoding="utf-8")
    script = EXPLORATORY_EXAMPLES["advanced_gan"]["script"].read_text(encoding="utf-8")

    for contents in (" ".join(page.split()), " ".join(script.split())):
        assert "Exploratory workflow" in contents
        assert "lower-level Artifex GAN building blocks" in contents
        assert "custom training loop" in contents
        assert (
            "does not instantiate the top-level `ConditionalGAN`, `WGAN`, `DCGAN`, "
            "or `LSGAN` owners end to end"
        ) in contents

    assert EXPLORATORY_EXAMPLES["advanced_gan"]["run_command"] in page
    assert EXPLORATORY_EXAMPLES["advanced_gan"]["notebook_command"] in page
    assert "source ./activate.sh" not in page

    for banned in [
        "production-ready implementations",
        "How to use Artifex's ConditionalGAN",
        "Using Artifex's ConditionalGAN for label-controlled generation",
    ]:
        assert banned not in page
        assert banned not in script


def test_protein_diffusion_pair_is_labeled_as_exploratory_direct_owner_work() -> None:
    """The protein diffusion tier should stop teaching placeholder or fake high-level surfaces."""
    page = EXPLORATORY_EXAMPLES["protein_diffusion"]["page"].read_text(encoding="utf-8")
    script = EXPLORATORY_EXAMPLES["protein_diffusion"]["script"].read_text(encoding="utf-8")

    for contents in (" ".join(page.split()), " ".join(script.split())):
        assert "Exploratory workflow" in contents
        assert "does not demonstrate a shipped high-level Artifex protein diffusion API" in contents
        assert "ProteinPointCloudModel" in contents
        assert "ProteinGraphModel" in contents
        assert "ProteinDataset" in contents
        assert "protein_collate_fn" in contents
        assert "create_protein_structure_loss" in contents

    assert EXPLORATORY_EXAMPLES["protein_diffusion"]["run_command"] in page
    assert EXPLORATORY_EXAMPLES["protein_diffusion"]["notebook_command"] in page
    assert "source ./activate.sh" not in page

    for banned in ["nnx.Module()", "dummy metrics", "dataset.collate_batch("]:
        assert banned not in page
        assert banned not in script


def test_protein_tech_validation_pair_is_labeled_as_validation_utility() -> None:
    """The protein tech validation tier should be grouped as environment validation, not canonical modeling."""
    page = VALIDATION_EXAMPLE["page"].read_text(encoding="utf-8")
    script = VALIDATION_EXAMPLE["script"].read_text(encoding="utf-8")

    for contents in (" ".join(page.split()), " ".join(script.split())):
        assert "Validation utility" in contents
        assert (
            "does not instantiate shipped Artifex protein model, modality, or data owners"
        ) in contents

    assert VALIDATION_EXAMPLE["run_command"] in page
    assert VALIDATION_EXAMPLE["notebook_command"] in page
    assert "source ./activate.sh" not in page

    for contents in (page, script):
        assert "pip install artifex" not in contents

    assert (
        "uv run python examples/generative_models/protein/protein_diffusion_tech_validation.py"
        not in page
    )
    assert (
        "uv run jupyter lab examples/generative_models/protein/protein_diffusion_tech_validation.ipynb"
        not in page
    )
