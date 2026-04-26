from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_quickstart_separates_package_users_from_source_contributors() -> None:
    """Quickstart should separate package-user and contributor workflows clearly."""
    quickstart = (REPO_ROOT / "docs/getting-started/quickstart.md").read_text()
    quickstart_lines = {line.strip() for line in quickstart.splitlines()}

    required_references = [
        "pip install avitai-artifex",
        'pip install "avitai-artifex[cuda12]"',
        "installed code is still imported",
        "./setup.sh",
        "source ./activate.sh",
        "python -c",
        "python train_vae.py",
        "uv run pytest",
    ]
    for reference in required_references:
        assert reference in quickstart

    banned_references = [
        "uv venv && source .venv/bin/activate",
        "python -m venv && source .venv/bin/activate",
        "uv pip install -e '.[dev]'",
        "uv pip install artifex",
        "uv pip install avitai-artifex",
        "pip install artifex",
        "uv run python -c",
        "uv run python train_vae.py",
    ]
    for reference in banned_references:
        assert reference not in quickstart

    banned_lines = {
        "uv sync --all-extras",
        "pytest tests/ -v",
        "ruff format src/",
        "pyright src/",
        "mkdocs serve",
    }
    for line in banned_lines:
        assert line not in quickstart_lines


def test_rendered_quickstart_tracks_executable_vae_first_contract() -> None:
    """Rendered onboarding docs should stay aligned with the executable VAE quickstart pair."""
    rendered = (REPO_ROOT / "docs/getting-started/quickstart.md").read_text()
    executable = (REPO_ROOT / "docs/getting-started/quickstart.py").read_text()

    shared_required_fragments = [
        "TFDSEagerSource",
        "TFDSEagerConfig",
        "VAEConfig",
        "VAE",
        "train_epoch_staged",
        "VAETrainer",
        "VAETrainingConfig",
        'loss_fn = trainer.create_loss_fn(loss_type="bce")',
        "base_step=step",
    ]
    for fragment in shared_required_fragments:
        assert fragment in rendered
        assert fragment in executable

    assert rendered.count('loss_fn = trainer.create_loss_fn(loss_type="bce")') == 1
    assert "`docs/getting-started/quickstart.py`" in rendered
    assert "`docs/getting-started/quickstart.ipynb`" in rendered

    banned_fragments = [
        "TfdsDataSourceConfig",
        "TFDSSource",
        "FlowModel",
        "GANConfig",
        "EnergyBasedModel",
        "FlowConfig",
        "DDPMModel",
        "DCGANConfig",
    ]
    for fragment in banned_fragments:
        assert fragment not in rendered


def test_contributing_docs_use_current_repository_setup() -> None:
    """Contributor docs should match the supported Artifex setup workflow."""
    docs_contributing = (REPO_ROOT / "docs/community/contributing.md").read_text()

    required_references = [
        "./setup.sh",
        "source ./activate.sh",
        "uv sync --extra cuda-dev",
        "uv run pytest",
    ]
    for reference in required_references:
        assert reference in docs_contributing

    banned_references = [
        "uv sync --all-extras",
        "source activate.sh",
    ]
    for reference in banned_references:
        assert reference not in docs_contributing


def test_homepage_and_faq_match_installation_guide_contract() -> None:
    """Public install docs should align with the package-user install contract."""
    homepage = (REPO_ROOT / "docs/index.md").read_text()
    faq = (REPO_ROOT / "docs/community/faq.md").read_text()

    for contents in (homepage, faq):
        assert "pip install avitai-artifex" in contents
        assert 'pip install "avitai-artifex[cuda12]"' in contents
        assert "pip install artifex" not in contents
        assert "uv sync --extra cuda12" not in contents
        assert "uv sync --all-extras" not in contents

    assert "./setup.sh" in faq
    assert "source ./activate.sh" in faq


def test_homepage_routes_readers_to_live_vae_first_quickstart() -> None:
    """Homepage quickstart copy should point to the live VAE-first onboarding path."""
    homepage = (REPO_ROOT / "docs/index.md").read_text()

    required_fragments = [
        "Start with the VAE Quickstart",
        "TFDSEagerSource",
        "VAEConfig",
        "VAETrainer",
        "train_epoch_staged",
        "getting-started/quickstart.md",
        "docs/getting-started/quickstart.py",
        "docs/getting-started/quickstart.ipynb",
    ]
    for fragment in required_fragments:
        assert fragment in homepage

    banned_fragments = [
        "Train a Diffusion Model on Fashion-MNIST",
        "from datarax import from_source",
        "ElementOperatorConfig",
        "OperatorNode",
        "from_tfds",
        "DDPMModel",
        "NoiseScheduleConfig",
    ]
    for fragment in banned_fragments:
        assert fragment not in homepage


def test_root_readme_distinguishes_package_users_from_contributors() -> None:
    """Root README should present package use and source contribution as separate paths."""
    readme = (REPO_ROOT / "README.md").read_text()

    required_references = [
        "pip install avitai-artifex",
        'pip install "avitai-artifex[cuda12]"',
        "The PyPI distribution is named `avitai-artifex`",
        "the Python import package remains",
        "./setup.sh",
        "source ./activate.sh",
    ]
    for reference in required_references:
        assert reference in readme

    banned_references = [
        "pip install artifex",
        "source activate.sh",
        "uv pip install artifex",
        "uv pip install avitai-artifex",
        "uv sync --all-extras",
    ]
    for reference in banned_references:
        assert reference not in readme
