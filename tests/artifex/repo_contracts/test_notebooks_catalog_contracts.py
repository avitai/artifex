from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH_PATTERN = re.compile(r"notebooks/[\w./-]+\.ipynb")


def test_published_notebooks_page_tracks_real_repo_root_inventory() -> None:
    """The published notebooks page should describe the real repo-root notebook inventory only."""
    contents = (REPO_ROOT / "docs/notebooks/README.md").read_text(encoding="utf-8")
    normalized = " ".join(contents.split())
    documented_notebooks = set(NOTEBOOK_PATH_PATTERN.findall(contents))
    repo_notebooks = {
        path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "notebooks").glob("*.ipynb")
    }

    assert "repo-root `notebooks/` directory" in normalized
    assert "source ./activate.sh" in contents
    assert "uv run jupyter lab" in contents
    assert documented_notebooks == repo_notebooks

    for relative_path in documented_notebooks:
        assert (REPO_ROOT / relative_path).exists()

    for banned in [
        "Energy-Based Models",
        "Advanced MCMC Sampling",
        "GPU Optimization",
        "Navigate to the notebook you want to run.",
    ]:
        assert banned not in contents


def test_published_notebooks_page_does_not_claim_docs_directory_holds_notebooks() -> None:
    """The docs/notebooks directory should remain a docs page, not a fake notebook inventory folder."""
    docs_notebooks_files = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "docs/notebooks").rglob("*")
        if path.is_file()
    }
    contents = (REPO_ROOT / "docs/notebooks/README.md").read_text(encoding="utf-8")

    assert docs_notebooks_files == {"docs/notebooks/README.md"}
    assert "This published page lives under `docs/notebooks/`" in contents
    assert "actual notebooks live in the repo-root `notebooks/` directory" in contents
