import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_example_authoring_guide_exists_and_matches_artifex_contract() -> None:
    """Artifex should ship a dedicated example authoring guide with current workflows."""
    guide = (REPO_ROOT / "docs/development/example-documentation-design.md").read_text()

    required_references = [
        "Artifex Example Documentation Design Guide",
        "source ./activate.sh",
        "uv run python",
        "uv run python scripts/jupytext_converter.py sync",
        "scripts/verify_gpu_setup.py --require-gpu",
        "docs/examples/templates/example_template.py",
        "artifex.configs",
        "Flax NNX",
    ]

    for reference in required_references:
        assert reference in guide

    banned_references = [
        "source activate.sh",
        "jupytext --to notebook",
        "uv pip install -e .",
        "Pydantic",
        "cuda,cpu",
        "/usr/local/cuda",
    ]

    for reference in banned_references:
        assert reference not in guide


def test_contributor_docs_link_to_example_authoring_guide() -> None:
    """Contributor entry points should direct authors to the dedicated example guide."""
    docs_contributing = (REPO_ROOT / "docs/community/contributing.md").read_text()
    repo_contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text()
    mkdocs = (REPO_ROOT / "mkdocs.yml").read_text()

    assert "development/example-documentation-design.md" in mkdocs
    assert "Example Documentation Design Guide" in docs_contributing
    assert "Example Documentation Design Guide" in repo_contributing


def test_example_template_and_examples_readme_use_current_setup_commands() -> None:
    """Contributor-facing example scaffolding should match the current repo setup contract."""
    template = (REPO_ROOT / "docs/examples/templates/example_template.py").read_text()
    examples_readme = (REPO_ROOT / "examples/README.md").read_text()

    for contents in (template, examples_readme):
        assert "source ./activate.sh" in contents
        assert "source activate.sh" not in contents
        assert "uv run python" in contents
        assert "jupytext --to notebook" not in contents
        assert "uv pip install -e ." not in contents

    assert "JAX_PLATFORMS=cpu" not in examples_readme


def test_example_pages_link_to_shipped_python_and_notebook_pairs() -> None:
    """Example docs should expose clickable links to their shipped source pairs."""
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    docs_root = REPO_ROOT / "docs" / "examples"
    examples_root = REPO_ROOT / "examples"

    for path in sorted(docs_root.rglob("*.md")):
        if path.name in {"index.md", "overview.md"}:
            continue

        contents = path.read_text()
        if "not yet available" in contents.lower():
            continue

        normalized = path.stem.replace("-", "_")
        candidate_paths = [
            str(candidate.relative_to(REPO_ROOT)).replace("\\", "/")
            for candidate in examples_root.rglob("*")
            if candidate.suffix in {".py", ".ipynb"}
            and (candidate.stem == normalized or candidate.stem.startswith(f"{normalized}_"))
        ]
        python_candidates = [
            candidate for candidate in candidate_paths if candidate.endswith(".py")
        ]
        notebook_candidates = [
            candidate for candidate in candidate_paths if candidate.endswith(".ipynb")
        ]

        if not (python_candidates and notebook_candidates):
            continue

        link_targets = set(link_pattern.findall(contents))

        assert any(
            any(candidate in target for target in link_targets) for candidate in python_candidates
        ), f"{path} is missing a markdown link to its Python source"
        assert any(
            any(candidate in target for target in link_targets) for candidate in notebook_candidates
        ), f"{path} is missing a markdown link to its notebook source"


def test_example_docs_do_not_use_placeholder_hash_links() -> None:
    """Tracked markdown docs should use real destinations or honest plain text."""
    for path in sorted(REPO_ROOT.rglob("*.md")):
        if "memory-bank" in path.parts:
            continue
        assert "](#)" not in path.read_text(), f"{path} still contains placeholder hash links"
