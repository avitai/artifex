import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


SUBSTRAX_EXAMPLE_GUIDE = "https://substrax.readthedocs.io/en/latest/guides/writing-examples/"
CONTRIBUTOR_EXAMPLE_DOCS = (
    "CONTRIBUTING.md",
    "docs/community/contributing.md",
    "examples/README.md",
    "examples/EXAMPLES_GUIDE.md",
)


def test_contributor_docs_send_example_authors_to_substrax_guide_and_tools() -> None:
    """Examples follow substrax's shared guide; substrax-notebooks writes notebooks and pages."""
    for relative_path in CONTRIBUTOR_EXAMPLE_DOCS:
        contents = (REPO_ROOT / relative_path).read_text()
        assert SUBSTRAX_EXAMPLE_GUIDE in contents, relative_path
        assert "substrax-notebooks convert" in contents, relative_path
        assert "jupytext_converter" not in contents, relative_path
        assert "example-documentation-design" not in contents, relative_path


def test_examples_readme_uses_current_setup_commands() -> None:
    """Contributor-facing example scaffolding should match the current repo setup contract."""
    examples_readme = (REPO_ROOT / "examples/README.md").read_text()

    assert "source ./activate.sh" in examples_readme
    assert "source activate.sh" not in examples_readme
    assert "uv run python" in examples_readme
    assert "jupytext --to notebook" not in examples_readme
    assert "uv pip install -e ." not in examples_readme
    assert "JAX_PLATFORMS=cpu" not in examples_readme


def test_example_pages_link_to_shipped_python_and_notebook_pairs() -> None:
    """Example docs should expose clickable links to their shipped source pairs."""
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    docs_root = REPO_ROOT / "docs" / "tutorials"
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
