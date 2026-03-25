import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BARE_EXAMPLE_PYTHON_PATTERN = re.compile(r"(?m)^python examples/[\w./-]+\.py$")
UV_EXAMPLE_PYTHON_PATTERN = re.compile(r"uv run python examples/[\w./-]+\.py")
UV_JUPYTER_PATTERN = re.compile(r"uv run jupyter\b")


def test_examples_readme_uses_source_checkout_uv_workflow() -> None:
    """The repo-root examples README should describe the contributor checkout workflow truthfully."""
    contents = (REPO_ROOT / "examples/README.md").read_text()

    required_references = [
        "./setup.sh --backend cpu",
        "source ./activate.sh",
        "uv run python",
        "docs/examples/index.md",
    ]
    for reference in required_references:
        assert reference in contents

    banned_references = [
        "source activate.sh",
        "uv sync --all-extras",
        "JAX_PLATFORMS=cpu",
    ]
    for reference in banned_references:
        assert reference not in contents

    assert not BARE_EXAMPLE_PYTHON_PATTERN.search(contents)


def test_user_example_pages_do_not_require_uv_or_activation() -> None:
    """Reader-facing example pages should use plain example commands, not contributor-only uv launchers."""
    example_docs_root = REPO_ROOT / "docs" / "examples"

    for path in example_docs_root.rglob("*.md"):
        if "templates" in path.parts or path.name == "overview.md":
            continue

        contents = path.read_text()
        banned_references = [
            "source activate.sh",
            "source ./activate.sh",
            "uv pip install",
            "uv sync",
        ]
        for reference in banned_references:
            assert reference not in contents, f"{path} still contains {reference!r}"

        assert UV_EXAMPLE_PYTHON_PATTERN.search(contents) is None, (
            f"{path} still launches example scripts through uv"
        )
        assert UV_JUPYTER_PATTERN.search(contents) is None, (
            f"{path} still launches notebooks through uv"
        )


def test_user_docs_use_pip_for_optional_dependencies() -> None:
    """User docs should not require uv for optional integration dependencies."""
    paths = [
        REPO_ROOT / "docs" / "user-guide" / "inference" / "overview.md",
        REPO_ROOT / "docs" / "user-guide" / "integrations" / "huggingface.md",
        REPO_ROOT / "docs" / "user-guide" / "integrations" / "tensorboard.md",
        REPO_ROOT / "docs" / "user-guide" / "integrations" / "wandb.md",
        REPO_ROOT / "docs" / "user-guide" / "training" / "logging.md",
        REPO_ROOT / "docs" / "user-guide" / "training" / "training-guide.md",
        REPO_ROOT / "docs" / "training" / "logging.md",
    ]

    for path in paths:
        contents = path.read_text()
        assert "uv pip install" not in contents, f"{path} still requires uv for package users"
