from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
README_PATH = REPO_ROOT / "examples" / "README.md"
GUIDE_PATH = REPO_ROOT / "examples" / "EXAMPLES_GUIDE.md"
COMMAND_PATTERN = re.compile(r"uv run python (?P<path>examples/[\w./-]+\.py)")
BARE_PYTHON_PATTERN = re.compile(r"(?m)^python examples/[\w./-]+\.py$")


def _extract_uv_commands(contents: str) -> set[str]:
    return set(COMMAND_PATTERN.findall(contents))


def test_root_example_docs_use_uv_and_avoid_backend_forcing() -> None:
    """Contributor-facing root example docs should follow the checked-in design guide."""
    for path in (README_PATH, GUIDE_PATH):
        contents = path.read_text(encoding="utf-8")
        assert "source ./activate.sh" in contents
        assert "uv run python" in contents
        assert not BARE_PYTHON_PATTERN.search(contents), (
            f"{path} still documents direct python execution"
        )
        assert "export JAX_PLATFORMS" not in contents
        assert "unset JAX_PLATFORMS" not in contents


def test_root_example_doc_commands_resolve_to_real_files() -> None:
    """Every documented root example command should point at a real checked-in example."""
    commands = _extract_uv_commands(README_PATH.read_text(encoding="utf-8"))
    commands.update(_extract_uv_commands(GUIDE_PATH.read_text(encoding="utf-8")))

    assert commands

    for relative_path in commands:
        assert (REPO_ROOT / relative_path).exists(), (
            f"Documented example path is missing: {relative_path}"
        )


def test_examples_resolve_outputs_outside_the_source_tree() -> None:
    """Examples, their notebooks and their docs pages name no repository-relative output directory."""
    pages = [
        *sorted((REPO_ROOT / "examples").rglob("*.py")),
        *sorted((REPO_ROOT / "examples").rglob("*.ipynb")),
        *sorted((REPO_ROOT / "docs" / "examples").rglob("*.md")),
    ]
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in pages
        if "examples_output" in path.read_text(encoding="utf-8")
    ]

    assert (
        REPO_ROOT / "examples" / "generative_models" / "image" / "vae" / "advanced_vae.py" in pages
    )
    assert offenders == []
