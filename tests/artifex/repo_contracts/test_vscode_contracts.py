from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_vscode_bundle_is_artifex_specific_not_portable() -> None:
    """The checked-in VS Code bundle should describe itself as Artifex-local only."""
    readme = (REPO_ROOT / ".vscode" / "README.md").read_text(encoding="utf-8")
    validator = (REPO_ROOT / ".vscode" / "validate-setup.py").read_text(encoding="utf-8")

    banned_portability_claims = [
        "fully portable",
        "any Python project",
        "guaranteed to work",
        "copy the `.vscode/` folder",
        "zero hardcoding",
    ]

    for contents in (readme.lower(), validator.lower()):
        for claim in banned_portability_claims:
            assert claim not in contents

    assert "Artifex" in readme
    assert "Artifex" in validator


def test_vscode_tasks_use_uv_run_repo_contract() -> None:
    """Workspace tasks should use the repo's uv-based contributor workflow."""
    contents = (REPO_ROOT / ".vscode" / "tasks.json").read_text(encoding="utf-8")

    assert '"command": "uv"' in contents

    banned_commands = [
        '"command": "pre-commit"',
        '"command": "ruff"',
        '"command": "pyright"',
        '"command": "mypy"',
        '"command": "pytest"',
        '"command": "nbqa"',
    ]
    for command in banned_commands:
        assert command not in contents


def test_vscode_launch_surface_avoids_non_artifex_entrypoints() -> None:
    """Launch configs should not advertise generic app entrypoints unrelated to Artifex."""
    contents = (REPO_ROOT / ".vscode" / "launch.json").read_text(encoding="utf-8")

    banned_references = [
        "Portable Debug Configurations",
        "Works with any Python project",
        "src/__main__.py",
        "main:app",
        "app.py",
        "FastAPI Server",
        "Flask Server",
    ]
    for banned_reference in banned_references:
        assert banned_reference not in contents

    assert "Pytest Current File" in contents


def test_vscode_validator_uses_uv_workspace_checks() -> None:
    """The validation helper should check the Artifex uv workspace instead of .venv/bin paths."""
    contents = (REPO_ROOT / ".vscode" / "validate-setup.py").read_text(encoding="utf-8")

    assert ".venv/bin" not in contents
    assert '"uv"' in contents
    assert ".artifex.env" in contents or "./setup.sh" in contents
