"""Contracts for the documented model-family surface check."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "derive_status.py"
MODELS_INDEX = REPO_ROOT / "docs" / "models" / "index.md"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={"JAX_PLATFORMS": "cpu", "PATH": "", "PYTHONPATH": ""},
    )


def test_documented_family_surface_matches_the_tree() -> None:
    """Every family the docs table names exists with every representative export."""
    result = _run("--check")

    assert result.returncode == 0, result.stdout + result.stderr
    for family in ("vae", "gan", "diffusion", "flow", "energy", "autoregressive", "geometric"):
        assert family in result.stdout


def test_check_fails_when_the_docs_name_an_export_that_does_not_exist(tmp_path: Path) -> None:
    """A representative export the package does not define is drift."""
    drifted = MODELS_INDEX.read_text(encoding="utf-8").replace("`VAE`", "`VAEThatDoesNotExist`")
    index = tmp_path / "index.md"
    index.write_text(drifted, encoding="utf-8")

    result = _run("--check", "--models-index", str(index))

    assert result.returncode == 1
    assert "VAEThatDoesNotExist" in result.stdout + result.stderr


def test_check_fails_when_a_family_package_is_missing_from_the_docs(tmp_path: Path) -> None:
    """A family package under models/ that the table does not list is drift."""
    lines = MODELS_INDEX.read_text(encoding="utf-8").splitlines(keepends=True)
    index = tmp_path / "index.md"
    index.write_text(
        "".join(line for line in lines if not line.startswith("| Energy |")), encoding="utf-8"
    )

    result = _run("--check", "--models-index", str(index))

    assert result.returncode == 1
    assert "energy" in result.stdout + result.stderr


def test_ci_runs_the_family_surface_check() -> None:
    """The quality job runs the check so the docs table cannot drift silently."""
    contents = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "uv run python scripts/derive_status.py --check" in contents
