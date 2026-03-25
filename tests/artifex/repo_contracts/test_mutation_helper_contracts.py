from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_python_script(
    relative_path: str,
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / relative_path), *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_clean_cache_preserves_repo_contract_targets(tmp_path: Path) -> None:
    """The cleanup script should remove only disposable artifacts."""
    for relative in (
        ".pytest_cache",
        ".ruff_cache",
        "temp",
        "test_artifacts",
        "htmlcov",
        "build",
        "dist",
        "package.egg-info",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)

    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".vscode" / "settings.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".vscode" / "session.log").write_text("log", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    (tmp_path / ".coverage").write_text("", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "clean_cache.sh")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for removed in (
        ".pytest_cache",
        ".ruff_cache",
        "temp",
        "test_artifacts",
        "htmlcov",
        "build",
        "dist",
        "package.egg-info",
    ):
        assert not (tmp_path / removed).exists()

    assert not (tmp_path / ".coverage").exists()
    assert not (tmp_path / ".vscode" / "session.log").exists()
    assert (tmp_path / ".vscode" / "settings.json").exists()
    assert (tmp_path / ".venv").exists()
    assert (tmp_path / ".git").exists()
    assert (tmp_path / "uv.lock").exists()


def test_orphaned_mutation_helpers_are_not_retained() -> None:
    """Unsupported write-capable helper scripts should not stay in the repo."""
    assert not (REPO_ROOT / "scripts" / "fix_imports.py").exists()
    assert not (REPO_ROOT / "scripts" / "migrate_batch_sizes.py").exists()

    scripts_readme = (REPO_ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert "fix_imports.py" not in scripts_readme
    assert "migrate_batch_sizes.py" not in scripts_readme


def test_jupytext_sync_creates_missing_notebook_pair(tmp_path: Path) -> None:
    """The retained jupytext helper should create the missing paired notebook."""
    python_file = tmp_path / "example.py"
    python_file.write_text("print('hello')\n", encoding="utf-8")

    result = _run_python_script("scripts/jupytext_converter.py", "sync", str(python_file))

    assert result.returncode == 0, result.stderr
    assert python_file.with_suffix(".ipynb").exists()
    assert "Created" in result.stdout


def test_jupytext_validate_reports_corrupt_notebook_without_traceback(
    tmp_path: Path,
) -> None:
    """Corrupt notebook metadata should fail cleanly without a traceback."""
    python_file = tmp_path / "example.py"
    notebook_file = tmp_path / "example.ipynb"
    python_file.write_text("print('hello')\n", encoding="utf-8")
    notebook_file.write_text("{not-json}\n", encoding="utf-8")

    result = _run_python_script("scripts/jupytext_converter.py", "validate", str(tmp_path))

    assert result.returncode == 1
    assert "Error checking" in result.stdout
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
