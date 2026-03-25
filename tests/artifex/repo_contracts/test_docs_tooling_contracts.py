from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ACTION = "./.github/actions/setup-artifex"


def _run_script(
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


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _load_yaml(relative_path: str) -> dict[str, object]:
    with (REPO_ROOT / relative_path).open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def test_validate_docs_fix_fails_when_unresolved_issues_remain(tmp_path: Path) -> None:
    """Fix mode must still fail when structural documentation issues remain."""
    _write(
        tmp_path / "mkdocs.yml",
        "site_name: Demo\nnav:\n  - Home: index.md\n  - Missing: missing.md\n",
    )
    _write(tmp_path / "docs" / "index.md", "# Home\n")
    _write(tmp_path / "src" / "artifex" / "__init__.py", '"""package"""\n')

    result = _run_script(
        "scripts/validate_docs.py",
        "--fix",
        "--config-path",
        "mkdocs.yml",
        "--docs-path",
        "docs",
        "--src-path",
        "src",
        cwd=tmp_path,
    )

    assert result.returncode == 1, result.stderr
    assert "Navigation references missing file: missing.md" in result.stderr


def test_validate_docs_fix_can_resolve_missing_custom_dir(tmp_path: Path) -> None:
    """Fix mode may repair safe scaffolding gaps and return success once issues are gone."""
    _write(
        tmp_path / "mkdocs.yml",
        "site_name: Demo\ntheme:\n  custom_dir: docs/_overrides\nnav:\n  - Home: index.md\n",
    )
    _write(tmp_path / "docs" / "index.md", "# Home\n")
    _write(tmp_path / "src" / "artifex" / "__init__.py", '"""package"""\n')

    result = _run_script(
        "scripts/validate_docs.py",
        "--fix",
        "--config-path",
        "mkdocs.yml",
        "--docs-path",
        "docs",
        "--src-path",
        "src",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "docs" / "_overrides").is_dir()


def test_validate_docs_accepts_mkdocstrings_object_paths(tmp_path: Path) -> None:
    """Mkdocstrings directives may point at objects within a real module."""
    _write(tmp_path / "mkdocs.yml", "site_name: Demo\nnav:\n  - Home: index.md\n")
    _write(
        tmp_path / "docs" / "index.md",
        "# Home\n\n::: artifex.pkg.configs.DemoConfig\n",
    )
    _write(tmp_path / "src" / "artifex" / "pkg" / "__init__.py", "")
    _write(
        tmp_path / "src" / "artifex" / "pkg" / "configs.py",
        "class DemoConfig:\n    pass\n",
    )

    result = _run_script(
        "scripts/validate_docs.py",
        "--check-only",
        "--config-path",
        "mkdocs.yml",
        "--docs-path",
        "docs",
        "--src-path",
        "src",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr


def test_validate_docs_rejects_relative_links_outside_docs_tree(tmp_path: Path) -> None:
    """Relative links must stay within the curated docs tree."""
    _write(tmp_path / "mkdocs.yml", "site_name: Demo\nnav:\n  - Home: index.md\n")
    _write(
        tmp_path / "docs" / "index.md",
        "# Home\n\n[Source](../src/artifex/pkg/configs.py)\n",
    )
    _write(tmp_path / "src" / "artifex" / "pkg" / "configs.py", "class DemoConfig:\n    pass\n")

    result = _run_script(
        "scripts/validate_docs.py",
        "--check-only",
        "--config-path",
        "mkdocs.yml",
        "--docs-path",
        "docs",
        "--src-path",
        "src",
        cwd=tmp_path,
    )

    assert result.returncode == 1, result.stderr
    assert "Relative link escapes the docs tree" in result.stderr


def test_validate_docs_ignores_code_and_math_link_lookalikes(tmp_path: Path) -> None:
    """Code fences and math expressions should not be parsed as markdown links."""
    _write(tmp_path / "mkdocs.yml", "site_name: Demo\nnav:\n  - Home: index.md\n")
    _write(
        tmp_path / "docs" / "index.md",
        "# Home\n\n"
        "```python\n"
        "outputs = model[x](y)\n"
        "```\n\n"
        "$$p_\\theta(y|x) = \\frac{e^{f_\\theta[x](y)}}{Z_\\theta}$$\n",
    )
    _write(tmp_path / "src" / "artifex" / "__init__.py", "")

    result = _run_script(
        "scripts/validate_docs.py",
        "--check-only",
        "--config-path",
        "mkdocs.yml",
        "--docs-path",
        "docs",
        "--src-path",
        "src",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr


def test_docs_build_surface_uses_validator_and_strict_build_only() -> None:
    """The retained docs wrapper should validate first and avoid generator coupling."""
    result = _run_script("scripts/build_docs.py", "--help")
    script_contents = (REPO_ROOT / "scripts" / "build_docs.py").read_text(encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert "--config-path" in result.stdout
    assert "--docs-path" in result.stdout
    assert "--src-path" in result.stdout
    assert "--serve" in result.stdout
    assert "--skip-generation" not in result.stdout
    assert "generate_docs.py" not in script_contents
    assert '"build"' in script_contents
    assert '"--clean"' in script_contents
    assert '"--strict"' in script_contents


def test_unsafe_docs_generator_is_not_in_the_retained_surface() -> None:
    """The removed docs generator should not remain documented or wired into workflows."""
    readme = (REPO_ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    docs_workflow = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

    assert not (REPO_ROOT / "scripts" / "generate_docs.py").exists()
    assert "generate_docs.py" not in readme
    assert "generate_docs.py" not in docs_workflow


def test_local_docs_serve_wrapper_uses_uv_run() -> None:
    """Local docs serving should use the same environment resolution path as the repo."""
    contents = (REPO_ROOT / "scripts" / "serve_docs_fast.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in contents
    assert ".venv/bin/mkdocs" not in contents
    assert "uv run mkdocs serve -f mkdocs-dev.yml --dirtyreload" in contents


def test_docs_deployment_workflow_uses_shared_setup_and_validator() -> None:
    """Docs deployment must reuse shared setup and the truthful validator contract."""
    workflow = _load_yaml(".github/workflows/docs.yml")
    contents = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

    assert set(workflow["on"]) == {"push", "workflow_dispatch"}
    for job in workflow["jobs"].values():
        setup_steps = [step for step in job["steps"] if step.get("uses") == SETUP_ACTION]
        assert len(setup_steps) == 1

    assert "actions/setup-python@" not in contents
    assert "astral-sh/setup-uv@" not in contents
    assert "uv sync --extra docs" not in contents
    assert (
        "uv run python scripts/validate_docs.py --config-path mkdocs.yml --docs-path docs --src-path src"
        in contents
    )
    assert "uv run mkdocs build --clean --strict" in contents
    assert "continue-on-error" not in contents


def test_scripts_readme_tracks_retained_script_contracts() -> None:
    """The scripts catalog should only advertise retained, truthful script surfaces."""
    contents = (REPO_ROOT / "scripts" / "README.md").read_text(encoding="utf-8")

    required_references = [
        "uv run python scripts/setup_env.py show",
        "uv run python scripts/verify_gpu_setup.py --require-gpu",
        "uv run python scripts/gpu_utils.py --test-critical",
        "uv run python scripts/check_jax_nnx_compatibility.py --json",
        "uv run python scripts/analyze_test_structure.py",
        "uv run python scripts/build_docs.py",
        "uv run python scripts/validate_docs.py --check-only",
        "./scripts/serve_docs_fast.sh",
    ]
    for reference in required_references:
        assert reference in contents

    banned_references = [
        "python scripts/build_docs.py --skip-generation",
        "python scripts/generate_docs.py",
        ".venv/bin/mkdocs",
        "--configure-generative",
        "--critical-only",
        "Recommended version reporting",
        "python scripts/fix_imports.py",
    ]
    for reference in banned_references:
        assert reference not in contents
