import json
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_analyze_dependencies_defaults_to_repo_internal_artifacts(tmp_path: Path) -> None:
    """Dependency analysis should avoid writing into the docs tree by default."""
    _write(tmp_path / "pkg_a" / "__init__.py", "")
    _write(tmp_path / "pkg_a" / "base.py", "from pkg_b import base\n")
    _write(tmp_path / "pkg_b" / "__init__.py", "")
    _write(tmp_path / "pkg_b" / "base.py", "from pkg_a import base\n")

    result = _run_script(
        "scripts/analyze_dependencies.py", "--source-dir", str(tmp_path), cwd=tmp_path
    )

    assert result.returncode == 0, result.stderr
    report_path = (
        tmp_path
        / "test_artifacts"
        / "code_analysis"
        / "dependency_analysis"
        / "dependency_report.md"
    )
    graph_path = (
        tmp_path / "test_artifacts" / "code_analysis" / "dependency_analysis" / "dependencies.svg"
    )
    assert report_path.exists()
    assert graph_path.exists()
    assert not (tmp_path / "docs").exists()

    report_contents = report_path.read_text(encoding="utf-8")
    assert "pkg_a.base" in report_contents
    assert "pkg_b.base" in report_contents


def test_find_circular_imports_reports_multi_module_cycles(tmp_path: Path) -> None:
    """The circular-import script should catch full repo-internal cycle groups."""
    source_dir = tmp_path / "src" / "artifex" / "pkg"
    _write(tmp_path / "src" / "artifex" / "__init__.py", "")
    _write(source_dir / "__init__.py", "")
    _write(source_dir / "a.py", "from artifex.pkg import b\n")
    _write(source_dir / "b.py", "from artifex.pkg import c\n")
    _write(source_dir / "c.py", "from artifex.pkg import a\n")

    output_file = tmp_path / "analysis" / "circular_imports.txt"
    result = _run_script(
        "scripts/find_circular_imports.py",
        "--source-dir",
        str(tmp_path / "src" / "artifex"),
        "--output-file",
        str(output_file),
    )

    assert result.returncode == 0, result.stderr
    assert output_file.exists()

    report_contents = output_file.read_text(encoding="utf-8")
    assert "Detected 1 circular dependency group(s)." in report_contents
    assert "- artifex.pkg.a -> artifex.pkg.b" in report_contents
    assert "- artifex.pkg.b -> artifex.pkg.c" in report_contents
    assert "- artifex.pkg.c -> artifex.pkg.a" in report_contents


def test_analyze_test_structure_defaults_to_repo_internal_artifacts(tmp_path: Path) -> None:
    """Test-structure analysis should avoid writing into tracked docs by default."""
    _write(tmp_path / "src" / "artifex" / "__init__.py", "")
    _write(tmp_path / "tests" / "test_demo.py", "from artifex import missing\n")

    result = _run_script("scripts/analyze_test_structure.py", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    report_path = (
        tmp_path / "test_artifacts" / "code_analysis" / "test_structure" / "test_structure.md"
    )
    json_path = (
        tmp_path / "test_artifacts" / "code_analysis" / "test_structure" / "test_analysis.json"
    )
    assert report_path.exists()
    assert json_path.exists()
    assert not (tmp_path / "docs").exists()
    assert "sys.path.insert" not in (REPO_ROOT / "scripts" / "analyze_test_structure.py").read_text(
        encoding="utf-8"
    )


def test_check_jax_nnx_compatibility_guidance_tracks_pyproject() -> None:
    """Dependency guidance should come from live project metadata and existing docs."""
    result = _run_script("scripts/check_jax_nnx_compatibility.py", "--json")

    assert result.returncode in {0, 1}, result.stderr
    payload = json.loads(result.stdout)

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = {
        requirement.split(">=", 1)[0].split("[", 1)[0]: requirement.split(">=", 1)[1]
        for requirement in pyproject["project"]["dependencies"]
        if requirement.startswith(("jax>=", "flax>=", "optax>=", "orbax-checkpoint>="))
    }

    assert payload["project_minimum_versions"] == requirements
    assert payload["docs_refs"] == [
        "docs/getting-started/installation.md",
        "docs/getting-started/hardware-setup-guide.md",
    ]
    for relative_path in payload["docs_refs"]:
        assert (REPO_ROOT / relative_path).exists()


def test_gpu_utils_help_uses_live_supported_surface_only() -> None:
    """GPU diagnostics help should stay import-safe and avoid dead commands."""
    result = _run_script("scripts/gpu_utils.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "--configure-generative" not in result.stdout
    assert "--detailed" in result.stdout
    assert "--test" in result.stdout
    assert "--test-critical" in result.stdout
    assert "sys.path.insert" not in (REPO_ROOT / "scripts" / "gpu_utils.py").read_text(
        encoding="utf-8"
    )
