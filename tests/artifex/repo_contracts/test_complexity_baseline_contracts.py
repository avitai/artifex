from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER = REPO_ROOT / "scripts" / "check_complexity_baseline.py"
BASELINE = REPO_ROOT / "quality" / "complexity_baseline.json"


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_complexity_baseline_includes_example_functions_and_methods() -> None:
    """The complexity ratchet should include example functions and methods."""
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert payload["minimum_rank"] == "C"
    assert payload["scan_paths"] == ["src", "examples"]
    assert payload["example_enforced_kinds"] == ["function", "method"]
    assert payload["entries"]
    example_records = [
        record for record in payload["entries"].values() if record["path"].startswith("examples/")
    ]
    assert example_records
    assert {record["kind"] for record in example_records} <= {"function", "method"}


def test_complexity_checker_blocks_new_source_and_example_function_debt(tmp_path: Path) -> None:
    """New source and example C-or-worse functions should both fail the ratchet."""
    source_dir = tmp_path / "src"
    examples_dir = tmp_path / "examples"
    baseline = tmp_path / "baseline.json"

    _write(source_dir / "clean.py", "def clean(value):\n    return value + 1\n")
    _write(
        examples_dir / "complex_demo.py",
        "def complex_demo(value):\n"
        "    if value == 0:\n        return 0\n"
        "    if value == 1:\n        return 1\n"
        "    if value == 2:\n        return 2\n"
        "    if value == 3:\n        return 3\n"
        "    if value == 4:\n        return 4\n"
        "    if value == 5:\n        return 5\n"
        "    if value == 6:\n        return 6\n"
        "    if value == 7:\n        return 7\n"
        "    if value == 8:\n        return 8\n"
        "    if value == 9:\n        return 9\n"
        "    if value == 10:\n        return 10\n"
        "    return value\n",
    )

    write_result = _run_checker(
        "--source-dir",
        str(tmp_path),
        "--baseline",
        str(baseline),
        "--write-baseline",
    )
    assert write_result.returncode == 0, write_result.stderr

    check_result = _run_checker("--source-dir", str(tmp_path), "--baseline", str(baseline))
    assert check_result.returncode == 0, check_result.stdout + check_result.stderr

    _write(
        source_dir / "complex_source.py",
        "def complex_source(value):\n"
        "    if value == 0:\n        return 0\n"
        "    if value == 1:\n        return 1\n"
        "    if value == 2:\n        return 2\n"
        "    if value == 3:\n        return 3\n"
        "    if value == 4:\n        return 4\n"
        "    if value == 5:\n        return 5\n"
        "    if value == 6:\n        return 6\n"
        "    if value == 7:\n        return 7\n"
        "    if value == 8:\n        return 8\n"
        "    if value == 9:\n        return 9\n"
        "    if value == 10:\n        return 10\n"
        "    return value\n",
    )

    failed_result = _run_checker("--source-dir", str(tmp_path), "--baseline", str(baseline))
    assert failed_result.returncode == 1
    assert "complex_source" in failed_result.stdout

    _write(
        examples_dir / "complex_new_demo.py",
        "def complex_new_demo(value):\n"
        "    if value == 0:\n        return 0\n"
        "    if value == 1:\n        return 1\n"
        "    if value == 2:\n        return 2\n"
        "    if value == 3:\n        return 3\n"
        "    if value == 4:\n        return 4\n"
        "    if value == 5:\n        return 5\n"
        "    if value == 6:\n        return 6\n"
        "    if value == 7:\n        return 7\n"
        "    if value == 8:\n        return 8\n"
        "    if value == 9:\n        return 9\n"
        "    if value == 10:\n        return 10\n"
        "    return value\n",
    )

    example_failed_result = _run_checker("--source-dir", str(tmp_path), "--baseline", str(baseline))
    assert example_failed_result.returncode == 1
    assert "complex_new_demo" in example_failed_result.stdout


def test_pre_commit_runs_the_complexity_baseline_gate() -> None:
    """Pre-commit should run the checked-in complexity ratchet."""
    contents = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "id: complexity-baseline" in contents
    assert "uv run --locked python scripts/check_complexity_baseline.py" in contents
