from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ACTION = "./.github/actions/setup-artifex"


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _load_uv_lock() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "uv.lock").read_text())


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _load_yaml(relative_path: str) -> dict[str, object]:
    with (REPO_ROOT / relative_path).open() as handle:
        return yaml.load(handle, Loader=yaml.BaseLoader)


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def _ci_policy() -> dict[str, object]:
    pyproject = _load_pyproject()
    tool = pyproject["tool"]
    artifex = tool["artifex"]
    return artifex["ci"]


def test_pyproject_declares_reviewed_ci_roles_and_security_triage_policy() -> None:
    """The CI role split and reviewed security suppressions should live in pyproject."""
    policy = _ci_policy()
    security = policy["security"]
    reviewed_ignores = security["reviewed_ignores"]

    assert policy["tooling_python"] == "3.12"
    assert policy["compatibility_python"] == ["3.12", "3.13"]
    assert policy["pyright_enforcement"] == "blocking"
    assert policy["smoke_package"] == "artifex"
    assert policy["smoke_exports"] == ["generative_models"]
    assert policy["blocking_workflows"] == [
        ".github/workflows/ci.yml",
        ".github/workflows/build-verification.yml",
        ".github/workflows/security.yml",
    ]
    assert policy["informational_workflows"] == [
        ".github/workflows/quality-checks.yml",
        ".github/workflows/upstream-compat.yml",
    ]
    assert security["mode"] == "blocking"

    # The contract on suppressions is that each one is justified, not that there are none.
    # This previously also asserted the list was empty, which was true on the day it was
    # written and made the loop below unreachable, so the per-entry schema was never enforced
    # once and an entry could be declared in any shape at all.
    for entry in reviewed_ignores:
        assert entry["owner"] == "repo-maintainers"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["review_after"])
        assert len(entry["rationale"]) >= 20


def test_lockfile_resolves_security_patch_floors_for_fixable_alerts() -> None:
    """Fixable audited vulnerabilities should resolve to patched lockfile versions."""
    packages = {pkg["name"].lower(): pkg["version"] for pkg in _load_uv_lock()["package"]}

    assert _version_key(packages["aiohttp"]) >= (3, 14, 1)
    assert _version_key(packages["bleach"]) >= (6, 4, 0)
    assert _version_key(packages["cryptography"]) >= (48, 0, 1)
    assert _version_key(packages["fastapi"]) >= (0, 139, 0)
    assert _version_key(packages["gitpython"]) >= (3, 1, 50)
    assert _version_key(packages["idna"]) >= (3, 15)
    assert _version_key(packages["jupyter-server"]) >= (2, 20, 0)
    assert _version_key(packages["jupyterlab"]) >= (4, 5, 9)
    assert _version_key(packages["mako"]) >= (1, 3, 12)
    assert _version_key(packages["mistune"]) >= (3, 3, 2)
    assert _version_key(packages["mlflow"]) >= (3, 14, 0)
    assert _version_key(packages["msgpack"]) >= (1, 2, 1)
    assert _version_key(packages["nbconvert"]) >= (7, 17, 1)
    assert _version_key(packages["notebook"]) >= (7, 6, 0)
    assert _version_key(packages["pillow"]) >= (12, 3, 0)
    assert _version_key(packages["pygments"]) >= (2, 20, 0)
    assert _version_key(packages["pymdown-extensions"]) >= (10, 21, 3)
    assert _version_key(packages["pytest"]) >= (9, 0, 3)
    assert _version_key(packages["python-multipart"]) >= (0, 0, 31)
    assert _version_key(packages["requests"]) >= (2, 33, 0)
    assert _version_key(packages["pyasn1"]) >= (0, 6, 3)
    assert _version_key(packages["starlette"]) >= (1, 3, 1)
    assert _version_key(packages["tornado"]) >= (6, 5, 7)
    assert _version_key(packages["urllib3"]) >= (2, 7, 0)


def test_policy_workflows_use_checked_in_setup_action_instead_of_inline_bootstrap() -> None:
    """Tracked CI workflows should reuse one setup owner instead of copy-pasting bootstrap."""
    policy = _ci_policy()

    for relative_path in policy["blocking_workflows"] + policy["informational_workflows"]:
        workflow = _load_yaml(relative_path)
        contents = _read(relative_path)

        assert "actions/setup-python@" not in contents
        assert "astral-sh/setup-uv@" not in contents
        assert "uv pip install -e" not in contents

        for job in workflow["jobs"].values():
            setup_steps = [step for step in job["steps"] if step.get("uses") == SETUP_ACTION]
            assert len(setup_steps) == 1


def test_shared_setup_action_uses_current_pinned_uv_toolchain() -> None:
    """The shared setup action should not regress to an old uv that rejects uv.lock."""
    action = _load_yaml(".github/actions/setup-artifex/action.yml")
    install_uv = next(step for step in action["runs"]["steps"] if step.get("name") == "Install uv")

    assert install_uv["uses"] == "astral-sh/setup-uv@v8.3.1"
    assert install_uv["with"]["version"] == "0.11.25"


def test_workflow_roles_and_blocking_quality_commands_are_explicit() -> None:
    """Blocking checks should stay explicit, and security should run on the automatic gate."""
    ci_workflow = _load_yaml(".github/workflows/ci.yml")
    build_workflow = _load_yaml(".github/workflows/build-verification.yml")
    quality_workflow = _load_yaml(".github/workflows/quality-checks.yml")
    security_workflow = _load_yaml(".github/workflows/security.yml")

    ci_contents = _read(".github/workflows/ci.yml")
    build_contents = _read(".github/workflows/build-verification.yml")
    quality_contents = _read(".github/workflows/quality-checks.yml")
    security_contents = _read(".github/workflows/security.yml")

    assert set(ci_workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert set(build_workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert set(quality_workflow["on"]) == {"workflow_dispatch"}
    assert set(security_workflow["on"]) == {"push", "pull_request", "schedule", "workflow_dispatch"}

    assert "uv run ruff check --output-format=github" in ci_contents
    assert "uv run ruff format --check" in ci_contents
    assert "uv run pyright --warnings" not in ci_contents

    for contents in (build_contents, quality_contents, security_contents):
        assert "uv run ruff check --output-format=github" not in contents
        assert "uv run ruff format --check" not in contents
        assert "continue-on-error" not in contents

    assert "uv run pyright --warnings" not in build_contents
    assert "uv run pyright --warnings --outputjson" in quality_contents
    assert "uv run pyright --warnings" not in security_contents
    assert "Advisory Security Audit" not in security_contents
    assert security_workflow["jobs"]["security-audit"]["name"] == "Security Audit"
    assert "automatic pull-request and push enforcement" in security_contents
    assert "scheduled and manual by design" not in security_contents


def test_partial_ci_coverage_jobs_defer_threshold_to_combined_report() -> None:
    """Partial coverage producers should not apply the repo-wide fail-under threshold."""
    workflow = _load_yaml(".github/workflows/ci.yml")

    for job_name, step_name in (
        ("integration_tests", "Run integration tests"),
        ("e2e_tests", "Run end-to-end tests"),
    ):
        step = next(
            step for step in workflow["jobs"][job_name]["steps"] if step.get("name") == step_name
        )
        command = step["run"]

        assert "--cov=src/artifex" in command
        assert "--cov-fail-under=0" in command

    coverage_step = next(
        step
        for step in workflow["jobs"]["coverage"]["steps"]
        if step.get("name") == "Combine coverage reports"
    )

    assert "uv run coverage report" in coverage_step["run"]


CAP_OVERRIDES = ("--cov-fail-under", "--no-cov", "-o addopts", "--override-ini")


def coverage_cap_violations(workflow: dict, pyproject: dict) -> list[str]:
    """Return why CI would not fail below the coverage cap, if it would not."""
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    addopts = " ".join(addopts) if isinstance(addopts, list) else addopts
    caps = [int(cap) for cap in re.findall(r"--cov-fail-under[= ](\d+)", addopts)]
    report_cap = pyproject["tool"]["coverage"]["report"].get("fail_under")
    unit_job = workflow["jobs"]["unit_tests"]
    unit_command = next(
        step["run"]
        for step in unit_job["steps"]
        if step.get("name") == "Run unit tests with coverage"
    )
    coverage_job = workflow["jobs"]["coverage"]
    combine_command = next(
        step["run"]
        for step in coverage_job["steps"]
        if step.get("name") == "Combine coverage reports"
    )

    problems = []
    if not caps or min(caps) < 80:
        problems.append(f"pytest addopts cap is {caps}, not at least 80")
    if report_cap is None or float(report_cap) < 80:
        problems.append(f"[tool.coverage.report] fail_under is {report_cap}, not at least 80")
    if not {"push", "pull_request"} <= set(workflow["on"]):
        problems.append(f"CI runs on {sorted(workflow['on'])}, not on both push and pull_request")
    problems += [
        f"job {name} only runs when {job['if']}"
        for name, job in (("unit_tests", unit_job), ("coverage", coverage_job))
        if "if" in job
    ]
    problems += [
        f"the unit test command overrides the cap with {override}"
        for override in CAP_OVERRIDES
        if override in unit_command
    ]
    if "uv run coverage report" not in combine_command or "--fail-under=0" in combine_command:
        problems.append("the combined coverage job does not run coverage report against fail_under")
    return problems


def test_ci_fails_below_the_coverage_cap() -> None:
    """Unit tests and the combined report both fail below pyproject's floor on every change."""
    pyproject = _load_pyproject()

    assert coverage_cap_violations(_load_yaml(".github/workflows/ci.yml"), pyproject) == []


def test_ci_uploads_no_coverage_to_codecov() -> None:
    """coverage.py in CI is the coverage gate; no workflow uploads to Codecov."""
    uses = [
        str(step.get("uses", ""))
        for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
        for job in _load_yaml(str(path.relative_to(REPO_ROOT))).get("jobs", {}).values()
        for step in job.get("steps", [])
    ]

    assert any(action.startswith("actions/checkout@") for action in uses)
    assert [action for action in uses if action.startswith("codecov/")] == []


def test_ci_coverage_artifacts_include_hidden_coverage_data_files() -> None:
    """Coverage aggregation needs the raw hidden .coverage files, not just XML reports."""
    workflow = _load_yaml(".github/workflows/ci.yml")

    for job_name in ("unit_tests", "integration_tests", "e2e_tests"):
        upload_step = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("name") == "Upload test results"
        )

        assert ".coverage" in upload_step["with"]["path"]
        assert upload_step["with"]["include-hidden-files"] == "true"


def test_build_verification_matches_compatibility_matrix_and_install_smoke_policy() -> None:
    """Build verification should cover the reviewed compatibility matrix and a clean import contract."""
    policy = _ci_policy()
    workflow = _load_yaml(".github/workflows/build-verification.yml")
    job = workflow["jobs"]["build"]
    smoke_step = next(
        step for step in job["steps"] if step.get("name") == "Run install smoke contract"
    )
    smoke_command = smoke_step["run"]

    assert job["strategy"]["matrix"]["python-version"] == policy["compatibility_python"]
    assert job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-14"]
    assert f"import {policy['smoke_package']}" in smoke_command
    assert "dir(artifex)" in smoke_command
    assert "'generative_models'" in smoke_command or '"generative_models"' in smoke_command
    assert "Successfully imported artifex" not in smoke_command
    assert "uv run pyright" not in _read(".github/workflows/build-verification.yml")


def test_security_workflow_reads_reviewed_ignores_from_pyproject_policy() -> None:
    """Security suppressions should come from reviewed policy, not inline workflow literals."""
    policy = _ci_policy()
    contents = _read(".github/workflows/security.yml")

    assert "tomllib" in contents
    assert "reviewed_ignores" in contents
    assert 'get("reviewed_ignores", [])' in contents
    assert "uv pip install pip-audit bandit" not in contents
    assert "uv run --with pip-audit" not in contents
    assert "uv run --with bandit" not in contents
    assert "uv run --locked pip-audit" in contents
    assert "uv run --locked bandit" in contents
    assert "uv run --locked detect-secrets scan --baseline .secrets.baseline" in contents

    for entry in policy["security"]["reviewed_ignores"]:
        assert entry["id"] not in contents


def test_root_readme_claims_match_the_reviewed_ci_policy() -> None:
    """README claims should reflect the current enforced typing and testing contract."""
    readme = _read("README.md")
    readme_lower = readme.lower()

    required_references = [
        "Pyright standard-mode checks block on the whole source tree",
        "blocking CI enforces repository contracts",
        "80% repo-wide coverage floor",
        "Security workflow checks are blocking",
    ]
    for reference in required_references:
        assert reference.lower() in readme_lower

    banned_references = [
        "complete type annotations",
        "full type annotations",
        "Well Tested",
        "70% repo-wide coverage floor",
        "coverage targets at 80% for new code",
        "security workflows remain reviewed but informational",
    ]
    for reference in banned_references:
        assert reference not in readme
