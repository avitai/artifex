from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Final

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ACTION = "./.github/actions/setup-artifex"

# Each fixable vulnerability the audit has reported, against the first release that
# resolves it.
SECURITY_PATCH_FLOORS: Final = {
    "aiohttp": (3, 14, 1),
    "bleach": (6, 4, 0),
    "cryptography": (48, 0, 1),
    "fastapi": (0, 139, 0),
    "gitpython": (3, 1, 50),
    "idna": (3, 15),
    "jupyter-server": (2, 20, 0),
    "jupyterlab": (4, 5, 9),
    "mako": (1, 3, 12),
    "mistune": (3, 3, 2),
    "mlflow": (3, 14, 0),
    "msgpack": (1, 2, 1),
    "nbconvert": (7, 17, 1),
    "notebook": (7, 6, 0),
    "pillow": (12, 3, 0),
    "pyasn1": (0, 6, 3),
    "pygments": (2, 20, 0),
    "pymdown-extensions": (10, 21, 3),
    "pytest": (9, 0, 3),
    "python-multipart": (0, 0, 31),
    "requests": (2, 33, 0),
    "starlette": (1, 3, 1),
    "tornado": (6, 5, 7),
    "urllib3": (2, 7, 0),
}


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
    """Fixable audited vulnerabilities should resolve to patched lockfile versions.

    A floor binds a package only while the lock holds it: a package that leaves the
    dependency closure carries no alert to fix. Its floor stays in the table so that
    re-entering the closure below the patched release is still a failure.
    """
    packages = {pkg["name"].lower(): pkg["version"] for pkg in _load_uv_lock()["package"]}

    unpatched = {
        name: packages[name]
        for name, floor in SECURITY_PATCH_FLOORS.items()
        if name in packages and _version_key(packages[name]) < floor
    }

    assert unpatched == {}


def test_policy_workflows_use_checked_in_setup_action_instead_of_inline_bootstrap() -> None:
    """Tracked CI workflows should reuse one setup owner instead of copy-pasting bootstrap."""
    policy = _ci_policy()

    for relative_path in policy["blocking_workflows"] + policy["informational_workflows"]:
        workflow = _load_yaml(relative_path)
        contents = _read(relative_path)

        assert "actions/setup-python@" not in contents
        assert "astral-sh/setup-uv@" not in contents
        assert "uv pip install -e" not in contents

        for name, job in workflow["jobs"].items():
            setup_steps = [step for step in job["steps"] if step.get("uses") == SETUP_ACTION]
            runs_uv = any("uv " in step.get("run", "") for step in job["steps"])
            # A job that needs no Python (the already-tested gate reads git and gh) sets up none.
            assert len(setup_steps) == (1 if runs_uv else 0), f"{relative_path}: {name}"


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
    # A merge whose tree its pull request already tested (and so already held to the cap)
    # may stand the combined report down; no other condition may.
    problems += [
        f"job {name} only runs when {job['if']}"
        for name, job in (("unit_tests", unit_job), ("coverage", coverage_job))
        if job.get("if", GATE_CONDITION) != GATE_CONDITION
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
    assert f"import {policy['smoke_package']}" in smoke_command
    assert "dir(artifex)" in smoke_command
    assert "'generative_models'" in smoke_command or '"generative_models"' in smoke_command
    assert "Successfully imported artifex" not in smoke_command
    assert "uv run pyright" not in _read(".github/workflows/build-verification.yml")


def _platform_matrix(expression: str) -> tuple[str, list[str], list[str]]:
    """Split a conditional runner matrix into its condition and its two platform lists."""
    match = re.fullmatch(
        r"\$\{\{ fromJSON\((?P<condition>.+?) && '(?P<when_true>\[.*?\])'"
        r" \|\| '(?P<when_false>\[.*?\])'\) \}\}",
        expression,
    )
    assert match is not None, expression
    return match["condition"], json.loads(match["when_true"]), json.loads(match["when_false"])


def test_macos_runners_join_the_platform_matrix_on_main_only() -> None:
    """macOS runners queue for hours; pushes to main measure both platforms, branches ubuntu only."""
    job = _load_yaml(".github/workflows/build-verification.yml")["jobs"]["build"]

    condition, on_main, elsewhere = _platform_matrix(job["strategy"]["matrix"]["os"])

    assert condition == "github.ref == 'refs/heads/main'"
    assert on_main == ["ubuntu-latest", "macos-14"]
    assert elsewhere == ["ubuntu-latest"]


def test_unit_tests_on_an_already_tested_tree_run_only_the_platform_no_pull_request_ran() -> None:
    """A merge of a tree its pull request tested repeats ubuntu; macOS it has never measured.

    Pull requests run ubuntu only, so the merge narrows to macOS rather than skipping, which
    also keeps the main-only performance job (it waits on the unit tests) running.
    """
    expression = _load_yaml(".github/workflows/ci.yml")["jobs"]["unit_tests"]["strategy"]["matrix"][
        "os"
    ]

    assert expression == (
        "${{ fromJSON(needs.already_tested.outputs.skip == 'true' && '[\"macos-14\"]'"
        " || (github.ref == 'refs/heads/main' && '[\"ubuntu-latest\", \"macos-14\"]'"
        " || '[\"ubuntu-latest\"]')) }}"
    )


GATE_JOB = "already_tested"
# The quality gate is not gated: every other job waits on it, including the main-only
# performance job, and GitHub skips a job whose dependency skipped.
UNGATED_JOBS = frozenset({GATE_JOB, "quality"})
GATE_CONDITION = f"needs.{GATE_JOB}.outputs.skip != 'true'"


def _ci_jobs() -> dict[str, dict]:
    return _load_yaml(".github/workflows/ci.yml")["jobs"]


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _runs_only_on_main(job: dict) -> bool:
    """Whether the job's own condition confines it to a push to main."""
    return "refs/heads/main" in str(job.get("if", ""))


def _consults_the_gate(job: dict) -> bool:
    return f"needs.{GATE_JOB}.outputs.skip" in yaml.safe_dump(job)


def _skipped_by_the_gate(job: dict) -> bool:
    return f"needs.{GATE_JOB}.outputs.skip" in str(job.get("if", ""))


def _transitive_needs(name: str, jobs: dict[str, dict]) -> set[str]:
    """Every job ``name`` waits on, directly or through another."""
    pending, seen = [name], set()
    while pending:
        for dependency in _needs(jobs.get(pending.pop(), {})):
            if dependency not in seen:
                seen.add(dependency)
                pending.append(dependency)
    return seen


def test_the_gate_reports_a_skip_only_for_a_push() -> None:
    """A manual run re-measures the tree on purpose and must not be skipped."""
    gate = _ci_jobs()[GATE_JOB]
    reported_by = gate["outputs"]["skip"]
    writers = [step for step in gate["steps"] if f"steps.{step.get('id')}.outputs" in reported_by]

    assert [step.get("if") for step in writers] == ["github.event_name == 'push'"]


def test_the_gate_skips_only_a_tree_its_pull_request_passed() -> None:
    """Skip needs the pull ref's tree to equal this tree and no failed check on that PR."""
    compare = next(step for step in _ci_jobs()[GATE_JOB]["steps"] if step.get("id") == "compare")
    script = compare["run"]

    assert 'git fetch --no-tags --depth=1 origin "refs/pull/$pull_request/head" || true' in script
    assert '[ "$tree" = "$tested" ] && [ "$failed" = "0" ]' in script
    assert "statusCheckRollup" in script
    assert compare["env"]["GH_TOKEN"] == "${{ github.token }}"


def test_a_job_that_repeats_the_pull_request_consults_the_gate() -> None:
    """Work a pull request already did over the same tree does not run again on the merge."""
    jobs = _ci_jobs()
    repeated = {
        name
        for name, job in jobs.items()
        if name not in UNGATED_JOBS and not _runs_only_on_main(job)
    }

    ungated = sorted(name for name in repeated if not _consults_the_gate(jobs[name]))

    assert repeated >= {"unit_tests", "integration_tests", "e2e_tests", "coverage"}
    assert ungated == [], f"these repeat the pull request without consulting the gate: {ungated}"


def test_nothing_a_main_only_job_waits_on_is_skipped_by_the_gate() -> None:
    """A job confined to main must not be skipped because something it needs was.

    The unit tests consult the gate only to narrow their platforms, so they still run and the
    performance job that waits on them still runs.
    """
    jobs = _ci_jobs()
    main_only = [name for name, job in jobs.items() if _runs_only_on_main(job)]

    assert main_only == ["performance_tests"]
    for name in main_only:
        skipped = sorted(
            dependency
            for dependency in _transitive_needs(name, jobs)
            if _skipped_by_the_gate(jobs[dependency])
        )
        assert skipped == [], f"{name} runs only on main but waits on gated {skipped}"


def test_an_unanswered_gate_leaves_the_work_running() -> None:
    """Where the gate answers nothing, every job runs as it would without it.

    The compare step does not run for a pull request or a manual run, and its lookups answer
    ``unknown`` rather than failing, so an empty output is ordinary. A consumer may only stand
    work down on ``'true'``.
    """
    for name, job in _ci_jobs().items():
        if name == GATE_JOB or not _consults_the_gate(job):
            continue
        assert GATE_JOB in _needs(job), name
        assert job.get("if") in (None, GATE_CONDITION), f"{name} runs only when {job.get('if')}"

        pattern = rf"needs\.{GATE_JOB}\.outputs\.skip\s*(==|!=)\s*'([a-z]+)'"
        compared = set(re.findall(pattern, yaml.safe_dump(job, width=10_000)))
        assert {value for _, value in compared} == {"true"}, f"{name} compares against {compared}"
