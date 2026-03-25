import re
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_obsolete_test_wrappers_are_not_exposed():
    """Ad hoc test wrappers should not remain in the supported repo surface."""
    assert not (REPO_ROOT / "scripts" / "blackjax_test_helper.py").exists()
    assert not (REPO_ROOT / "scripts" / "smart_test_runner.sh").exists()
    assert not (
        REPO_ROOT
        / "tests"
        / "artifex"
        / "generative_models"
        / "core"
        / "sampling"
        / "run_blackjax_tests.py"
    ).exists()

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project_scripts = pyproject["project"]["scripts"]
    assert "artifex-smart-test" not in project_scripts

    docs_to_check = [
        "README.md",
        "CONTRIBUTING.md",
        "TESTING.md",
        "docs/development/testing.md",
        "docs/development/blackjax-testing.md",
        "scripts/README.md",
        "tests/artifex/generative_models/core/sampling/README_BLACKJAX_TESTS.md",
    ]

    for relative_path in docs_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        assert "blackjax_test_helper.py" not in contents
        assert "smart_test_runner.sh" not in contents


def test_gpu_guidance_uses_supported_cli_flags():
    """GPU setup docs should reference the supported backend verifier CLI."""
    verifier_contents = (REPO_ROOT / "scripts/verify_gpu_setup.py").read_text()
    assert "--critical-only" not in verifier_contents
    assert "--configure-first" not in verifier_contents
    assert "--require-gpu" in verifier_contents
    assert "--json" in verifier_contents

    docs_contents = (REPO_ROOT / "src/artifex/generative_models/core/README.md").read_text()
    assert "JAX_PLATFORMS=cuda,cpu" not in docs_contents


def test_curated_docs_use_uv_run_for_gpu_verification() -> None:
    """Curated docs should use the supported uv-based GPU verifier command."""
    unsupported_invocation = re.compile(r"(?<!uv run )python scripts/verify_gpu_setup\.py\b")
    for path in REPO_ROOT.joinpath("docs").rglob("*.md"):
        contents = path.read_text()
        assert unsupported_invocation.search(contents) is None


def test_blackjax_is_not_env_gated():
    """BlackJAX is a first-class dependency and should not be hidden behind env toggles."""
    files_to_check = [
        "pyproject.toml",
        "tests/conftest.py",
        "tests/artifex/generative_models/core/sampling/__init__.py",
        "README.md",
        "TESTING.md",
        "docs/development/testing.md",
        "docs/development/blackjax-testing.md",
        "docs/getting-started/installation.md",
        "tests/artifex/generative_models/core/sampling/README.md",
        "tests/artifex/generative_models/core/sampling/README_BLACKJAX_TESTS.md",
    ]

    for relative_path in files_to_check:
        contents = (REPO_ROOT / relative_path).read_text()
        assert "ENABLE_BLACKJAX_TESTS" not in contents
        assert "SKIP_BLACKJAX_TESTS" not in contents


def test_stale_backend_entrypoints_are_not_exposed():
    """Broken setup/test entrypoints should not remain in the public package surface."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project_scripts = pyproject["project"]["scripts"]
    assert "artifex-cuda-test" not in project_scripts
    assert "artifex-fresh-setup" not in project_scripts
    assert "artifex-setup-env" not in project_scripts


def test_root_contributor_and_testing_guides_match_current_repo_contract():
    """Top-level contributor and testing guides should describe the current toolchain only."""
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text()
    testing = (REPO_ROOT / "TESTING.md").read_text()

    assert "88 characters (Black formatter)" not in contributing
    assert "100 characters (Ruff formatter)" in contributing

    banned_testing_references = [
        "git reset --hard",
        "blackjax_test_helper.py",
        "smart_test_runner.sh",
        "scripts/run_tests.sh",
        "ENABLE_BLACKJAX_TESTS",
        "SKIP_BLACKJAX_TESTS",
    ]
    for banned_reference in banned_testing_references:
        assert banned_reference not in testing

    required_testing_references = [
        "uv run pytest",
        "source ./activate.sh",
        "./setup.sh --force-clean",
        "uv run python scripts/verify_gpu_setup.py --require-gpu",
    ]
    for required_reference in required_testing_references:
        assert required_reference in testing


def test_scripts_readme_table_of_contents_uses_live_section_targets() -> None:
    """Scripts README navigation should point at headings that still exist."""
    contents = (REPO_ROOT / "scripts" / "README.md").read_text()

    assert "[Development Utilities](#development-utilities)" not in contents
    assert "[Script Guidelines](#-script-guidelines)" in contents
