from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_dead_tests_utils_helper_framework_is_removed() -> None:
    """The stale local test runner and env-gating helper surface should not remain."""
    removed_paths = [
        "tests/utils/README.md",
        "tests/utils/cli.py",
        "tests/utils/discovery.py",
        "tests/utils/test_discovery.py",
        "tests/utils/test_helpers.py",
        "tests/utils/test_test_helpers.py",
    ]

    for relative_path in removed_paths:
        assert not (REPO_ROOT / relative_path).exists()


def test_integration_modules_use_marker_contract_not_env_toggles() -> None:
    """Retained integration tests should use pytest markers instead of RUN_/SKIP_ toggles."""
    for relative_path in (
        "tests/artifex/generative_models/integration/test_flow_integration.py",
        "tests/artifex/generative_models/integration/test_gan_integration.py",
    ):
        contents = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "tests.utils.test_helpers" not in contents
        assert "should_run_" not in contents
        assert "get_mock_reason" not in contents
        assert "pytest.mark.integration" in contents


def test_testing_guidance_does_not_advertise_dead_runner_or_env_selection() -> None:
    """Contributor testing docs should stay on the direct pytest and marker contract."""
    docs_to_check = [
        "TESTING.md",
        "docs/development/testing.md",
        "README.md",
    ]

    banned_references = [
        "./scripts/run_tests.py",
        "RUN_INTEGRATION_TESTS",
        "SKIP_INTEGRATION_TESTS",
        "RUN_GAN_TESTS",
        "SKIP_GAN_TESTS",
        "RUN_FLOW_TESTS",
        "SKIP_FLOW_TESTS",
    ]

    for relative_path in docs_to_check:
        contents = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for banned_reference in banned_references:
            assert banned_reference not in contents

    assert "uv run pytest" in (REPO_ROOT / "TESTING.md").read_text(encoding="utf-8")
