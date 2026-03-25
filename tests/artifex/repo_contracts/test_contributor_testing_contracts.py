from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def normalized_text(path: Path) -> str:
    """Collapse insignificant whitespace so wording checks survive line wrapping."""
    return " ".join(path.read_text(encoding="utf-8").split())


def test_development_testing_guide_matches_live_pytest_contract() -> None:
    """The development testing guide should describe only the supported pytest surface."""
    contents = normalized_text(REPO_ROOT / "docs/development/testing.md")

    assert "uv run pytest" in contents
    assert "test_device" in contents
    assert "gpu_test_fixture" in contents
    assert "70%" in contents
    assert "80%" in contents
    assert "pytest.mark.gpu" in contents
    assert "pytest.mark.blackjax" in contents

    for banned in [
        "complete test coverage",
        "tests/standalone",
        "device_manager.cleanup()",
        "def test_with_device(device)",
        "'device' fixture",
    ]:
        assert banned not in contents


def test_contributor_guides_do_not_publish_shadow_test_topology() -> None:
    """Contributor-facing guides should not preserve the deleted standalone suite story."""
    philosophy = normalized_text(REPO_ROOT / "docs/development/philosophy.md")
    docs_contributing = normalized_text(REPO_ROOT / "docs/community/contributing.md")
    root_contributing = normalized_text(REPO_ROOT / "CONTRIBUTING.md")

    for contents in (philosophy, docs_contributing, root_contributing):
        assert "tests/standalone" not in contents
        assert "./test.py standalone" not in contents
        assert "./scripts/run_tests.sh --standalone" not in contents

    assert "tests/artifex/" in philosophy
    assert "tests/unit/" in philosophy
    assert "live Artifex owners" in philosophy

    assert "uv run pytest" in docs_contributing
    assert "uv run pytest" in root_contributing
    assert "tests/artifex/" in root_contributing
    assert "tests/unit/" in root_contributing
    assert "test_device" in root_contributing
    assert "gpu_test_fixture" in root_contributing


def test_standalone_shadow_suite_is_removed() -> None:
    """The dead local-replica standalone suite should not remain collected under tests/."""
    assert not (REPO_ROOT / "tests/standalone").exists()
