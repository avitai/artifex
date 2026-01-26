"""Tests for the test discovery module."""

import os
import tempfile
from pathlib import Path

import pytest

from tests.utils.discovery import (
    find_all_test_files,
    find_tests_by_feature,
    find_tests_by_module,
    find_tests_by_pattern,
    find_tests_by_type,
    get_pytest_args_for_tests,
)


@pytest.fixture
def test_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create some test files
        Path(tmp_dir, "test_basic.py").write_text("# Basic test")
        Path(tmp_dir, "test_util.py").write_text("# Utility test")
        Path(tmp_dir, "some_file.py").write_text("# Not a test")
        Path(tmp_dir, "__init__.py").write_text("")

        # Create a subdirectory with more test files
        sub_dir = Path(tmp_dir, "subdir")
        sub_dir.mkdir()
        Path(sub_dir, "__init__.py").write_text("")
        Path(sub_dir, "test_feature.py").write_text(
            "# Feature test\nfrom artifex.generative_models.core import utils"
        )
        Path(sub_dir, "test_unit.py").write_text("# Unit test")
        Path(sub_dir, "test_integration.py").write_text("# Integration test")

        yield tmp_dir


def test_find_all_test_files(test_dir):
    """Test finding all test files in a directory."""
    test_files = find_all_test_files(test_dir)
    assert len(test_files) == 5
    assert all(file.endswith(".py") for file in test_files)
    assert all(os.path.basename(file).startswith("test_") for file in test_files)
    assert not any("__init__.py" in file for file in test_files)


def test_find_tests_by_module(test_dir):
    """Test finding tests by module."""
    # Create a test file with imports
    feature_test = Path(test_dir, "subdir", "test_feature.py")
    feature_test.write_text(
        "from artifex.generative_models.core import utils\n"
        "import artifex.generative_models.core.layers"
    )

    # Test finding tests by module
    module_tests = find_tests_by_module("artifex.generative_models.core", test_dir)
    assert len(module_tests) == 1
    assert os.path.basename(module_tests[0]) == "test_feature.py"


def test_find_tests_by_pattern(test_dir):
    """Test finding tests by pattern."""
    pattern_tests = find_tests_by_pattern("util", test_dir)
    assert len(pattern_tests) == 1
    assert os.path.basename(pattern_tests[0]) == "test_util.py"


def test_find_tests_by_feature(test_dir):
    """Test finding tests by feature."""
    feature_tests = find_tests_by_feature("feature", test_dir)
    assert len(feature_tests) == 1
    assert os.path.basename(feature_tests[0]) == "test_feature.py"


def test_find_tests_by_type(test_dir):
    """Test finding tests by type."""
    # Test finding unit tests
    unit_tests = find_tests_by_type("unit", test_dir)
    assert len(unit_tests) == 1
    assert os.path.basename(unit_tests[0]) == "test_unit.py"

    # Test finding integration tests
    integration_tests = find_tests_by_type("integration", test_dir)
    assert len(integration_tests) == 1
    assert os.path.basename(integration_tests[0]) == "test_integration.py"

    # Test finding a non-existent type
    with pytest.raises(ValueError):
        find_tests_by_type("invalid_type", test_dir)


def test_get_pytest_args_for_tests():
    """Test generating pytest arguments."""
    # Test with default options
    test_files = ["test1.py", "test2.py"]
    args = get_pytest_args_for_tests(test_files)
    assert args == ["test1.py", "test2.py"]

    # Test with verbose option
    args = get_pytest_args_for_tests(test_files, verbose=True)
    assert args == ["-v", "test1.py", "test2.py"]

    # Test with no capture option
    args = get_pytest_args_for_tests(test_files, capture_output=False)
    assert args == ["-s", "test1.py", "test2.py"]

    # Test with both options
    args = get_pytest_args_for_tests(test_files, verbose=True, capture_output=False)
    assert args == ["-v", "-s", "test1.py", "test2.py"]
