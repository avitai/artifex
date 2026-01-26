"""Tests for the pytest hooks."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tests.utils.pytest_hooks import (
    _get_artifact_manager,
    _get_module_name,
    _get_test_type,
    pytest_configure,
    pytest_runtest_logreport,
    pytest_runtest_setup,
    pytest_sessionfinish,
    pytest_terminal_summary,
)
from tests.utils.test_output import WSArtifactManager, WSTestResult


@pytest.fixture
def mock_config():
    """Create a mock pytest config."""
    config = MagicMock()
    reporter = MagicMock()
    config.pluginmanager.getplugin.return_value = reporter
    return config


@pytest.fixture
def mock_item():
    """Create a mock pytest item."""
    item = MagicMock()
    item.nodeid = "tests/utils/test_example.py::test_function"
    item.fspath = "tests/utils/test_example.py"
    return item


@pytest.fixture
def mock_report():
    """Create a mock pytest report."""
    report = MagicMock()
    report.when = "call"
    report.nodeid = "tests/utils/test_example.py::test_function"
    report.fspath = "tests/utils/test_example.py"
    report.passed = True
    report.failed = False
    report.skipped = False
    return report


class TestPytestHooks:
    """Tests for the pytest hooks."""

    def test_get_artifact_manager(self):
        """Test getting the artifact manager."""
        with patch("tests.utils.pytest_hooks._artifact_manager", None):
            manager = _get_artifact_manager()
            assert isinstance(manager, WSArtifactManager)
            assert manager.base_dir == "test_artifacts"

    def test_get_test_type_from_marker(self, mock_item):
        """Test getting the test type from a marker."""
        # Set up a marker
        mock_item.get_closest_marker.return_value = MagicMock()

        # Test getting the type
        test_type = _get_test_type(mock_item)
        assert test_type == "unit"

        # Check that the correct marker was checked
        mock_item.get_closest_marker.assert_called_with("unit")

    def test_get_test_type_from_path(self, mock_item):
        """Test getting the test type from a path."""
        # Remove all markers
        mock_item.get_closest_marker.return_value = None

        # Test with different paths
        paths = {
            "tests/unit/test_example.py": "unit",
            "tests/integration/test_example.py": "integration",
            "tests/functional/test_example.py": "functional",
            "tests/benchmark/test_example.py": "benchmark",
            "tests/e2e/test_example.py": "e2e",
            "tests/end_to_end/test_example.py": "e2e",
            "tests/other/test_example.py": None,
        }

        for path, expected_type in paths.items():
            mock_item.fspath = path
            test_type = _get_test_type(mock_item)
            assert test_type == expected_type

    def test_get_module_name(self, mock_item):
        """Test getting the module name."""
        # Test with different paths
        paths = {
            "tests/artifex/generative_models/core/test_example.py": "artifex.generative_models.core.test_example",
            "tests/utils/test_example.py": "utils.test_example",
            "src/artifex/example.py": "example",
        }

        for path, expected_name in paths.items():
            mock_item.fspath = path
            module_name = _get_module_name(mock_item)
            assert module_name == expected_name

    def test_pytest_configure(self, mock_config):
        """Test pytest_configure."""
        # Run the hook
        pytest_configure(mock_config)

        # Check that the reporter was retrieved
        mock_config.pluginmanager.getplugin.assert_called_once_with("terminalreporter")
        assert hasattr(mock_config, "_artifex_reporter")

    def test_pytest_runtest_setup(self, mock_item):
        """Test pytest_runtest_setup."""
        start_times_dict = {}
        with patch("tests.utils.pytest_hooks._start_times", start_times_dict):
            # Run the hook
            pytest_runtest_setup(mock_item)

            # Check that the start time was recorded
            assert mock_item.nodeid in start_times_dict

    def test_pytest_runtest_logreport_pass(self, mock_report):
        """Test pytest_runtest_logreport with a passing test."""
        start_times_dict = {mock_report.nodeid: 100}
        with (
            patch("tests.utils.pytest_hooks._start_times", start_times_dict),
            patch("tests.utils.pytest_hooks._test_results", []) as mock_results,
            patch("time.time", return_value=100.5),
        ):
            # Run the hook
            pytest_runtest_logreport(mock_report)

            # Check that the test result was added
            assert len(mock_results) == 1
            result = mock_results[0]
            assert isinstance(result, WSTestResult)
            assert result.name == "test_function"
            assert result.path == mock_report.fspath
            assert result.status == "pass"
            assert result.execution_time == 0.5
            assert result.error_message is None

            # Check that the start time was cleaned up
            assert mock_report.nodeid not in start_times_dict

    def test_pytest_runtest_logreport_fail(self, mock_report):
        """Test pytest_runtest_logreport with a failing test."""
        # Set up a failing report
        mock_report.passed = False
        mock_report.failed = True
        mock_report.longrepr = "Test error message"

        start_times_dict = {mock_report.nodeid: 100}
        with (
            patch("tests.utils.pytest_hooks._start_times", start_times_dict),
            patch("tests.utils.pytest_hooks._test_results", []) as mock_results,
            patch("time.time", return_value=100.5),
        ):
            # Run the hook
            pytest_runtest_logreport(mock_report)

            # Check that the test result was added
            assert len(mock_results) == 1
            result = mock_results[0]
            assert result.status == "fail"
            assert result.error_message == "Test error message"

    def test_pytest_terminal_summary(self, mock_config):
        """Test pytest_terminal_summary."""
        # Create a mock test result
        test_result = WSTestResult(
            name="test_function",
            path="tests/utils/test_example.py",
            status="pass",
            execution_time=0.5,
        )

        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch("tests.utils.pytest_hooks._test_results", [test_result]),
            patch("tests.utils.pytest_hooks._get_artifact_manager") as mock_get_manager,
        ):
            # Create a mock artifact manager
            artifact_manager = WSArtifactManager(base_dir=tmp_dir)
            mock_get_manager.return_value = artifact_manager

            # Create mock summary paths
            paths = {
                "text": os.path.join(tmp_dir, "summary.txt"),
                "json": os.path.join(tmp_dir, "summary.json"),
                "html": os.path.join(tmp_dir, "summary.html"),
            }
            for path in paths.values():
                with open(path, "w") as f:
                    f.write("Test summary")

            # Mock the summary generator
            mock_generator = MagicMock()
            mock_generator.generate_text_summary.return_value = "Test summary"
            mock_generator.save_summaries.return_value = paths

            with patch("tests.utils.test_output.WSSummaryGenerator", return_value=mock_generator):
                # Run the hook
                pytest_terminal_summary(mock_config._artifex_reporter, 0, mock_config)

                # Check that the terminal reporter was used
                reporter = mock_config._artifex_reporter
                assert reporter.write_sep.call_count == 2
                assert reporter.write_line.call_count >= 4

    def test_pytest_sessionfinish(self):
        """Test pytest_sessionfinish."""
        with patch("tests.utils.pytest_hooks._test_results") as mock_results:
            # Set the mock results list
            mock_results.__len__.return_value = 2

            # Run the hook
            pytest_sessionfinish(None, 0)

            # Check that the test results were cleared
            mock_results.clear.assert_called_once()
