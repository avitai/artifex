"""Pytest hooks for standardized test output."""

import os
import time

import pytest

from tests.utils.test_output import (
    WSArtifactManager,
    WSSummaryGenerator,
    WSTestResult,
)


# Global test results list
_test_results: list[WSTestResult] = []
_start_times: dict[str, float] = {}
_artifact_manager: WSArtifactManager | None = None


def _get_artifact_manager() -> WSArtifactManager:
    """Get or create a test artifact manager.

    Returns:
        Test artifact manager instance
    """
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = WSArtifactManager()
    return _artifact_manager


def _get_test_type(item) -> str | None:
    """Get the type of a test from its markers or path.

    Args:
        item: Pytest item

    Returns:
        Test type or None if not specified
    """
    # Check for markers first
    for marker_name in ["unit", "integration", "functional", "benchmark", "e2e"]:
        marker = item.get_closest_marker(marker_name)
        if marker:
            return marker_name

    # Check path
    path = str(item.fspath)
    if "unit" in path:
        return "unit"
    elif "integration" in path:
        return "integration"
    elif "functional" in path:
        return "functional"
    elif "benchmark" in path:
        return "benchmark"
    elif "e2e" in path or "end_to_end" in path:
        return "e2e"

    # Default to None if no type can be determined
    return None


def _get_module_name(item) -> str:
    """Get the module name from a pytest item.

    Args:
        item: Pytest item

    Returns:
        Module name
    """
    # Try to get the module name from the path
    path = str(item.fspath)
    if "tests/artifex/" in path:
        rel_path = path.split("tests/artifex/")[1]
        # Convert path to module name
        module_name = rel_path.replace("/", ".").replace(".py", "")
        return f"artifex.{module_name}"
    elif "tests/" in path:
        rel_path = path.split("tests/")[1]
        # Convert path to module name
        module_name = rel_path.replace("/", ".").replace(".py", "")
        return module_name
    else:
        # Fallback to the file name
        file_name = os.path.basename(path)
        return file_name.replace(".py", "")


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Configure pytest with our custom output plugin.

    Args:
        config: Pytest config
    """
    global _test_results
    _test_results = []

    # Create a terminal reporter that captures test session info
    reporter = config.pluginmanager.getplugin("terminalreporter")
    config._artifex_reporter = reporter


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item):
    """Set up a test run.

    Args:
        item: Pytest item
    """
    # Record the start time
    _start_times[item.nodeid] = time.time()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Clean up after a test run.

    Args:
        item: Pytest item
    """
    # If no start time was recorded, do nothing
    if item.nodeid not in _start_times:
        return


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Process a test report.

    Args:
        report: Pytest report
    """
    # Skip setup/teardown reports
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return

    # Get the execution time
    if report.nodeid in _start_times:
        start_time = _start_times[report.nodeid]
        execution_time = time.time() - start_time
    else:
        execution_time = getattr(report, "duration", 0)

    # Get test path and name
    path = report.fspath
    name = report.nodeid.split("::")[-1]

    # Determine test status
    if report.passed:
        status = "pass"
        error_message = None
    elif report.failed:
        status = "error" if report.when != "call" else "fail"
        error_message = str(report.longrepr) if report.longrepr else None
    elif report.skipped:
        status = "skip"
        # Handle different types of longrepr objects safely
        if report.longrepr:
            if isinstance(report.longrepr, tuple) and len(report.longrepr) > 2:
                error_message = report.longrepr[2]
            else:
                error_message = str(report.longrepr)
        else:
            error_message = None
    else:
        status = "unknown"
        error_message = None

    # Extract test type from markers or path
    test_type = None
    if hasattr(report, "item"):
        test_type = _get_test_type(report.item)
        module_name = _get_module_name(report.item)
    else:
        module_name = os.path.basename(path).replace(".py", "")

    # Create and append test result
    test_result = WSTestResult(
        name=name,
        path=str(path),
        status=status,
        execution_time=execution_time,
        test_type=test_type,
        error_message=error_message,
        metadata={"module": module_name},
    )
    _test_results.append(test_result)

    # Clean up the start time
    if report.nodeid in _start_times:
        del _start_times[report.nodeid]


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate a summary after all tests have run.

    Args:
        terminalreporter: Terminal reporter
        exitstatus: Exit status
        config: Pytest config
    """
    # Don't generate a summary if no tests were run
    if not _test_results:
        return

    # Create a test artifact manager and summary generator
    artifact_manager = _get_artifact_manager()
    summary_generator = WSSummaryGenerator(_test_results, artifact_manager)

    # Generate a summary
    summary = summary_generator.generate_text_summary()

    # Print the summary
    terminalreporter.write_sep("=", "Test Summary")
    terminalreporter.write_line(summary)

    # Save summary files
    paths = summary_generator.save_summaries()
    terminalreporter.write_sep("=", "Test Summary Files")
    terminalreporter.write_line(f"Text summary: {paths['text']}")
    terminalreporter.write_line(f"JSON summary: {paths['json']}")
    terminalreporter.write_line(f"HTML summary: {paths['html']}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Clean up after all tests have run.

    Args:
        session: Pytest session
        exitstatus: Exit status
    """
    # Clear the test results
    global _test_results
    _test_results.clear()
