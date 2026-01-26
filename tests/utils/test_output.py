"""Test output utilities for standardized test result handling.

This module provides functions to create, format, and manage test output artifacts
in a consistent manner across different test types.
"""

import datetime
import json
import os
from typing import Any


# Default location for test artifacts
DEFAULT_ARTIFACTS_DIR = "test_artifacts"


class WSTestResult:
    """Class representing a test result with standardized output format."""

    def __init__(
        self,
        name: str,
        path: str,
        status: str,
        execution_time: float,
        test_type: str | None = None,
        error_message: str | None = None,
        stack_trace: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a TestResult object.

        Args:
            name: Name of the test
            path: Path to the test file
            status: Test status (pass, fail, error, skip)
            execution_time: Test execution time in seconds
            test_type: Type of the test (unit, integration, etc.)
            error_message: Error message if the test failed
            stack_trace: Stack trace if the test failed
            metadata: Additional metadata for the test
        """
        self.name = name
        self.path = path
        self.status = status
        self.execution_time = execution_time
        self.test_type = test_type
        self.error_message = error_message
        self.stack_trace = stack_trace
        self.metadata = metadata or {}
        self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert the test result to a dictionary.

        Returns:
            Dictionary representation of the test result
        """
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status,
            "execution_time": self.execution_time,
            "test_type": self.test_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WSTestResult":
        """Create a WSTestResult object from a dictionary.

        Args:
            data: Dictionary containing test result data

        Returns:
            TestResult object
        """
        result = cls(
            name=data["name"],
            path=data["path"],
            status=data["status"],
            execution_time=data["execution_time"],
            test_type=data.get("test_type"),
            error_message=data.get("error_message"),
            stack_trace=data.get("stack_trace"),
            metadata=data.get("metadata", {}),
        )
        result.timestamp = data.get("timestamp", result.timestamp)
        return result


class WSArtifactManager:
    """Manager for test artifacts."""

    def __init__(self, base_dir: str = DEFAULT_ARTIFACTS_DIR):
        """Initialize a TestArtifactManager.

        Args:
            base_dir: Base directory for test artifacts
        """
        self.base_dir = base_dir
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensure that the artifacts directory exists."""
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_artifact_path(
        self, test_type: str | None, module_name: str, artifact_name: str
    ) -> str:
        """Get the path for a test artifact.

        Args:
            test_type: Type of the test (unit, integration, etc.)
            module_name: Name of the module being tested
            artifact_name: Name of the artifact

        Returns:
            Path to the artifact
        """
        # Create directory path with test type and module name
        dir_parts = [self.base_dir]
        if test_type:
            dir_parts.append(test_type)

        # Convert module name to directory structure
        module_parts = module_name.split(".")
        dir_parts.extend(module_parts)

        # Create the directory
        artifact_dir = os.path.join(*dir_parts)
        os.makedirs(artifact_dir, exist_ok=True)

        # Return the full path to the artifact
        return os.path.join(artifact_dir, artifact_name)

    def save_artifact(
        self,
        content: str | bytes | dict[str, Any] | list[Any],
        artifact_name: str,
        test_type: str | None = None,
        module_name: str = "general",
        timestamp: bool = True,
    ) -> str:
        """Save a test artifact to the artifacts directory.

        Args:
            content: Content of the artifact
            artifact_name: Name of the artifact
            test_type: Type of the test (unit, integration, etc.)
            module_name: Name of the module being tested
            timestamp: Whether to add a timestamp to the artifact name

        Returns:
            Path to the saved artifact
        """
        # Add timestamp to artifact name if requested
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name_parts = artifact_name.split(".")
            if len(name_parts) > 1:
                artifact_name = f"{name_parts[0]}_{timestamp_str}.{'.'.join(name_parts[1:])}"
            else:
                artifact_name = f"{artifact_name}_{timestamp_str}"

        # Get the path for the artifact
        artifact_path = self._get_artifact_path(test_type, module_name, artifact_name)

        # Save the artifact based on its type
        if isinstance(content, (str, bytes)):
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(artifact_path, mode) as f:
                f.write(content)
        elif isinstance(content, (dict, list)):
            with open(artifact_path, "w") as f:
                json.dump(content, f, indent=2)
        else:
            raise TypeError(f"Unsupported content type: {type(content)}")

        return artifact_path

    def load_artifact(
        self,
        artifact_name: str,
        test_type: str | None = None,
        module_name: str = "general",
    ) -> Any:
        """Load a test artifact from the artifacts directory.

        Args:
            artifact_name: Name of the artifact
            test_type: Type of the test (unit, integration, etc.)
            module_name: Name of the module being tested

        Returns:
            Content of the artifact
        """
        # Get the path for the artifact
        artifact_path = self._get_artifact_path(test_type, module_name, artifact_name)

        # Check if the artifact exists
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        # Load the artifact based on its extension
        if artifact_path.endswith(".json"):
            with open(artifact_path, "r") as f:
                return json.load(f)
        else:
            # Try to load as text, fall back to binary if it fails
            try:
                with open(artifact_path, "r") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(artifact_path, "rb") as f:
                    return f.read()

    def list_artifacts(
        self, test_type: str | None = None, module_name: str | None = None
    ) -> list[str]:
        """list all artifacts in the artifacts directory.

        Args:
            test_type: Type of the test to filter by
            module_name: Name of the module to filter by

        Returns:
            list of artifact paths
        """
        # Get the base directory for listing
        base_path = self.base_dir
        if test_type:
            base_path = os.path.join(base_path, test_type)

        # list all files in the directory recursively
        artifacts = []
        for root, _, files in os.walk(base_path):
            for file in files:
                artifact_path = os.path.join(root, file)
                # If module_name is specified, check if it's in the path
                if module_name:
                    # Convert module name to path format for comparison
                    module_path_part = module_name.replace(".", os.path.sep)
                    if module_path_part in artifact_path:
                        artifacts.append(artifact_path)
                else:
                    artifacts.append(artifact_path)

        return artifacts


class WSSummaryGenerator:
    """Generator for test summary reports."""

    def __init__(self, results: list[WSTestResult], artifact_manager: WSArtifactManager):
        """Initialize a TestSummaryGenerator.

        Args:
            results: list of test results
            artifact_manager: Test artifact manager
        """
        self.results = results
        self.artifact_manager = artifact_manager

    def _calculate_statistics(self) -> dict[str, Any]:
        """Calculate statistics from test results.

        Returns:
            Dictionary containing statistics
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        errors = sum(1 for r in self.results if r.status == "error")
        skipped = sum(1 for r in self.results if r.status == "skip")

        # Calculate pass rate
        pass_rate = (passed / total) * 100 if total > 0 else 0

        # Calculate execution time statistics
        execution_times = [r.execution_time for r in self.results]
        avg_execution_time = sum(execution_times) / total if total > 0 else 0
        max_execution_time = max(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0

        # Group by test type
        by_type: dict[str, dict[str, Any]] = {}
        for result in self.results:
            test_type = result.test_type or "unknown"
            by_type.setdefault(
                test_type,
                {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0},
            )
            by_type[test_type]["total"] += 1
            if result.status == "pass":
                by_type[test_type]["passed"] += 1
            elif result.status == "fail":
                by_type[test_type]["failed"] += 1
            elif result.status == "error":
                by_type[test_type]["errors"] += 1
            elif result.status == "skip":
                by_type[test_type]["skipped"] += 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "pass_rate": pass_rate,
            "avg_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            "min_execution_time": min_execution_time,
            "by_type": by_type,
        }

    def generate_text_summary(self) -> str:
        """Generate a text summary of the test results.

        Returns:
            Text summary
        """
        stats = self._calculate_statistics()

        # Format the summary
        summary = [
            "Test Summary",
            "============",
            f"Total tests: {stats['total']}",
            f"Passed: {stats['passed']} ({stats['pass_rate']:.2f}%)",
            f"Failed: {stats['failed']}",
            f"Errors: {stats['errors']}",
            f"Skipped: {stats['skipped']}",
            "",
            "Execution Time",
            "==============",
            f"Average: {stats['avg_execution_time']:.4f}s",
            f"Maximum: {stats['max_execution_time']:.4f}s",
            f"Minimum: {stats['min_execution_time']:.4f}s",
            "",
            "Test Types",
            "==========",
        ]

        # Add test type statistics
        for test_type, type_stats in stats["by_type"].items():
            pass_rate = (
                (type_stats["passed"] / type_stats["total"]) * 100 if type_stats["total"] > 0 else 0
            )
            summary.append(
                f"{test_type.capitalize()}: {type_stats['total']} tests, "
                f"{type_stats['passed']} passed ({pass_rate:.2f}%)"
            )

        # Add failing tests section
        if stats["failed"] > 0 or stats["errors"] > 0:
            summary.extend(["", "Failing Tests", "============="])
            for result in self.results:
                if result.status in ["fail", "error"]:
                    summary.append(f"{result.name} ({result.path}): {result.status.upper()}")
                    if result.error_message:
                        summary.append(f"  Error: {result.error_message}")

        return "\n".join(summary)

    def generate_json_summary(self) -> dict[str, Any]:
        """Generate a JSON summary of the test results.

        Returns:
            JSON summary as a dictionary
        """
        stats = self._calculate_statistics()
        results = [result.to_dict() for result in self.results]

        return {
            "statistics": stats,
            "results": results,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def generate_html_summary(self) -> str:
        """Generate an HTML summary of the test results.

        Returns:
            HTML summary
        """
        stats = self._calculate_statistics()

        # Generate the HTML summary
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <title>Test Summary</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 20px; }",
            "    h1, h2 { color: #333; }",
            "    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background-color: #f2f2f2; }",
            "    tr:nth-child(even) { background-color: #f9f9f9; }",
            "    .pass { color: green; }",
            "    .fail { color: red; }",
            "    .error { color: red; font-weight: bold; }",
            "    .skip { color: orange; }",
            "    .summary-box { border: 1px solid #ddd; padding: 15px; "
            "margin-bottom: 20px; background-color: #f9f9f9; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Test Summary</h1>",
            "  <div class='summary-box'>",
            f"    <p><strong>Total tests:</strong> {stats['total']}</p>",
            f"    <p><strong>Passed:</strong> {stats['passed']} ({stats['pass_rate']:.2f}%)</p>",
            f"    <p><strong>Failed:</strong> {stats['failed']}</p>",
            f"    <p><strong>Errors:</strong> {stats['errors']}</p>",
            f"    <p><strong>Skipped:</strong> {stats['skipped']}</p>",
            "  </div>",
            "",
            "  <h2>Execution Time</h2>",
            "  <table>",
            "    <tr><th>Metric</th><th>Time (seconds)</th></tr>",
            f"    <tr><td>Average</td><td>{stats['avg_execution_time']:.4f}</td></tr>",
            f"    <tr><td>Maximum</td><td>{stats['max_execution_time']:.4f}</td></tr>",
            f"    <tr><td>Minimum</td><td>{stats['min_execution_time']:.4f}</td></tr>",
            "  </table>",
            "",
            "  <h2>Test Types</h2>",
            "  <table>",
            "    <tr><th>Type</th><th>Total</th><th>Passed</th><th>Failed</th>"
            "<th>Errors</th><th>Skipped</th><th>Pass Rate</th></tr>",
        ]

        # Add test type statistics
        for test_type, type_stats in stats["by_type"].items():
            pass_rate = (
                (type_stats["passed"] / type_stats["total"]) * 100 if type_stats["total"] > 0 else 0
            )
            html.append(
                f"    <tr>"
                f"<td>{test_type.capitalize()}</td>"
                f"<td>{type_stats['total']}</td>"
                f"<td>{type_stats['passed']}</td>"
                f"<td>{type_stats['failed']}</td>"
                f"<td>{type_stats['errors']}</td>"
                f"<td>{type_stats['skipped']}</td>"
                f"<td>{pass_rate:.2f}%</td>"
                f"</tr>"
            )

        html.append("  </table>")

        # Add results table
        html.extend(
            [
                "  <h2>Test Results</h2>",
                "  <table>",
                "    <tr><th>Test</th><th>Status</th><th>Execution Time</th><th>Type</th></tr>",
            ]
        )

        for result in self.results:
            status_class = {
                "pass": "pass",
                "fail": "fail",
                "error": "error",
                "skip": "skip",
            }.get(result.status, "")

            html.append(
                f"    <tr>"
                f"<td>{result.name}</td>"
                f"<td class='{status_class}'>{result.status.upper()}</td>"
                f"<td>{result.execution_time:.4f}s</td>"
                f"<td>{result.test_type or 'unknown'}</td>"
                f"</tr>"
            )

            # Add error message if available
            if result.error_message and result.status in ["fail", "error"]:
                html.append(
                    f"    <tr><td colspan='4' class='{status_class}'>"
                    f"<strong>Error:</strong> {result.error_message}</td></tr>"
                )

        html.extend(
            [
                "  </table>",
                "</body>",
                "</html>",
            ]
        )

        return "\n".join(html)

    def save_summaries(self, prefix: str = "test_summary") -> dict[str, str]:
        """Save test summaries in different formats.

        Args:
            prefix: Prefix for the summary files

        Returns:
            Dictionary with paths to the saved summary files
        """
        # Generate summaries
        text_summary = self.generate_text_summary()
        json_summary = self.generate_json_summary()
        html_summary = self.generate_html_summary()

        # Save summaries to files
        text_path = self.artifact_manager.save_artifact(
            text_summary, f"{prefix}.txt", module_name="summaries"
        )
        json_path = self.artifact_manager.save_artifact(
            json_summary, f"{prefix}.json", module_name="summaries"
        )
        html_path = self.artifact_manager.save_artifact(
            html_summary, f"{prefix}.html", module_name="summaries"
        )

        return {
            "text": text_path,
            "json": json_path,
            "html": html_path,
        }
