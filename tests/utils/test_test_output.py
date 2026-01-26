"""Tests for the test output module."""

import json
import os
import tempfile

import pytest

from tests.utils.test_output import (
    WSArtifactManager,
    WSSummaryGenerator,
    WSTestResult,
)


@pytest.fixture
def test_result():
    """Create a test result for testing."""
    return WSTestResult(
        name="test_example",
        path="tests/example_test.py",
        status="pass",
        execution_time=0.123,
        test_type="unit",
        metadata={"module": "example"},
    )


@pytest.fixture
def test_results():
    """Create a list of test results for testing."""
    return [
        WSTestResult(
            name="test_passing",
            path="tests/test_1.py",
            status="pass",
            execution_time=0.1,
            test_type="unit",
        ),
        WSTestResult(
            name="test_failing",
            path="tests/test_2.py",
            status="fail",
            execution_time=0.2,
            test_type="unit",
            error_message="Assertion failed",
        ),
        WSTestResult(
            name="test_error",
            path="tests/test_3.py",
            status="error",
            execution_time=0.3,
            test_type="integration",
            error_message="Exception raised",
        ),
        WSTestResult(
            name="test_skipped",
            path="tests/test_4.py",
            status="skip",
            execution_time=0.05,
            test_type="unit",
        ),
    ]


@pytest.fixture
def artifact_manager():
    """Create a test artifact manager with a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield WSArtifactManager(base_dir=tmp_dir)


class TestWSTestResult:
    """Tests for the WSTestResult class."""

    def test_initialization(self, test_result):
        """Test that a TestResult can be initialized correctly."""
        assert test_result.name == "test_example"
        assert test_result.path == "tests/example_test.py"
        assert test_result.status == "pass"
        assert test_result.execution_time == 0.123
        assert test_result.test_type == "unit"
        assert test_result.metadata == {"module": "example"}
        assert test_result.error_message is None
        assert test_result.stack_trace is None
        assert test_result.timestamp is not None

    def test_to_dict(self, test_result):
        """Test converting a TestResult to a dictionary."""
        result_dict = test_result.to_dict()
        assert result_dict["name"] == "test_example"
        assert result_dict["path"] == "tests/example_test.py"
        assert result_dict["status"] == "pass"
        assert result_dict["execution_time"] == 0.123
        assert result_dict["test_type"] == "unit"
        assert result_dict["metadata"] == {"module": "example"}
        assert result_dict["error_message"] is None
        assert result_dict["stack_trace"] is None
        assert "timestamp" in result_dict

    def test_from_dict(self):
        """Test creating a TestResult from a dictionary."""
        data = {
            "name": "test_from_dict",
            "path": "tests/test_dict.py",
            "status": "fail",
            "execution_time": 0.456,
            "test_type": "integration",
            "error_message": "Test error",
            "stack_trace": "Traceback...",
            "metadata": {"key": "value"},
            "timestamp": "2023-01-01T00:00:00",
        }
        result = WSTestResult.from_dict(data)
        assert result.name == "test_from_dict"
        assert result.path == "tests/test_dict.py"
        assert result.status == "fail"
        assert result.execution_time == 0.456
        assert result.test_type == "integration"
        assert result.error_message == "Test error"
        assert result.stack_trace == "Traceback..."
        assert result.metadata == {"key": "value"}
        assert result.timestamp == "2023-01-01T00:00:00"


class TestWSArtifactManager:
    """Tests for the WSArtifactManager class."""

    def test_initialization(self, artifact_manager):
        """Test that a TestArtifactManager can be initialized correctly."""
        assert artifact_manager.base_dir is not None
        assert os.path.exists(artifact_manager.base_dir)

    def test_save_and_load_text_artifact(self, artifact_manager):
        """Test saving and loading a text artifact."""
        content = "This is a test artifact"
        path = artifact_manager.save_artifact(
            content,
            "test.txt",
            test_type="unit",
            module_name="example",
            timestamp=False,
        )
        assert os.path.exists(path)
        loaded = artifact_manager.load_artifact("test.txt", "unit", "example")
        assert loaded == content

    def test_save_and_load_json_artifact(self, artifact_manager):
        """Test saving and loading a JSON artifact."""
        content = {"key": "value", "list": [1, 2, 3]}
        path = artifact_manager.save_artifact(
            content,
            "test.json",
            test_type="unit",
            module_name="example",
            timestamp=False,
        )
        assert os.path.exists(path)
        loaded = artifact_manager.load_artifact("test.json", "unit", "example")
        assert loaded == content

    def test_timestamp_in_artifact_name(self, artifact_manager):
        """Test that timestamps are added to artifact names."""
        path = artifact_manager.save_artifact(
            "content",
            "test.txt",
            test_type="unit",
            module_name="example",
        )
        # Extract the filename from the path
        filename = os.path.basename(path)
        # Check that the filename contains a timestamp
        assert "_2" in filename  # This will match timestamps like _20230101_120000

    def test_listing_artifacts(self, artifact_manager):
        """Test listing artifacts."""
        # Create some artifacts
        artifact_manager.save_artifact(
            "content1",
            "test1.txt",
            test_type="unit",
            module_name="example",
            timestamp=False,
        )
        artifact_manager.save_artifact(
            "content2",
            "test2.txt",
            test_type="unit",
            module_name="example",
            timestamp=False,
        )
        artifact_manager.save_artifact(
            "content3",
            "test3.txt",
            test_type="integration",
            module_name="example",
            timestamp=False,
        )

        # List all artifacts
        all_artifacts = artifact_manager.list_artifacts()
        assert len(all_artifacts) == 3

        # List artifacts by test type
        unit_artifacts = artifact_manager.list_artifacts(test_type="unit")
        assert len(unit_artifacts) == 2
        integration_artifacts = artifact_manager.list_artifacts(test_type="integration")
        assert len(integration_artifacts) == 1

        # List artifacts by module name
        module_artifacts = artifact_manager.list_artifacts(module_name="example")
        assert len(module_artifacts) == 3  # All artifacts have "example" in their path

        # Test filtering by both test type and module name
        unit_example_artifacts = artifact_manager.list_artifacts(
            test_type="unit", module_name="example"
        )
        assert len(unit_example_artifacts) == 2

        # This test is just to show that filtering on a non-existent module returns an empty list
        different_artifacts = artifact_manager.list_artifacts(module_name="different")
        assert len(different_artifacts) == 0


class TestWSSummaryGenerator:
    """Tests for the WSSummaryGenerator class."""

    def test_initialize_summary_generator(self, test_results, artifact_manager):
        """Test that a WSSummaryGenerator can be initialized correctly."""
        generator = WSSummaryGenerator(test_results, artifact_manager)
        assert generator.results == test_results
        assert generator.artifact_manager == artifact_manager

    def test_calculate_statistics(self, test_results, artifact_manager):
        """Test calculating statistics from test results."""
        generator = WSSummaryGenerator(test_results, artifact_manager)
        stats = generator._calculate_statistics()

        assert stats["total"] == 4
        assert stats["passed"] == 1
        assert stats["failed"] == 1
        assert stats["errors"] == 1
        assert stats["skipped"] == 1
        assert stats["pass_rate"] == 25.0
        assert stats["avg_execution_time"] == 0.1625
        assert stats["max_execution_time"] == 0.3
        assert stats["min_execution_time"] == 0.05

        # Check test type statistics
        assert "unit" in stats["by_type"]
        assert "integration" in stats["by_type"]
        assert stats["by_type"]["unit"]["total"] == 3
        assert stats["by_type"]["unit"]["passed"] == 1
        assert stats["by_type"]["unit"]["failed"] == 1
        assert stats["by_type"]["unit"]["skipped"] == 1
        assert stats["by_type"]["integration"]["total"] == 1
        assert stats["by_type"]["integration"]["errors"] == 1

    def test_generate_text_summary(self, test_results, artifact_manager):
        """Test generating a text summary."""
        generator = WSSummaryGenerator(test_results, artifact_manager)
        summary = generator.generate_text_summary()

        # Check that the summary contains the expected information
        assert "Test Summary" in summary
        assert "Total tests: 4" in summary
        assert "Passed: 1 (25.00%)" in summary
        assert "Failed: 1" in summary
        assert "Errors: 1" in summary
        assert "Skipped: 1" in summary
        assert "Unit: 3 tests, 1 passed" in summary
        assert "Integration: 1 tests, 0 passed" in summary
        assert "Failing Tests" in summary
        assert "test_failing" in summary
        assert "test_error" in summary
        assert "Assertion failed" in summary
        assert "Exception raised" in summary

    def test_generate_json_summary(self, test_results, artifact_manager):
        """Test generating a JSON summary."""
        generator = WSSummaryGenerator(test_results, artifact_manager)
        summary = generator.generate_json_summary()

        # Check that the summary contains the expected information
        assert "statistics" in summary
        assert "results" in summary
        assert "timestamp" in summary
        assert summary["statistics"]["total"] == 4
        assert summary["statistics"]["pass_rate"] == 25.0
        assert len(summary["results"]) == 4
        assert summary["results"][0]["name"] == "test_passing"
        assert summary["results"][1]["name"] == "test_failing"
        assert summary["results"][1]["error_message"] == "Assertion failed"

    def test_generate_html_summary(self, test_results, artifact_manager):
        """Test generating an HTML summary."""
        generator = WSSummaryGenerator(test_results, artifact_manager)
        summary = generator.generate_html_summary()

        # Check that the summary contains the expected HTML elements
        assert "<!DOCTYPE html>" in summary
        assert "<html>" in summary
        assert "<title>Test Summary</title>" in summary
        assert "<style>" in summary
        assert "<table>" in summary
        assert "test_passing" in summary
        assert "test_failing" in summary
        assert "Assertion failed" in summary
        assert "PASS" in summary
        assert "FAIL" in summary
        assert "ERROR" in summary
        assert "SKIP" in summary

    def test_save_summaries(self, test_results, artifact_manager):
        """Test saving summaries in different formats."""
        generator = WSSummaryGenerator(test_results, artifact_manager)
        paths = generator.save_summaries(prefix="test")

        # Check that the expected files were created
        assert "text" in paths
        assert "json" in paths
        assert "html" in paths
        assert os.path.exists(paths["text"])
        assert os.path.exists(paths["json"])
        assert os.path.exists(paths["html"])

        # Verify the contents of the files
        with open(paths["json"], "r") as f:
            json_content = json.load(f)
            assert "statistics" in json_content
            assert "results" in json_content

        with open(paths["text"], "r") as f:
            text_content = f.read()
            assert "Test Summary" in text_content
            assert "Total tests: 4" in text_content
