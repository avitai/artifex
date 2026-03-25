"""End-to-end tests for the retained CLI command catalog."""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run the Artifex CLI and capture the result."""
    return subprocess.run(
        [sys.executable, "-m", "artifex.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.mark.e2e
class TestCLIWorkflows:
    """End-to-end tests for the surviving CLI workflows."""

    def test_top_level_help_lists_only_config_command(self):
        """Top-level help should publish the retained config-only catalog."""
        result = _run_cli("--help")

        assert result.returncode == 0
        assert "config" in result.stdout

        for removed_command in (
            "train",
            "generate",
            "evaluate",
            "serve",
            "benchmark",
            "convert",
            "list-models",
            "validate-config",
            "generate-config",
            "run-benchmark",
        ):
            assert removed_command not in result.stdout

    def test_config_help_lists_retained_subcommands(self):
        """The config sub-app should expose the real supported workflows."""
        result = _run_cli("config", "--help")

        assert result.returncode == 0
        for subcommand in ("create", "validate", "show", "diff", "version", "list", "get"):
            assert subcommand in result.stdout

    def test_config_create_and_validate_workflow(self, temp_workspace):
        """A retained config workflow should execute end to end."""
        config_file = temp_workspace / "generated_config.yaml"

        create_result = _run_cli(
            "config",
            "create",
            "simple_training",
            str(config_file),
            "--param",
            "batch_size=4",
            "--param",
            "learning_rate=0.001",
        )

        assert create_result.returncode == 0, create_result.stderr
        assert config_file.exists()

        validate_result = _run_cli("config", "validate", str(config_file))

        assert validate_result.returncode == 0, validate_result.stderr
        assert "Configuration is valid" in validate_result.stdout
        assert "Typed document: TrainingConfig" in validate_result.stdout

    @pytest.mark.parametrize(
        "command",
        [
            "train",
            "generate",
            "evaluate",
            "serve",
            "benchmark",
            "convert",
            "list-models",
            "validate-config",
            "generate-config",
            "run-benchmark",
        ],
    )
    def test_removed_legacy_top_level_commands_fail_explicitly(self, command):
        """Removed top-level commands should fail instead of being silently skipped."""
        result = _run_cli(command)

        assert result.returncode != 0
        combined_output = result.stdout + result.stderr
        assert "No such command" in combined_output
