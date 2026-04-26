from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from artifex.benchmarks.cli import benchmark_runner, optimization_benchmark


def test_benchmark_cli_lists_retained_benchmarks() -> None:
    """The retained benchmark CLI should expose registrations without running them."""
    result = CliRunner().invoke(benchmark_runner.app, ["run", "--list"])

    assert result.exit_code == 0
    assert "Retained benchmark registrations:" in result.output


def test_benchmark_cli_rejects_public_benchmark_execution() -> None:
    """The retained benchmark CLI should fail closed for execution paths."""
    result = CliRunner().invoke(benchmark_runner.app, ["run"])

    assert result.exit_code == 1
    assert benchmark_runner.UNSUPPORTED_BENCHMARK_CLI_MESSAGE in result.output


def test_benchmark_cli_lists_retained_suites() -> None:
    """The retained suite CLI should expose suite registrations without running them."""
    result = CliRunner().invoke(benchmark_runner.app, ["suite", "--list"])

    assert result.exit_code == 0
    assert "Retained benchmark suites:" in result.output


def test_benchmark_suite_cli_rejects_public_execution() -> None:
    """The retained suite CLI should fail closed for execution paths."""
    result = CliRunner().invoke(benchmark_runner.app, ["suite"])

    assert result.exit_code == 1
    assert benchmark_runner.UNSUPPORTED_BENCHMARK_CLI_MESSAGE in result.output


def test_optimization_benchmark_parser_has_retained_status_description() -> None:
    """The deprecated optimization benchmark helper should remain status-only."""
    parser = optimization_benchmark.create_parser()

    assert "Public CLI execution is not shipped" in parser.description


def test_optimization_benchmark_main_exits_with_status_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The optimization benchmark helper should reject execution from its main path."""
    monkeypatch.setattr(sys, "argv", ["optimization_benchmark"])

    with pytest.raises(SystemExit) as exc_info:
        optimization_benchmark.main()

    assert exc_info.value.code == 2
    assert (
        optimization_benchmark.UNSUPPORTED_OPTIMIZATION_BENCHMARK_MESSAGE in capsys.readouterr().err
    )
