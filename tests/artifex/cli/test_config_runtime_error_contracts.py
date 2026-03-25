"""Runtime error-boundary tests for config CLI helpers."""

from __future__ import annotations

from types import SimpleNamespace

from artifex.generative_models.core.cli.config_commands import (
    show_config,
    validate_config_file,
)


def test_validate_config_file_returns_error_code_for_missing_config(capsys) -> None:
    """Validation should report a controlled loader failure instead of raising."""
    args = SimpleNamespace(config_file="missing-config.yaml")

    result = validate_config_file(args)
    captured = capsys.readouterr()

    assert result == 1
    assert "Configuration 'missing-config.yaml' could not be found" in captured.err


def test_show_config_returns_error_code_for_invalid_yaml(tmp_path, capsys) -> None:
    """Display should keep invalid YAML inside the repo-owned config error boundary."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("name: [unterminated\n", encoding="utf-8")
    args = SimpleNamespace(config_file=str(config_file), format="yaml")

    result = show_config(args)
    captured = capsys.readouterr()

    assert result == 1
    assert "Failed to load configuration" in captured.err
    assert "invalid.yaml" in captured.err
