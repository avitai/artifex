"""Typer sub-app for configuration management commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import typer

from artifex.generative_models.core.cli.config_commands import (
    create_config,
    diff_config,
    get_config,
    list_configs,
    show_config,
    validate_config_file,
    version_config,
)


app = typer.Typer(help="Configuration management commands")


@app.command()
def create(
    template: str = typer.Argument(..., help="Supported typed template identifier"),
    output: Path = typer.Argument(..., help="Output configuration file path"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format (yaml or json)"),
    param: list[str] | None = typer.Option(
        None,
        "--param",
        "-p",
        help="Flat key=value template parameters",
    ),
) -> None:
    """Create a new configuration from a template."""
    args = SimpleNamespace(
        template=template,
        output=str(output),
        format=format,
        param=param or [],
    )
    code = create_config(args)
    raise typer.Exit(code=code)


@app.command()
def validate(
    config_file: Path = typer.Argument(..., help="Configuration file to validate"),
) -> None:
    """Validate a configuration file."""
    args = SimpleNamespace(config_file=str(config_file))
    code = validate_config_file(args)
    raise typer.Exit(code=code)


@app.command()
def show(
    config_file: Path = typer.Argument(..., help="Configuration file to display"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format (yaml or json)"),
) -> None:
    """Show a configuration in a readable format."""
    args = SimpleNamespace(config_file=str(config_file), format=format)
    code = show_config(args)
    raise typer.Exit(code=code)


@app.command()
def diff(
    config1: Path = typer.Argument(..., help="First configuration file"),
    config2: Path = typer.Argument(..., help="Second configuration file"),
) -> None:
    """Show differences between two configurations."""
    args = SimpleNamespace(config1=str(config1), config2=str(config2))
    code = diff_config(args)
    raise typer.Exit(code=code)


@app.command()
def version(
    config_file: Path = typer.Argument(..., help="Configuration file to version"),
    description: str = typer.Option("", "--description", "-d", help="Version description"),
    registry: str = typer.Option(
        "./config_registry", "--registry", "-r", help="Registry directory"
    ),
) -> None:
    """Version a configuration file."""
    args = SimpleNamespace(
        config_file=str(config_file),
        description=description,
        registry=registry,
    )
    code = version_config(args)
    raise typer.Exit(code=code)


@app.command("list")
def list_cmd(
    registry: str = typer.Option(
        "./config_registry", "--registry", "-r", help="Registry directory"
    ),
    limit: int = typer.Option(0, "--limit", "-n", help="Maximum entries to show (0=all)"),
) -> None:
    """List versioned configurations."""
    args = SimpleNamespace(registry=registry, limit=limit)
    code = list_configs(args)
    raise typer.Exit(code=code)


@app.command()
def get(
    version_or_hash: str = typer.Argument(..., help="Version ID or hash to retrieve"),
    registry: str = typer.Option(
        "./config_registry", "--registry", "-r", help="Registry directory"
    ),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format (yaml or json)"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Write to file instead of stdout"
    ),
) -> None:
    """Get a specific versioned configuration."""
    args = SimpleNamespace(
        version_or_hash=version_or_hash,
        version=version_or_hash,
        registry=registry,
        format=format,
        output=output,
    )
    code = get_config(args)
    raise typer.Exit(code=code)
