"""CLI entry point for artifex generative models."""

from __future__ import annotations

from typing import Optional

import typer

from artifex.cli.config import app as config_app


app = typer.Typer(
    name="artifex",
    help="Artifex generative models CLI.",
    no_args_is_help=True,
)

app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"artifex {version('avitai-artifex')}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Optional[bool] = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Artifex generative models CLI."""


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
