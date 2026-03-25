"""Retained benchmark CLI status helper."""

from __future__ import annotations

import typer

from artifex.benchmarks.registry import list_benchmarks
from artifex.benchmarks.suites.registry import list_suites


UNSUPPORTED_BENCHMARK_CLI_MESSAGE = (
    "There is no supported public benchmark CLI runner. Use the retained "
    "CalibraX-backed benchmark runtime from Python, or the explicit demo scripts "
    "under examples/ for demo-only workflows."
)

app = typer.Typer(
    help=(
        "Retained benchmark status helper. Public benchmark execution is not shipped from the CLI."
    )
)


def _fail_public_execution() -> None:
    typer.echo(UNSUPPORTED_BENCHMARK_CLI_MESSAGE, err=True)
    raise typer.Exit(code=1)


@app.command("run")
def run_command(
    benchmark_name: str = typer.Option(
        "", "--benchmark-name", "-b", help="Name of the benchmark to inspect"
    ),
    model_path: str = typer.Option("", "--model-path", "-m", help="Ignored status field"),
    dataset_path: str | None = typer.Option(
        None, "--dataset-path", "-d", help="Ignored status field"
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Ignored status field"
    ),
    list_benchmarks_flag: bool = typer.Option(
        False, "--list", "-l", help="List retained benchmark registrations"
    ),
) -> None:
    """Report the retained benchmark CLI status."""
    del benchmark_name, model_path, dataset_path, output_path

    if list_benchmarks_flag:
        benchmarks = list_benchmarks()
        typer.echo("Retained benchmark registrations:")
        for name in benchmarks:
            typer.echo(f"  - {name}")
        if not benchmarks:
            typer.echo("  (none registered)")
        return

    _fail_public_execution()


@app.command("suite")
def suite_command(
    suite_name: str = typer.Option("", "--suite-name", "-s", help="Name of the suite to inspect"),
    model_path: str = typer.Option("", "--model-path", "-m", help="Ignored status field"),
    dataset_path: str | None = typer.Option(
        None, "--dataset-path", "-d", help="Ignored status field"
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", "-o", help="Ignored status field"),
    list_suites_flag: bool = typer.Option(False, "--list", "-l", help="List retained suites"),
) -> None:
    """Report the retained benchmark-suite CLI status."""
    del suite_name, model_path, dataset_path, output_dir

    if list_suites_flag:
        suites = list_suites()
        typer.echo("Retained benchmark suites:")
        for name in suites:
            typer.echo(f"  - {name}")
        if not suites:
            typer.echo("  (none registered)")
        return

    _fail_public_execution()


if __name__ == "__main__":
    app()
