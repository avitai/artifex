"""CLI runner for benchmarks."""

import os
import sys

import typer

from artifex.benchmarks import (
    Benchmark,
    BenchmarkResult,
    DatasetProtocol,
    get_benchmark,
    list_benchmarks,
    ModelProtocol,
)
from artifex.benchmarks.datasets.dataset_loaders import load_dataset
from artifex.benchmarks.model_adapters import adapt_model
from artifex.benchmarks.suites.registry import get_suite, list_suites
from artifex.utils.file_utils import ensure_valid_output_path, get_valid_output_dir


app = typer.Typer(help="Benchmark runner for generative models")


def load_model(model_path: str) -> ModelProtocol:
    """Load a model from a file.

    Args:
        model_path: Path to the model file.

    Returns:
        The loaded model adapted to the ModelProtocol interface.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        ValueError: If the model can't be loaded or adapted.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # TODO: Implement actual model loading from files
    # For now, return a dummy model for testing
    class DummyModel:
        def __init__(self):
            self._model_name = os.path.basename(model_path).split(".")[0]

        @property
        def model_name(self) -> str:
            return self._model_name

        def predict(self, x, *, rngs=None):
            return x

        def sample(self, rng_key, batch_size=1, *, rngs=None):
            import jax.numpy as jnp

            return jnp.ones((batch_size, 10))

    # Adapt the model to the ModelProtocol interface
    return adapt_model(DummyModel())


def run_benchmark(
    benchmark: Benchmark,
    model: ModelProtocol,
    dataset: DatasetProtocol | None = None,
    output_path: str | None = None,
) -> BenchmarkResult:
    """Run a benchmark on a model.

    Args:
        benchmark: Benchmark to run.
        model: Model to benchmark.
        dataset: Dataset to use for benchmarking.
        output_path: Path to save the result to.

    Returns:
        The benchmark result.
    """
    # Run the benchmark
    result = benchmark.run(model=model, dataset=dataset)

    # Save the result if an output path is provided
    if output_path:
        # Ensure output path is within benchmark_results directory
        valid_output_path = ensure_valid_output_path(output_path, "benchmark_results")
        result.save(valid_output_path)

    return result


def run_benchmark_suite(
    benchmarks: list[Benchmark],
    model: ModelProtocol,
    dataset: DatasetProtocol | None = None,
    output_dir: str | None = None,
) -> list[BenchmarkResult]:
    """Run multiple benchmarks on a model.

    Args:
        benchmarks: List of benchmarks to run.
        model: Model to benchmark.
        dataset: Dataset to use for benchmarking.
        output_dir: Directory to save the results to.

    Returns:
        List of benchmark results.
    """
    results = []

    for benchmark in benchmarks:
        # Determine the output path
        output_path = None
        if output_dir:
            # Ensure output directory is within benchmark_results directory
            valid_output_dir = get_valid_output_dir(output_dir, "benchmark_results")
            output_path = os.path.join(valid_output_dir, f"{benchmark.config.name}.json")

        # Run the benchmark
        result = run_benchmark(
            benchmark=benchmark,
            model=model,
            dataset=dataset,
            output_path=output_path,
        )

        results.append(result)

    return results


@app.command("run")
def run_command(
    benchmark_name: str = typer.Option(
        ..., "--benchmark-name", "-b", help="Name of the benchmark to run"
    ),
    model_path: str = typer.Option(..., "--model-path", "-m", help="Path to the model file"),
    dataset_path: str | None = typer.Option(
        None, "--dataset-path", "-d", help="Path to the dataset file"
    ),
    output_path: str | None = typer.Option(
        None, "--output-path", "-o", help="Path to save the result to"
    ),
    list_benchmarks_flag: bool = typer.Option(
        False, "--list", "-l", help="List available benchmarks"
    ),
) -> None:
    """Run a benchmark on a model."""
    # List available benchmarks if requested
    if list_benchmarks_flag:
        benchmarks = list_benchmarks()
        typer.echo("Available benchmarks:")
        for name in benchmarks:
            typer.echo(f"  - {name}")
        return

    try:
        # Get the benchmark
        try:
            benchmark = get_benchmark(benchmark_name)
        except KeyError:
            typer.echo(f"Benchmark '{benchmark_name}' not found", err=True)
            available = list_benchmarks()
            if available:
                typer.echo("Available benchmarks:", err=True)
                for name in available:
                    typer.echo(f"  - {name}", err=True)
            sys.exit(1)

        # Load the model
        try:
            model = load_model(model_path)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading model: {e}", err=True)
            sys.exit(1)

        # Load the dataset if provided
        dataset = None
        if dataset_path:
            try:
                dataset = load_dataset(dataset_path)  # nosec B615
            except ValueError as e:
                typer.echo(f"Error loading dataset: {e}", err=True)
                sys.exit(1)

        # Run the benchmark
        typer.echo(f"Running benchmark: {benchmark_name}")
        result = run_benchmark(
            benchmark=benchmark,
            model=model,
            dataset=dataset,
            output_path=output_path,
        )

        # Print the result
        typer.echo("Benchmark completed successfully")
        typer.echo(f"Model: {result.model_name}")
        typer.echo("Metrics:")
        for name, value in result.metrics.items():
            typer.echo(f"  - {name}: {value:.4f}")

        if output_path:
            typer.echo(f"Result saved to: {output_path}")

    except Exception as e:
        typer.echo(f"Error running benchmark: {e}", err=True)
        sys.exit(1)


@app.command("suite")
def suite_command(
    suite_name: str = typer.Option(
        ..., "--suite-name", "-s", help="Name of the benchmark suite to run"
    ),
    model_path: str = typer.Option(..., "--model-path", "-m", help="Path to the model file"),
    dataset_path: str | None = typer.Option(
        None, "--dataset-path", "-d", help="Path to the dataset file"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", "-o", help="Directory to save the results to"
    ),
    list_suites_flag: bool = typer.Option(False, "--list", "-l", help="List available suites"),
) -> None:
    """Run a benchmark suite on a model."""
    # List available suites if requested
    if list_suites_flag:
        suites = list_suites()
        typer.echo("Available benchmark suites:")
        for name in suites:
            typer.echo(f"  - {name}")
        return

    try:
        # Get the suite
        try:
            benchmarks = get_suite(suite_name)
        except KeyError:
            typer.echo(f"Benchmark suite '{suite_name}' not found", err=True)
            available = list_suites()
            if available:
                typer.echo("Available benchmark suites:", err=True)
                for name in available:
                    typer.echo(f"  - {name}", err=True)
            sys.exit(1)

        # Load the model
        try:
            model = load_model(model_path)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error loading model: {e}", err=True)
            sys.exit(1)

        # Load the dataset if provided
        dataset = None
        if dataset_path:
            try:
                dataset = load_dataset(dataset_path)  # nosec B615
            except ValueError as e:
                typer.echo(f"Error loading dataset: {e}", err=True)
                sys.exit(1)

        # Run the benchmarks
        typer.echo(f"Running benchmark suite: {suite_name}")
        typer.echo(f"Number of benchmarks: {len(benchmarks)}")

        results = run_benchmark_suite(
            benchmarks=benchmarks,
            model=model,
            dataset=dataset,
            output_dir=output_dir,
        )

        # Print a summary of the results
        typer.echo("Benchmark suite completed successfully")
        typer.echo(f"Model: {model.model_name}")
        typer.echo("Results:")
        for result in results:
            typer.echo(f"  - {result.benchmark_name}:")
            for name, value in result.metrics.items():
                typer.echo(f"    - {name}: {value:.4f}")

        if output_dir:
            typer.echo(f"Results saved to: {output_dir}")

    except Exception as e:
        typer.echo(f"Error running benchmark suite: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
