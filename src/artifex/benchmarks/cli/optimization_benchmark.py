"""Command-line interface for running optimization benchmarks."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from artifex.benchmarks.performance import (
    OptimizationBenchmark,
    TrainingConvergenceBenchmark,
)
from artifex.utils.file_utils import get_valid_output_dir


def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run optimization benchmarks on training pipelines"
    )

    parser.add_argument("--model", type=str, required=True, help="Model trainer to benchmark")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for training")

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training (default: 32)"
    )

    parser.add_argument(
        "--num-epochs", type=int, default=10, help="Number of epochs to train for (default: 10)"
    )

    parser.add_argument(
        "--target-loss",
        type=float,
        default=None,
        help="Target loss for convergence (default: None)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations (default: 1000)",
    )

    parser.add_argument(
        "--eval-frequency", type=int, default=10, help="Evaluation frequency (default: 10)"
    )

    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (default: 5)"
    )

    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (default: 42)")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results (default: benchmark_results)",
    )

    parser.add_argument(
        "--convergence-mode",
        action="store_true",
        help="Use convergence benchmark mode (requires --target-loss)",
    )

    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")

    return parser


def main():
    """Run the optimization benchmark CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Create output directory
    valid_output_dir = get_valid_output_dir(args.output_dir, "benchmark_results")
    output_dir = Path(valid_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = output_dir / f"optimization_benchmark_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    # This is just for demonstration purposes in the CLI tool

    # Load model and dataset (in a real CLI, these would be dynamically loaded)
    print(f"Loading model: {args.model}")
    print(f"Loading dataset: {args.dataset}")

    # For demonstration purposes, we're not implementing actual model loading
    # In a real CLI, you would load the model and dataset based on the arguments
    print("Note: This is a demonstration CLI. In a real implementation,")
    print("the model and dataset would be loaded from the provided arguments.")

    # Create the benchmark
    if args.convergence_mode:
        if args.target_loss is None:
            parser.error("Convergence mode requires --target-loss to be specified")

        print(f"Running TrainingConvergenceBenchmark with target loss: {args.target_loss}")
        benchmark = TrainingConvergenceBenchmark(
            target_loss=args.target_loss,
            max_iterations=args.max_iterations,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            random_seed=args.random_seed,
        )
    else:
        print("Running OptimizationBenchmark")
        benchmark = OptimizationBenchmark(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            random_seed=args.random_seed,
        )

    # Simulate running the benchmark
    print("Benchmark configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {args.num_epochs}")
    print(f"  Maximum iterations: {args.max_iterations}")
    print(f"  Evaluation frequency: {args.eval_frequency}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Random seed: {args.random_seed}")

    # In a real CLI, you would run the benchmark with the loaded model and dataset
    print("\nNote: Benchmark execution is simulated in this demonstration.")
    print("In a real implementation, the benchmark would be executed with:")
    print("  result = benchmark.run(model, dataset)")

    # Create a sample result for demonstration purposes
    sample_result = {
        "benchmark_name": benchmark.config.name,
        "model_name": args.model,
        "metrics": {
            "iterations_to_convergence": 100,
            "time_to_convergence": 20.5,
            "final_loss": 0.05,
            "training_throughput": 500.0,
        },
        "metadata": {
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "training_curve": [
                {
                    "iteration": i * args.eval_frequency,
                    "timestamp": i * 2.0,
                    "metrics": {"loss": 1.0 * (0.9**i), "accuracy": 1.0 - 1.0 * (0.9**i)},
                }
                for i in range(10)
            ],
            "total_iterations": 100,
            "total_time": 20.5,
            "examples_processed": 3200,
        },
    }

    # Save the result
    result_file = result_dir / "benchmark_result.json"
    with open(result_file, "w") as f:
        json.dump(sample_result, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    # Generate plots
    if not args.no_plots:
        try:
            # Create sample plots for demonstration
            plt.figure(figsize=(10, 6))
            plt.plot(
                [p["iteration"] for p in sample_result["metadata"]["training_curve"]],
                [p["metrics"]["loss"] for p in sample_result["metadata"]["training_curve"]],
                "o-",
                label="Loss",
            )
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.title(f"Training Curve for {args.model}")
            plt.grid(True)
            plt.savefig(result_dir / "training_curve.png")

            print(f"Plots saved to: {result_dir}")
        except Exception as e:
            print(f"Error generating plots: {e}")

    print("\nOptimization benchmark complete!")


if __name__ == "__main__":
    main()
