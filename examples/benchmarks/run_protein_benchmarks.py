#!/usr/bin/env python
"""Run benchmarks on protein models and establish baseline metrics.

This script applies the benchmark suite to protein models and generates
visualization outputs.
"""

import argparse
import os

import flax.nnx as nnx
import jax
import matplotlib.pyplot as plt

from artifex.benchmarks.datasets.protein_dataset import (
    create_synthetic_protein_dataset,
)
from artifex.benchmarks.suites.protein_benchmarks import ProteinBenchmarkSuite
from artifex.generative_models.core.configuration import (
    PointCloudNetworkConfig,
    ProteinPointCloudConfig,
)
from artifex.generative_models.models.geometric.protein_point_cloud import ProteinPointCloudModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarks on protein generative models")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate for evaluation",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible results",
    )

    return parser.parse_args()


def create_test_model(config, seed=42):
    """Create a test protein model for benchmarking.

    Args:
        config: Model configuration dict with model parameters
        seed: Random seed

    Returns:
        Initialized protein model
    """
    key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(params=key, dropout=key, sample=key)

    # Create network config for the point cloud model
    network_config = PointCloudNetworkConfig(
        name="protein_network",
        hidden_dims=(config["embed_dim"], config["embed_dim"] * 2),
        activation="gelu",
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout_rate=config.get("dropout", 0.1),
    )

    # Create ProteinPointCloudConfig with proper nested config
    model_config = ProteinPointCloudConfig(
        name=f"protein_model_{config.get('model_variant', 'base')}",
        network=network_config,
        num_points=config["num_residues"] * config["num_atoms"],
        num_residues=config["num_residues"],
        num_atoms_per_residue=config["num_atoms"],
        backbone_indices=tuple(config["backbone_indices"]),
        use_constraints=config.get("use_constraints", True),
        dropout_rate=config.get("dropout", 0.1),
    )

    # Create protein point cloud model with proper config
    model = ProteinPointCloudModel(model_config, rngs=rngs)

    # Set model name for better reporting
    model.model_name = f"protein_model_{config.get('model_variant', 'base')}"

    return model


def main():
    """Run the protein model benchmarks."""
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up random seed
    seed = args.random_seed

    # Create synthetic protein dataset for testing
    # In a real scenario, this would be a real protein dataset
    from pathlib import Path

    from flax import nnx

    from artifex.generative_models.core.configuration import DataConfig
    from artifex.generative_models.core.device_manager import DeviceManager

    DeviceManager()
    rngs = nnx.Rngs(seed)

    data_config = DataConfig(
        name="synthetic_protein_data",
        dataset_name="synthetic_protein",
        data_dir=Path("./protein_data"),
        metadata={
            "num_samples": 500,
            "num_residues": 10,
            "num_atoms": 4,
            "seed": seed,
            "batch_size": 32,
        },
    )

    train_dataset = create_synthetic_protein_dataset(
        config=data_config, rngs=rngs, data_path="./protein_data"
    )

    # Create configurations for different model variants
    model_configs = [
        {
            "model_variant": "base",
            "num_residues": 10,
            "num_atoms": 4,
            "backbone_indices": [0, 1, 2, 3],
            "use_constraints": True,
            "embed_dim": 64,
            "num_layers": 3,
            "num_heads": 4,
            "dropout": 0.1,
        },
        {
            "model_variant": "large",
            "num_residues": 10,
            "num_atoms": 4,
            "backbone_indices": [0, 1, 2, 3],
            "use_constraints": True,
            "embed_dim": 128,
            "num_layers": 6,
            "num_heads": 8,
            "dropout": 0.1,
        },
    ]

    # Initialize the benchmark suite
    benchmark_suite = ProteinBenchmarkSuite(num_samples=args.num_samples, random_seed=seed)

    # Run benchmarks for each model configuration
    for config in model_configs:
        print(f"Running benchmarks for {config['model_variant']} model...")

        # Create model
        model = create_test_model(config, seed=seed)

        # Run all benchmarks
        results = benchmark_suite.run_all(model, train_dataset)

        # Print results
        for benchmark_name, result in results.items():
            print(f"\n{benchmark_name} metrics:")
            for metric_name, value in result.metrics.items():
                print(f"  {metric_name}: {value:.4f}")

            # Save results to file
            result_path = os.path.join(
                args.output_dir, f"{config['model_variant']}_{benchmark_name}_results.json"
            )
            result.save(result_path)
            print(f"Results saved to {result_path}")

    # Visualize results
    benchmark_suite.visualize_results()

    # Save visualization
    vis_path = os.path.join(args.output_dir, "protein_benchmark_results.png")
    plt.savefig(vis_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {vis_path}")

    # Create a markdown summary report
    summary_path = os.path.join(args.output_dir, "protein_benchmark_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Protein Model Benchmark Results\n\n")

        f.write("## Models Evaluated\n\n")
        for config in model_configs:
            f.write(f"- {config['model_variant']}: ")
            f.write(f"{config['num_layers']} layers, ")
            f.write(f"{config['embed_dim']} embedding dim, ")
            f.write(f"{config['num_heads']} attention heads\n")

        f.write("\n## Benchmark Results\n\n")

        for model_name in benchmark_suite.results:
            f.write(f"### {model_name}\n\n")

            for benchmark_name, result in benchmark_suite.results[model_name].items():
                f.write(f"#### {benchmark_name}\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for metric_name, value in result.metrics.items():
                    f.write(f"| {metric_name} | {value:.4f} |\n")

                f.write("\n")

        f.write("\n## Visualization\n\n")
        f.write("![Protein Benchmark Results](protein_benchmark_results.png)\n")

    print(f"Summary report saved to {summary_path}")
    print("Benchmark completed successfully")


if __name__ == "__main__":
    main()
