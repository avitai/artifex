#!/usr/bin/env python
"""Demo of protein model benchmarks with NNX-compatible clustering.

This script demonstrates the protein benchmark suite with NNX models.
"""

import argparse
import os

import flax.nnx as nnx
import jax
import matplotlib.pyplot as plt

from artifex.benchmarks.datasets.protein_dataset import (
    create_synthetic_protein_dataset,
)
from artifex.benchmarks.model_adapters import adapt_model
from artifex.benchmarks.suites.protein_benchmarks import (
    ProteinBenchmarkSuite,
)


class MockProteinModel(nnx.Module):
    """Mock protein model for benchmark demonstration.

    This model returns random protein structures in 3D format.
    """

    def __init__(self, config, *, rngs=None):
        """Initialize the mock protein model.

        Args:
            config: Model configuration
            rngs: Random number generators
        """
        super().__init__()
        self.num_residues = config.get("num_residues", 10)
        self.num_atoms = config.get("num_atoms", 4)
        self.model_name = config.get("model_variant", "mock")

        # Store RNG key for reproducibility (wrapped for Flax NNX compatibility)
        init_key_value = jax.random.key(0)  # Default
        if rngs is not None and "params" in rngs:
            init_key_value = rngs.params()  # Use method call, not .key.value
        self.init_key = nnx.Variable(init_key_value)

    def sample(self, batch_size=1, *, rngs=None):
        """Generate protein structure samples.

        Args:
            batch_size: Number of samples to generate
            rngs: Random number generators

        Returns:
            Randomly generated protein structures of shape:
            [batch_size, num_residues, num_atoms, 3]
        """
        # Get RNG key
        key = jax.random.key(0)  # Default fallback
        if rngs is not None and "sample" in rngs:
            key = rngs.sample()  # Use method call, not .key.value
        elif hasattr(self, "init_key"):
            key = self.init_key.value  # Access wrapped value

        # Generate full 4D coordinates for protein structures
        # Shape: [batch_size, num_residues, num_atoms, 3]
        samples = jax.random.normal(key, shape=(batch_size, self.num_residues, self.num_atoms, 3))

        # Make the structures more protein-like
        samples = self._create_protein_like_coords(samples, key)

        return samples

    def __call__(self, x, *, rngs=None):
        """Forward pass through the model.

        Args:
            x: Input data of shape [batch, residues, atoms, 3]
            rngs: Random number generators

        Returns:
            Dictionary with model outputs.
        """
        # Get RNG key
        key = jax.random.key(0)  # Default fallback
        if rngs is not None and "sample" in rngs:
            key = rngs.sample()  # Use method call, not .key.value
        elif hasattr(self, "init_key"):
            key = self.init_key.value  # Access wrapped value

        # Add slight variations to the input
        # Preserving the original shape
        output = x + 0.1 * jax.random.normal(key, x.shape)

        return {"coordinates": output}

    def _create_protein_like_coords(self, coords, key):
        """Make random coordinates more protein-like.

        Args:
            coords: Random coordinates [batch, residues, atoms, 3]
            key: Random key

        Returns:
            Coordinates with protein-like structure
        """
        batch_size, num_residues, num_atoms, _ = coords.shape

        # Create a backbone-like helical structure
        t = jax.numpy.arange(num_residues) * 1.5
        backbone_x = 2.5 * jax.numpy.cos(t)
        backbone_y = 2.5 * jax.numpy.sin(t)
        backbone_z = t

        # Combine into backbone coordinates [num_residues, 3]
        backbone = jax.numpy.stack([backbone_x, backbone_y, backbone_z], axis=1)

        # Create a template for all samples
        template = jax.numpy.zeros_like(coords)

        # Add backbone structure
        for i in range(num_residues):
            for j in range(num_atoms):
                # Offset each atom based on its index
                offset = jax.numpy.array([j * 0.5, j * 0.1, j * 0.1])
                # Set position based on backbone + offset
                template = template.at[:, i, j].set(backbone[i] + offset)

        # Add random variations
        key1, key2 = jax.random.split(key)
        noise = 0.2 * jax.random.normal(key2, shape=coords.shape)

        return template + noise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarks on NNX protein models")

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


def main():
    """Run the benchmark demo."""
    # Parse command line arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up random seed
    seed = args.random_seed

    # Create dataset with protein coordinates
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
            "num_samples": 200,
            "num_residues": 10,
            "num_atoms": 4,
            "seed": seed,
            "batch_size": 32,
        },
    )

    train_dataset = create_synthetic_protein_dataset(
        config=data_config, rngs=rngs, data_path="./protein_data"
    )

    # Create model configurations for different variants
    model_configs = [
        {
            "model_variant": "small",
            "num_residues": 10,
            "num_atoms": 4,
        },
        {
            "model_variant": "medium",
            "num_residues": 10,
            "num_atoms": 4,
        },
    ]

    # Initialize the benchmark suite
    benchmark_suite = ProteinBenchmarkSuite(num_samples=args.num_samples, random_seed=seed)

    # Run benchmarks for each model configuration
    for config in model_configs:
        print(f"Running benchmarks for {config['model_variant']} model...")

        # Create mock model
        rngs = nnx.Rngs(params=jax.random.PRNGKey(seed))
        model = MockProteinModel(config, rngs=rngs)

        # Adapt the model for the benchmark system
        adapted_model = adapt_model(model)

        # Run all benchmarks
        results = benchmark_suite.run_all(adapted_model, train_dataset)

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
            f.write(f"{config['num_residues']} residues, ")
            f.write(f"{config['num_atoms']} atoms per residue\n")

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
        f.write("![Protein Benchmark Results](protein_benchmark_results.png)")

    print(f"Summary report saved to {summary_path}")
    print("Benchmark demo completed successfully")


if __name__ == "__main__":
    main()
