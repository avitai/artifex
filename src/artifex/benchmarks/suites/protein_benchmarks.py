"""Protein model benchmarks.

This module provides benchmark suites for protein generative models, including
quality metrics and performance metrics.
"""

from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from matplotlib.figure import Figure

from artifex.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
)
from artifex.benchmarks.metrics.precision_recall import PrecisionRecallBenchmark
from artifex.benchmarks.model_adapters import adapt_model
from artifex.generative_models.core.protocols.evaluation import (
    DatasetProtocol,
    ModelProtocol,
)
from artifex.utils.file_utils import ensure_valid_output_path


class ProteinStructureBenchmark(Benchmark):
    """Benchmark for evaluating protein structure models.

    This benchmark evaluates protein structure models based on geometric
    properties and structural quality metrics.
    """

    def __init__(self, num_samples: int = 100, random_seed: int | None = 42) -> None:
        """Initialize the protein structure benchmark.

        Args:
            num_samples: Number of samples to generate for evaluation.
            random_seed: Random seed for sampling and evaluation.
        """
        config = BenchmarkConfig(
            name="protein_structure_quality",
            description="Quality metrics for protein structure models",
            metric_names=[
                "bond_length_rmsd",
                "angle_rmsd",
                "clash_score",
                "precision",
                "recall",
                "f1_score",
            ],
        )
        super().__init__(config=config)

        self.num_samples = num_samples
        self.random_seed = random_seed

        # Initialize the precision-recall benchmark as a component
        self.pr_benchmark = PrecisionRecallBenchmark(
            num_clusters=10, num_samples=num_samples, random_seed=random_seed
        )

    def run(self, model: ModelProtocol, dataset: DatasetProtocol | None = None) -> BenchmarkResult:
        """Run the protein structure benchmark.

        Args:
            model: NNX model to benchmark.
            dataset: Dataset containing real protein structures.

        Returns:
            Benchmark result with structure quality metrics.
        """
        if dataset is None:
            raise ValueError("Dataset is required for protein structure benchmark")

        # Create proper RNG for sampling
        key = jax.random.PRNGKey(0)  # Default fallback
        if self.random_seed is not None:
            key = jax.random.PRNGKey(self.random_seed)

        # Get model name if available
        model_name = getattr(model, "model_name", None)
        if model_name is None:
            model_name = getattr(model.model, "model_name", "unknown")

        # Run precision-recall benchmark to evaluate distribution matching
        pr_result = self.pr_benchmark.run(model, dataset)

        # Extract precision, recall, and F1 scores
        precision = pr_result.metrics["precision"]
        recall = pr_result.metrics["recall"]
        f1_score = pr_result.metrics["f1_score"]

        # Calculate protein-specific metrics
        bond_length_rmsd, angle_rmsd, clash_score = self._calculate_structure_metrics(
            model, dataset, key
        )

        # Create metrics dictionary
        metrics = {
            "bond_length_rmsd": bond_length_rmsd,
            "angle_rmsd": angle_rmsd,
            "clash_score": clash_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        # Create result
        result = BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=model_name,
            metrics=metrics,
        )

        return result

    def _calculate_structure_metrics(
        self, model: ModelProtocol, dataset: DatasetProtocol, key: jax.Array
    ) -> tuple[float, float, float]:
        """Calculate protein structure quality metrics.

        Args:
            model: Model to evaluate.
            dataset: Dataset with real protein structures.
            key: JAX random key.

        Returns:
            tuple of (bond_length_rmsd, angle_rmsd, clash_score).
        """
        # Create Rngs for sampling
        rngs = nnx.Rngs(sample=key)

        # Generate samples (key is already in rngs, don't pass it twice)
        generated_samples = model.sample(batch_size=self.num_samples, rngs=rngs)

        # For now, calculate simplified structure metrics
        # In a real implementation, these would use proper biophysical
        # calculations

        # Simplified bond length RMSD (distance between adjacent points)
        # This is a placeholder for actual bond length calculations
        bond_length_rmsd = self._calculate_bond_length_rmsd(generated_samples)

        # Simplified angle RMSD (angles between triplets of points)
        # This is a placeholder for proper angle calculations
        angle_rmsd = self._calculate_angle_rmsd(generated_samples)

        # Simplified clash score (points too close together)
        # This is a placeholder for proper clash detection
        clash_score = self._calculate_clash_score(generated_samples)

        return bond_length_rmsd, angle_rmsd, clash_score

    def _calculate_bond_length_rmsd(self, samples: np.ndarray | jax.Array) -> float:
        """Calculate RMSD of bond lengths.

        Args:
            samples: Generated protein structures.

        Returns:
            RMSD of bond lengths compared to ideal values.
        """
        # In a real implementation, this would compare to ideal bond lengths
        # For now, we'll use a simplified metric based on consecutive points

        # Convert to numpy if needed
        if isinstance(samples, jax.Array):
            samples = np.array(samples)

        # Calculate distances between consecutive points
        if samples.ndim == 3:  # [batch, points, 3]
            diffs = samples[:, 1:] - samples[:, :-1]
            distances = np.sqrt(np.sum(diffs**2, axis=2))

            # Ideal C-alpha distance is around 3.8 Å
            ideal_distance = 3.8

            # RMSD from ideal
            rmsd = np.sqrt(np.mean((distances - ideal_distance) ** 2))
            return float(rmsd)
        else:
            # Return placeholder for other shapes
            return 1.0

    def _calculate_angle_rmsd(self, samples: np.ndarray | jax.Array) -> float:
        """Calculate RMSD of bond angles.

        Args:
            samples: Generated protein structures.

        Returns:
            RMSD of bond angles compared to ideal values.
        """
        # In a real implementation, this would calculate proper bond angles
        # For now, we'll use a simplified placeholder

        # Convert to numpy if needed
        if isinstance(samples, jax.Array):
            samples = np.array(samples)

        # For simplicity, return a constant value
        # This would be replaced with actual angle calculations
        return 15.0

    def _calculate_clash_score(self, samples: np.ndarray | jax.Array) -> float:
        """Calculate clash score.

        Args:
            samples: Generated protein structures.

        Returns:
            Clash score (lower is better).
        """
        # In a real implementation, this would detect atoms that are too close
        # For now, we'll use a simplified placeholder

        # Convert to numpy if needed
        if isinstance(samples, jax.Array):
            samples = np.array(samples)

        # For simplicity, return a constant value
        # This would be replaced with actual clash detection
        return 10.0


class ProteinBenchmarkSuite:
    """Suite of benchmarks for protein generative models.

    This class runs multiple benchmarks on protein models and provides
    visualization utilities.
    """

    def __init__(self, num_samples: int = 100, random_seed: int | None = 42) -> None:
        """Initialize the protein benchmark suite.

        Args:
            num_samples: Number of samples to generate for evaluation.
            random_seed: Random seed for deterministic evaluation.
        """
        self.num_samples = num_samples
        self.random_seed = random_seed

        # Create benchmarks
        self.benchmarks = [
            ProteinStructureBenchmark(num_samples=num_samples, random_seed=random_seed),
            PrecisionRecallBenchmark(
                num_clusters=10, num_samples=num_samples, random_seed=random_seed
            ),
        ]

        # Store results
        self.results: dict[str, BenchmarkResult | None] = {}

    def run_all(self, model: Any, dataset: DatasetProtocol) -> dict[str, BenchmarkResult | None]:
        """Run all benchmarks for a protein model.

        Args:
            model: Model to benchmark (will be adapted if needed).
            dataset: Dataset with real protein structures.

        Returns:
            dictionary of benchmark results.
        """
        # Adapt model if necessary
        if not isinstance(model, ModelProtocol):
            model = adapt_model(model)

        results: dict[str, BenchmarkResult | None] = {}
        for benchmark in self.benchmarks:
            result = benchmark.timed_run(model, dataset)
            results[benchmark.config.name] = result

        # Store results
        model_name = results[self.benchmarks[0].config.name].model_name
        self.results[model_name] = results

        return results

    def visualize_results(self, output_path: str | None = None) -> Figure:
        """Visualize benchmark results.

        Args:
            output_path: Optional path to save the figure.

        Returns:
            Matplotlib figure.
        """
        if not self.results:
            raise ValueError("No benchmark results available")

        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Extract model names and metrics
        model_names = list(self.results.keys())

        # Plot structure quality metrics
        quality_metrics = ["bond_length_rmsd", "angle_rmsd", "clash_score"]
        quality_values = []

        for model_name in model_names:
            # Get structure quality benchmark result
            struct_result = self.results[model_name].get("protein_structure_quality")
            if struct_result:
                model_values = [struct_result.metrics.get(m, 0) for m in quality_metrics]
                quality_values.append(model_values)

        # Only plot if we have structure quality results
        if quality_values:
            x = np.arange(len(quality_metrics))
            width = 0.8 / len(model_names)

            for i, (model_name, values) in enumerate(zip(model_names, quality_values)):
                axs[0].bar(x + i * width, values, width, label=model_name)

            axs[0].set_ylabel("Value")
            axs[0].set_title("Structure Quality Metrics")
            axs[0].set_xticks(x + width * (len(model_names) - 1) / 2)
            axs[0].set_xticklabels(quality_metrics)
            axs[0].legend()

        # Plot precision-recall metrics
        pr_metrics = ["precision", "recall", "f1_score"]
        pr_values = []

        for model_name in model_names:
            # Try to get PR metrics from either benchmark
            struct_result = self.results[model_name].get("protein_structure_quality")
            pr_result = self.results[model_name].get("precision_recall")

            if struct_result and all(m in struct_result.metrics for m in pr_metrics):
                model_values = [struct_result.metrics.get(m, 0) for m in pr_metrics]
                pr_values.append(model_values)
            elif pr_result:
                model_values = [pr_result.metrics.get(m, 0) for m in pr_metrics]
                pr_values.append(model_values)

        # Only plot if we have PR results
        if pr_values:
            x = np.arange(len(pr_metrics))
            width = 0.8 / len(model_names)

            for i, (model_name, values) in enumerate(zip(model_names, pr_values)):
                axs[1].bar(x + i * width, values, width, label=model_name)

            axs[1].set_ylabel("Value")
            axs[1].set_title("Precision-Recall Metrics")
            axs[1].set_xticks(x + width * (len(model_names) - 1) / 2)
            axs[1].set_xticklabels(pr_metrics)
            axs[1].legend()

        plt.tight_layout()

        # Save if path is provided
        if output_path:
            # Ensure the path is in the benchmark_results directory
            valid_path = ensure_valid_output_path(output_path, base_dir="benchmark_results")
            plt.savefig(valid_path, dpi=300, bbox_inches="tight")

        return fig
