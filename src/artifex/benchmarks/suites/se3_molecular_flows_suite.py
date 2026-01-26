"""SE(3)-Equivariant Molecular Flows benchmark suite."""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
from flax import nnx

from artifex.benchmarks.base import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
)
from artifex.benchmarks.datasets.qm9 import QM9Dataset
from artifex.benchmarks.metrics.molecular_flows import (
    MolecularFlowsMetrics,
)
from artifex.generative_models.core.configuration import (
    DataConfig,
    EvaluationConfig,
)
from artifex.generative_models.models.flow.se3_molecular import SE3MolecularFlow


class SE3MolecularFlowsBenchmark(Benchmark):
    """Benchmark for SE(3)-equivariant molecular flows using QM9 dataset.

    This benchmark evaluates molecular conformation generation using SE(3)-equivariant
    normalizing flows on the QM9 dataset, focusing on chemical validity, conformational
    diversity, and energy consistency.

    Target Performance:
    - Chemical validity: >95% on QM9 molecules
    - Conformational diversity: >0.8 (RMSD clustering)
    - Energy consistency: <0.1 kcal/mol RMSE
    - Training efficiency: Reasonable convergence on available hardware
    """

    def __init__(self, *, rngs: nnx.Rngs, config: EvaluationConfig):
        """Initialize SE(3) molecular flows benchmark.

        Args:
            config: Benchmark configuration (must be EvaluationConfig)
            rngs: Random number generators


        Raises:
            TypeError: If config is not EvaluationConfig
        """
        if not isinstance(config, EvaluationConfig):
            raise TypeError(f"config must be EvaluationConfig, got {type(config).__name__}")

        self.original_config = config

        # Extract performance targets from config metadata
        performance_targets = config.metadata.get(
            "performance_targets",
            {
                "chemical_validity": 0.95,
                "conformational_diversity": 0.8,
                "energy_consistency": 0.1,  # kcal/mol
                "training_time_per_epoch": 3600,  # seconds
            },
        )

        # Create benchmark configuration
        benchmark_config = BenchmarkConfig(
            name="se3_molecular_flows",
            description="SE(3)-equivariant molecular flows benchmark",
            metric_names=["chemical_validity", "conformational_diversity", "energy_consistency"],
            metadata={
                "dataset_name": "QM9",
                "model_type": "SE3MolecularFlow",
                "performance_targets": performance_targets,
            },
        )

        super().__init__(config=benchmark_config)

        self.eval_config = config
        self.rngs = rngs

        # Initialize dataset using DataConfig from metadata
        dataset_config = config.metadata.get("dataset_config")
        if dataset_config is None:
            raise ValueError("dataset_config must be provided in EvaluationConfig metadata")

        self.dataset = QM9Dataset(
            data_path=str(dataset_config.data_dir), config=dataset_config, rngs=rngs
        )

        # Initialize metrics with the evaluation configuration
        self.metrics = MolecularFlowsMetrics(rngs=rngs, config=self.eval_config)

        # Store performance targets
        self.performance_targets = performance_targets

    def evaluate_model(
        self, model: SE3MolecularFlow, num_samples: int = 100, warmup: bool = True, **kwargs
    ) -> dict[str, Any]:
        """Evaluate SE(3) molecular flow model performance.

        Args:
            model: SE(3) molecular flow model to evaluate
            num_samples: Number of samples to generate for evaluation
            warmup: If True, do a warmup run to trigger JIT compilation
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation metrics and results
        """
        # Get a batch of data for evaluation
        batch = next(iter(self.dataset))
        coordinates = batch["coordinates"]
        atom_types = batch["atom_types"]
        atom_mask = batch["atom_mask"]
        reference_energies = batch.get("energies")

        # Limit to specified number of samples
        if num_samples < coordinates.shape[0]:
            coordinates = coordinates[:num_samples]
            atom_types = atom_types[:num_samples]
            atom_mask = atom_mask[:num_samples]
            if reference_energies is not None:
                reference_energies = reference_energies[:num_samples]
        elif num_samples > coordinates.shape[0] and reference_energies is not None:
            # Expand reference energies to match num_samples
            repeat_factor = (num_samples + coordinates.shape[0] - 1) // coordinates.shape[0]
            reference_energies = jnp.tile(reference_energies, repeat_factor)[:num_samples]

        # Warmup run to trigger JIT compilation (discard results)
        if warmup:
            _ = model.sample(
                atom_types=atom_types[:2],
                atom_mask=atom_mask[:2],
                num_samples=2,
                rngs=self.rngs,
            )
            _ = model.log_prob(coordinates[:2], atom_types[:2], atom_mask[:2])

        # Sample molecular conformations
        generated_coords = model.sample(
            atom_types=atom_types, atom_mask=atom_mask, num_samples=num_samples, rngs=self.rngs
        )

        # Generate samples for evaluation
        # batch = next(iter(self.dataset)) # This line is removed as per the new_code
        # coordinates = generated_coords # This line is removed as per the new_code
        # atom_types = batch["atom_types"] # This line is removed as per the new_code
        # atom_mask = batch["atom_mask"] # This line is removed as per the new_code

        # Prepare data for metrics
        real_data = batch  # Real molecular data from dataset
        generated_data = {
            "coordinates": generated_coords,
            "atom_types": atom_types,
            "atom_mask": atom_mask,
        }

        # Compute molecular flow metrics
        results = self.metrics.compute(
            real_data=real_data,
            generated_data=generated_data,
            reference_energies=reference_energies,
        )

        # Evaluate likelihood on real data
        log_probs = model.log_prob(coordinates, atom_types, atom_mask)
        results["average_log_likelihood"] = float(jnp.mean(log_probs))
        results["log_likelihood_std"] = float(jnp.std(log_probs))

        # Add model metadata
        results["model_name"] = getattr(model, "name", "se3_molecular_flow")
        results["num_evaluated_samples"] = coordinates.shape[0]

        return results

    def meets_performance_targets(self, results: dict[str, Any]) -> bool:
        """Check if results meet performance targets.

        Args:
            results: Evaluation results dictionary

        Returns:
            True if all targets are met, False otherwise
        """
        targets = self.performance_targets

        # Check chemical validity
        if results.get("chemical_validity", 0.0) < targets.get("chemical_validity", 0.95):
            return False

        # Check conformational diversity
        if results.get("conformational_diversity", 0.0) < targets.get(
            "conformational_diversity", 0.8
        ):
            return False

        # Check energy consistency (lower is better)
        if results.get("energy_consistency", float("inf")) > targets.get("energy_consistency", 0.1):
            return False

        return True

    def get_benchmark_info(self) -> dict[str, Any]:
        """Get benchmark information and configuration."""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "dataset": {
                "name": "QM9",
                "num_molecules": self.dataset.num_molecules,
                "max_atoms": self.dataset.max_atoms,
                "atom_types": self.dataset.num_atom_types,
            },
            "performance_targets": self.performance_targets,
            "metrics": [
                "chemical_validity",
                "conformational_diversity",
                "energy_consistency",
                "average_log_likelihood",
            ],
        }

    def run(
        self,
        model: SE3MolecularFlow,
        dataset: QM9Dataset | None = None,
    ) -> BenchmarkResult:
        """Run the SE(3) molecular flows benchmark.

        Args:
            model: SE(3) molecular flow model to benchmark
            dataset: QM9 dataset (if None, uses the configured dataset)

        Returns:
            Benchmark result with molecular flow metrics
        """
        if dataset is None:
            dataset = self.dataset

        # Evaluate model performance
        results = self.evaluate_model(model, num_samples=100)

        # Check if performance targets are met
        meets_targets = self.meets_performance_targets(results)

        # Create benchmark result
        return BenchmarkResult(
            benchmark_name="SE3MolecularFlows",
            model_name=getattr(model, "name", "SE3MolecularFlow"),
            metrics=results,
            metadata={
                "config": self.config.metadata,
                "passed": meets_targets,
                "execution_time": 0.0,  # TODO: Add timing
            },
        )


class SE3MolecularFlowsSuite(BenchmarkSuite):
    """Complete benchmark suite for SE(3)-equivariant molecular flows.

    This suite provides comprehensive evaluation of SE(3)-equivariant molecular
    flow models across multiple configurations and model sizes.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        """Initialize SE(3) molecular flows benchmark suite.

        Args:
            rngs: Random number generators for model creation
        """
        super().__init__(
            name="SE3MolecularFlowsSuite",
            description="Benchmark suite for SE(3)-equivariant molecular flows",
        )
        self.rngs = rngs

        # Define benchmark configurations
        self.benchmark_configs = self._create_benchmark_configs()

        # Initialize benchmarks
        self.benchmarks = {}
        for name, config in self.benchmark_configs.items():
            self.benchmarks[name] = SE3MolecularFlowsBenchmark(rngs=rngs, config=config)

    def _create_benchmark_configs(self) -> dict[str, EvaluationConfig]:
        """Create benchmark configurations for different model sizes."""
        # Create DataConfig for QM9 dataset
        dataset_config = DataConfig(
            name="qm9_dataset_config",
            dataset_name="qm9",
            data_dir=Path("test_results/test_data/mock_qm9_data"),
            split="test",
            metadata={
                "max_atoms": 29,
                "batch_size": 16,
                "num_conformations": 100,
            },
        )

        # Define model configuration
        model_config = {
            "hidden_dim": 64,
            "num_layers": 3,
            "num_coupling_layers": 4,
            "max_atoms": 29,
            "atom_types": 5,
            "use_attention": True,
            "equivariant_layers": True,
        }

        # Create EvaluationConfig
        eval_config = EvaluationConfig(
            name="se3_molecular_flows_benchmark_config",
            metrics=["chemical_validity", "conformational_diversity", "energy_consistency"],
            eval_batch_size=16,
            metadata={
                "dataset_config": dataset_config,
                "model_config": model_config,
                "training": {
                    "learning_rate": 1e-4,
                    "num_epochs": 5,
                    "warmup_steps": 100,
                    "gradient_clip": 1.0,
                },
                "performance_targets": {
                    "chemical_validity": 0.95,
                    "conformational_diversity": 0.8,
                    "energy_consistency": 0.1,
                    "training_time_per_epoch": 3600,
                },
            },
        )

        configs = {"se3_molecular_flows": eval_config}

        return configs

    def run_benchmarks(
        self, models: dict[str, SE3MolecularFlow], num_samples: int = 50, **kwargs
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Run all benchmarks on provided models.

        Args:
            models: dictionary mapping model names to SE(3) flow models
            num_samples: Number of samples to evaluate per model
            **kwargs: Additional evaluation parameters

        Returns:
            Nested dictionary: {model_name: {benchmark_name: results}}
        """
        all_results = {}

        for model_name, model in models.items():
            model_results = {}

            for benchmark_name, benchmark in self.benchmarks.items():
                print(f"Running {benchmark_name} on {model_name}...")

                try:
                    results = benchmark.evaluate_model(model, num_samples=num_samples, **kwargs)

                    # Add benchmark metadata
                    results["benchmark_name"] = benchmark_name
                    results["meets_targets"] = benchmark.meets_performance_targets(results)

                    model_results[benchmark_name] = results

                    chemical_validity = results.get("chemical_validity", 0.0)
                    conformational_div = results.get("conformational_diversity", 0.0)
                    energy_consistency = results.get("energy_consistency", float("inf"))

                    print(f"  Chemical validity: {chemical_validity:.3f}")
                    print(f"  Conformational diversity: {conformational_div:.3f}")
                    print(f"  Energy consistency: {energy_consistency:.3f}")
                    print(f"  Meets targets: {results['meets_targets']}")

                except Exception as e:
                    print(f"  Error: {e}")
                    model_results[benchmark_name] = {
                        "error": str(e),
                        "benchmark_name": benchmark_name,
                        "meets_targets": False,
                    }

            all_results[model_name] = model_results

        return all_results

    def compare_models(self, results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        """Compare performance across models.

        Args:
            results: Results from run_benchmarks

        Returns:
            Comparison summary and rankings
        """
        if not results:
            return {"error": "No results to compare"}

        # Extract performance metrics for comparison
        model_scores = {}

        for model_name, model_results in results.items():
            scores = []

            for benchmark_name, benchmark_results in model_results.items():
                if "error" not in benchmark_results:
                    # Composite score based on all metrics
                    validity = benchmark_results.get("chemical_validity", 0.0)
                    diversity = benchmark_results.get("conformational_diversity", 0.0)
                    # Energy consistency: lower is better, so invert it
                    energy = 1.0 / (1.0 + benchmark_results.get("energy_consistency", float("inf")))

                    composite_score = (validity + diversity + energy) / 3.0
                    scores.append(composite_score)

            if scores:
                model_scores[model_name] = sum(scores) / len(scores)
            else:
                model_scores[model_name] = 0.0

        # Find best model
        best_model = (
            max(model_scores.keys(), key=lambda k: model_scores[k]) if model_scores else None
        )

        # Create comparison summary
        comparison = {
            "best_model": best_model,
            "model_scores": model_scores,
            "metric_comparison": {},
        }

        # Detailed metric comparison
        metrics = ["chemical_validity", "conformational_diversity", "energy_consistency"]

        for metric in metrics:
            metric_values = {}
            for model_name, model_results in results.items():
                values = []
                for benchmark_results in model_results.values():
                    if metric in benchmark_results and "error" not in benchmark_results:
                        values.append(benchmark_results[metric])

                if values:
                    metric_values[model_name] = sum(values) / len(values)

            comparison["metric_comparison"][metric] = metric_values

        return comparison

    def get_suite_info(self) -> dict[str, Any]:
        """Get suite information and available benchmarks."""
        return {
            "name": "SE(3) Molecular Flows Suite",
            "description": "Comprehensive evaluation of SE(3)-equivariant molecular flows",
            "benchmarks": {
                name: benchmark.get_benchmark_info() for name, benchmark in self.benchmarks.items()
            },
            "num_benchmarks": len(self.benchmarks),
            "supported_models": ["SE3MolecularFlow"],
        }

    def create_default_models(self) -> dict[str, SE3MolecularFlow]:
        """Create default models for testing and comparison.

        Returns:
            dictionary of default models with different configurations
        """
        models = {}

        # Small model for quick testing
        models["se3_flow_small"] = SE3MolecularFlow(
            hidden_dim=32,
            num_layers=2,
            num_coupling_layers=2,
            max_atoms=29,
            atom_types=5,
            use_attention=False,
            equivariant_layers=True,
            rngs=self.rngs,
        )

        # Medium model (default configuration)
        models["se3_flow_medium"] = SE3MolecularFlow(
            hidden_dim=64,
            num_layers=3,
            num_coupling_layers=4,
            max_atoms=29,
            atom_types=5,
            use_attention=True,
            equivariant_layers=True,
            rngs=self.rngs,
        )

        # Large model for best performance
        models["se3_flow_large"] = SE3MolecularFlow(
            hidden_dim=128,
            num_layers=4,
            num_coupling_layers=6,
            max_atoms=29,
            atom_types=5,
            use_attention=True,
            equivariant_layers=True,
            rngs=self.rngs,
        )

        return models
