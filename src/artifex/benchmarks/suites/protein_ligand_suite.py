"""Protein-ligand co-design benchmark suite.

This module provides a complete benchmark suite for evaluating
protein-ligand co-design models, targeting the v0.5-v0.8 objectives:
- Binding affinity RMSE <1.0 kcal/mol
- Molecular validity >95%
- Drug-likeness QED >0.7
"""

import logging
from typing import Any

import numpy as np
from flax import nnx

from artifex.benchmarks.core import Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite
from artifex.benchmarks.datasets.crossdocked import CrossDockedDataset
from artifex.benchmarks.metrics.protein_ligand import (
    BindingAffinityMetric,
    DrugLikenessMetric,
    MolecularValidityMetric,
)
from artifex.benchmarks.runtime_guards import demo_mode_from_mapping, require_demo_mode
from artifex.generative_models.core.configuration import DataConfig, EvaluationConfig


logger = logging.getLogger(__name__)


class ProteinLigandCoDesignBenchmark(Benchmark):
    """Benchmark for protein-ligand co-design models.

    This benchmark evaluates models on their ability to:
    1. Generate valid molecular structures
    2. Predict accurate binding affinities
    3. Produce drug-like ligands
    """

    def __init__(
        self,
        dataset: CrossDockedDataset,
        num_samples: int = 100,
        batch_size: int = 16,
        demo_mode: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the protein-ligand co-design benchmark.

        Args:
            dataset: CrossDocked dataset for evaluation
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            demo_mode: Whether to allow retained synthetic/demo benchmark paths
            rngs: Random number generator keys
        """
        config = BenchmarkConfig(
            name="protein_ligand_codesign",
            description="Complete evaluation of protein-ligand co-design models",
            metric_names=[
                "binding_affinity_rmse",
                "binding_affinity_mae",
                "binding_affinity_r2",
                "binding_affinity_pearson_r",
                "molecular_validity_rate",
                "bond_validity_rate",
                "angle_validity_rate",
                "clash_free_rate",
                "qed_score",
                "lipinski_compliance",
                "molecular_weight",
                "num_rotatable_bonds",
            ],
            metadata={
                "target_metrics": {
                    "binding_affinity_rmse": 1.0,  # Target: <1.0 kcal/mol
                    "molecular_validity_rate": 0.95,  # Target: >95%
                    "qed_score": 0.7,  # Target: >0.7
                }
            },
        )

        super().__init__(config)
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.demo_mode = demo_mode or getattr(dataset, "demo_mode", False)
        self.rngs = rngs

        # Initialize metrics
        affinity_config = EvaluationConfig(
            name="binding_affinity",
            metrics=["binding_affinity"],
            metric_params={"higher_is_better": False},
            eval_batch_size=batch_size,
        )
        validity_config = EvaluationConfig(
            name="molecular_validity",
            metrics=["molecular_validity"],
            metric_params={"higher_is_better": True},
            eval_batch_size=batch_size,
        )
        drug_config = EvaluationConfig(
            name="drug_likeness",
            metrics=["drug_likeness"],
            metric_params={
                "drug_likeness": {"higher_is_better": True, "demo_mode": self.demo_mode}
            },
            eval_batch_size=batch_size,
        )
        self.binding_affinity_metric = BindingAffinityMetric(rngs=rngs, config=affinity_config)
        self.molecular_validity_metric = MolecularValidityMetric(rngs=rngs, config=validity_config)
        self.drug_likeness_metric = DrugLikenessMetric(rngs=rngs, config=drug_config)

    def run(self, model, **kwargs) -> BenchmarkResult:
        """Run the protein-ligand co-design benchmark.

        Args:
            model: Model to benchmark
            **kwargs: Additional benchmark parameters

        Returns:
            Benchmark results with complete metrics
        """
        logger.info("Running protein-ligand co-design benchmark with %d samples", self.num_samples)

        all_metrics: dict[str, list[Any]] = {}

        # Process samples in batches
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            current_batch_size = min(self.batch_size, self.num_samples - start_idx)

            logger.debug("Processing batch %d/%d", batch_idx + 1, num_batches)

            # Get batch from dataset
            batch_data = self.dataset.get_batch(batch_size=current_batch_size)

            # Run model on batch
            batch_results = self._evaluate_batch(model, batch_data, **kwargs)

            # Accumulate metrics
            for metric_name, value in batch_results.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Aggregate metrics across batches
        final_metrics: dict[str, float] = {}
        for metric_name, values in all_metrics.items():
            if isinstance(values[0], dict):
                # Handle nested metrics (e.g., from multi-metric computations)
                final_metrics.update(self._aggregate_nested_metrics(values))
            else:
                # Simple averaging for scalar metrics
                final_metrics[metric_name] = float(np.mean(values))

        # Create benchmark result
        result = BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=getattr(model, "model_name", str(type(model).__name__)),
            metrics=final_metrics,
            metadata={
                "num_samples": self.num_samples,
                "batch_size": self.batch_size,
                "dataset_size": len(self.dataset),
                "target_metrics": self.config.metadata.get("target_metrics", {}),
            },
        )

        logger.info("Benchmark completed. Key results:")
        logger.info(
            "  Binding affinity RMSE: %.3f kcal/mol",
            final_metrics.get("binding_affinity_rmse", float("nan")),
        )
        logger.info(
            "  Molecular validity: %.3f",
            final_metrics.get("molecular_validity_rate", float("nan")),
        )
        logger.info("  QED score: %.3f", final_metrics.get("qed_score", float("nan")))

        return result

    def _evaluate_batch(self, model, batch_data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Evaluate model on a single batch.

        Args:
            model: Model to evaluate
            batch_data: Batch of protein-ligand complexes
            **kwargs: Additional evaluation parameters

        Returns:
            dictionary of batch metrics
        """
        # Extract data
        # We extract these variables for completeness, but they're used in _run_model_inference
        # via the batch_data dictionary rather than directly
        ligand_coords = batch_data["ligand_coords"]
        ligand_types = batch_data["ligand_types"]
        ligand_masks = batch_data["ligand_masks"]  # Use plural form as returned by dataset
        true_affinities = batch_data["binding_affinities"]  # Use plural form as returned by dataset

        model_outputs = self._run_model_inference(model, batch_data, **kwargs)

        predicted_affinities = model_outputs.get("binding_affinity")
        generated_ligand_coords = model_outputs.get("ligand_coords", ligand_coords)
        generated_ligand_types = model_outputs.get("ligand_types", ligand_types)

        batch_metrics: dict[str, float] = {}

        # 1. Binding affinity metrics
        if predicted_affinities is not None:
            affinity_metrics = self.binding_affinity_metric.compute(
                predictions=predicted_affinities, targets=true_affinities
            )
            # Add prefix to distinguish metric types
            for key, value in affinity_metrics.items():
                batch_metrics[f"binding_affinity_{key}"] = value

        # 2. Molecular validity metrics
        validity_metrics = self.molecular_validity_metric.compute(
            coordinates=generated_ligand_coords,
            atom_types=generated_ligand_types,
            masks=ligand_masks,
        )
        # Add prefix for clarity
        for key, value in validity_metrics.items():
            if key == "validity_rate":
                batch_metrics["molecular_validity_rate"] = value
            else:
                batch_metrics[f"{key}"] = value

        # 3. Drug-likeness metrics
        drug_metrics = self.drug_likeness_metric.compute(
            coordinates=generated_ligand_coords,
            atom_types=generated_ligand_types,
            masks=ligand_masks,
        )
        batch_metrics.update(drug_metrics)

        return batch_metrics

    def _run_model_inference(self, model, batch_data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Run model inference on batch data.

        Args:
            model: Model to run
            batch_data: Input batch
            **kwargs: Additional parameters

        Returns:
            Model outputs
        """
        # This is a placeholder for model inference
        # The actual implementation would depend on the model interface

        if not hasattr(model, "predict_binding_affinity"):
            raise ValueError(
                "ProteinLigandCoDesignBenchmark requires a model with "
                "predict_binding_affinity(...)."
            )

        predicted_affinities = model.predict_binding_affinity(
            protein_coords=batch_data["protein_coords"],
            protein_types=batch_data["protein_types"],
            ligand_coords=batch_data["ligand_coords"],
            ligand_types=batch_data["ligand_types"],
            **kwargs,
        )
        if predicted_affinities is None:
            raise ValueError(
                "ProteinLigandCoDesignBenchmark received no binding-affinity predictions"
            )

        if not hasattr(model, "generate_ligand"):
            raise ValueError(
                "ProteinLigandCoDesignBenchmark requires a model with generate_ligand(...)."
            )

        generated_ligands = model.generate_ligand(
            protein_coords=batch_data["protein_coords"],
            protein_types=batch_data["protein_types"],
            **kwargs,
        )
        if not isinstance(generated_ligands, dict):
            raise TypeError(
                "generate_ligand(...) must return a dict with coordinates and atom_types"
            )

        generated_ligand_coords = generated_ligands.get("coordinates")
        generated_ligand_types = generated_ligands.get("atom_types")
        if generated_ligand_coords is None or generated_ligand_types is None:
            raise ValueError(
                "generate_ligand(...) must return both coordinates and atom_types for evaluation"
            )

        return {
            "binding_affinity": predicted_affinities,
            "ligand_coords": generated_ligand_coords,
            "ligand_types": generated_ligand_types,
        }

    def _aggregate_nested_metrics(self, nested_values: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate nested metric dictionaries.

        Args:
            nested_values: list of metric dictionaries

        Returns:
            Aggregated metrics
        """
        aggregated: dict[str, float] = {}

        # Get all unique keys
        all_keys: set[str] = set()
        for value_dict in nested_values:
            all_keys.update(value_dict.keys())

        # Aggregate each metric
        for key in all_keys:
            values = [v.get(key, 0.0) for v in nested_values if key in v]
            if values:
                aggregated[key] = float(np.mean(values))

        return aggregated


class ProteinLigandBenchmarkSuite(BenchmarkSuite):
    """Complete benchmark suite for protein-ligand co-design evaluation.

    This suite includes all benchmarks needed for v0.5-v0.8 objectives.
    """

    def __init__(
        self,
        dataset_config: dict[str, Any] | None = None,
        benchmark_config: dict[str, Any] | None = None,
        demo_mode: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the protein-ligand benchmark suite.

        Args:
            dataset_config: Configuration for CrossDocked dataset
            benchmark_config: Configuration for benchmarks
            demo_mode: Whether to allow retained synthetic/demo benchmark paths
            rngs: Random number generator keys
        """
        super().__init__(
            name="protein_ligand_codesign_suite",
            description="Complete protein-ligand co-design evaluation suite",
        )

        self.rngs = rngs
        self.demo_mode = (
            demo_mode
            or demo_mode_from_mapping(dataset_config or {})
            or demo_mode_from_mapping(benchmark_config or {})
        )
        require_demo_mode(
            enabled=self.demo_mode,
            component="ProteinLigandBenchmarkSuite",
            detail=(
                "This retained suite still runs against mock CrossDocked data and demo-only "
                "drug-likeness heuristics."
            ),
        )

        # Default configurations
        self.dataset_config = dataset_config or {
            "num_samples": 1000,
            "max_protein_atoms": 1000,
            "max_ligand_atoms": 50,
            "pocket_radius": 10.0,
        }

        self.benchmark_config = benchmark_config or {
            "num_samples": 100,
            "batch_size": 16,
        }

        # Initialize dataset
        # Convert dict config to DataConfig if needed
        if isinstance(self.dataset_config, dict):
            data_config = DataConfig(
                name="crossdocked_suite",
                dataset_name="CrossDocked2020",
                split="train",
                metadata={**self.dataset_config, "demo_mode": self.demo_mode},
            )
        else:
            data_config = self.dataset_config

        self.dataset = CrossDockedDataset(
            data_path="test_results/test_data/mock_crossdocked", config=data_config, rngs=rngs
        )

        # Initialize benchmarks
        self._setup_benchmarks()

    def _setup_benchmarks(self) -> None:
        """Set up all benchmarks in the suite."""
        # Main co-design benchmark
        codesign_benchmark = ProteinLigandCoDesignBenchmark(
            dataset=self.dataset, demo_mode=self.demo_mode, **self.benchmark_config, rngs=self.rngs
        )

        self.add_benchmark(codesign_benchmark)

        logger.info(
            "Protein-ligand benchmark suite initialized with %d benchmarks",
            len(self.benchmarks),
        )
        logger.info("Dataset: %d samples", len(self.dataset))
        logger.info("Target metrics:")
        logger.info("  - Binding affinity RMSE: <1.0 kcal/mol")
        logger.info("  - Molecular validity: >95%")
        logger.info("  - QED score: >0.7")

    def run_all(self, model, **kwargs) -> dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite.

        Args:
            model: Model to benchmark
            **kwargs: Additional parameters

        Returns:
            dictionary mapping benchmark names to results
        """
        logger.info("Running protein-ligand co-design benchmark suite")
        logger.info("Benchmarks: %s", [b.config.name for b in self.benchmarks])

        results = super().run_all(model, **kwargs)

        # Log summary
        logger.info("=" * 60)
        logger.info("PROTEIN-LIGAND CO-DESIGN BENCHMARK SUMMARY")
        logger.info("=" * 60)

        for benchmark_name, result in results.items():
            logger.info("%s:", benchmark_name)

            # Key metrics
            rmse = result.metrics.get("binding_affinity_rmse")
            validity = result.metrics.get("molecular_validity_rate")
            qed = result.metrics.get("qed_score")

            if rmse is not None:
                status = "PASS" if rmse < 1.0 else "FAIL"
                logger.info("  Binding Affinity RMSE: %.3f kcal/mol %s", rmse, status)

            if validity is not None:
                status = "PASS" if validity > 0.95 else "FAIL"
                logger.info("  Molecular Validity: %.3f %s", validity, status)

            if qed is not None:
                status = "PASS" if qed > 0.7 else "FAIL"
                logger.info("  QED Score: %.3f %s", qed, status)

        return results
