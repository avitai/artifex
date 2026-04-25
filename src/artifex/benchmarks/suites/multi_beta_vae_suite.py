"""Multi-β VAE controllable generation benchmark suite.

This module provides a complete benchmark suite for evaluating
multi-β VAE controllable generation models, targeting the v0.9-v0.12 objectives:
- MIG Score >0.3 (Mutual Information Gap for disentanglement)
- FID Score <50 on CelebA (Fréchet Inception Distance)
- Reconstruction Quality: LPIPS <0.2, SSIM >0.8
- Training Efficiency: <8h per epoch on CelebA subset
"""

import logging
from typing import Any, cast

import numpy as np
from flax import nnx

from artifex.benchmarks.core import Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite
from artifex.benchmarks.datasets.celeba import CelebADataset
from artifex.benchmarks.metrics.disentanglement import (
    DisentanglementMetric,
    MutualInformationGapMetric,
    SeparationMetric,
)
from artifex.benchmarks.metrics.image import (
    FIDMetric,
    LPIPSMetric,
    SSIMMetric,
)
from artifex.generative_models.core.protocols.evaluation import (
    BatchableDatasetProtocol,
    BenchmarkModelProtocol,
    DatasetProtocol,
)


logger = logging.getLogger(__name__)


class MultiBetaVAEBenchmark(Benchmark):
    """Benchmark for Multi-β VAE controllable generation models.

    This benchmark evaluates models on their ability to:
    1. Learn disentangled representations (MIG, SAP, DCI metrics)
    2. Generate high-quality images (FID, LPIPS, SSIM)
    3. Train efficiently on standard hardware
    """

    def __init__(
        self,
        dataset: CelebADataset,
        num_samples: int = 1000,
        batch_size: int = 32,
        demo_mode: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Multi-β VAE controllable generation benchmark.

        Args:
            dataset: CelebA dataset for evaluation
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            demo_mode: Whether to allow retained synthetic/demo benchmark paths
            rngs: Random number generator keys
        """
        config = BenchmarkConfig(
            name="multi_beta_vae_controllable_generation",
            description="Complete evaluation of Multi-β VAE controllable generation models",
            metric_names=[
                "mig_score",
                "sap_score",
                "dci_score",
                "fid_score",
                "lpips_score",
                "ssim_score",
                "training_time_per_epoch",
                "reconstruction_loss",
            ],
            metadata={
                "target_metrics": {
                    "mig_score": 0.3,  # Target: >0.3
                    "fid_score": 50.0,  # Target: <50
                    "lpips_score": 0.2,  # Target: <0.2
                    "ssim_score": 0.8,  # Target: >0.8
                    "training_time_per_epoch": 8.0,  # Target: <8h
                }
            },
        )

        super().__init__(config)
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.demo_mode = demo_mode
        self.rngs = rngs

        # Import EvaluationConfig for metrics
        from artifex.generative_models.core.configuration import EvaluationConfig

        # Initialize metrics with proper configurations
        mig_config = EvaluationConfig(
            name="mig_metric",
            metrics=["mig"],
            metric_params={"mig": {"higher_is_better": True}},
            eval_batch_size=batch_size,
        )
        self.mig_metric = MutualInformationGapMetric(config=mig_config, rngs=rngs)

        sap_config = EvaluationConfig(
            name="sap_metric",
            metrics=["sap"],
            metric_params={"sap": {"higher_is_better": True}},
            eval_batch_size=batch_size,
        )
        self.sap_metric = SeparationMetric(config=sap_config, rngs=rngs)

        dci_config = EvaluationConfig(
            name="dci_metric",
            metrics=["dci"],
            metric_params={"dci": {"higher_is_better": True}},
            eval_batch_size=batch_size,
        )
        self.dci_metric = DisentanglementMetric(config=dci_config, rngs=rngs)

        fid_config = EvaluationConfig(
            name="fid_metric",
            metrics=["fid"],
            metric_params={
                "fid": {
                    "higher_is_better": False,
                    "mock_inception": demo_mode,
                    "demo_mode": demo_mode,
                }
            },
            eval_batch_size=batch_size,
        )
        self.fid_metric = FIDMetric(config=fid_config, rngs=rngs)

        lpips_config = EvaluationConfig(
            name="lpips_metric",
            metrics=["lpips"],
            metric_params={
                "lpips": {
                    "higher_is_better": False,
                    "mock_implementation": demo_mode,
                    "demo_mode": demo_mode,
                }
            },
            eval_batch_size=batch_size,
        )
        self.lpips_metric = LPIPSMetric(config=lpips_config, rngs=rngs)

        ssim_config = EvaluationConfig(
            name="ssim_metric",
            metrics=["ssim"],
            metric_params={"ssim": {"higher_is_better": True}},
            eval_batch_size=batch_size,
        )
        self.ssim_metric = SSIMMetric(config=ssim_config, rngs=rngs)

    def run(
        self,
        model: BenchmarkModelProtocol,
        dataset: DatasetProtocol | BatchableDatasetProtocol | None = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Run the Multi-β VAE controllable generation benchmark.

        Args:
            model: Model to benchmark
            dataset: Dataset to use for benchmarking (if None, uses self.dataset)
            **kwargs: Additional benchmark parameters

        Returns:
            Benchmark results with complete metrics
        """
        logger.info(
            "Running Multi-beta VAE controllable generation benchmark with %d samples",
            self.num_samples,
        )

        # Use the provided dataset or fall back to self.dataset
        dataset_to_use = dataset or self.dataset

        # Ensure dataset is compatible with BatchableDatasetProtocol
        if not hasattr(dataset_to_use, "get_batch"):
            raise ValueError(
                "Dataset must implement the BatchableDatasetProtocol with a get_batch method."
            )
        dataset_to_use = cast(BatchableDatasetProtocol, dataset_to_use)

        all_metrics: dict[str, list[Any]] = {}

        # Process samples in batches
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            current_batch_size = min(self.batch_size, self.num_samples - start_idx)

            logger.debug("Processing batch %d/%d", batch_idx + 1, num_batches)

            # Get batch from dataset
            batch_data = dataset_to_use.get_batch(
                batch_size=current_batch_size, start_idx=start_idx
            )

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
        # Handle metrics that might be N/A safely
        mig_score = final_metrics.get("mig_score", "N/A")
        mig_str = f"{mig_score:.3f}" if isinstance(mig_score, (float, int)) else mig_score
        logger.info("  MIG Score: %s", mig_str)

        fid_score = final_metrics.get("fid_score", "N/A")
        fid_str = f"{fid_score:.3f}" if isinstance(fid_score, (float, int)) else fid_score
        logger.info("  FID Score: %s", fid_str)

        lpips_score = final_metrics.get("lpips_distance", "N/A")  # Updated to use lpips_distance
        lpips_str = f"{lpips_score:.3f}" if isinstance(lpips_score, (float, int)) else lpips_score
        logger.info("  LPIPS Score: %s", lpips_str)

        ssim_score = final_metrics.get("ssim_score", "N/A")
        ssim_str = f"{ssim_score:.3f}" if isinstance(ssim_score, (float, int)) else ssim_score
        logger.info("  SSIM Score: %s", ssim_str)

        return result

    def _evaluate_batch(self, model, batch_data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Evaluate model on a single batch.

        Args:
            model: Model to evaluate
            batch_data: Batch of image data
            **kwargs: Additional evaluation parameters

        Returns:
            dictionary of batch metrics
        """
        # Extract data
        images = batch_data["images"]
        attributes = batch_data.get("attributes")

        model_outputs = self._run_model_inference(model, batch_data, **kwargs)

        reconstructed_images = model_outputs.get("reconstructions")
        latent_codes = model_outputs.get("latent_codes")
        generated_images = model_outputs.get("generated_images")

        batch_metrics = {}

        # 1. Disentanglement metrics
        if latent_codes is not None and attributes is not None:
            # MIG Score
            mig_result = self.mig_metric.compute(real_data=attributes, generated_data=latent_codes)
            batch_metrics.update(mig_result)

            # SAP Score
            sap_result = self.sap_metric.compute(real_data=attributes, generated_data=latent_codes)
            batch_metrics.update(sap_result)

            # DCI Score
            dci_result = self.dci_metric.compute(real_data=attributes, generated_data=latent_codes)
            batch_metrics.update(dci_result)

        # 2. Image quality metrics
        if reconstructed_images is not None:
            # LPIPS Distance
            lpips_result = self.lpips_metric.compute(
                real_data=images, generated_data=reconstructed_images
            )
            batch_metrics.update(lpips_result)

            # SSIM Score
            ssim_result = self.ssim_metric.compute(
                real_data=images, generated_data=reconstructed_images
            )
            batch_metrics.update(ssim_result)

        # 3. Generation quality metrics
        if generated_images is not None:
            # FID Score
            fid_result = self.fid_metric.compute(real_data=images, generated_data=generated_images)
            batch_metrics.update(fid_result)

        # 4. Training efficiency (if available)
        if "training_time_per_epoch" in kwargs:
            batch_metrics["training_time_per_epoch"] = kwargs["training_time_per_epoch"]

        # 5. Reconstruction loss
        if "reconstruction_loss" in model_outputs:
            batch_metrics["reconstruction_loss"] = float(model_outputs["reconstruction_loss"])

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

        if hasattr(model, "encode_decode"):
            results = model.encode_decode(images=batch_data["images"], rngs=self.rngs)
        elif hasattr(model, "__call__"):
            results = model(batch_data["images"], rngs=self.rngs)
        else:
            raise ValueError(
                "MultiBetaVAEBenchmark requires a model with encode_decode(...) or __call__(...) "
                "returning benchmark outputs."
            )

        if not isinstance(results, dict):
            raise TypeError("MultiBetaVAEBenchmark model outputs must be returned as a dict")

        if not any(
            key in results for key in ("reconstructions", "latent_codes", "generated_images")
        ):
            raise ValueError(
                "MultiBetaVAEBenchmark model outputs must expose reconstructions, latent_codes, "
                "or generated_images."
            )

        return results

    def _aggregate_nested_metrics(self, nested_values: list[dict]) -> dict[str, float]:
        """Aggregate nested metric dictionaries.

        Args:
            nested_values: list of metric dictionaries

        Returns:
            Aggregated metrics
        """
        aggregated = {}

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


class MultiBetaVAEBenchmarkSuite(BenchmarkSuite):
    """Complete benchmark suite for Multi-β VAE controllable generation evaluation.

    This suite includes all benchmarks needed for v0.9-v0.12 objectives.
    """

    def __init__(
        self,
        dataset_config: dict[str, Any] | None = None,
        benchmark_config: dict[str, Any] | None = None,
        demo_mode: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the Multi-β VAE benchmark suite.

        Args:
            dataset_config: Configuration for CelebA dataset
            benchmark_config: Configuration for benchmarks
            demo_mode: Whether to allow retained synthetic/demo benchmark paths
            rngs: Random number generator keys
        """
        super().__init__(
            name="multi_beta_vae_controllable_generation_suite",
            description="Complete Multi-β VAE controllable generation evaluation suite",
        )

        self.rngs = rngs
        self.demo_mode = demo_mode

        # Default configurations
        self.dataset_config = dataset_config or {
            "num_samples": 10000,
            "image_size": 128,
            "include_attributes": True,
        }

        self.benchmark_config = benchmark_config or {
            "num_samples": 1000,
            "batch_size": 32,
        }

        # Initialize dataset
        self.dataset = CelebADataset(**self.dataset_config, rngs=rngs)

        # Initialize benchmarks
        self._setup_benchmarks()

    def _setup_benchmarks(self):
        """Set up all benchmarks in the suite."""
        # Main controllable generation benchmark
        vae_benchmark = MultiBetaVAEBenchmark(
            dataset=self.dataset, demo_mode=self.demo_mode, **self.benchmark_config, rngs=self.rngs
        )

        self.add_benchmark(vae_benchmark)

        logger.info(
            "Multi-beta VAE benchmark suite initialized with %d benchmarks",
            len(self.benchmarks),
        )
        logger.info("Dataset: %d samples", len(self.dataset))
        logger.info("Target metrics:")
        logger.info("  - MIG Score: >0.3")
        logger.info("  - FID Score: <50")
        logger.info("  - LPIPS Distance: <0.2")
        logger.info("  - SSIM Score: >0.8")
        logger.info("  - Training time: <8h per epoch")

    def run_all(self, model, **kwargs) -> dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite.

        Args:
            model: Model to benchmark
            **kwargs: Additional parameters

        Returns:
            dictionary mapping benchmark names to results
        """
        logger.info("Running Multi-beta VAE controllable generation benchmark suite")
        logger.info("Benchmarks: %s", [b.config.name for b in self.benchmarks])

        results = super().run_all(model, **kwargs)

        # Log summary
        logger.info("=" * 60)
        logger.info("MULTI-BETA VAE CONTROLLABLE GENERATION BENCHMARK SUMMARY")
        logger.info("=" * 60)

        for benchmark_name, result in results.items():
            logger.info("%s:", benchmark_name)

            # Key metrics
            mig = result.metrics.get("mig_score")
            fid = result.metrics.get("fid_score")
            lpips = result.metrics.get("lpips_score")
            ssim = result.metrics.get("ssim_score")
            training_time = result.metrics.get("training_time_per_epoch")

            if mig is not None:
                status = "PASS" if mig > 0.3 else "FAIL"
                logger.info("  MIG Score: %.3f %s", mig, status)

            if fid is not None:
                status = "PASS" if fid < 50.0 else "FAIL"
                logger.info("  FID Score: %.3f %s", fid, status)

            if lpips is not None:
                status = "PASS" if lpips < 0.2 else "FAIL"
                logger.info("  LPIPS Score: %.3f %s", lpips, status)

            if ssim is not None:
                status = "PASS" if ssim > 0.8 else "FAIL"
                logger.info("  SSIM Score: %.3f %s", ssim, status)

            if training_time is not None:
                status = "PASS" if training_time < 8.0 else "FAIL"
                logger.info("  Training Time: %.2fh per epoch %s", training_time, status)

        return results
