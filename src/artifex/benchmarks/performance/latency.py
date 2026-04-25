"""Latency benchmark for generative models.

Provides benchmarks for measuring inference latency of generative
models, supporting both sampling-based and prediction-based inference.
"""

import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from artifex.benchmarks import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkModelProtocol,
    BenchmarkResult,
    DatasetProtocol,
)


def measure_inference_latency(
    model: BenchmarkModelProtocol,
    method: Literal["sample", "predict"] = "sample",
    num_runs: int = 100,
    warmup_runs: int = 10,
    batch_size: int = 1,
    rngs: nnx.Rngs | None = None,
    inputs: jax.Array | None = None,
) -> tuple[float, float]:
    """Measure the inference latency of a model.

    Args:
        model: Model to benchmark.
        method: Method to use for inference ("sample" or "predict").
        num_runs: Number of runs to average over.
        warmup_runs: Number of warmup runs to exclude from timing.
        batch_size: Batch size for sampling.
        rngs: NNX Rngs for stochastic operations.
        inputs: Input data for prediction.

    Returns:
        Tuple of (average latency in seconds, standard deviation).
    """
    if method not in ["sample", "predict"]:
        raise ValueError(f"Unknown method: {method}")

    # Prepare inputs for prediction
    if method == "predict" and inputs is None:
        raise ValueError("Inputs are required for prediction method")

    resolved_rngs = rngs if rngs is not None else nnx.Rngs(sample=jax.random.PRNGKey(0))
    resolved_inputs = jnp.asarray(inputs) if inputs is not None else None

    # Warmup runs
    for _ in range(warmup_runs):
        if method == "sample":
            model.sample(batch_size=batch_size, rngs=resolved_rngs)
        elif resolved_inputs is not None:
            model.predict(resolved_inputs, rngs=resolved_rngs)

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()

        if method == "sample":
            model.sample(batch_size=batch_size, rngs=resolved_rngs)
        elif resolved_inputs is not None:
            model.predict(resolved_inputs, rngs=resolved_rngs)

        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    # Calculate statistics
    avg_latency = float(np.mean(latencies))
    std_dev = float(np.std(latencies))

    return avg_latency, std_dev


def _coerce_prediction_input(item: Any) -> jax.Array:
    """Normalize dataset items into array inputs for prediction latency."""
    if isinstance(item, dict):
        for key in ("images", "image", "inputs", "input", "data", "x"):
            value = item.get(key)
            if value is not None:
                return jnp.asarray(value)
        raise TypeError("Dataset item mapping does not expose a known prediction input key")
    return jnp.asarray(item)


class LatencyBenchmark(Benchmark):
    """Benchmark for measuring inference latency of generative models.

    This benchmark measures the time it takes for a model to generate samples
    or make predictions, providing insights into its computational efficiency.
    """

    def __init__(
        self,
        method: Literal["sample", "predict"] = "sample",
        batch_size: int = 1,
        num_runs: int = 100,
        warmup_runs: int = 10,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the latency benchmark.

        Args:
            method: Method to use for inference ("sample" or "predict").
            batch_size: Batch size for sampling.
            num_runs: Number of runs to average over.
            warmup_runs: Number of warmup runs to exclude from timing.
            random_seed: Random seed for sampling.
        """
        config = BenchmarkConfig(
            name="latency",
            description="Inference latency for generative models",
            metric_names=[
                "inference_latency_ms",
                "latency_std_dev_ms",
                "samples_per_second",
            ],
        )
        super().__init__(config=config)

        self.method: Literal["sample", "predict"] = method
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.random_seed = random_seed

    def run(
        self,
        model: BenchmarkModelProtocol,
        dataset: DatasetProtocol | None = None,
    ) -> BenchmarkResult:
        """Run the latency benchmark.

        Args:
            model: Model to benchmark.
            dataset: Dataset for prediction inputs.

        Returns:
            Benchmark result with latency metrics.
        """
        # Prepare inputs for prediction
        inputs: jax.Array | None = None
        if self.method == "predict":
            if dataset is None:
                raise ValueError("Dataset is required for latency benchmark with predict method")

            input_items = [
                _coerce_prediction_input(dataset[i])
                for i in range(min(self.batch_size, len(dataset)))
            ]
            inputs = jnp.stack(input_items)

        # Prepare random key for sampling
        if self.random_seed is not None:
            key = jax.random.PRNGKey(self.random_seed)
        else:
            key = jax.random.PRNGKey(0)

        # Create rngs object for NNX models
        rngs = nnx.Rngs(sample=key)

        # Measure latency
        avg_latency, std_dev = measure_inference_latency(
            model=model,
            method=self.method,
            num_runs=self.num_runs,
            warmup_runs=self.warmup_runs,
            batch_size=self.batch_size,
            rngs=rngs,
            inputs=inputs,
        )

        # Convert to milliseconds for better readability
        avg_latency_ms = avg_latency * 1000
        std_dev_ms = std_dev * 1000

        # Calculate samples per second
        samples_per_second = self.batch_size / avg_latency

        # Create metrics
        metrics = {
            "inference_latency_ms": avg_latency_ms,
            "latency_std_dev_ms": std_dev_ms,
            "samples_per_second": samples_per_second,
        }

        # Create result
        result = BenchmarkResult(
            benchmark_name=self.config.name,
            model_name=getattr(model, "model_name", "unknown"),
            metrics=metrics,
            metadata={
                "method": self.method,
                "batch_size": self.batch_size,
                "num_runs": self.num_runs,
                "warmup_runs": self.warmup_runs,
            },
        )

        return result
