"""Common test models for artifex tests.

This module provides centralized, reusable nnx.Module subclasses for testing purposes.
All test models support proper NNX compatibility with required rngs parameter handling.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MockModel(nnx.Module):
    """Mock NNX model for testing with flexible configuration.

    Combines functionality from:
    - tests/artifex/generative_models/test_precision_recall.py
    - tests/artifex/generative_models/benchmarks/test_base.py
    """

    def __init__(self, mock_samples=None, *, rngs=None, model_name="mock_model"):
        """Initialize with optional predetermined samples.

        Args:
            mock_samples: Samples to return from sample method. If None, generates default ones.
            rngs: Required RNG dict for NNX compatibility.
            model_name: Name of the model.
        """
        super().__init__()
        self.model_name = model_name
        self.predict_called = False
        self.sample_called = False

        if mock_samples is not None:
            self.mock_samples = jnp.array(mock_samples)
        else:
            # Default mock samples for basic testing
            self.mock_samples = jnp.ones((10, 10))

    def __call__(self, x, *, rngs=None):
        """Forward pass.

        Args:
            x: Input data.
            rngs: Required RNG dict for NNX compatibility.

        Returns:
            Mock output based on input shape or predetermined samples.
        """
        if hasattr(self, "mock_samples") and self.mock_samples.ndim > 1:
            return self.mock_samples[0]
        return jnp.ones_like(x)

    def predict(self, x, *, rngs=None):
        """Mock predict method.

        Args:
            x: Input data.
            rngs: Required RNG dict for NNX compatibility.

        Returns:
            Prediction output.
        """
        self.predict_called = True
        if hasattr(self, "mock_samples") and self.mock_samples.ndim > 1:
            return self.mock_samples[0]
        return jnp.ones_like(x)

    def sample(self, rng_key=None, batch_size=1, *, rngs=None):
        """Mock sample method returning predetermined or generated samples.

        Args:
            rng_key: JAX random key (optional, for compatibility).
            batch_size: Number of samples to generate.
            rngs: Required RNG dict for NNX compatibility.

        Returns:
            Generated or predetermined samples.
        """
        self.sample_called = True

        # We're fully NNX compliant, so we require rngs and ignore rng_key
        if rngs is None:
            raise ValueError("rngs is required for NNX models")

        # Return predetermined samples if available
        if hasattr(self, "mock_samples") and self.mock_samples.ndim > 1:
            actual_batch_size = min(batch_size, self.mock_samples.shape[0])
            return self.mock_samples[:actual_batch_size]

        # Otherwise generate default samples
        return jnp.ones((batch_size, 10))


class SimpleNNXModel(nnx.Module):
    """Simple NNX model for testing with flexible input/output dimensions.

    Combines functionality from:
    - tests/artifex/generative_models/test_precision_recall.py
    - tests/artifex/generative_models/benchmarks/test_nnx_adapter.py
    - tests/artifex/generative_models/benchmarks/performance/test_latency_nnx.py
    - tests/artifex/generative_models/benchmarks/test_model_adapters.py
    """

    def __init__(self, features, in_features=None, *, rngs=None):
        """Initialize the model.

        Args:
            features: Number of output features.
            in_features: Number of input features. If None, defaults to 2 or 10 based on usage.
            rngs: Optional RNG object for initialization.
        """
        # Handle backward compatibility for different input dimensions
        if in_features is None:
            in_features = 10  # Most common usage

        self.dense = nnx.Linear(in_features=in_features, out_features=features, rngs=rngs)
        self.model_name = "SimpleNNXModel"
        self.in_features = in_features
        self.out_features = features

    def __call__(self, x, *, rngs=None):
        """Forward pass.

        Args:
            x: Input data.
            rngs: Optional RNG object.

        Returns:
            The model output.
        """
        return self.dense(x)

    def predict(self, x, *, rngs=None):
        """Make predictions.

        Args:
            x: Input data.
            rngs: Optional RNG object.

        Returns:
            The model predictions.
        """
        return self(x, rngs=rngs)

    def sample(self, rng_key=None, batch_size=1, *, rngs=None):
        """Generate samples.

        Args:
            rng_key: JAX random key (legacy).
            batch_size: Number of samples to generate.
            rngs: Optional RNG object.

        Returns:
            Generated samples.
        """
        # Use the provided rng_key as default
        sample_rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)

        # Proper RNG handling following guidelines
        if rngs is not None and hasattr(rngs, "sample"):
            sample_rng = rngs.sample.key.value

        # Generate samples based on configured input features
        if self.in_features == 2:
            # For precision/recall testing - generate two clusters
            centers = jnp.array([[-2.0, -2.0], [2.0, 2.0]])
            center_indices = jax.random.randint(sample_rng, (batch_size,), 0, 2)
            centers_selected = centers[center_indices]
            noise = jax.random.normal(sample_rng, (batch_size, 2)) * 0.5
            x = centers_selected + noise
        else:
            # For general testing
            x = jax.random.normal(sample_rng, (batch_size, self.in_features))

        return self(x, rngs=rngs)

    def generate(self, batch_size=1, *, rngs=None):
        """Generate samples (alternative interface).

        Args:
            batch_size: Number of samples to generate.
            rngs: Optional RNG dict.

        Returns:
            Generated samples.
        """
        sample_rng = jax.random.key(0)
        if rngs is not None and hasattr(rngs, "sample"):
            sample_rng = rngs.sample.key.value

        x = jax.random.normal(sample_rng, (batch_size, self.in_features))
        return self.predict(x, rngs=rngs)


class SimpleModel(nnx.Module):
    """Simple NNX model for basic testing needs.

    Combines functionality from:
    - tests/artifex/generative_models/core/test_checkpointing.py
    - tests/artifex/generative_models/core/losses/test_losses.py
    - examples/loss_examples.py
    """

    def __init__(self, features=2, *, rngs=None):
        """Initialize the model.

        Args:
            features: Number of output features.
            rngs: Optional RNG object for initialization.
        """
        self.dense1 = nnx.Linear(in_features=10, out_features=features, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=features, out_features=features, rngs=rngs)
        self.model_name = "SimpleModel"

    def __call__(self, x, *, rngs=None):
        """Forward pass.

        Args:
            x: Input data.
            rngs: Optional RNG object.

        Returns:
            The model output.
        """
        h = self.dense1(x)
        return self.dense2(h)

    def predict(self, x, *, rngs=None):
        """Make predictions.

        Args:
            x: Input data.
            rngs: Optional RNG object.

        Returns:
            The model predictions.
        """
        return self(x, rngs=rngs)

    def sample(self, rng_key=None, batch_size=1, *, rngs=None):
        """Generate samples.

        Args:
            rng_key: JAX random key (legacy).
            batch_size: Number of samples to generate.
            rngs: Optional RNG object.

        Returns:
            Generated samples.
        """
        # Use proper RNG handling
        sample_rng = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        if rngs is not None and hasattr(rngs, "sample"):
            sample_rng = rngs.sample.key.value

        # Generate random input and pass through model
        x = jax.random.normal(sample_rng, (batch_size, 10))
        return self(x, rngs=rngs)


class MockProteinModel(nnx.Module):
    """Mock protein model for benchmark testing.

    Combines functionality from:
    - scripts/simple_protein_benchmark.py
    - scripts/benchmark_demo.py
    - scripts/protein_benchmark_demo.py
    - scripts/minimal_benchmark.py
    """

    def __init__(self, config=None, *, rngs=None):
        """Initialize the mock model.

        Args:
            config: Configuration dictionary or None for defaults
            rngs: Random number generators
        """
        super().__init__()

        # Handle both dict config and direct parameter passing
        if config is None:
            config = {}

        self.model_name = f"mock_protein_model_{config.get('model_variant', 'base')}"
        self.num_residues = config.get("num_residues", 10)
        self.num_atoms = config.get("num_atoms", 4)
        self.quality = config.get("quality", "high")

        # For precision-recall benchmark, different distributions based on model quality
        self.precision_factor = 1.0 if self.quality == "high" else 0.7
        self.recall_factor = 1.0 if self.quality == "high" else 0.5

    def __call__(self, x, *, rngs=None):
        """Mock forward pass."""
        # Just return a slightly transformed version of the input
        return x * 0.9 + 0.1

    def predict(self, x, *, rngs=None):
        """Predict using the mock model."""
        # Ensure input is 2D for the model
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        return self(x, rngs=rngs)

    def sample(self, rng_key=None, batch_size=1, *, rngs=None):
        """Generate samples of protein structures.

        Args:
            rng_key: JAX random key (legacy support)
            batch_size: Number of samples to generate
            rngs: Random number generators

        Returns:
            Mock protein structures flattened to 2D
        """
        # Get proper random key
        key = jax.random.PRNGKey(42)
        if rng_key is not None:
            key = rng_key
        elif rngs is not None and hasattr(rngs, "sample"):
            key = rngs.sample.key.value

        # Generate random samples
        # Shape: [batch_size, num_residues, num_atoms, 3]
        base_shape = (batch_size, self.num_residues, self.num_atoms, 3)

        # For better precision-recall results, create distinct clusters based on quality
        if self.quality == "high":
            # Two clusters for high recall
            cluster_key, sample_key = jax.random.split(key)
            cluster = jax.random.choice(cluster_key, jnp.array([0, 1]), (batch_size,))

            # Generate samples from either positive or negative cluster
            samples = []
            for i in range(batch_size):
                if cluster[i] == 0:
                    # Cluster 1 - positive values
                    sample = (
                        jax.random.normal(sample_key, (1, self.num_residues, self.num_atoms, 3))
                        + 2.0
                    )
                else:
                    # Cluster 2 - negative values
                    sample = (
                        jax.random.normal(sample_key, (1, self.num_residues, self.num_atoms, 3))
                        - 2.0
                    )
                samples.append(sample)

            samples = jnp.concatenate(samples, axis=0)
        else:
            # Just one cluster for low recall
            samples = jax.random.normal(key, base_shape) + 2.0

        # Flatten to 2D (batch_size, num_residues*num_atoms*3)
        # This is required for precision-recall benchmark which uses KMeans
        samples = samples.reshape(batch_size, -1)

        return samples
