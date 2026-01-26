"""Test JAX compliance for trainer module.

These tests ensure that trainer module properly uses JAX
and avoids numpy in NNX context.
"""

import jax
import jax.numpy as jnp
import numpy as np


class TestTrainerJAXCompliance:
    """Test that trainer module is JAX-compliant."""

    def test_numpy_usage_in_trainer(self):
        """Test to identify numpy usage in trainer."""
        import ast
        import inspect

        from artifex.generative_models.training import trainer as trainer_module

        # Get the source code of the module
        source = inspect.getsource(trainer_module)

        # Parse the AST to check imports
        tree = ast.parse(source)

        # Check imports
        has_numpy = False
        has_jax = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "numpy":
                        has_numpy = True
                    elif alias.name == "jax":
                        has_jax = True
                    elif alias.name == "jax.numpy" or alias.name == "jnp":
                        pass
            elif isinstance(node, ast.ImportFrom):
                if node.module == "jax" and any(alias.name == "numpy" for alias in node.names):
                    pass

        # Check that numpy is NOT imported
        assert not has_numpy, "numpy should not be imported directly"

        # Check that JAX is imported
        assert has_jax, "jax should be imported"

        # Note: jax.numpy import is not required if trainer uses pure Python operations
        # or delegates array operations to model/loss functions

    def test_numpy_mean_in_epoch_metrics(self):
        """Test that numpy mean is used for averaging metrics."""
        # This test documents current behavior that needs to be fixed
        import numpy as np

        epoch_metrics = [
            {"loss": 0.5, "accuracy": 0.8, "step": 0},
            {"loss": 0.4, "accuracy": 0.85, "step": 1},
            {"loss": 0.3, "accuracy": 0.9, "step": 2},
        ]

        # Current implementation uses np.mean
        avg_loss = np.mean([m["loss"] for m in epoch_metrics])
        assert isinstance(avg_loss, (float, np.floating))

        # JAX-compatible alternative
        avg_loss_jax = jnp.mean(jnp.array([m["loss"] for m in epoch_metrics]))
        assert isinstance(avg_loss_jax, jax.Array)

        # They should be equivalent
        assert jnp.allclose(avg_loss, avg_loss_jax)

    def test_numpy_random_permutation(self):
        """Test numpy random permutation usage."""
        # Current implementation uses numpy
        data_len = 100
        perm_np = np.random.permutation(data_len)
        assert isinstance(perm_np, np.ndarray)

        # JAX-compatible alternative
        key = jax.random.PRNGKey(42)
        perm_jax = jax.random.permutation(key, data_len)
        assert isinstance(perm_jax, jax.Array)

        # Both create valid permutations
        assert len(perm_np) == data_len
        assert len(perm_jax) == data_len
        assert set(perm_np) == set(range(data_len))
        assert set(np.array(perm_jax)) == set(range(data_len))


class TestJAXPatterns:
    """Test patterns for JAX-compatible code."""

    def test_averaging_metrics_with_jax(self):
        """Test JAX-compatible metric averaging."""
        # Sample metrics
        metrics = [0.5, 0.4, 0.3, 0.6, 0.2]

        # Pure Python approach (no numpy needed)
        avg_python = sum(metrics) / len(metrics)
        assert isinstance(avg_python, float)

        # JAX approach if we need array operations
        avg_jax = jnp.mean(jnp.array(metrics))
        assert isinstance(avg_jax, jax.Array)

        # Convert to Python float for logging/display
        avg_float = float(avg_jax)
        assert isinstance(avg_float, float)

        # All should be equivalent
        assert abs(avg_python - avg_float) < 1e-6

    def test_data_shuffling_with_jax(self):
        """Test JAX-compatible data shuffling."""
        # Create sample data
        data = {"x": jnp.arange(10), "y": jnp.arange(10, 20)}

        # JAX shuffling with proper key management
        key = jax.random.PRNGKey(42)
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, len(data["x"]))

        # Apply permutation
        shuffled_data = {
            "x": data["x"][perm],
            "y": data["y"][perm],
        }

        # Verify shuffling
        assert len(shuffled_data["x"]) == len(data["x"])
        assert set(np.array(shuffled_data["x"])) == set(np.array(data["x"]))

        # Verify correspondence is maintained
        for i in range(len(perm)):
            orig_idx = perm[i]
            assert shuffled_data["x"][i] == data["x"][orig_idx]
            assert shuffled_data["y"][i] == data["y"][orig_idx]
