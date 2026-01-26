"""Tests for ExtensionDict class.

ExtensionDict is a subclass of nnx.Dict that properly implements __contains__
to fix a bug where the `in` operator raises AttributeError for missing keys.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.extensions.base import ExtensionDict


class TestExtensionDictContains:
    """Tests for __contains__ fix."""

    def test_contains_existing_key(self):
        """Test that existing keys return True."""
        ext_dict = ExtensionDict({"key1": jnp.array([1.0]), "key2": jnp.array([2.0])})
        assert "key1" in ext_dict
        assert "key2" in ext_dict

    def test_contains_missing_key(self):
        """Test that missing keys return False (not raise AttributeError)."""
        ext_dict = ExtensionDict({"key1": jnp.array([1.0])})
        # This was the bug: nnx.Dict raises AttributeError here
        assert "nonexistent" not in ext_dict
        assert "missing_key" not in ext_dict

    def test_contains_non_string_key(self):
        """Test that non-string keys return False."""
        ext_dict = ExtensionDict({"key1": jnp.array([1.0])})
        assert 123 not in ext_dict
        assert None not in ext_dict
        assert ("tuple",) not in ext_dict

    def test_contains_after_modification(self):
        """Test __contains__ after adding/removing items."""
        ext_dict = ExtensionDict({})

        # Initially empty
        assert "key1" not in ext_dict

        # Add an item
        ext_dict["key1"] = jnp.array([1.0])
        assert "key1" in ext_dict

        # Delete the item
        del ext_dict["key1"]
        assert "key1" not in ext_dict


class TestExtensionDictPytree:
    """Tests for pytree compatibility."""

    def test_flatten_unflatten(self):
        """Test that ExtensionDict can be flattened and unflattened."""
        ext_dict = ExtensionDict({"key1": jnp.array([1.0, 2.0]), "key2": jnp.array([3.0, 4.0])})

        leaves, treedef = jax.tree_util.tree_flatten(ext_dict)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert isinstance(reconstructed, ExtensionDict)
        assert jnp.allclose(reconstructed["key1"], ext_dict["key1"])
        assert jnp.allclose(reconstructed["key2"], ext_dict["key2"])

    def test_tree_map(self):
        """Test that jax.tree_util.tree_map works."""
        ext_dict = ExtensionDict({"scale": jnp.array(2.0), "bias": jnp.array(1.0)})

        scaled = jax.tree_util.tree_map(lambda x: x * 2.0, ext_dict)

        assert jnp.allclose(scaled["scale"], jnp.array(4.0))
        assert jnp.allclose(scaled["bias"], jnp.array(2.0))

    def test_contains_after_pytree_operations(self):
        """Test that __contains__ works after pytree operations."""
        ext_dict = ExtensionDict({"key1": jnp.array([1.0])})

        leaves, treedef = jax.tree_util.tree_flatten(ext_dict)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert "key1" in reconstructed
        assert "nonexistent" not in reconstructed


class TestExtensionDictJaxTransforms:
    """Tests for JAX transformation compatibility."""

    def test_jax_jit(self):
        """Test that ExtensionDict works with jax.jit."""
        ext_dict = ExtensionDict({"weights": jnp.array([1.0, 2.0, 3.0])})

        @jax.jit
        def compute_sum(d):
            return jnp.sum(d["weights"])

        result = compute_sum(ext_dict)
        expected = jnp.sum(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result, expected)


class TestExtensionDictNNXTransforms:
    """Tests for NNX transformation compatibility."""

    @pytest.fixture
    def model_with_extensions(self):
        """Create a model with ExtensionDict for testing."""

        class ModelWithExtensions(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                super().__init__()
                self.linear = nnx.Linear(4, 4, rngs=rngs)
                self.extensions = ExtensionDict({"scale": nnx.Param(jnp.array(2.0))})

            def __call__(self, x: jax.Array) -> jax.Array:
                x = self.linear(x)
                return x * self.extensions["scale"]

        rngs = nnx.Rngs(params=42)
        return ModelWithExtensions(rngs=rngs)

    def test_nnx_jit(self, model_with_extensions):
        """Test that nnx.jit works with model containing ExtensionDict."""
        model = model_with_extensions
        x = jnp.ones((2, 4))

        # Regular forward
        y_regular = model(x)

        # JIT compiled forward
        @nnx.jit
        def forward_jitted(model, x):
            return model(x)

        y_jitted = forward_jitted(model, x)
        assert jnp.allclose(y_regular, y_jitted)

    def test_nnx_value_and_grad(self, model_with_extensions):
        """Test that nnx.value_and_grad works with ExtensionDict."""
        model = model_with_extensions

        def loss_fn(model, x, target):
            pred = model(x)
            return jnp.mean((pred - target) ** 2)

        x = jnp.ones((2, 4))
        target = jnp.zeros((2, 4))

        loss, grads = nnx.value_and_grad(loss_fn)(model, x, target)

        assert loss > 0
        assert grads is not None

    def test_nnx_split_merge(self, model_with_extensions):
        """Test that nnx.split/merge works with ExtensionDict."""
        model = model_with_extensions
        x = jnp.ones((2, 4))

        # Get output before split
        y_before = model(x)

        # Split and merge
        graphdef, state = nnx.split(model)
        model_restored = nnx.merge(graphdef, state)

        # Get output after merge
        y_after = model_restored(x)

        assert jnp.allclose(y_before, y_after)


class TestExtensionDictInheritance:
    """Tests verifying ExtensionDict properly inherits from nnx.Dict."""

    def test_is_instance_of_nnx_dict(self):
        """Test that ExtensionDict is an instance of nnx.Dict."""
        ext_dict = ExtensionDict({})
        assert isinstance(ext_dict, nnx.Dict)

    def test_dict_operations(self):
        """Test that standard dict operations still work."""
        ext_dict = ExtensionDict({"a": 1, "b": 2})

        # __getitem__
        assert ext_dict["a"] == 1

        # __setitem__
        ext_dict["c"] = 3
        assert ext_dict["c"] == 3

        # __delitem__
        del ext_dict["c"]
        assert "c" not in ext_dict

        # __iter__
        keys = list(ext_dict)
        assert "a" in keys
        assert "b" in keys

        # __len__
        assert len(ext_dict) == 2
