"""Tests for gradient checkpointing utility module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.gradient_checkpointing import (
    apply_remat,
    resolve_checkpoint_policy,
)


class TestResolveCheckpointPolicy:
    """Tests for resolve_checkpoint_policy."""

    def test_none_returns_none(self) -> None:
        """Passing None returns None (no policy)."""
        assert resolve_checkpoint_policy(None) is None

    def test_callable_passes_through(self) -> None:
        """A callable policy is returned unchanged."""
        custom_policy = lambda *args: True
        result = resolve_checkpoint_policy(custom_policy)
        assert result is custom_policy

    def test_invalid_string_raises_value_error(self) -> None:
        """An unknown string raises ValueError with available policies."""
        with pytest.raises(ValueError, match="Unknown checkpoint policy"):
            resolve_checkpoint_policy("nonexistent_policy")

    @pytest.mark.parametrize(
        "name",
        [
            "dots_saveable",
            "everything_saveable",
            "nothing_saveable",
            "checkpoint_dots",
            "checkpoint_dots_no_batch",
        ],
    )
    def test_all_named_policies_resolve(self, name: str) -> None:
        """All 5 named policies resolve to callables."""
        result = resolve_checkpoint_policy(name)
        assert isinstance(result, Callable)  # type: ignore[arg-type]


class TestApplyRemat:
    """Tests for apply_remat wrapping nnx.remat."""

    @pytest.fixture
    def simple_module(self) -> nnx.Module:
        """Create a simple NNX module for testing."""

        class SimpleBlock(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                super().__init__()
                self.linear = nnx.Linear(in_features=8, out_features=8, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                return nnx.relu(self.linear(x))

        return SimpleBlock(rngs=nnx.Rngs(0))

    def test_output_matches_non_remated(self, simple_module: nnx.Module) -> None:
        """Remated closure produces the same output as the original."""
        x = jnp.ones((2, 8))

        output_original = simple_module(x)

        # apply_remat wraps plain functions / closures (not bound methods)
        def forward(x_in: jax.Array) -> jax.Array:
            return simple_module(x_in)

        remated_fn = apply_remat(forward)
        output_remated = remated_fn(x)

        assert jnp.allclose(output_original, output_remated, atol=1e-6)

    @pytest.mark.parametrize(
        "policy_name",
        [
            "dots_saveable",
            "everything_saveable",
            "nothing_saveable",
            "checkpoint_dots",
            "checkpoint_dots_no_batch",
        ],
    )
    def test_with_named_policies(self, simple_module: nnx.Module, policy_name: str) -> None:
        """All 5 named policies produce valid output through apply_remat."""
        x = jnp.ones((2, 8))

        def forward(x_in: jax.Array) -> jax.Array:
            return simple_module(x_in)

        remated_fn = apply_remat(forward, policy=policy_name)
        output = remated_fn(x)

        assert output.shape == (2, 8)
        assert jnp.isfinite(output).all()

    def test_gradient_correctness(self, simple_module: nnx.Module) -> None:
        """Gradients with remat match gradients without remat within tolerance."""
        x = jnp.ones((2, 8))

        def loss_no_remat(model: nnx.Module, x: jax.Array) -> jax.Array:
            return jnp.mean(model(x) ** 2)

        def loss_with_remat(model: nnx.Module, x: jax.Array) -> jax.Array:
            def forward(x_in: jax.Array) -> jax.Array:
                return model(x_in)

            remated_fn = apply_remat(forward)
            return jnp.mean(remated_fn(x) ** 2)

        grads_original = nnx.grad(loss_no_remat)(simple_module, x)
        grads_remated = nnx.grad(loss_with_remat)(simple_module, x)

        # Compare gradient values on the linear layer kernel
        orig_kernel_grad = grads_original.linear.kernel.value
        remat_kernel_grad = grads_remated.linear.kernel.value

        assert jnp.allclose(orig_kernel_grad, remat_kernel_grad, atol=1e-5)
