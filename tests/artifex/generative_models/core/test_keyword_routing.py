"""A composite passes a call-time keyword only to children whose signature accepts it.

The rule is flax's own for a container of heterogeneous children (``nnx.Sequential`` filters
``**kwargs`` per layer, flax commit 7a24e6a7, released in 0.12.11): a layer without randomness
declares no ``rngs`` and is called without one.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from artifex.generative_models.core.keyword_routing import call_with_accepted_keywords


def _takes_rngs(x: jax.Array, *, rngs: nnx.Rngs | None = None) -> tuple[jax.Array, bool]:
    return x, rngs is not None


def _takes_nothing(x: jax.Array) -> jax.Array:
    return x


def _takes_everything(x: jax.Array, **keywords: Any) -> tuple[jax.Array, set[str]]:
    return x, set(keywords)


def _positional_or_keyword(x: jax.Array, scale: float = 1.0) -> jax.Array:
    return x * scale


def test_a_child_declaring_the_keyword_receives_it() -> None:
    _, received = call_with_accepted_keywords(_takes_rngs, jnp.ones(2), rngs=nnx.Rngs(0))

    assert received


def test_a_child_without_the_keyword_is_called_without_it() -> None:
    out = call_with_accepted_keywords(_takes_nothing, jnp.ones(2), rngs=nnx.Rngs(0))

    assert jnp.array_equal(out, jnp.ones(2))


def test_a_child_taking_keyword_variadics_receives_every_keyword() -> None:
    _, names = call_with_accepted_keywords(
        _takes_everything, jnp.ones(2), rngs=nnx.Rngs(0), deterministic=True
    )

    assert names == {"rngs", "deterministic"}


def test_a_parameter_bound_by_a_positional_argument_is_not_passed_again() -> None:
    out = call_with_accepted_keywords(_positional_or_keyword, jnp.ones(2), 3.0, scale=5.0)

    assert jnp.array_equal(out, 3.0 * jnp.ones(2))


def _positional_only_and_variadic(x: jax.Array, /, **keywords: Any) -> set[str]:
    del x
    return set(keywords)


def test_a_positional_only_name_stays_free_as_a_variadic_keyword() -> None:
    """A positional fills ``x``, but ``x`` is still a legal ``**kwargs`` key, as in flax's rule."""
    names = call_with_accepted_keywords(_positional_only_and_variadic, jnp.ones(2), x=1, rngs=2)

    assert names == {"x", "rngs"}


def test_an_unbound_positional_or_keyword_parameter_receives_the_keyword() -> None:
    out = call_with_accepted_keywords(_positional_or_keyword, jnp.ones(2), scale=5.0)

    assert jnp.array_equal(out, 5.0 * jnp.ones(2))


@pytest.mark.parametrize("seed", [0, 1])
def test_flax_layers_route_as_nnx_sequential_routes_them(seed: int) -> None:
    """Dropout takes the call-time key, Linear none: the outputs equal the direct calls."""
    linear = nnx.Linear(4, 4, rngs=nnx.Rngs(params=seed))
    dropout = nnx.Dropout(0.5)
    x = jnp.ones((3, 4))

    routed = call_with_accepted_keywords(
        dropout,
        call_with_accepted_keywords(linear, x, rngs=nnx.Rngs(dropout=seed)),
        rngs=nnx.Rngs(dropout=seed),
    )
    direct = dropout(linear(x), rngs=nnx.Rngs(dropout=seed))

    assert jnp.array_equal(routed, direct)


class _Block(nnx.Module):
    """A dropout child and a deterministic child, called through the router."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(4, 4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs | None = None) -> jax.Array:
        h = call_with_accepted_keywords(self.linear, x, rngs=rngs)
        return call_with_accepted_keywords(self.dropout, h, rngs=rngs)


def test_routing_under_nnx_jit_matches_eager_and_traces_once() -> None:
    """The signature is read while tracing, so a second call of the same shapes does not retrace."""
    from substrax.testing.compiles import expect_compiles
    from substrax.testing.traces import TraceCounter

    block = _Block(rngs=nnx.Rngs(params=0))
    x = jnp.ones((3, 4))
    eager = block(x, rngs=nnx.Rngs(dropout=7))

    counter = TraceCounter()
    compiled = nnx.jit(counter.wrap(lambda module, x, key: module(x, rngs=nnx.Rngs(dropout=key))))
    first = compiled(block, x, jax.random.key(7))
    with expect_compiles(0):
        compiled(block, x, jax.random.key(8))

    assert counter.count == 1
    assert jnp.array_equal(first, eager)


def test_routing_under_jax_grad_gives_the_direct_gradient() -> None:
    block = _Block(rngs=nnx.Rngs(params=0))
    graphdef, params = nnx.split(block, nnx.Param)
    x = jnp.ones((3, 4))

    def routed(p: nnx.State) -> jax.Array:
        return jnp.sum(nnx.merge(graphdef, p)(x, rngs=nnx.Rngs(dropout=3)))

    def direct(p: nnx.State) -> jax.Array:
        module = nnx.merge(graphdef, p)
        return jnp.sum(module.dropout(module.linear(x), rngs=nnx.Rngs(dropout=3)))

    routed_grads, direct_grads = jax.grad(routed)(params), jax.grad(direct)(params)

    assert jax.tree.all(jax.tree.map(jnp.array_equal, routed_grads, direct_grads))
