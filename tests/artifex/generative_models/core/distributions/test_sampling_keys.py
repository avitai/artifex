"""Distributions take their sampling key at call time and hold none of their own.

A distribution builds its parameters from values, so its constructor takes no key (flax's
``nnx.PReLU`` does the same), and ``sample`` requires ``rngs``: flax's explicit style, "we
recommend that new flax projects stick to the explicit style" (flax
``docs_nnx/guides/randomness.md``). The draw is one key of the stream the instance names in
``rng_collection`` (flax ``nnx.Dropout(rng_collection=...)``), ``sample`` unless told otherwise,
which falls back to the ``default`` stream when the ``nnx.Rngs`` does not name it
(``substrax.rng.stream_key``).
"""

from __future__ import annotations

import inspect
from typing import Protocol

import distrax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.rng import SAMPLE_STREAM

from artifex.generative_models.core.distributions import (
    AffineTransform,
    Bernoulli,
    Beta,
    Categorical,
    Distribution,
    Mixture,
    MixtureOfGaussians,
    Normal,
    OneHotCategorical,
    TransformedDistribution,
)


class Builder(Protocol):
    """Builds one distribution; ``rng_collection`` reaches its constructor when given."""

    def __call__(self, **collection: str) -> Distribution: ...


def _normal() -> Normal:
    return Normal(jnp.array([0.1, -0.2]), jnp.array([1.0, 0.7]))


BUILDERS: dict[str, Builder] = {
    "Normal": lambda **c: Normal(jnp.array([0.1, -0.2]), jnp.array([1.0, 0.7]), **c),
    "Beta": lambda **c: Beta(jnp.array([2.0, 1.5]), jnp.array([1.2, 2.5]), **c),
    "Bernoulli": lambda **c: Bernoulli(logits=jnp.array([0.3, -0.5]), **c),
    "Categorical": lambda **c: Categorical(logits=jnp.array([0.2, -0.1, 0.4]), **c),
    "OneHotCategorical": lambda **c: OneHotCategorical(logits=jnp.array([0.2, -0.1, 0.4]), **c),
    "MixtureOfGaussians": lambda **c: MixtureOfGaussians(
        jnp.array([[0.0, 1.0], [2.0, -0.5]]), jnp.array([[1.0, 0.8], [0.6, 1.0]]), **c
    ),
    "Mixture": lambda **c: Mixture(
        [Normal(jnp.array([0.0]), jnp.array([1.0])), Normal(jnp.array([3.0]), jnp.array([0.5]))],
        **c,
    ),
    "AffineTransform": lambda **c: AffineTransform(_normal(), jnp.array([0.5, 0.0]), 2.0, **c),
    "TransformedDistribution": lambda **c: TransformedDistribution(_normal(), distrax.Tanh(), **c),
}
CLASSES = (
    Distribution,
    Normal,
    Beta,
    Bernoulli,
    Categorical,
    OneHotCategorical,
    MixtureOfGaussians,
    Mixture,
    AffineTransform,
    TransformedDistribution,
)
SHAPE = (64,)


def _held_keys(module: nnx.Module) -> list[str]:
    return [
        "/".join(map(str, path))
        for path, node in nnx.iter_graph(module)
        if isinstance(node, (nnx.Rngs, nnx.RngStream))
    ]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_constructor_takes_no_key(cls: type[Distribution]) -> None:
    assert "rngs" not in inspect.signature(cls.__init__).parameters


@pytest.mark.parametrize("name", BUILDERS)
def test_sample_requires_a_key_at_call_time(name: str) -> None:
    rngs = inspect.signature(type(BUILDERS[name]()).sample).parameters["rngs"]
    assert rngs.kind is inspect.Parameter.KEYWORD_ONLY
    assert rngs.default is inspect.Parameter.empty


@pytest.mark.parametrize("name", BUILDERS)
def test_one_key_gives_one_sample_and_two_keys_two(name: str) -> None:
    dist = BUILDERS[name]()
    key, other = jax.random.split(jax.random.key(0))
    first = dist.sample(SHAPE, rngs=nnx.Rngs(sample=key))
    again = dist.sample(SHAPE, rngs=nnx.Rngs(sample=key))
    different = dist.sample(SHAPE, rngs=nnx.Rngs(sample=other))
    assert first.shape[: len(SHAPE)] == SHAPE
    assert jnp.array_equal(first, again)
    assert not jnp.array_equal(first, different)


@pytest.mark.parametrize("name", BUILDERS)
def test_the_draw_falls_back_to_the_default_stream(name: str) -> None:
    dist = BUILDERS[name]()
    key = jax.random.key(1)
    named = dist.sample(SHAPE, rngs=nnx.Rngs(sample=key))
    default = dist.sample(SHAPE, rngs=nnx.Rngs(default=key))
    assert jnp.array_equal(named, default)


@pytest.mark.parametrize("name", BUILDERS)
def test_one_draw_takes_one_key_from_the_stream(name: str) -> None:
    """A composite derives its parts' keys from that one key (flax ``stochastic_test.py``)."""
    rngs = nnx.Rngs(sample=jax.random.key(7), default=jax.random.key(8))
    BUILDERS[name]().sample(SHAPE, rngs=rngs)
    assert int(rngs.sample.count[...]) == 1
    assert int(rngs.default.count[...]) == 0


@pytest.mark.parametrize("name", BUILDERS)
def test_no_key_is_held_after_sampling(name: str) -> None:
    dist = BUILDERS[name]()
    dist.sample(SHAPE, rngs=nnx.Rngs(0))
    assert _held_keys(dist) == []


@pytest.mark.parametrize("name", BUILDERS)
def test_the_stream_is_the_one_the_instance_names(name: str) -> None:
    key, other = jax.random.split(jax.random.key(4))
    assert BUILDERS[name]().rng_collection == SAMPLE_STREAM
    noise = BUILDERS[name](rng_collection="noise")
    named = noise.sample(SHAPE, rngs=nnx.Rngs(noise=key, sample=other, default=other))
    assert jnp.array_equal(named, BUILDERS[name]().sample(SHAPE, rngs=nnx.Rngs(sample=key)))
    assert jnp.array_equal(named, noise.sample(SHAPE, rngs=nnx.Rngs(default=key)))


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_the_stream_name_is_a_keyword_defaulting_to_sample(cls: type[Distribution]) -> None:
    collection = inspect.signature(cls.__init__).parameters["rng_collection"]
    assert collection.kind is inspect.Parameter.KEYWORD_ONLY
    assert collection.default == SAMPLE_STREAM


NOT_AN_RNGS = {
    "a typed key": lambda: jax.random.key(0),
    "a legacy key": lambda: jax.random.PRNGKey(0),  # substrax-allow: seed-literal: refused input
    "a stream": lambda: nnx.Rngs(0).sample,
    "None": lambda: None,
}


@pytest.mark.parametrize("what", NOT_AN_RNGS)
@pytest.mark.parametrize("name", BUILDERS)
def test_anything_but_an_rngs_is_refused_by_type(name: str, what: str) -> None:
    with pytest.raises(TypeError, match="rngs must be an nnx.Rngs"):
        BUILDERS[name]().sample(SHAPE, rngs=NOT_AN_RNGS[what]())


@pytest.mark.parametrize("name", BUILDERS)
def test_an_rngs_with_neither_the_stream_nor_a_default_is_refused(name: str) -> None:
    with pytest.raises(KeyError, match="No RngStream named 'sample'"):
        BUILDERS[name]().sample(SHAPE, rngs=nnx.Rngs(params=0))


@pytest.mark.parametrize("name", BUILDERS)
def test_a_distribution_is_not_callable(name: str) -> None:
    """One method per job: ``log_prob`` evaluates, ``sample`` draws."""
    assert not callable(BUILDERS[name]())


@pytest.mark.parametrize("name", BUILDERS)
def test_rows_differ_under_vmap_with_one_key_per_row(name: str) -> None:
    dist = BUILDERS[name]()
    keys = jax.random.split(jax.random.key(2), 4)

    def draw(module: Distribution, key: jax.Array) -> jax.Array:
        return module.sample((8,), rngs=nnx.Rngs(sample=key))

    rows = nnx.vmap(draw, in_axes=(None, 0))(dist, keys)
    for i in range(1, rows.shape[0]):
        assert not jnp.array_equal(rows[0], rows[i])
    assert jnp.array_equal(rows[3], draw(dist, keys[3]))


@pytest.mark.parametrize("name", BUILDERS)
def test_two_jit_calls_with_one_rngs_draw_different_keys(name: str) -> None:
    dist = BUILDERS[name]()
    draw = nnx.jit(lambda module, rngs: module.sample(SHAPE, rngs=rngs))
    rngs = nnx.Rngs(3)
    assert not jnp.array_equal(draw(dist, rngs), draw(dist, rngs))


@pytest.mark.parametrize("name", BUILDERS)
def test_a_bound_step_draws_from_a_key_folded_by_step(name: str) -> None:
    """The training binding: ``nnx.jit_partial`` takes no ``nnx.Rngs`` at run time.

    The step builds its ``nnx.Rngs`` inside the compiled function from the run key folded with
    the step index, so one step repeats exactly and two steps differ, with one compilation.
    """
    traces = 0

    def step(module: Distribution, key: jax.Array, index: jax.Array) -> jax.Array:
        nonlocal traces
        traces += 1
        return module.sample(SHAPE, rngs=nnx.Rngs(sample=jax.random.fold_in(key, index)))

    bound = nnx.jit_partial(step, BUILDERS[name](), graph=True, graph_updates=False)
    key = jax.random.key(5)
    first, again, second = (bound(key, jnp.asarray(i)) for i in (0, 0, 1))
    assert jnp.array_equal(first, again)
    assert not jnp.array_equal(first, second)
    assert traces == 1


def test_a_mixture_gives_each_component_and_the_choice_a_key_of_their_own() -> None:
    """numpyro's scheme: one split for choice and draws, then one key per component.

    ``MixtureGeneral`` (numpyro ``distributions/mixtures.py``, ``sample_with_intermediates`` and
    ``component_sample``). TFP's ``Mixture._sample_n`` splits once into a salted n+1 instead;
    neither reuses a key.
    """
    component = Normal(jnp.array([0.0]), jnp.array([1.0]))
    mixture = Mixture([component, Normal(jnp.array([0.0]), jnp.array([1.0]))])
    key = jax.random.key(6)
    # A stream hands out its seed key folded with its count, never the seed key itself
    # (flax ``RngStream.__call__``), so the mixture's key is the stream's first one.
    select, draws = jax.random.split(nnx.Rngs(sample=key).sample())
    first_key, second_key = jax.random.split(draws, 2)
    choice = jax.random.categorical(select, jnp.log(jnp.array([0.5, 0.5])), shape=SHAPE)
    expected = jnp.where(
        (choice == 0)[:, None],
        component.sample(SHAPE, rngs=nnx.Rngs(first_key)),
        component.sample(SHAPE, rngs=nnx.Rngs(second_key)),
    )
    assert jnp.array_equal(mixture.sample(SHAPE, rngs=nnx.Rngs(sample=key)), expected)
