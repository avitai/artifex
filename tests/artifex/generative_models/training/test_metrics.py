"""Record-weighted metric means accumulated on the device.

A step or an evaluation batch reports mean metrics over its records; ``MeanAccumulator`` adds
each mean weighted by the batch's record count, so the mean it computes over any run of batches,
short ones included, is the mean over every record. It lives on the device and updates inside
a compiled step, so reading it is the only host transfer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from artifex.generative_models.training.metrics import MeanAccumulator


def _batches() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.normal(size=size).astype(np.float32) for size in (4, 4, 3)]


def test_the_mean_over_batches_is_the_mean_over_every_record() -> None:
    accumulator = MeanAccumulator(("loss",))
    batches = _batches()

    for values in batches:
        accumulator.update({"loss": jnp.mean(values)}, jnp.int32(values.size))

    records = np.concatenate(batches).astype(np.float64)
    # float32 sums of O(1) values: a few ULP at the data's scale, absolute (the mean is near
    # zero, so a relative bound measures the cancellation; measured 4.3e-8 here).
    atol = 4 * np.finfo(np.float32).eps * np.abs(records).max()
    np.testing.assert_allclose(accumulator.compute()["loss"], records.mean(), rtol=0, atol=atol)
    assert int(accumulator.records[...]) == 11


def test_reset_starts_a_new_window() -> None:
    accumulator = MeanAccumulator(("loss",))
    accumulator.update({"loss": jnp.float32(5.0)}, jnp.int32(2))

    accumulator.reset()
    accumulator.update({"loss": jnp.float32(1.0)}, jnp.int32(3))

    np.testing.assert_allclose(accumulator.compute()["loss"], 1.0)
    assert int(accumulator.records[...]) == 3


def test_it_accumulates_inside_a_compiled_step_and_compiles_once() -> None:
    accumulator = MeanAccumulator(("loss", "accuracy"))
    traces = {"count": 0}

    @nnx.jit(graph=False, graph_updates=False)
    def record(accumulator: MeanAccumulator, values: jax.Array) -> None:
        traces["count"] += 1
        accumulator.update(
            {"loss": jnp.mean(values), "accuracy": jnp.mean(values > 0)},
            jnp.int32(values.shape[0]),
        )

    for values in [np.ones(4, np.float32), np.zeros(4, np.float32), np.ones(4, np.float32)]:
        record(accumulator, values)

    assert traces["count"] == 1
    np.testing.assert_allclose(accumulator.compute()["loss"], 2 / 3, rtol=1e-6)


def test_a_mean_before_any_record_is_nan_not_zero() -> None:
    assert np.isnan(float(MeanAccumulator(("loss",)).compute()["loss"]))


def test_a_metric_it_was_not_built_for_is_refused() -> None:
    with pytest.raises(KeyError, match=r"accuracy"):
        MeanAccumulator(("loss",)).update(
            {"loss": jnp.float32(1.0), "accuracy": jnp.float32(0.5)}, jnp.int32(1)
        )
