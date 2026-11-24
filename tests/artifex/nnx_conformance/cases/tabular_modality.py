"""Cases for the tabular modality, ``artifex.generative_models.modalities.tabular``.

The processors are fitted when built, on a fixed table of ``_FIT_ROWS`` rows, and their cases
check the fitted transform: fitting replaces ``None`` statistics with arrays, so a case that
fitted inside its call would be measuring the fit, not the transform.
"""

from __future__ import annotations

import jax
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import BATCH


_TABULAR = "artifex.generative_models.modalities.tabular"
_NUMERICAL, _CATEGORICAL, _CATEGORIES = ("height", "weight"), ("colour",), 3
_FIT_ROWS = 16


def _table(key: jax.Array, rows: int = BATCH) -> tuple[jax.Array, ...]:
    """Two numerical columns and one categorical column of ``rows`` rows."""
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.normal(k1, (rows,)),
        jax.random.normal(k2, (rows,)) * 3.0 + 1.0,
        jax.random.randint(k3, (rows,), 0, _CATEGORIES),
    )


def _columns(*columns: jax.Array) -> dict[str, jax.Array]:
    return dict(zip((*_NUMERICAL, *_CATEGORICAL), columns, strict=True))


def _fitted(module: nnx.Module) -> nnx.Module:
    module.fit(_columns(*_table(jax.random.key(0), _FIT_ROWS)))
    return module


def _round_trip(module: nnx.Module, *columns: jax.Array) -> PyTree:
    encoded = module.encode(_columns(*columns))
    return encoded, module.decode(encoded)


def cases() -> dict[str, Case]:
    """Return the tabular-modality cases, by the qualified name of the class each builds."""

    from artifex.generative_models.modalities.tabular import (  # noqa: PLC0415
        base,
        evaluation,
        representations,
    )

    config = base.TabularModalityConfig(
        num_features=len(_NUMERICAL) + len(_CATEGORICAL),
        numerical_features=_NUMERICAL,
        categorical_features=_CATEGORICAL,
        categorical_vocab_sizes={"colour": _CATEGORIES},
    )
    return {
        f"{_TABULAR}.representations.NumericalProcessor": Case(
            build=lambda r: _fitted(representations.NumericalProcessor(list(_NUMERICAL), rngs=r)),
            inputs=lambda key: _table(key)[:2],
            call=lambda module, height, weight: module.inverse_transform(
                module.transform({"height": height, "weight": weight})
            ),
            is_differentiable=False,
        ),
        f"{_TABULAR}.representations.CategoricalEncoder": Case(
            build=lambda r: representations.CategoricalEncoder(
                list(_CATEGORICAL), {"colour": _CATEGORIES}, rngs=r
            ),
            inputs=lambda key: _table(key)[2:],
            call=lambda module, colour: module.decode(module.encode({"colour": colour})),
            is_differentiable=False,
        ),
        f"{_TABULAR}.representations.TabularProcessor": Case(
            build=lambda r: _fitted(representations.TabularProcessor(config, rngs=r)),
            inputs=_table,
            call=_round_trip,
            is_differentiable=False,
        ),
        f"{_TABULAR}.evaluation.TabularEvaluationSuite": Case(
            build=lambda r: evaluation.TabularEvaluationSuite(config, rngs=r),
            inputs=lambda key: (*_table(key), *_table(jax.random.fold_in(key, 1))),
            call=lambda module, *columns: module.evaluate_batch(
                _columns(*columns[:3]), _columns(*columns[3:])
            ),
            is_differentiable=False,
            is_rowwise=False,
        ),
    }


EXEMPT: dict[str, str] = {}
