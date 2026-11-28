"""artifex's ``nnx.Module`` classes against substrax's NNX conformance checks.

The ``cases`` package says how to build and call each class, one module per family;
``known_failures.py`` lists the checks a case is known to fail, each with its defect, as strict
expected failures: the check must fail with a ``ConformanceError``, not crash, and the fix that
repairs a defect must delete its entries. ``cases.NOT_APPLICABLE`` names the checks a module does
not owe by design, each skipped with its reason. The checks are ``substrax.testing.nnx_conformance.CHECKS``.
"""

from __future__ import annotations

import pytest
from flax import nnx
from substrax.rng import rngs_from_seed
from substrax.testing.nnx_conformance import CHECKS, ConformanceError, module_classes, STREAMS

import artifex
from tests.artifex.nnx_conformance.cases import all_cases, EXEMPT, NOT_APPLICABLE
from tests.artifex.nnx_conformance.known_failures import KNOWN_FAILURES


CASES = all_cases()


def _class_of(name: str) -> str:
    """The qualified class name of a case: a method's case is keyed ``<class>#<method>``."""
    return name.partition("#")[0]


def _parameters() -> list[pytest.param]:
    parameters = []
    for name, case in sorted(CASES.items()):
        for check, (_, applies) in sorted(CHECKS.items()):
            if not applies(case):
                continue
            reason = KNOWN_FAILURES.get((name, check))
            xfail = pytest.mark.xfail(strict=True, raises=ConformanceError, reason=reason)
            marks = [xfail] if reason else []
            if (name, check) in NOT_APPLICABLE:
                marks = [pytest.mark.skip(reason=NOT_APPLICABLE[name, check])]
            parameters.append(
                pytest.param(name, check, id=f"{name.rsplit('.', 1)[1]}-{check}", marks=marks)
            )
    return parameters


@pytest.mark.parametrize(("name", "check"), _parameters())
def test_module_conforms(name: str, check: str) -> None:
    """The case's module has the property the check requires."""
    run, _ = CHECKS[check]
    run(CASES[name])


@pytest.mark.parametrize(
    "entries", [KNOWN_FAILURES, NOT_APPLICABLE], ids=["known_failures", "not_applicable"]
)
def test_every_entry_names_a_case_and_a_check_that_runs(
    entries: dict[tuple[str, str], str],
) -> None:
    """An entry for a case or check that no longer exists, or that never runs, is stale."""
    stale = [
        (name, check)
        for name, check in entries
        if name not in CASES or check not in CHECKS or not CHECKS[check][1](CASES[name])
    ]

    assert stale == []


def test_no_check_is_both_a_known_failure_and_not_applicable() -> None:
    """A defect to fix and a property not owed are different claims about one check."""
    assert sorted(set(KNOWN_FAILURES) & set(NOT_APPLICABLE)) == []


def test_every_nnx_module_in_the_package_has_a_case_or_an_exemption() -> None:
    """A new module class is checked from the moment it exists, or says why it is not."""
    classes = set(module_classes(artifex))

    assert "artifex.generative_models.core.base.MLP" in classes
    covered = {_class_of(name) for name in CASES}
    assert sorted(classes - covered - set(EXEMPT)) == []
    assert sorted(covered & set(EXEMPT)) == []


def test_every_case_and_exemption_names_a_class_of_its_own() -> None:
    """A case keyed by an alias (``ResidualBlock = Conv2DResidualBlock``) runs a class twice."""
    classes = set(module_classes(artifex))

    assert "artifex.generative_models.core.base.MLP" in classes
    assert sorted(({_class_of(name) for name in CASES} | set(EXEMPT)) - classes) == []


def _holds_dropout(module: nnx.Module) -> bool:
    return any(
        isinstance(node, nnx.Dropout)
        or (isinstance(node, nnx.MultiHeadAttention) and node.dropout_rate > 0)
        for _, node in nnx.iter_graph(module)
    )


def test_a_module_with_dropout_takes_its_dropout_key_at_call_time() -> None:
    """Dropout follows the caller's key, so the checks can tell a leaf the key never reaches.

    The built module is asked, not its class name: a dropout leaf reached through a child a
    factory builds counts too.
    """
    owed = [
        name
        for name, case in sorted(CASES.items())
        if case.has_dropout or _holds_dropout(case.build(rngs_from_seed(0, streams=STREAMS)))
    ]
    undeclared = [name for name in owed if "dropout" not in CASES[name].rng_streams]

    assert "artifex.generative_models.core.base.MLP" in owed
    assert undeclared == []


def _held_dropout_streams(module: nnx.Module) -> list[str]:
    """Paths of the dropout and attention layers in ``module`` that hold a stream of their own.

    flax's ``nnx.Dropout`` keeps one when built with ``rngs`` and ``nnx.MultiHeadAttention`` when
    built with ``keep_rngs=True`` and a nonzero rate (flax 0.12.12 ``stochastic.py:85-90``,
    ``attention.py:731``). Either then draws silently whenever a caller omits the key.
    """
    return [
        "/".join(map(str, path))
        for path, node in nnx.iter_graph(module)
        if isinstance(node, nnx.Dropout | nnx.MultiHeadAttention) and node.rngs is not None
    ]


def test_no_dropout_or_attention_layer_holds_a_key_of_its_own() -> None:
    """A training call without a key raises instead of drawing from a held stream.

    With a held stream, a forgotten or misspelled key, or a composite that does not pass it
    down, goes unnoticed; without one, flax raises (``stochastic.py:140-145``,
    ``attention.py:891-896``).
    """
    holders = {
        name: held
        for name, case in sorted(CASES.items())
        if (held := _held_dropout_streams(case.build(rngs_from_seed(0, streams=STREAMS))))
    }

    assert holders == {}
