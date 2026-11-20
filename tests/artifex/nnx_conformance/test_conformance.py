"""artifex's ``nnx.Module`` classes against substrax's NNX conformance checks.

The ``cases`` package says how to build and call each class, one module per family;
``known_failures.py`` lists the checks a case is known to fail, each with its defect, as strict
expected failures: the check must fail with a ``ConformanceError``, not crash, and the fix that
repairs a defect must delete its entries. ``cases.NOT_APPLICABLE`` names the checks a module does
not owe by design, each skipped with its reason. The checks are ``substrax.testing.nnx_conformance.CHECKS``.
"""

from __future__ import annotations

import pytest
from substrax.testing.nnx_conformance import CHECKS, ConformanceError, module_classes

import artifex
from tests.artifex.nnx_conformance.cases import all_cases, EXEMPT, NOT_APPLICABLE
from tests.artifex.nnx_conformance.known_failures import KNOWN_FAILURES


CASES = all_cases()


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


def test_every_case_and_exemption_names_a_class_of_its_own() -> None:
    """A case keyed by an alias (``ResidualBlock = Conv2DResidualBlock``) runs a class twice."""
    classes = set(module_classes(artifex))

    assert "artifex.generative_models.core.base.MLP" in classes
    assert sorted((set(CASES) | set(EXEMPT)) - classes) == []
