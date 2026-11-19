"""artifex's ``nnx.Module`` classes against substrax's NNX conformance checks.

``cases.py`` says how to build and call each class; ``known_failures.py`` lists the checks a
case is known to fail, each with its defect, as strict expected failures: the check must fail
with a ``ConformanceError``, not crash, and the fix that repairs a defect must delete its
entries. The checks are ``substrax.testing.nnx_conformance.CHECKS``.
"""

from __future__ import annotations

import pytest
from substrax.testing.nnx_conformance import CHECKS, ConformanceError, module_classes

import artifex
from tests.artifex.nnx_conformance.cases import all_cases, EXEMPT
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
            parameters.append(
                pytest.param(name, check, id=f"{name.rsplit('.', 1)[1]}-{check}", marks=marks)
            )
    return parameters


@pytest.mark.parametrize(("name", "check"), _parameters())
def test_module_conforms(name: str, check: str) -> None:
    """The case's module has the property the check requires."""
    run, _ = CHECKS[check]
    run(CASES[name])


def test_every_known_failure_names_a_case_and_an_applicable_check() -> None:
    """A known failure for a case or check that no longer exists, or that never runs, is stale."""
    stale = [
        (name, check)
        for name, check in KNOWN_FAILURES
        if name not in CASES or check not in CHECKS or not CHECKS[check][1](CASES[name])
    ]

    assert stale == []


def test_every_case_and_exemption_names_a_class_of_its_own() -> None:
    """A case keyed by an alias (``ResidualBlock = Conv2DResidualBlock``) runs a class twice."""
    classes = set(module_classes(artifex))

    assert "artifex.generative_models.core.base.MLP" in classes
    assert sorted((set(CASES) | set(EXEMPT)) - classes) == []
