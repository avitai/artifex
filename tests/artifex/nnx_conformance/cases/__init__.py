"""One conformance case per artifex ``nnx.Module`` class: how to build it and call it.

A case is a :class:`substrax.testing.nnx_conformance.Case`: it builds the module, makes inputs
with a leading batch axis of ``BATCH``, and states the capabilities the module claims (dropout,
batch statistics, rows computed independently). ``test_conformance.py`` runs every substrax
check whose claim a case makes. Each family has its own module, whose ``cases()`` this package
gathers in :func:`all_cases`. ``EXEMPT`` names the classes with no case of their own, each with
the reason.

Where a module's activation is configurable, its case uses ``nnx.silu``: a finite difference
taken across a ReLU kink is no derivative (ResNetBlock's float64 check disagrees by 0.64% with
``relu`` and agrees with ``silu``), and the activation is a library function, not this code.
"""

from __future__ import annotations

from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases import backbones, core, diffusion, flow, gan, vae


def all_cases() -> dict[str, Case]:
    """Return every case, by the qualified name of the class it builds."""
    return {
        **core.layer_cases(),
        **core.base_cases(),
        **backbones.cases(),
        **vae.cases(),
        **flow.cases(),
        **gan.cases(),
        **diffusion.cases(),
    }


EXEMPT: dict[str, str] = {**core.EXEMPT, **flow.EXEMPT}
"""Classes with no case of their own, each with the reason."""

NOT_APPLICABLE: dict[tuple[str, str], str] = {**vae.NOT_APPLICABLE}
"""Checks a case's module does not owe, keyed like the known failures, each with the reason.

A check belongs here when the method the module implements makes the property false by design.
A defect belongs in ``known_failures.py``, which a fix empties.
"""
