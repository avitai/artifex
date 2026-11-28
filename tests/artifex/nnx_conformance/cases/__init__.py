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

from tests.artifex.nnx_conformance.cases import (
    audio,
    autoregressive,
    backbones,
    benchmarks,
    core,
    diffusion,
    distributions,
    energy,
    extensions,
    flow,
    gan,
    geometric,
    image_modality,
    multi_modal,
    objectives,
    signal_modalities,
    tabular_modality,
    text_modality,
    vae,
)


def all_cases() -> dict[str, Case]:
    """Return every case, by the qualified name of the class it builds.

    A class whose method draws has a further case for it, keyed ``<class>#<method>``.
    """
    return {
        **core.layer_cases(),
        **core.base_cases(),
        **backbones.cases(),
        **vae.cases(),
        **flow.cases(),
        **gan.cases(),
        **diffusion.cases(),
        **autoregressive.cases(),
        **energy.cases(),
        **geometric.cases(),
        **audio.cases(),
        **objectives.cases(),
        **distributions.cases(),
        **distributions.draw_cases(),
        **extensions.cases(),
        **image_modality.cases(),
        **multi_modal.cases(),
        **text_modality.cases(),
        **tabular_modality.cases(),
        **signal_modalities.cases(),
        **benchmarks.cases(),
    }


EXEMPT: dict[str, str] = {
    **core.EXEMPT,
    **flow.EXEMPT,
    **autoregressive.EXEMPT,
    **energy.EXEMPT,
    **geometric.EXEMPT,
    **audio.EXEMPT,
    **objectives.EXEMPT,
    **distributions.EXEMPT,
    **extensions.EXEMPT,
    **image_modality.EXEMPT,
    **multi_modal.EXEMPT,
    **text_modality.EXEMPT,
    **tabular_modality.EXEMPT,
    **signal_modalities.EXEMPT,
    **benchmarks.EXEMPT,
}
"""Classes with no case of their own, each with the reason."""

NOT_APPLICABLE: dict[tuple[str, str], str] = {**vae.NOT_APPLICABLE, **distributions.NOT_APPLICABLE}
"""Checks a case's module does not owe, keyed like the known failures, each with the reason.

A check belongs here when the method the module implements makes the property false by design.
A defect belongs in ``known_failures.py``, which a fix empties.
"""
