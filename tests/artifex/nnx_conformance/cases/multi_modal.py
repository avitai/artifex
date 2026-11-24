"""Cases for the multi-modal modality, ``artifex.generative_models.modalities.multi_modal``.

``MultiModalProcessor``, ``CrossModalProcessor``, ``ModalityFusionProcessor`` and the adapters
take one sample per call: they flatten their whole input (``x.reshape(-1)``), and artifex's own
tests call them that way (tests/artifex/generative_models/modalities/
test_multi_modal_modality.py:140-146, 159-165, 178-184, 331-337: an image of (32, 32, 3) and
text of (100,) or (50,), no batch axis). Their cases call them on each row in turn and stack the
outputs, which is that contract applied to a batch. R385 records that a batch crashes
MultiModalProcessor; whether these modules should take a batch is the fix branch's decision.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import BATCH, each_row


_MULTI_MODAL = "artifex.generative_models.modalities.multi_modal"
_IMAGE, _TEXT, _AUDIO = (32, 32, 3), 100, 256
_HIDDEN = 16


def _image_and_text(key: jax.Array) -> tuple[jax.Array, ...]:
    k1, k2 = jax.random.split(key)
    return jax.random.normal(k1, (BATCH, *_IMAGE)), jax.random.normal(k2, (BATCH, _TEXT))


def _features(*widths: int) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    def inputs(key: jax.Array) -> tuple[jax.Array, ...]:
        keys = jax.random.split(key, len(widths))
        return tuple(jax.random.normal(k, (BATCH, w)) for k, w in zip(keys, widths, strict=True))

    return inputs


def cases() -> dict[str, Case]:
    """Return the multi-modal cases, by the qualified name of the class each builds."""

    from artifex.generative_models.modalities.multi_modal import (  # noqa: PLC0415
        adapters,
        evaluation,
        representations,
    )

    vae_config = adapters.MultiModalVAEAdapterConfig(
        name="vae_adapter", modalities=("image", "text"), shared_latent_dim=8, latent_dim=8
    )
    adapter_config = adapters.MultiModalAdapterConfig(
        name="adapter",
        modalities=("image", "text"),
        model_type="vae",
        shared_latent_dim=8,
        latent_dim=8,
    )

    def image_and_text(module: nnx.Module, image: jax.Array, text: jax.Array) -> PyTree:
        return module({"image": image, "text": text})

    def adapter_forward(module: nnx.Module, image: jax.Array, text: jax.Array) -> PyTree:
        return module.forward({"image": image, "text": text})

    return {
        f"{_MULTI_MODAL}.representations.MultiModalProcessor": Case(
            build=lambda r: representations.MultiModalProcessor(
                ["image", "text"], _HIDDEN, hidden_dims=[_HIDDEN], dropout_rate=0.5, rngs=r
            ),
            inputs=_image_and_text,
            call=each_row(
                lambda module, image, text, *, rngs: module(
                    {"image": image, "text": text}, rngs=rngs
                )
            ),
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_MULTI_MODAL}.representations.CrossModalProcessor": Case(
            build=lambda r: representations.CrossModalProcessor(
                "image", "text", _HIDDEN, hidden_dims=[_HIDDEN], rngs=r
            ),
            # R537: both projectors take 512 features whatever the modalities are named
            # (representations.py:177, 193), as the module's own test passes them.
            inputs=_features(512, 512),
            call=each_row(lambda module, source, target: module(source, target)),
        ),
        f"{_MULTI_MODAL}.representations.ModalityFusionProcessor": Case(
            build=lambda r: representations.ModalityFusionProcessor(
                ["image", "text"], output_dim=_HIDDEN, hidden_dim=_HIDDEN, num_heads=2, rngs=r
            ),
            inputs=_features(_HIDDEN, _HIDDEN),
            call=each_row(image_and_text),
        ),
        f"{_MULTI_MODAL}.representations.CrossModalAttention": Case(
            build=lambda r: representations.CrossModalAttention(
                _HIDDEN, _HIDDEN, _HIDDEN, num_heads=2, dropout_rate=0.5, rngs=r
            ),
            inputs=lambda key: tuple(
                jax.random.normal(k, (BATCH, 3, _HIDDEN)) for k in jax.random.split(key, 3)
            ),
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_MULTI_MODAL}.representations.HierarchicalFusion": Case(
            build=lambda r: representations.HierarchicalFusion(
                [["image", "text"], ["audio"]],
                [_HIDDEN, _HIDDEN],
                _HIDDEN,
                {"image": 12, "text": 10, "audio": 8},
                rngs=r,
            ),
            inputs=_features(12, 10, 8),
            call=lambda module, image, text, audio: module(
                {"image": image, "text": text, "audio": audio}
            ),
        ),
        f"{_MULTI_MODAL}.representations.ModalityDropout": Case(
            # The three modalities its own test drops (test_multi_modal_modality.py:467-469).
            build=lambda r: representations.ModalityDropout(["image", "text", "audio"], rngs=r),
            inputs=_features(_HIDDEN, _HIDDEN, _HIDDEN),
            call=lambda module, image, text, audio, *, rngs: module(
                {"image": image, "text": text, "audio": audio}, rngs=rngs
            ),
            is_differentiable=False,
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_MULTI_MODAL}.adapters.MultiModalAdapter": Case(
            build=lambda r: adapters.MultiModalAdapter(adapter_config, rngs=r),
            inputs=_image_and_text,
            call=each_row(adapter_forward),
        ),
        f"{_MULTI_MODAL}.adapters.MultiModalVAEAdapter": Case(
            build=lambda r: adapters.MultiModalVAEAdapter(vae_config, rngs=r),
            inputs=_image_and_text,
            call=each_row(adapter_forward),
        ),
        f"{_MULTI_MODAL}.evaluation.CrossModalContrastiveLoss": Case(
            build=lambda r: evaluation.CrossModalContrastiveLoss(),
            inputs=_features(_HIDDEN, _HIDDEN),
            is_differentiable=False,
            # The loss contrasts every pair in the batch.
            is_rowwise=False,
        ),
        f"{_MULTI_MODAL}.evaluation.MultiModalEvaluationSuite": Case(
            build=lambda r: evaluation.MultiModalEvaluationSuite(
                ["image", "audio"], alignment_dim=_HIDDEN, rngs=r
            ),
            inputs=lambda key: (
                *_image_and_text(key)[:1],
                jax.random.normal(jax.random.fold_in(key, 1), (BATCH, _AUDIO)),
                *_image_and_text(jax.random.fold_in(key, 2))[:1],
                jax.random.normal(jax.random.fold_in(key, 3), (BATCH, _AUDIO)),
            ),
            call=lambda module, image, audio, reference_image, reference_audio: module.evaluate(
                {"image": image, "audio": audio},
                {"image": reference_image, "audio": reference_audio},
            ),
            is_differentiable=False,
            is_rowwise=False,
        ),
    }


EXEMPT: dict[str, str] = {}
