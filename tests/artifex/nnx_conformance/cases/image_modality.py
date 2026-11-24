"""Cases for the image modality, ``artifex.generative_models.modalities.image``.

Its modules hold no ``nnx.Param``: processors transform a batch of images and metrics compare
two batches. A metric's value is a reduction over the batch, so metric cases do not claim
independent rows.
"""

from __future__ import annotations

import jax
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import uniform


_IMAGE = "artifex.generative_models.modalities.image"
_SIDE, _CHANNELS = 32, 3


_IMAGE_SHAPE = (_SIDE, _SIDE, _CHANNELS)
_images = uniform(_IMAGE_SHAPE)
_image_pairs = uniform(_IMAGE_SHAPE, _IMAGE_SHAPE)
"""Generated and reference images in [0, 1]."""


def _processed(module: nnx.Module, images: jax.Array) -> PyTree:
    normalized = module.normalize(images)
    return normalized, module.denormalize(normalized), module.rgb_to_grayscale(images)


def _pyramid(module: nnx.Module, images: jax.Array) -> PyTree:
    pyramid = module.create_multi_scale_pyramid(images)
    return pyramid, module.combine_scales(pyramid)


def cases() -> dict[str, Case]:
    """Return the image-modality cases, by the qualified name of the class each builds."""

    from artifex.generative_models.modalities.image import (  # noqa: PLC0415
        base,
        evaluation,
        representations,
    )

    config = base.ImageModalityConfig(height=_SIDE, width=_SIDE, channels=_CHANNELS)
    metrics = {
        "MSEMetric": evaluation.MSEMetric,
        "PSNRMetric": evaluation.PSNRMetric,
        "SSIMMetric": evaluation.SSIMMetric,
        "MSSSIMMetric": evaluation.MSSSIMMetric,
        "VendiScoreMetric": evaluation.VendiScoreMetric,
        "PerceptualDistanceMetric": evaluation.PerceptualDistanceMetric,
    }
    return {
        f"{_IMAGE}.base.ImageModality": Case(
            build=lambda r: base.ImageModality(config, rngs=r),
            inputs=_images,
            call=lambda module, images: module.process(images),
            is_differentiable=False,
        ),
        **{
            f"{_IMAGE}.evaluation.{name}": Case(
                build=lambda r, metric=metric: metric(config, rngs=r),
                inputs=_image_pairs,
                call=lambda module, generated, reference: module.compute_metric(
                    generated, reference
                ),
                is_differentiable=False,
                is_rowwise=False,
            )
            for name, metric in metrics.items()
        },
        f"{_IMAGE}.evaluation.ImageEvaluationSuite": Case(
            build=lambda r: evaluation.ImageEvaluationSuite(config, rngs=r),
            inputs=_image_pairs,
            call=lambda module, generated, reference: module.evaluate(generated, reference),
            is_differentiable=False,
            is_rowwise=False,
        ),
        f"{_IMAGE}.representations.ImageProcessor": Case(
            build=lambda r: representations.ImageProcessor(config, rngs=r),
            inputs=_images,
            call=_processed,
            is_differentiable=False,
        ),
        f"{_IMAGE}.representations.AugmentationProcessor": Case(
            build=lambda r: representations.AugmentationProcessor(config, rngs=r),
            inputs=_images,
            call=lambda module, images: module.augment_batch(images),
            is_differentiable=False,
        ),
        f"{_IMAGE}.representations.MultiScaleProcessor": Case(
            build=lambda r: representations.MultiScaleProcessor(config, rngs=r),
            inputs=_images,
            call=_pyramid,
            is_differentiable=False,
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_IMAGE}.evaluation.ImageMetrics": (
        "base class whose compute_metric raises NotImplementedError; each metric's case runs it"
    ),
}
