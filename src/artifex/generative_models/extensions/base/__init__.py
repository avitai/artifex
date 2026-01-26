"""Base extension interfaces for generative models."""

from artifex.generative_models.extensions.base.extensions import (
    AugmentationExtension,
    CallbackExtension,
    ConstraintExtension,
    EvaluationExtension,
    Extension,
    ExtensionDict,
    LossExtension,
    ModalityExtension,
    ModelExtension,
    SamplingExtension,
)


__all__ = [
    "Extension",
    "ExtensionDict",
    "ModelExtension",
    "ConstraintExtension",
    "AugmentationExtension",
    "SamplingExtension",
    "LossExtension",
    "EvaluationExtension",
    "CallbackExtension",
    "ModalityExtension",
]
