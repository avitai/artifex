"""Cases for the extensions, ``artifex.generative_models.extensions``.

An extension is called as ``extension(inputs, model_outputs)`` and returns a dict of metrics.
Each case calls it the way its model does, and where it has a loss, also ``loss_fn``: a loss is a
batch reduction, so those cases do not claim independent rows. The outputs are returned as the
extension builds them, string entries included.
"""

from __future__ import annotations

import jax
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import (
    BATCH,
    each_row,
    integers,
    normal,
    uniform,
)


_EXTENSIONS = "artifex.generative_models.extensions"
_RESIDUES, _ATOMS = 3, 4
_AUDIO = 4096


def _backbones(key: jax.Array) -> tuple[jax.Array, ...]:
    """Backbone atom positions (N, CA, C, O) of a short chain, one chain per row."""
    return (jax.random.normal(key, (BATCH, _RESIDUES, _ATOMS, 3)),)


def _metrics_and_loss(module: nnx.Module, positions: jax.Array) -> PyTree:
    outputs = {"atom_positions": positions}
    return module({}, outputs), module.loss_fn({}, outputs)


def _sampling(module: nnx.Module, score: jax.Array, t: jax.Array) -> PyTree:
    return (
        module.modify_score(score, t, {}),
        module.filter_samples(score, {}),
        module.post_process_sample(score, {}),
    )


def _analyse(module: nnx.Module, signal: jax.Array) -> PyTree:
    """Analyse one signal.

    ``__call__`` takes a single signal: its test asserts that a batch returns the "single audio"
    error dict (tests/artifex/generative_models/extensions/audio_processing/test_temporal.py:
    537-538), though the docstrings of its compute methods accept a batch (R442).
    """
    return module({}, signal)


def _describe(module: nnx.Module, molecule: jax.Array) -> PyTree:
    """Describe one molecule: every test of MolecularFeatures passes one (N, 3) molecule
    (tests/artifex/generative_models/extensions/chemical/test_features.py:311, 503)."""
    return module({}, molecule)


def cases() -> dict[str, Case]:
    """Return the extension cases, by the qualified name of the class each builds."""

    from artifex.generative_models.core.configuration.extension_config import (  # noqa: PLC0415
        ChemicalConstraintConfig,
        ExtensionConfig,
        ImageAugmentationConfig,
        ProteinDihedralConfig,
        ProteinExtensionConfig,
        ProteinMixinConfig,
        SamplingExtensionConfig,
    )
    from artifex.generative_models.extensions.audio_processing import (  # noqa: PLC0415
        spectral,
        temporal,
    )
    from artifex.generative_models.extensions.base import extensions  # noqa: PLC0415
    from artifex.generative_models.extensions.chemical import (  # noqa: PLC0415
        constraints as chemical,
        features,  # noqa: PLC0415
    )
    from artifex.generative_models.extensions.nlp import (  # noqa: PLC0415
        embeddings,
        tokenization,
    )
    from artifex.generative_models.extensions.protein import (  # noqa: PLC0415
        backbone,
        constraints,
        mixin,
    )
    from artifex.generative_models.extensions.vision import augmentation  # noqa: PLC0415

    protein = ProteinExtensionConfig(name="protein")
    plain = ExtensionConfig(name="extension")
    protein_cases = {
        f"{_EXTENSIONS}.protein.backbone.BondLengthExtension": backbone.BondLengthExtension,
        f"{_EXTENSIONS}.protein.backbone.BondAngleExtension": backbone.BondAngleExtension,
        f"{_EXTENSIONS}.protein.constraints.ProteinBackboneConstraint": (
            constraints.ProteinBackboneConstraint
        ),
    }
    return {
        f"{_EXTENSIONS}.base.extensions.SamplingExtension": Case(
            build=lambda r: extensions.SamplingExtension(
                SamplingExtensionConfig(name="sampling"), rngs=r
            ),
            inputs=lambda key: (
                jax.random.normal(key, (BATCH, 5)),
                jax.random.uniform(jax.random.fold_in(key, 1), (BATCH,)),
            ),
            call=_sampling,
            is_differentiable=False,
        ),
        **{
            name: Case(
                build=lambda r, extension=extension: extension(protein, rngs=r),
                inputs=_backbones,
                call=_metrics_and_loss,
                is_differentiable=False,
                is_rowwise=False,
            )
            for name, extension in protein_cases.items()
        },
        f"{_EXTENSIONS}.protein.constraints.ProteinDihedralConstraint": Case(
            build=lambda r: constraints.ProteinDihedralConstraint(
                ProteinDihedralConfig(name="dihedral"), rngs=r
            ),
            inputs=_backbones,
            call=_metrics_and_loss,
            is_differentiable=False,
            is_rowwise=False,
        ),
        f"{_EXTENSIONS}.protein.mixin.ProteinMixinExtension": Case(
            build=lambda r: mixin.ProteinMixinExtension(
                ProteinMixinConfig(name="mixin", use_one_hot=False, embedding_dim=4), rngs=r
            ),
            inputs=integers(0, 21, _RESIDUES),
            call=lambda module, aatype: module({"aatype": aatype}, {}),
        ),
        f"{_EXTENSIONS}.chemical.constraints.ChemicalConstraints": Case(
            build=lambda r: chemical.ChemicalConstraints(
                ChemicalConstraintConfig(name="chemical"), rngs=r
            ),
            # One molecule per call: validate_molecular_structure documents [n_atoms, 3]
            # (constraints.py:111), as the module's test passes it (test_molecular_modality.py:
            # 229). A batch, which GeometricModel.apply_extensions passes, is R538: below four
            # molecules it scores 1.0 unchecked, from four it crashes.
            inputs=normal(5, 3),
            call=each_row(lambda module, molecule: module({}, molecule)),
            is_differentiable=False,
        ),
        f"{_EXTENSIONS}.chemical.features.MolecularFeatures": Case(
            build=lambda r: features.MolecularFeatures(plain, rngs=r),
            inputs=normal(5, 3),
            call=each_row(_describe),
            is_differentiable=False,
        ),
        f"{_EXTENSIONS}.nlp.embeddings.TextEmbeddings": Case(
            # R534: the config cannot set the sizes, so this builds the defaults (50,000 x 512).
            build=lambda r: embeddings.TextEmbeddings(plain, rngs=r),
            inputs=integers(0, 100, 6),
            call=lambda module, tokens, *, rngs: module({"tokens": tokens}, {}, rngs=rngs),
            has_dropout=True,
            rng_streams=("dropout",),
        ),
        f"{_EXTENSIONS}.nlp.tokenization.AdvancedTokenization": Case(
            build=lambda r: tokenization.AdvancedTokenization(plain, rngs=r),
            inputs=integers(0, 100, 6),
            call=lambda module, tokens: module({}, {"tokens": tokens}),
            is_differentiable=False,
            # Token frequencies are counted over the batch.
            is_rowwise=False,
        ),
        f"{_EXTENSIONS}.audio_processing.spectral.SpectralAnalysis": Case(
            build=lambda r: spectral.SpectralAnalysis(plain, rngs=r),
            inputs=normal(_AUDIO),
            call=lambda module, audio: module({}, audio),
            is_differentiable=False,
        ),
        f"{_EXTENSIONS}.audio_processing.temporal.TemporalAnalysis": Case(
            build=lambda r: temporal.TemporalAnalysis(plain, rngs=r),
            inputs=normal(_AUDIO),
            call=each_row(_analyse),
            is_differentiable=False,
        ),
        f"{_EXTENSIONS}.vision.augmentation.AdvancedImageAugmentation": Case(
            build=lambda r: augmentation.AdvancedImageAugmentation(
                ImageAugmentationConfig(
                    name="augmentation", random_rotation=10.0, color_jitter=True
                ),
                rngs=r,
            ),
            inputs=uniform((6, 6, 3)),
            call=lambda module, images: module.augment(images),
            is_differentiable=False,
        ),
    }


_ABSTRACT = "abstract base class: its {method} raises NotImplementedError; each subclass's case"
_BASE = f"{_EXTENSIONS}.base.extensions"
EXEMPT: dict[str, str] = {
    f"{_BASE}.Extension": "base class with no __call__; each subclass's case runs it",
    f"{_BASE}.ModelExtension": _ABSTRACT.format(method="__call__") + " runs it",
    f"{_BASE}.ConstraintExtension": _ABSTRACT.format(method="validate") + " runs it",
    f"{_BASE}.AugmentationExtension": _ABSTRACT.format(method="augment") + " runs it",
    f"{_BASE}.LossExtension": _ABSTRACT.format(method="compute_loss") + " runs it",
    f"{_BASE}.EvaluationExtension": _ABSTRACT.format(method="compute_metrics") + " runs it",
    f"{_BASE}.ModalityExtension": _ABSTRACT.format(method="preprocess") + " runs it",
    f"{_BASE}.CallbackExtension": (
        "training-lifecycle hooks that return None and compute no arrays, so no check applies"
    ),
    f"{_BASE}.ExtensionDict": (
        "an nnx.Dict container of extensions; each extension's case runs it"
    ),
}
