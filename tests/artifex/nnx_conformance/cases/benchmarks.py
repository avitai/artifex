"""Cases for the benchmark modules (``artifex.benchmarks``) and the protein dataset.

Each metric is built as its tests build it, an ``EvaluationConfig`` whose ``metric_params``
are keyed by the metric (benchmarks/metrics/core.py:30-61), and called through ``compute`` with
the arrays its tests and suites pass. A metric is a reduction over its inputs, so no metric case
claims independent rows. Where a metric needs a model, it gets the callable its tests use.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.testing.nnx_conformance import Case

from tests.artifex.nnx_conformance.cases.inputs import arrays, BATCH, integers, normal, uniform


_BENCHMARKS = "artifex.benchmarks"
_METRICS = f"{_BENCHMARKS}.metrics"
_SAMPLES, _FACTORS, _LATENTS = 40, 3, 6
_IMAGE = (16, 16, 3)
_ATOMS = 10


def _config(key: str | None, **params: object) -> object:
    from artifex.generative_models.core.configuration import EvaluationConfig  # noqa: PLC0415

    metric_params = {key: dict(params)} if key is not None else dict(params)
    return EvaluationConfig(
        name=key or "metric", metrics=(key or "metric",), metric_params=metric_params
    )


def _compute(module: nnx.Module, *inputs: jax.Array) -> object:
    return module.compute(*inputs)


_images = uniform(_IMAGE, _IMAGE)
"""Real and generated images in [0, 1]."""


def _factors_and_latents(key: jax.Array) -> tuple[jax.Array, ...]:
    """Ground-truth factors (one binary, the rest continuous) and latents, one sample per row."""
    k1, k2, k3 = jax.random.split(key, 3)
    binary = jax.random.bernoulli(k1, shape=(_SAMPLES, 1)).astype(jnp.float32)
    continuous = jax.random.normal(k2, (_SAMPLES, _FACTORS - 1))
    return jnp.concatenate([binary, continuous], axis=1), jax.random.normal(
        k3, (_SAMPLES, _LATENTS)
    )


def _molecules(key: jax.Array) -> tuple[jax.Array, ...]:
    """Coordinates, atom types in [1, 8) and an all-valid mask, one molecule per row."""
    k1, k2 = jax.random.split(key)
    return (
        jax.random.normal(k1, (BATCH, _ATOMS, 3)) * 2.0,
        jax.random.randint(k2, (BATCH, _ATOMS), 1, 8),
        jnp.ones((BATCH, _ATOMS), dtype=bool),
    )


def _flatten_features(width: int) -> Callable[[jax.Array], jax.Array]:
    """A stand-in feature extractor or classifier: the first ``width`` pixel values of each image,
    as the metrics' tests pass (tests/artifex/benchmarks/metrics/test_calibrax_delegation.py:41-50).
    """
    return lambda images: images.reshape(images.shape[0], -1)[:, :width]


class _Predictor(nnx.Module):
    """A model the generative adapter wraps: ``__call__(x, *, rngs)``, as the adapter calls it."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(5, 4, rngs=rngs)

    def __call__(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        del rngs
        return self.linear(x)


class _PointCloudModel(nnx.Module):
    """A protein model the point-cloud adapter wraps: ``predict(x, *, rngs)`` to (B, P, 3)."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(5, 4 * 3, rngs=rngs)

    def predict(self, x: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        del rngs
        return self.linear(x).reshape(x.shape[0], 4, 3)


def cases() -> dict[str, Case]:
    """Return the benchmark cases, by the qualified name of the class each builds."""

    from artifex.benchmarks.metrics import (  # noqa: PLC0415
        audio,
        disentanglement,
        geometric,
        image,
        molecular_flows,
        precision_recall,
        protein_ligand,
        style_metrics,
        text,
    )
    from artifex.benchmarks.model_adapters import (  # noqa: PLC0415
        generative,
        protein_adapters,
    )
    from artifex.data.protein import dataset  # noqa: PLC0415

    def metric(cls: type, key: str | None, **params: object) -> Callable[[nnx.Rngs], nnx.Module]:
        return lambda r: cls(config=_config(key, **params), rngs=r)

    def statistic(**claims: bool) -> dict[str, bool]:
        return {"is_differentiable": False, "is_rowwise": False, **claims}

    def log_probabilities(tokens: jax.Array) -> jax.Array:
        # The test's model (test_calibrax_delegation.py:52-53).
        return -0.5 * jnp.ones(tokens.shape) - 0.1 * tokens

    synthetic_proteins = 8
    return {
        f"{_METRICS}.audio.SpectralMetric": Case(
            build=metric(audio.SpectralMetric, "spectral", n_fft=32, hop_length=16),
            inputs=arrays((128,), (128,)),
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.audio.MelCepstralMetric": Case(
            build=metric(audio.MelCepstralMetric, "mcd", n_mels=8, n_fft=32),
            inputs=arrays((128,), (128,)),
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.disentanglement.MutualInformationGapMetric": Case(
            build=metric(disentanglement.MutualInformationGapMetric, "mig"),
            inputs=_factors_and_latents,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.disentanglement.SeparationMetric": Case(
            build=metric(disentanglement.SeparationMetric, "sap"),
            inputs=_factors_and_latents,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.disentanglement.DisentanglementMetric": Case(
            build=metric(disentanglement.DisentanglementMetric, "dci"),
            inputs=_factors_and_latents,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.geometric.PointCloudMetrics": Case(
            build=metric(geometric.PointCloudMetrics, "point_cloud"),
            inputs=arrays((16, 3), (16, 3)),
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.image.FIDMetric": Case(
            build=metric(image.FIDMetric, "fid", feature_extractor=_flatten_features(8)),
            inputs=_images,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.image.ISMetric": Case(
            build=metric(
                image.ISMetric, "inception_score", classifier=_flatten_features(5), splits=2
            ),
            inputs=_images,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.image.LPIPSMetric": Case(
            # Its only path: without the demo flags it raises (image.py:356-365).
            build=metric(image.LPIPSMetric, "lpips", mock_implementation=True, demo_mode=True),
            inputs=_images,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.image.SSIMMetric": Case(
            build=metric(image.SSIMMetric, "ssim", window_size=11),
            inputs=_images,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.molecular_flows.MolecularFlowsMetrics": Case(
            build=metric(molecular_flows.MolecularFlowsMetrics, None),
            inputs=_molecules,
            # Its suite passes the generated molecules as a dict; real_data is unused
            # (se3_molecular_flows_suite.py:164-168).
            call=lambda module, coordinates, atom_types, mask: module.compute(
                None,
                {"coordinates": coordinates, "atom_types": atom_types, "atom_mask": mask},
            ),
            **statistic(),
        ),
        f"{_METRICS}.precision_recall.PrecisionRecallMetric": Case(
            build=metric(precision_recall.PrecisionRecallMetric, "precision_recall", k=2),
            inputs=arrays((2,), (2,)),
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.protein_ligand.BindingAffinityMetric": Case(
            build=metric(protein_ligand.BindingAffinityMetric, None, higher_is_better=False),
            inputs=arrays((), ()),
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.protein_ligand.MolecularValidityMetric": Case(
            build=metric(protein_ligand.MolecularValidityMetric, None, higher_is_better=True),
            inputs=_molecules,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.protein_ligand.DrugLikenessMetric": Case(
            # Its only path: without demo_mode it raises (protein_ligand.py:282-290).
            build=metric(protein_ligand.DrugLikenessMetric, "drug_likeness", demo_mode=True),
            inputs=_molecules,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.style_metrics.InceptionV3Feature": Case(
            build=lambda r: style_metrics.InceptionV3Feature(rngs=r),
            inputs=uniform(_IMAGE, minval=-1.0),
        ),
        f"{_METRICS}.style_metrics._StyleLPIPS": Case(
            build=lambda r: style_metrics._StyleLPIPS(rngs=r),  # noqa: SLF001
            inputs=_images,
            call=lambda module, first, second: module.compute_distance(first, second),
        ),
        f"{_METRICS}.style_metrics.StyleGANMetrics": Case(
            # As the StyleGAN3 suite's tests build it (test_stylegan3_suite.py:441-447), called
            # through compute on real and generated images, which runs its FID features.
            build=lambda r: style_metrics.StyleGANMetrics(
                config=_config(None, demo_mode=True, higher_is_better=False), rngs=r
            ),
            inputs=_images,
            call=_compute,
            **statistic(),
        ),
        f"{_METRICS}.text.PerplexityMetric": Case(
            build=metric(text.PerplexityMetric, "perplexity", model=log_probabilities),
            inputs=integers(0, 20, 5),
            call=lambda module, tokens: module.compute(None, tokens),
            **statistic(),
        ),
        f"{_BENCHMARKS}.model_adapters.generative.NNXGenerativeModelAdapter": Case(
            build=lambda r: generative.NNXGenerativeModelAdapter(_Predictor(rngs=r)),
            inputs=normal(5),
            # predict requires rngs (generative.py:68-71), as its tests pass them.
            call=lambda module, x, *, rngs: module.predict(x, rngs=rngs),
            rng_streams=("dropout",),
        ),
        f"{_BENCHMARKS}.model_adapters.protein_adapters.ProteinPointCloudAdapter": Case(
            build=lambda r: protein_adapters.ProteinPointCloudAdapter(_PointCloudModel(rngs=r)),
            inputs=normal(5),
            call=lambda module, x, *, rngs: module.predict(x, rngs=rngs),
            rng_streams=("sample",),
        ),
        "artifex.data.protein.dataset.ProteinDataset": Case(
            # Built as create_synthetic_protein_dataset builds it, and read through get_records,
            # the traced-index path datarax's compiled pipeline uses (dataset.py:401-409).
            build=lambda r: dataset.create_synthetic_protein_dataset(
                num_proteins=synthetic_proteins, min_seq_length=5, max_seq_length=8, random_seed=0
            ),
            inputs=lambda key: (jax.random.randint(key, (BATCH,), 0, synthetic_proteins),),
            call=lambda module, indices: module.get_records(indices),
            is_differentiable=False,
        ),
    }


_STRINGS = (
    "its compute takes Python lists of strings and returns host floats, with no array"
    " computation for a check to run ({tests}); its defects are {rows}"
)
_NEEDS_GENERATOR = (
    "its computation samples a StyleGAN3 generator passed per call ({where}), and its plain"
    " compute without one returns constant zeros ({row}); the generator's own computation is"
    " checked by the StyleGAN3Generator case, its feature networks by the InceptionV3Feature"
    " and _StyleLPIPS cases"
)
EXEMPT: dict[str, str] = {
    f"{_METRICS}.text.BLEUMetric": _STRINGS.format(
        tests="test_text_metrics_unified_config.py:20-33", rows="R407, R408"
    ),
    f"{_METRICS}.text.ROUGEMetric": _STRINGS.format(
        tests="test_text_metrics_calibrax_semantics.py:35-43", rows="R407"
    ),
    f"{_METRICS}.text.DiversityMetric": _STRINGS.format(
        tests="test_text_metrics_calibrax_semantics.py:124-130", rows="R325"
    ),
    f"{_METRICS}.style_metrics.StyleMixingMetric": _NEEDS_GENERATOR.format(
        where="style_metrics.py:169-186", row="R546"
    ),
    f"{_METRICS}.style_metrics.EquivarianceMetric": _NEEDS_GENERATOR.format(
        where="style_metrics.py:446-447", row="R546"
    ),
    f"{_METRICS}.style_metrics.FewShotAdaptationMetric": _NEEDS_GENERATOR.format(
        where="two generators and a dataset, style_metrics.py:320-336", row="R406"
    ),
    f"{_BENCHMARKS}.suites.stylegan3_suite.StyleGAN3Benchmark": (
        "a demo-only benchmark runner (RuntimeError without demo_mode, stylegan3_suite.py:142-149)"
        " that builds a full StyleGAN3 generator, discriminator and two datasets and whose"
        " run_training returns constants (R333) and whose discriminator crashes on its own images"
        " (R413); its networks are checked by the GAN family's cases"
    ),
}
