"""Cases for the timeseries and audio modalities, and the molecular modality's extension.

``artifex.generative_models.modalities.{timeseries, audio, molecular}``.
"""

from __future__ import annotations

import jax
from flax import nnx
from substrax.testing.nnx_conformance import Case
from substrax.typing import PyTree

from tests.artifex.nnx_conformance.cases.inputs import arrays, normal


_MODALITIES = "artifex.generative_models.modalities"
_STEPS, _FEATURES, _PERIOD = 16, 2, 4
_SAMPLE_RATE, _N_FFT, _HOP = 64, 16, 4


def _decomposed(module: nnx.Module, x: jax.Array) -> PyTree:
    components = module(x)
    return components, module.reconstruct(components)


def _round_trip(module: nnx.Module, audio: jax.Array) -> PyTree:
    features = module.to_representation(audio)
    return features, module.from_representation(features)


def cases() -> dict[str, Case]:
    """Return the timeseries and audio cases, by the qualified name of the class each builds."""

    from artifex.generative_models.modalities.audio import (  # noqa: PLC0415  # noqa: PLC0415
        base as audio_base,
        representations as audio,
    )
    from artifex.generative_models.modalities.timeseries import (  # noqa: PLC0415  # noqa: PLC0415  # noqa: PLC0415
        base as timeseries_base,
        evaluation as timeseries_evaluation,
        representations as timeseries,
    )

    series = timeseries_base.TimeseriesModalityConfig(
        sequence_length=_STEPS, num_features=_FEATURES, univariate=False
    )
    waveform = audio_base.AudioModalityConfig(sample_rate=_SAMPLE_RATE, duration=1.0)
    spectrogram = audio_base.AudioModalityConfig(
        representation=audio_base.AudioRepresentation.MEL_SPECTROGRAM,
        sample_rate=_SAMPLE_RATE,
        duration=1.0,
        n_fft=_N_FFT,
        hop_length=_HOP,
        n_mel_channels=4,
    )
    ts = f"{_MODALITIES}.timeseries"
    return {
        f"{ts}.representations.TimeseriesProcessor": Case(
            build=lambda r: timeseries.TimeseriesProcessor(series, _STEPS, _FEATURES, rngs=r),
            inputs=normal(_STEPS, _FEATURES),
            call=lambda module, x: (module(x), module.reverse(module(x))),
            is_differentiable=False,
        ),
        f"{ts}.representations.FourierProcessor": Case(
            build=lambda r: timeseries.FourierProcessor(num_frequencies=4, rngs=r),
            inputs=normal(_STEPS, _FEATURES),
        ),
        f"{ts}.representations.MultiScaleProcessor": Case(
            build=lambda r: timeseries.MultiScaleProcessor([1, 2, 4], rngs=r),
            inputs=normal(_STEPS, _FEATURES),
            call=lambda module, x: (module(x), module.reconstruct(module(x))),
            is_differentiable=False,
        ),
        f"{ts}.representations.TrendDecompositionProcessor": Case(
            build=lambda r: timeseries.TrendDecompositionProcessor(_PERIOD, rngs=r),
            inputs=normal(_STEPS, _FEATURES),
            call=_decomposed,
            is_differentiable=False,
        ),
        f"{ts}.evaluation.TimeseriesEvaluationSuite": Case(
            build=lambda r: timeseries_evaluation.TimeseriesEvaluationSuite(
                series, _STEPS, _FEATURES, rngs=r
            ),
            inputs=arrays((_STEPS, _FEATURES), (_STEPS, _FEATURES)),
            call=lambda module, generated, reference: module.evaluate_batch(generated, reference),
            is_differentiable=False,
            is_rowwise=False,
        ),
        f"{_MODALITIES}.audio.representations.WaveformProcessor": Case(
            build=lambda r: audio.WaveformProcessor(waveform, rngs=r),
            inputs=normal(_SAMPLE_RATE),
            call=_round_trip,
            is_differentiable=False,
        ),
        f"{_MODALITIES}.audio.representations.SpectrogramProcessor": Case(
            build=lambda r: audio.SpectrogramProcessor(spectrogram, rngs=r),
            inputs=normal(_SAMPLE_RATE),
            call=_round_trip,
            is_differentiable=False,
        ),
        f"{_MODALITIES}.audio.base.AudioModality": Case(
            # loss_fn is its array computation; generate is a draw method, whose output must
            # depend on its key, so it is tested with the F1.3 draw methods, not here.
            build=lambda r: audio_base.AudioModality(waveform, rngs=r),
            inputs=arrays((_SAMPLE_RATE,), (_SAMPLE_RATE,)),
            call=lambda module, target, prediction: module.loss_fn(
                {"audio": target}, {"audio": prediction}
            ),
            is_differentiable=False,
            # The loss is a mean over the batch.
            is_rowwise=False,
        ),
    }


EXEMPT: dict[str, str] = {
    f"{_MODALITIES}.audio.representations.AudioProcessor": (
        "base class whose to_representation and from_representation raise"
        " NotImplementedError; WaveformProcessor's and SpectrogramProcessor's cases run it"
    ),
    f"{_MODALITIES}.molecular.modality.PharmacophoreExtension": (
        "R437 placeholder: its __call__ is ModelExtension's, which raises NotImplementedError,"
        " and apply returns its input unchanged, so no check has a computation to run; the"
        " drug-discovery plan deletes modalities/molecular after W8"
    ),
}
