"""Golden behavior tests for Calibrax-adjacent audio metrics."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from calibrax.metrics.functional.audio import (
    mel_cepstral_distortion as calibrax_mel_cepstral_distortion,
    spectral_convergence as calibrax_spectral_convergence,
)
from flax import nnx

from artifex.benchmarks.metrics.audio import MelCepstralMetric, SpectralMetric
from artifex.generative_models.core.configuration import EvaluationConfig


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Create deterministic test RNGs."""
    return nnx.Rngs(321)


@pytest.fixture
def audio_pair() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create deterministic batched waveforms with different spectra."""
    time = jnp.arange(2048)
    real = jnp.stack(
        [
            jnp.sin(2 * jnp.pi * 3 * time / 2048),
            jnp.cos(2 * jnp.pi * 7 * time / 2048),
        ]
    )
    generated = jnp.stack(
        [
            jnp.sin(2 * jnp.pi * 5 * time / 2048),
            jnp.cos(2 * jnp.pi * 11 * time / 2048),
        ]
    )
    return real, generated


def _metric_config(metric_name: str, params: dict) -> EvaluationConfig:
    return EvaluationConfig(
        name=f"{metric_name}_calibrax_semantics",
        metrics=[metric_name],
        metric_params={metric_name: params},
        eval_batch_size=2,
    )


def test_spectral_metric_preserves_artifex_batched_stft_semantics(
    rngs: nnx.Rngs, audio_pair: tuple[jnp.ndarray, jnp.ndarray]
) -> None:
    """Direct Calibrax full-FFT spectral convergence is not a drop-in swap."""
    metric = SpectralMetric(
        rngs=rngs,
        config=_metric_config("spectral", {"n_fft": 512, "hop_length": 128}),
    )
    real, generated = audio_pair

    result = metric.compute(real, generated)
    direct_calibrax = float(
        jnp.mean(
            jnp.array(
                [calibrax_spectral_convergence(gen, ref) for ref, gen in zip(real, generated)]
            )
        )
    )

    assert set(result) == {"spectral_convergence"}
    assert result["spectral_convergence"] == pytest.approx(0.6263080835)
    assert result["spectral_convergence"] != pytest.approx(direct_calibrax)


def test_mcd_metric_preserves_artifex_mfcc_pipeline_semantics(
    rngs: nnx.Rngs, audio_pair: tuple[jnp.ndarray, jnp.ndarray]
) -> None:
    """Direct Calibrax cepstral distance lacks Artifex STFT/mel/MFCC ownership."""
    metric = MelCepstralMetric(
        rngs=rngs,
        config=_metric_config("mcd", {"n_mels": 16, "n_fft": 512}),
    )
    real, generated = audio_pair

    result = metric.compute(real, generated)
    direct_calibrax = float(
        jnp.mean(
            jnp.array(
                [
                    calibrax_mel_cepstral_distortion(gen, ref, num_mels=16)
                    for ref, gen in zip(real, generated)
                ]
            )
        )
    )

    assert set(result) == {"mel_cepstral_distortion"}
    # The simplified MFCC path uses float32 FFT/log operations whose last
    # digits differ between CPU and CUDA; keep this as a semantic-scale golden.
    assert result["mel_cepstral_distortion"] == pytest.approx(2.7821, rel=1e-3, abs=2e-3)
    assert result["mel_cepstral_distortion"] != pytest.approx(direct_calibrax)


def test_audio_metrics_return_zero_for_identical_artifex_inputs(
    rngs: nnx.Rngs, audio_pair: tuple[jnp.ndarray, jnp.ndarray]
) -> None:
    """Artifex public metrics keep exact zero for identical batched inputs."""
    real, _ = audio_pair
    spectral = SpectralMetric(
        rngs=rngs,
        config=_metric_config("spectral", {"n_fft": 512, "hop_length": 128}),
    )
    mcd = MelCepstralMetric(
        rngs=rngs,
        config=_metric_config("mcd", {"n_mels": 16, "n_fft": 512}),
    )

    assert spectral.compute(real, real)["spectral_convergence"] == pytest.approx(0.0)
    assert mcd.compute(real, real)["mel_cepstral_distortion"] == pytest.approx(0.0)
