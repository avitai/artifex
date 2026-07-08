"""Golden behavior tests for Calibrax-compatible text metrics."""

from __future__ import annotations

import pytest
from calibrax.metrics.functional.text import (
    bleu as calibrax_bleu,
    distinct_n as calibrax_distinct_n,
    rouge_l as calibrax_rouge_l,
    rouge_n as calibrax_rouge_n,
)
from flax import nnx

from artifex.benchmarks.metrics.text import BLEUMetric, DiversityMetric, ROUGEMetric
from artifex.generative_models.core.configuration import EvaluationConfig


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Create deterministic test RNGs."""
    return nnx.Rngs(123)


def _metric_config(metric_name: str, params: dict) -> EvaluationConfig:
    return EvaluationConfig(
        name=f"{metric_name}_calibrax_semantics",
        metrics=[metric_name],
        metric_params={metric_name: params},
        eval_batch_size=4,
    )


def test_default_rouge_names_compute_nonzero_scores(rngs: nnx.Rngs) -> None:
    """Default public ROUGE keys must compute real scores, not silent zeros."""
    metric = ROUGEMetric(
        rngs=rngs,
        config=_metric_config("rouge", {"rouge_types": ["rouge1", "rouge2", "rougeL"]}),
    )

    result = metric.compute(
        ["The quick brown fox jumps over the dog"],
        ["The quick brown fox jumps over the dog"],
    )

    assert set(result) == {"rouge1", "rouge2", "rougeL"}
    assert result["rouge1"] == pytest.approx(1.0)
    assert result["rouge2"] == pytest.approx(1.0)
    assert result["rougeL"] == pytest.approx(1.0)


def test_hyphenated_rouge_aliases_remain_supported(rngs: nnx.Rngs) -> None:
    """Existing hyphenated ROUGE consumers keep their requested output keys."""
    metric = ROUGEMetric(
        rngs=rngs,
        config=_metric_config("rouge", {"rouge_types": ["rouge-1", "rouge-2", "rouge-l"]}),
    )

    result = metric.compute(
        ["alpha beta gamma delta"],
        ["alpha beta gamma delta"],
    )

    assert set(result) == {"rouge-1", "rouge-2", "rouge-l"}
    assert result["rouge-1"] == pytest.approx(1.0)
    assert result["rouge-2"] == pytest.approx(1.0)
    assert result["rouge-l"] == pytest.approx(1.0)


def test_rouge_uses_calibrax_primitives_with_artifex_tokenization(rngs: nnx.Rngs) -> None:
    """ROUGE values should match Calibrax when fed Artifex-normalized tokens."""
    metric = ROUGEMetric(
        rngs=rngs,
        config=_metric_config("rouge", {"rouge_types": ["rouge1", "rouge2", "rougeL"]}),
    )
    reference = "Alpha, beta beta gamma!"
    generated = "alpha beta gamma beta"

    result = metric.compute([reference], [generated])

    reference_tokens = metric._tokenize(reference)
    generated_tokens = metric._tokenize(generated)
    assert result["rouge1"] == pytest.approx(
        calibrax_rouge_n(generated_tokens, reference_tokens, n=1)
    )
    assert result["rouge2"] == pytest.approx(
        calibrax_rouge_n(generated_tokens, reference_tokens, n=2)
    )
    assert result["rougeL"] == pytest.approx(calibrax_rouge_l(generated_tokens, reference_tokens))


def test_bleu_uses_calibrax_primitives_and_configured_weights(rngs: nnx.Rngs) -> None:
    """BLEU should preserve Artifex tokenization while honoring public weights."""
    metric = BLEUMetric(
        rngs=rngs,
        config=_metric_config(
            "bleu",
            {
                "max_n": 2,
                "weights": [0.8, 0.2],
                "smooth": True,
            },
        ),
    )
    reference = "The cat sat on the mat"
    generated = "the cat sat on mat"

    result = metric.compute([reference], [generated])

    reference_tokens = metric._tokenize(reference)
    generated_tokens = metric._tokenize(generated)
    expected = calibrax_bleu(
        generated_tokens,
        [reference_tokens],
        max_n=2,
        weights=(0.8, 0.2),
    )
    assert result["bleu_score"] == pytest.approx(expected)


def test_diversity_corpus_semantics_are_not_single_sequence_distinct_n(
    rngs: nnx.Rngs,
) -> None:
    """Artifex diversity aggregates per-document n-grams, unlike distinct_n."""
    metric = DiversityMetric(
        rngs=rngs,
        config=_metric_config("diversity", {"n_gram_sizes": [2]}),
    )
    generated = ["alpha beta alpha", "alpha beta alpha"]

    result = metric.compute(generated, generated)

    concatenated_tokens = [token for text in generated for token in metric._tokenize(text)]
    assert result["diversity_2gram"] == pytest.approx(0.5)
    assert result["diversity_2gram"] != pytest.approx(calibrax_distinct_n(concatenated_tokens, n=2))
