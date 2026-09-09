"""Contracts for the evaluation code artifex delegates to calibrax."""

from __future__ import annotations

import importlib

import pytest


HELPER_MODULES = (
    "artifex.generative_models.core.evaluation",
    "artifex.generative_models.core.evaluation.metrics",
    "artifex.generative_models.core.evaluation.metrics.base",
    "artifex.generative_models.core.evaluation.metrics.pipeline",
    "artifex.generative_models.core.evaluation.metrics.registry",
    "artifex.generative_models.core.evaluation.metrics.image.fid",
    "artifex.generative_models.core.evaluation.metrics.image.inception_score",
    "artifex.generative_models.core.evaluation.metrics.general.precision_recall",
    "artifex.generative_models.core.evaluation.metrics.text.perplexity",
    "artifex.generative_models.core.evaluation.metrics.metric_ops",
    "artifex.generative_models.core.evaluation.metrics.statistical",
    "artifex.generative_models.core.evaluation.metrics.distance",
    "artifex.generative_models.core.evaluation.metrics.quality",
    "artifex.generative_models.core.evaluation.metrics.information",
)
DELEGATIONS = (
    (
        "artifex.generative_models.modalities.tabular.evaluation",
        "kolmogorov_smirnov_distance",
        "calibrax.metrics.functional.divergence",
    ),
    (
        "artifex.generative_models.modalities.tabular.evaluation",
        "correlation_preservation",
        "calibrax.metrics.functional.statistical",
    ),
    (
        "artifex.generative_models.modalities.tabular.evaluation",
        "distance_to_closest_record",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.generative_models.modalities.tabular.evaluation",
        "memorization_rate",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.generative_models.modalities.timeseries.evaluation",
        "autocorrelation",
        "calibrax.metrics.functional.statistical",
    ),
    (
        "artifex.generative_models.modalities.timeseries.evaluation",
        "skewness",
        "calibrax.metrics.functional.statistical",
    ),
    (
        "artifex.benchmarks.metrics.molecular_flows",
        "pairwise_rmsd",
        "calibrax.metrics.functional.geometric",
    ),
    (
        "artifex.benchmarks.metrics.image",
        "frechet_feature_distance",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.benchmarks.metrics.image",
        "inception_score_per_split",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.benchmarks.metrics.precision_recall",
        "manifold_precision",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.benchmarks.metrics.precision_recall",
        "manifold_recall",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.benchmarks.metrics.precision_recall",
        "density_weighted_precision",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.benchmarks.metrics.precision_recall",
        "density_weighted_recall",
        "calibrax.metrics.functional.generative",
    ),
    (
        "artifex.benchmarks.metrics.text",
        "perplexity",
        "calibrax.metrics.functional.text",
    ),
    (
        "artifex.generative_models.modalities.text.evaluation",
        "perplexity",
        "calibrax.metrics.functional.text",
    ),
)


@pytest.mark.parametrize("module", HELPER_MODULES)
def test_the_helper_modules_are_gone(module: str) -> None:
    """The core evaluation package and its helpers are gone; the metric math is calibrax's."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize(("consumer", "name", "owner"), DELEGATIONS)
def test_consumers_bind_the_calibrax_function(consumer: str, name: str, owner: str) -> None:
    """Each former helper call site binds calibrax's function, not a local copy."""
    consumer_module = importlib.import_module(consumer)
    owner_module = importlib.import_module(owner)

    assert getattr(consumer_module, name) is getattr(owner_module, name)
