"""Repo contracts for modality runtime configuration surfaces."""

import dataclasses

import pytest

from artifex.generative_models.core.configuration import BaseModalityConfig
from artifex.generative_models.modalities import base as modality_base
from artifex.generative_models.modalities.audio.base import AudioModalityConfig
from artifex.generative_models.modalities.image.base import ImageModalityConfig
from artifex.generative_models.modalities.multi_modal.base import MultiModalModalityConfig
from artifex.generative_models.modalities.tabular.base import TabularModalityConfig
from artifex.generative_models.modalities.timeseries.base import TimeseriesModalityConfig


def test_base_modality_config_is_the_shared_runtime_config_surface() -> None:
    """The modality package should expose the same BaseModalityConfig type."""
    assert modality_base.BaseModalityConfig is BaseModalityConfig
    assert modality_base.ModalityConfig is BaseModalityConfig


def test_base_modality_config_is_frozen_and_supports_from_dict() -> None:
    """BaseModalityConfig must follow the frozen typed config standard."""
    config = BaseModalityConfig.from_dict({"name": "runtime"})

    assert config.name == "runtime"
    assert config.normalize is True
    assert config.augmentation is False
    assert config.batch_size == 32

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "changed"


@pytest.mark.parametrize(
    ("config_class", "payload"),
    [
        (
            AudioModalityConfig,
            {
                "representation": "mel_spectrogram",
                "sample_rate": 22050,
                "duration": 1.5,
            },
        ),
        (
            ImageModalityConfig,
            {
                "representation": "rgba",
                "height": 128,
            },
        ),
        (
            TimeseriesModalityConfig,
            {
                "name": "timeseries",
                "sequence_length": 32,
                "num_features": 2,
                "univariate": False,
                "multi_scale_factors": [1, 2, 8],
            },
        ),
        (
            TabularModalityConfig,
            {
                "name": "tabular",
                "num_features": 2,
                "numerical_features": ["age"],
                "binary_features": ["flag"],
            },
        ),
        (
            MultiModalModalityConfig,
            {
                "name": "multi_modal",
                "modalities": ["image", "text"],
                "fusion_strategy": "concatenate",
            },
        ),
    ],
)
def test_modality_runtime_configs_are_typed_frozen_documents(
    config_class: type, payload: dict[str, object]
) -> None:
    """Runtime modality configs must support typed dict loading and immutability."""
    config = config_class.from_dict(payload)

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, next(iter(dataclasses.asdict(config))), None)
