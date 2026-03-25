"""Focused tests for ExperimentConfig's model-creation boundary."""

import pytest

from artifex.generative_models.core.configuration import (
    DataConfig,
    ExperimentConfig,
    OptimizerConfig,
    TrainingConfig,
    VAEConfig,
)


def _training_config() -> TrainingConfig:
    optimizer = OptimizerConfig(name="adam", optimizer_type="adam", learning_rate=1e-3)
    return TrainingConfig(name="training", optimizer=optimizer)


def _data_config() -> DataConfig:
    return DataConfig(name="mnist", dataset_name="mnist")


def test_experiment_config_from_dict_materializes_family_specific_model_config() -> None:
    """ExperimentConfig should keep the nested model config on the typed family surface."""
    payload = {
        "name": "vae_experiment",
        "model_cfg": {
            "name": "vae_model",
            "encoder": {
                "name": "encoder",
                "input_shape": [28, 28, 1],
                "latent_dim": 16,
                "hidden_dims": [64, 32],
                "activation": "relu",
            },
            "decoder": {
                "name": "decoder",
                "latent_dim": 16,
                "output_shape": [28, 28, 1],
                "hidden_dims": [32, 64],
                "activation": "relu",
            },
        },
        "training_cfg": {
            "name": "training",
            "optimizer": {
                "name": "adam",
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
            },
        },
        "data_cfg": {
            "name": "mnist",
            "dataset_name": "mnist",
        },
    }

    config = ExperimentConfig.from_dict(payload)

    assert isinstance(config.model_cfg, VAEConfig)
    assert config.model_cfg.encoder.latent_dim == 16


def test_experiment_config_rejects_legacy_generic_model_payload() -> None:
    """ExperimentConfig should not revive the removed ModelConfig-style contract."""
    payload = {
        "name": "legacy_experiment",
        "model_cfg": {
            "name": "legacy_model",
            "model_class": "artifex.generative_models.models.vae.VAE",
            "input_dim": [28, 28, 1],
        },
        "training_cfg": {
            "name": "training",
            "optimizer": {
                "name": "adam",
                "optimizer_type": "adam",
                "learning_rate": 1e-3,
            },
        },
        "data_cfg": {
            "name": "mnist",
            "dataset_name": "mnist",
        },
    }

    with pytest.raises((TypeError, ValueError), match="model"):
        ExperimentConfig.from_dict(payload)


def test_experiment_config_rejects_non_model_runtime_configs() -> None:
    """ExperimentConfig.model_cfg must be one of the supported model families."""
    with pytest.raises(TypeError, match="model_cfg"):
        ExperimentConfig(
            name="invalid_experiment",
            model_cfg=_training_config(),
            training_cfg=_training_config(),
            data_cfg=_data_config(),
        )
